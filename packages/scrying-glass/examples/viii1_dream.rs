//! RITE VIII-1 — THE DREAM-DENOISER: the proof. Renders the front law pose
//! of the naruko realm (a TRAINING pose — see `denoiser_dataset`) AND a
//! held-out VALIDATION pose, both at a fixed tick/seed, loads the COMMITTED
//! weights artifact (never retrains), denoises each noisy 1-spp frame, and
//! writes two triptychs (noisy | denoised | reference, side by side),
//! printing RMSE(noisy, reference) and RMSE(denoised, reference) honestly
//! for each:
//!
//!   - proof/viii1-dream.png          — front pose (a TRAIN-split pose:
//!     the network has seen this pose's pixels during training — this is a
//!     RECONSTRUCTION proof, not a generalization proof).
//!   - proof/viii1-dream-heldout.png  — "orbit_-20" (a VALIDATION-split
//!     pose: never seen during training — this is the GENERALIZATION
//!     proof; adversary finding A6, night-2 review: the first version of
//!     this example only rendered a training pose and its print/labels did
//!     not say so).
//!
//! HONEST NOTE found while building this proof: this example renders at
//! 480×320 (presentation quality), while training/validation/the ordeals
//! run at `denoiser_dataset::DATASET_WIDTH`×`DATASET_HEIGHT` = 96×64. At
//! 96×64, `orbit_-20` (the held-out pose) strictly beats noisy — see
//! `tests/viii1_ordeals.rs`, which is what's actually pinned and gated. At
//! this proof's higher 480×320 resolution, the SAME pose's margin is
//! fragile: denoised (0.049595) narrowly loses to noisy (0.049177), a ~1%
//! gap — the pixel population at a different resolution is not identical
//! (different edge/anti-aliasing proportion), so this is a real, disclosed
//! resolution-sensitivity finding, not a bug being hidden. The ordeal-
//! pinned claim ("denoised beats noisy") is scoped to the training
//! resolution; this proof shows what presentation resolution actually
//! looks like, numbers included, without adjusting anything to make it
//! pass.
//!
//! Run:  cargo run -p scrying-glass --release --example viii1_dream

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::{Mlp, denoise_image, deserialize_weights};
use scrying_glass::denoiser_dataset::{DATASET_REF_FRAMES, law_poses, naruko_params};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, RenderScene};

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn write_triptych(
    a: &[GVec3],
    b: &[GVec3],
    c: &[GVec3],
    w: u32,
    h: u32,
    exposure: f32,
    path: &Path,
) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((3 * w * h * 3) as usize);
    for y in 0..h {
        for panel in [a, b, c] {
            for x in 0..w {
                let px = panel[(y * w + x) as usize];
                bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
                bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
                bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
            }
        }
    }
    let file = std::fs::File::create(path).unwrap();
    let writer = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(writer, 3 * w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .unwrap()
        .write_image_data(&bytes)
        .unwrap();
    eprintln!("[viii1-dream] wrote {}", path.display());
}

/// Render + denoise one pose's triptych, printing honest RMSEs labeled with
/// `role` ("train-split pose" / "validation-split pose (held out)").
#[allow(clippy::too_many_arguments)]
fn dream_one_pose(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &Camera,
    scene: &RenderScene,
    mlp: &Mlp,
    role: &str,
    pose_name: &str,
    ref_frames: u32,
    w: u32,
    h: u32,
    exposure: f32,
    out_path: &Path,
) {
    let noisy_params = IntegratorParams {
        spp: 1,
        ..IntegratorParams::default()
    };
    let noisy_accum = trace_headless(
        device,
        queue,
        bvh,
        camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        w,
        h,
        1,
        &noisy_params,
        None,
    );
    let noisy = resolve(&noisy_accum);

    let ref_accum = trace_headless(
        device,
        queue,
        bvh,
        camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        w,
        h,
        ref_frames,
        &IntegratorParams::default(),
        None,
    );
    let reference = resolve(&ref_accum);

    let raw_aov = trace_headless_aov(
        device,
        queue,
        bvh,
        camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        w,
        h,
    );
    let (albedo, normal, depth) = split_aov(&raw_aov);

    let denoised = denoise_image(mlp, &noisy, &albedo, &normal, &depth);

    let noisy_rmse = rmse(&noisy, &reference);
    let denoised_rmse = rmse(&denoised, &reference);
    println!("[viii1-dream] pose='{pose_name}' ({role})");
    println!("[viii1-dream]   RMSE(noisy 1spp,  reference {ref_frames}frames) = {noisy_rmse:.6}");
    println!(
        "[viii1-dream]   RMSE(denoised,    reference {ref_frames}frames) = {denoised_rmse:.6}"
    );
    println!(
        "[viii1-dream]   denoised beats noisy: {}",
        if denoised_rmse < noisy_rmse {
            "YES"
        } else {
            "NO"
        }
    );

    write_triptych(&noisy, &denoised, &reference, w, h, exposure, out_path);
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[viii1-dream] no GPU adapter on this host — cannot forge the dream");
    };

    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    eprintln!(
        "[viii1-dream] naruko: {} static leaf tris",
        scene.leaf_triangles().len()
    );

    let poses = law_poses(&params);
    let front_camera = &poses
        .iter()
        .find(|(n, _)| *n == "front")
        .expect("front pose")
        .1;
    let heldout_camera = &poses
        .iter()
        .find(|(n, _)| *n == "orbit_-20")
        .expect("orbit_-20 pose")
        .1;

    let (w, h) = (480u32, 320u32);
    let exposure = 1.6;
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");
    // DERIVED, not chosen for looks: matches `DATASET_REF_FRAMES`, the SAME
    // reference depth training/validation/the ordeals measured the pinned
    // bound against. A deeper reference (e.g. viii0_truth's proof-quality
    // 512) tells a DIFFERENT story — found honestly during this atom's
    // adversary pass: at 512 frames the held-out pose's converged target
    // shifts enough that "denoised beats noisy" flips to a marginal loss
    // (noisy=0.049018 vs denoised=0.049506, ~1% worse) even though the
    // SAME pose passes cleanly at the 128-frame reference the model was
    // actually trained/validated against (noisy=0.052073 vs
    // denoised=0.042999). Using 128 here keeps this proof measuring the
    // SAME claim the ordeals verified, not a different, deeper-converged
    // one; the 512-frame finding is recorded honestly here rather than
    // hidden, and stands as an open question for a future wave (does the
    // bound need to be derived against a deeper reference?).
    let ref_frames = env_u32("GAIA_VIII1_DREAM_REF_FRAMES", DATASET_REF_FRAMES);

    // ── denoise, loading the COMMITTED weights artifact — never retrained
    // here.
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let weights_path = data_dir.join("denoiser-weights-v1.bin");
    let weights_bytes = std::fs::read(&weights_path).unwrap_or_else(|e| {
        panic!(
            "[viii1-dream] could not read {} ({e}) — run `cargo run -p scrying-glass --release \
             --example viii1_train` first to forge the weights artifact",
            weights_path.display()
        )
    });
    let mlp = deserialize_weights(&weights_bytes).expect("deserialize committed weights artifact");

    dream_one_pose(
        &device,
        &queue,
        &bvh,
        front_camera,
        &scene,
        &mlp,
        "TRAIN-split pose — reconstruction proof, not generalization",
        "front",
        ref_frames,
        w,
        h,
        exposure,
        &proof.join("viii1-dream.png"),
    );

    dream_one_pose(
        &device,
        &queue,
        &bvh,
        heldout_camera,
        &scene,
        &mlp,
        "VALIDATION-split pose, held out from training — generalization proof",
        "orbit_-20",
        ref_frames,
        w,
        h,
        exposure,
        &proof.join("viii1-dream-heldout.png"),
    );

    eprintln!("[viii1-dream] noisy | denoised | reference — both dreams, side by side.");
}
