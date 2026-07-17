//! RITE VIII-1 — THE DREAM-DENOISER: the proof. Renders the front law pose
//! of the naruko realm (the SAME "front" pose `viii0_truth.rs`/`perf_audit.rs`
//! use) at a fixed tick/seed, loads the COMMITTED weights artifact (never
//! retrains), denoises the noisy 1-spp frame, and writes
//! proof/viii1-dream.png — noisy | denoised | reference, side by side —
//! printing RMSE(noisy, reference) and RMSE(denoised, reference) honestly.
//!
//! Run:  cargo run -p scrying-glass --release --example viii1_dream

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::{denoise_image, deserialize_weights};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

/// Naruko authoring dials — the SAME front pose `viii0_truth.rs` uses.
fn naruko_params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: 60.0,
        near: 0.1,
        far: 4_000.0,
        sky_top: "#20152f".into(),
        sky_horizon: "#9a627d".into(),
        mesh_color: "#9aa0a6".into(),
        radial_segments: 24,
        camera_position: [0.0, 2.0, 22.0],
        camera_yaw: 0.0,
        camera_pitch: 0.0,
        cluster_error_threshold: 1.0,
        tick_dt: 1.0 / 60.0,
        sun: SunDefaults {
            sun_color: "#ffe2b0".into(),
            sun_intensity: 1.1,
            sun_position: [60.0, 90.0, 30.0],
            ambient_intensity: 0.32,
        },
        emission_intensity: 2.5,
    }
}

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
        "[viii1-dream] naruko front pose: {} static leaf tris",
        scene.leaf_triangles().len()
    );

    let camera = Camera {
        eye: GVec3::from_array(params.camera_position),
        yaw: params.camera_yaw,
        pitch: params.camera_pitch,
        fov_y_radians: params.fov_y_degrees.to_radians(),
        near: params.near,
        far: params.far,
    };

    let (w, h) = (480u32, 320u32);
    let exposure = 1.6;
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");

    // ── noisy: 1 frame at spp 1 — the minimal achievable accumulation.
    let noisy_params = IntegratorParams {
        spp: 1,
        ..IntegratorParams::default()
    };
    let noisy_accum = trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
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

    // ── reference: converged N-frame accumulation (matches viii0_truth's
    // default argument for depth of convergence at this same scene).
    let ref_frames = env_u32("GAIA_VIII1_DREAM_REF_FRAMES", 512);
    let ref_accum = trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
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

    // ── current-frame AOVs (albedo/normal/depth) for the SAME pose.
    let raw_aov = trace_headless_aov(
        &device,
        &queue,
        &bvh,
        &camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        w,
        h,
    );
    let (albedo, normal, depth) = split_aov(&raw_aov);

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
    let denoised = denoise_image(&mlp, &noisy, &albedo, &normal, &depth);

    let noisy_rmse = rmse(&noisy, &reference);
    let denoised_rmse = rmse(&denoised, &reference);
    println!("[viii1-dream] RMSE(noisy 1spp, reference {ref_frames}frames)     = {noisy_rmse:.6}");
    println!(
        "[viii1-dream] RMSE(denoised,   reference {ref_frames}frames)     = {denoised_rmse:.6}"
    );
    println!(
        "[viii1-dream] denoised beats noisy at this pose: {}",
        if denoised_rmse < noisy_rmse {
            "YES"
        } else {
            "NO"
        }
    );

    write_triptych(
        &noisy,
        &denoised,
        &reference,
        w,
        h,
        exposure,
        &proof.join("viii1-dream.png"),
    );
    eprintln!("[viii1-dream] noisy | denoised | reference — the dream, side by side.");
}
