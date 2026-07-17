//! RITE VIII-3 — THE UPSCALER: the proof. Renders a held-out VALIDATION pose
//! of the naruko realm at a PRESENTATION resolution family (low internal →
//! native via `UPSCALE_SCALE`), loads the COMMITTED weights (never retrains),
//! and writes one triptych — NAIVE BILINEAR | NEURAL | TRUTH (converged
//! reference) — printing RMSE(bilinear, truth) and RMSE(neural, truth)
//! honestly, plus the beats-bilinear margin.
//!
//!   - proof/viii3-upscale.png  — "orbit_-20" (a VALIDATION-split pose,
//!     never seen in training — the GENERALIZATION proof).
//!
//! The presentation resolution is a DIFFERENT pixel population than the
//! 96×64 the pinned bound/ordeals measure (VIII-1 disclosed the same
//! resolution-sensitivity honestly); the numbers printed here are what
//! presentation actually looks like, unadjusted.
//!
//! Run:  cargo run -p scrying-glass --release --example viii3_upscale

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::RenderScene;
use scrying_glass::upscaler::{bilinear_upsample, deserialize_weights, upscale_image};
use scrying_glass::upscaler_dataset::{
    DATASET_REF_FRAMES, UPSCALE_SCALE, law_poses, naruko_params, target_dims,
};

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

fn write_triptych(a: &[GVec3], b: &[GVec3], c: &[GVec3], w: u32, h: u32, exposure: f32, path: &Path) {
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
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("[viii3-upscale] wrote {}", path.display());
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[viii3-upscale] no GPU adapter on this host — cannot forge the proof");
    };

    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());

    let poses = law_poses(&params);
    let camera = &poses
        .iter()
        .find(|(n, _)| *n == "orbit_-20")
        .expect("orbit_-20 held-out pose")
        .1;

    // Presentation resolution family: a low internal res × UPSCALE_SCALE,
    // derived (never a frozen target). Default low 240×160 → native 480×320.
    let low_w = env_u32("GAIA_VIII3_LOW_W", 240);
    let low_h = env_u32("GAIA_VIII3_LOW_H", 160);
    let (target_w, target_h) = target_dims(low_w, low_h, UPSCALE_SCALE);
    let ref_frames = env_u32("GAIA_VIII3_REF_FRAMES", DATASET_REF_FRAMES);
    let exposure = 1.6;

    // LOW internal noisy frame.
    let noisy_params = IntegratorParams {
        spp: 1,
        ..IntegratorParams::default()
    };
    let low_noisy = resolve(&trace_headless(
        &device, &queue, &bvh, camera, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h,
        1, &noisy_params, None,
    ));

    // TARGET AOVs + TARGET converged reference (truth).
    let raw_aov = trace_headless_aov(
        &device, &queue, &bvh, camera, &scene.sun, scene.sky_top, scene.sky_horizon, target_w,
        target_h,
    );
    let (hi_albedo, hi_normal, hi_depth) = split_aov(&raw_aov);
    let reference = resolve(&trace_headless(
        &device, &queue, &bvh, camera, &scene.sun, scene.sky_top, scene.sky_horizon, target_w,
        target_h, ref_frames, &IntegratorParams::default(), None,
    ));

    // Load COMMITTED weights (never retrained here).
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let weights_bytes = std::fs::read(data_dir.join("upscaler-weights-v1.bin")).unwrap_or_else(|e| {
        panic!(
            "[viii3-upscale] could not read upscaler-weights-v1.bin ({e}) — run \
             `cargo run -p scrying-glass --release --example viii3_train` first"
        )
    });
    let mlp = deserialize_weights(&weights_bytes).expect("deserialize committed weights");

    let bilinear = bilinear_upsample(&low_noisy, low_w, low_h, target_w, target_h);
    let neural = upscale_image(
        &mlp, &low_noisy, low_w, low_h, &hi_albedo, &hi_normal, &hi_depth, target_w, target_h,
    );

    let bilinear_rmse = rmse(&bilinear, &reference);
    let neural_rmse = rmse(&neural, &reference);
    println!("[viii3-upscale] pose='orbit_-20' (VALIDATION-split, held out) — GENERALIZATION proof");
    println!("[viii3-upscale]   low internal = {low_w}x{low_h}, native target = {target_w}x{target_h}, scale = {UPSCALE_SCALE}");
    println!("[viii3-upscale]   RMSE(naive bilinear, truth {ref_frames}f) = {bilinear_rmse:.6}");
    println!("[viii3-upscale]   RMSE(neural upscale,  truth {ref_frames}f) = {neural_rmse:.6}");
    println!(
        "[viii3-upscale]   neural beats bilinear: {} (margin {:.6})",
        if neural_rmse < bilinear_rmse { "YES" } else { "NO" },
        bilinear_rmse - neural_rmse
    );

    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof/viii3-upscale.png");
    write_triptych(&bilinear, &neural, &reference, target_w, target_h, exposure, &proof);
    eprintln!("[viii3-upscale] naive bilinear | neural | truth — side by side.");
}
