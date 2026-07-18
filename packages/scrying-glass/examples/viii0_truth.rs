//! RITE VIII-0 — THE NOISE AND THE TRUTH: the reference oracle. Before any
//! net lands (that is VIII-1), this is the baseline every later denoiser
//! claim measures against — see docs/proposals/RITE-VIII-THE-DREAM-DENOISER.md
//! §VIII-0.
//!
//! Renders the merged Naruko realm at a fixed law pose (the "front" authored
//! camera — the same front-pose scaffolding `perf_audit.rs`/`composed_coexist.rs`
//! use):
//!
//!   (a) "noisy"     — 1 accumulation frame at spp 1 (the minimal achievable
//!                     sample count: 1 frame × 1 sample/pixel).
//!   (b) "reference" — GAIA_VIII0_REF_FRAMES accumulation frames (default
//!                     below, argued in a comment at its definition),
//!                     converged. Printed alongside it: RMSE(ref_N, ref_N/2)
//!                     — the diminishing-error evidence, measured, not
//!                     asserted.
//!
//! Also exports AOV buffers (albedo/normal/depth of the primary hit) and
//! dumps them as PNGs, and writes proof/viii0-truth.png (noisy | reference,
//! side by side), printing RMSE(noisy, reference) honestly.
//!
//! Run:  cargo run -p scrying-glass --release --example viii0_truth
//!       GAIA_VIII0_REF_FRAMES=1024 cargo run -p scrying-glass --release --example viii0_truth

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

/// Naruko authoring dials — the SAME front pose `perf_audit.rs`/
/// `composed_coexist.rs` render from (reused verbatim, not reinvented).
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

/// Reference frame count, env-parameterized (`GAIA_VIII0_REF_FRAMES`), never
/// a bare literal at the call site. DEFAULT ARGUED: path-traced Monte Carlo
/// error falls off as 1/sqrt(samples); doubling frames halves RMSE against
/// the true (infinite-sample) image in expectation. 512 frames × spp 2
/// (`IntegratorParams::default().spp`) = 1024 samples/pixel is deep enough
/// that the remaining residual is dominated by the render's own numerical
/// floor (f32 accumulation, GGX importance-sampling variance in specular
/// lobes) rather than un-converged noise — while still finishing in a few
/// seconds at the resolution below on a discrete GPU, so this proof stays
/// something a builder actually runs, not a multi-minute CI tax. The
/// printed RMSE(ref_N, ref_N/2) is the honest evidence for THIS run, not an
/// assumption baked into the choice of 512.
fn default_ref_frames() -> u32 {
    512
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

/// Write a radiance image (HDR linear, tonemapped through sRGB display
/// encoding + exposure) to disk.
fn write_radiance_png(img: &[GVec3], w: u32, h: u32, exposure: f32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in img {
        bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
    }
    let file = std::fs::File::create(path).unwrap();
    let writer = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(writer, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .unwrap()
        .write_image_data(&bytes)
        .unwrap();
    eprintln!("[viii0] wrote {}", path.display());
}

/// Write a pre-normalized [0,1] DATA image (not radiance — no sRGB display
/// gamma; these are AOV values, not light) directly to 8-bit.
fn write_data_png(img: &[GVec3], w: u32, h: u32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in img {
        bytes.push((px.x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        bytes.push((px.y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        bytes.push((px.z.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
    }
    let file = std::fs::File::create(path).unwrap();
    let writer = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(writer, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .unwrap()
        .write_image_data(&bytes)
        .unwrap();
    eprintln!("[viii0] wrote {}", path.display());
}

/// Stitch two equal-sized radiance images side by side (left | right) and
/// write the result (display-encoded).
fn write_side_by_side(left: &[GVec3], right: &[GVec3], w: u32, h: u32, exposure: f32, path: &Path) {
    let mut stitched = vec![GVec3::ZERO; (2 * w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            stitched[(y * 2 * w + x) as usize] = left[(y * w + x) as usize];
            stitched[(y * 2 * w + w + x) as usize] = right[(y * w + x) as usize];
        }
    }
    write_radiance_png(&stitched, 2 * w, h, exposure, path);
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[viii0] no GPU adapter on this host — cannot forge the reference oracle");
    };

    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");

    // The fixed law pose: the front authored camera — the exact same "front"
    // pose `perf_audit.rs` builds (params.camera_position/yaw/pitch as
    // authored) — over the realm's static leaf triangles only, no ticking,
    // so the pose is trivially reproducible from (seed, coords) alone with
    // no physics/gait replay dependency.
    let camera = Camera {
        eye: GVec3::from_array(params.camera_position),
        yaw: params.camera_yaw,
        pitch: params.camera_pitch,
        fov_y_radians: params.fov_y_degrees.to_radians(),
        near: params.near,
        far: params.far,
    };

    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    eprintln!(
        "[viii0] naruko front pose: {} static leaf tris",
        scene.leaf_triangles().len()
    );

    let (w, h) = (480u32, 320u32);
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");
    let exposure = 1.6;

    // ── (a) noisy: 1 frame at spp 1 — the minimal achievable accumulation.
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

    // ── (b) reference: N-frame converged accumulation (fixed seed, matching
    // the existing 0x5eed convention via IntegratorParams::default()).
    let ref_frames = env_u32("GAIA_VIII0_REF_FRAMES", default_ref_frames());
    let ref_params = IntegratorParams::default();
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
        &ref_params,
        None,
    );
    let reference = resolve(&ref_accum);

    // Convergence evidence: RMSE(ref_N, ref_N/2) — diminishing error
    // DEMONSTRATED, not asserted.
    let half_frames = (ref_frames / 2).max(1);
    let half_accum = trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        w,
        h,
        half_frames,
        &ref_params,
        None,
    );
    let half = resolve(&half_accum);
    let convergence_rmse = rmse(&reference, &half);
    let noisy_vs_ref_rmse = rmse(&noisy, &reference);

    println!("[viii0] RMSE(noisy 1spp, reference {ref_frames}frames) = {noisy_vs_ref_rmse:.6}");
    println!(
        "[viii0] RMSE(reference {ref_frames}frames, reference {half_frames}frames) = {convergence_rmse:.6}  (convergence evidence)"
    );

    write_side_by_side(
        &noisy,
        &reference,
        w,
        h,
        exposure,
        &proof.join("viii0-truth.png"),
    );

    // ── AOV export + dumps ──────────────────────────────────────────────
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
    let (albedo, normal_raw, depth_raw) = split_aov(&raw_aov);

    // Normal AOV: remapped from [-1,1] to [0,1] for display (stated here and
    // in the file name — the raw signed values are what `split_aov` returns,
    // this remap is display-only).
    let normal_display: Vec<GVec3> = normal_raw
        .iter()
        .map(|n| *n * 0.5 + GVec3::splat(0.5))
        .collect();

    // Depth AOV: normalized by ITS OWN max in this frame (stated here and in
    // the file name) — not a fixed world-space scale, so it visualizes at
    // full contrast regardless of scene extent.
    let max_depth = depth_raw.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let depth_display: Vec<GVec3> = depth_raw
        .iter()
        .map(|d| GVec3::splat(d / max_depth))
        .collect();

    write_data_png(&albedo, w, h, &proof.join("viii0-aov-albedo.png"));
    write_data_png(&normal_display, w, h, &proof.join("viii0-aov-normal.png"));
    write_data_png(&depth_display, w, h, &proof.join("viii0-aov-depth.png"));

    eprintln!("[viii0] the noise and the truth, side by side — the baseline is set.");
}
