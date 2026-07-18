//! TEACHER_SHOT — converged ground-truth still renderer (lab machinery).
//!
//! Reuses the trainer's long-accumulation reference path (rdirect_train_v2's
//! `reference`) as a standalone binary: load a world dir, place a fixed camera,
//! trace to convergence, write a tonemapped 640×480 PNG. Checkpoint-friendly —
//! each invocation runs GAIA_ORDEAL_FRAMES more accumulation frames (a fresh
//! independent seed) and MERGES into a persisted accum buffer, so a 300s command
//! cap never loses progress. NEVER the app present — trainer/ordeal side only.
//!
//! IRON params (env, never hardcoded at the site):
//!   GAIA_ORDEAL_SCENE      ocean | labyrinth
//!   GAIA_ORDEAL_OUT        output PNG path
//!   GAIA_ORDEAL_STATE      accum checkpoint path (default <OUT>.accum)
//!   GAIA_ORDEAL_W / _H     640 / 480
//!   GAIA_ORDEAL_SPP        samples/pixel/frame (default 4)
//!   GAIA_ORDEAL_FRAMES     accumulation frames THIS invocation (default 48)
//!   GAIA_ORDEAL_MAX_BOUNCES  hard path-length cap (IRON; e.g. 8 ocean / 1024 lab)
//!   GAIA_ORDEAL_RR_START   bounce index russian-roulette begins (default =
//!                          max_bounces → RR OFF: every path runs full depth)
//!   GAIA_ORDEAL_EMISSIVE   emission_intensity (lantern/beam glow scale)
//!   GAIA_NATIVE_ORDEAL_MIRROR_R  perfect-mirror reflectance override (default 0.999)
//!   GAIA_ORDEAL_EXPOSURE   tonemap exposure (default 1.0)
//!   GAIA_ORDEAL_SEED_BASE  master seed (default 0x5eed)
//!
//! Run: GAIA_ORDEAL_SCENE=ocean cargo run -p scrying-glass --release --example teacher_shot

use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, trace_headless,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene, SceneParameters, SunDefaults};

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn write_png(mean: &[GVec3], w: u32, h: u32, exposure: f32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in mean {
        bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
    }
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("[teacher] wrote {}", path.display());
}

/// Persisted accumulation: header (w,h,total_frames,spp) + raw [f32;4] sum/count.
fn load_state(path: &Path, w: u32, h: u32) -> (Vec<[f32; 4]>, u32) {
    let n = (w * h) as usize;
    let Ok(raw) = std::fs::read(path) else { return (vec![[0.0; 4]; n], 0); };
    if raw.len() < 16 + n * 16 { return (vec![[0.0; 4]; n], 0); }
    let u = |i: usize| u32::from_le_bytes([raw[i], raw[i + 1], raw[i + 2], raw[i + 3]]);
    if u(0) != w || u(4) != h { return (vec![[0.0; 4]; n], 0); }
    let total_frames = u(8);
    let mut buf = vec![[0.0f32; 4]; n];
    for (i, px) in buf.iter_mut().enumerate() {
        for c in 0..4 {
            let o = 16 + i * 16 + c * 4;
            px[c] = f32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);
        }
    }
    (buf, total_frames)
}

fn save_state(path: &Path, w: u32, h: u32, total_frames: u32, spp: u32, buf: &[[f32; 4]]) {
    if let Some(dir) = path.parent() { std::fs::create_dir_all(dir).ok(); }
    let mut out = Vec::with_capacity(16 + buf.len() * 16);
    for v in [w, h, total_frames, spp] { out.extend_from_slice(&v.to_le_bytes()); }
    for px in buf { for c in 0..4 { out.extend_from_slice(&px[c].to_le_bytes()); } }
    std::fs::write(path, out).unwrap();
}

struct Setup {
    world: &'static str,
    camera: Camera,
    emission_intensity: f32,
    max_bounces: u32,
    exposure: f32,
}

fn main() {
    let scene_name = env_str("GAIA_ORDEAL_SCENE", "ocean");
    let w = env_u32("GAIA_ORDEAL_W", 640);
    let h = env_u32("GAIA_ORDEAL_H", 480);
    let spp = env_u32("GAIA_ORDEAL_SPP", 4);
    let frames = env_u32("GAIA_ORDEAL_FRAMES", 48);
    let seed_base = env_u32("GAIA_ORDEAL_SEED_BASE", 0x5eed);
    let mirror_r = env_f32("GAIA_NATIVE_ORDEAL_MIRROR_R", 0.999);
    // Per-scene FOV default (labyrinth wants a tight tunnel crop).
    let fov_default = if scene_name == "labyrinth" { 40.0 } else { 58.0 };
    let fov = env_f32("GAIA_ORDEAL_FOV", fov_default);

    let deg = std::f32::consts::PI / 180.0;
    let mk = |eye: [f32; 3], yaw: f32, pitch: f32| Camera {
        eye: GVec3::new(
            env_f32("GAIA_ORDEAL_EYE_X", eye[0]),
            env_f32("GAIA_ORDEAL_EYE_Y", eye[1]),
            env_f32("GAIA_ORDEAL_EYE_Z", eye[2]),
        ),
        yaw: env_f32("GAIA_ORDEAL_YAW", yaw),
        pitch: env_f32("GAIA_ORDEAL_PITCH", pitch),
        fov_y_radians: fov * deg,
        near: 0.05,
        far: 6000.0,
    };

    let setup = match scene_name.as_str() {
        "ocean" => Setup {
            world: "ordeal-ocean",
            camera: mk([0.0, 3.0, 34.0], 0.0, -0.145),
            emission_intensity: env_f32("GAIA_ORDEAL_EMISSIVE", 7.0),
            max_bounces: env_u32("GAIA_ORDEAL_MAX_BOUNCES", 8),
            exposure: env_f32("GAIA_ORDEAL_EXPOSURE", 1.0),
        },
        "labyrinth" => Setup {
            world: "ordeal-labyrinth",
            camera: mk([0.0, 4.0, 3.4], 0.055, -0.008),
            emission_intensity: env_f32("GAIA_ORDEAL_EMISSIVE", 10.0),
            max_bounces: env_u32("GAIA_ORDEAL_MAX_BOUNCES", 1024),
            exposure: env_f32("GAIA_ORDEAL_EXPOSURE", 0.7),
        },
        other => panic!("[teacher] unknown GAIA_ORDEAL_SCENE '{other}' (ocean|labyrinth)"),
    };
    // RR OFF by default (rr_start = max_bounces): every path runs to full depth
    // so the mirror tunnel keeps every echo — unbiased, just zero-variance depth.
    let rr_start = env_u32("GAIA_ORDEAL_RR_START", setup.max_bounces);

    let exposure = setup.exposure;
    let out = env_str(
        "GAIA_ORDEAL_OUT",
        &format!("proof/ordeal-light/{scene_name}.png"),
    );
    let out = PathBuf::from(&out);
    let state = env_str(
        "GAIA_ORDEAL_STATE",
        &format!("{}.accum", out.display()),
    );
    let state = PathBuf::from(&state);

    let Some((device, queue)) = headless_device() else {
        panic!("[teacher] no GPU adapter — cannot forge the reference");
    };

    // Dark base parameters — the scene `env` overrides sky/sun; sun off.
    let params = SceneParameters {
        fov_y_degrees: fov,
        near: 0.05,
        far: 6000.0,
        sky_top: "#000000".into(),
        sky_horizon: "#000000".into(),
        mesh_color: "#9aa0a6".into(),
        radial_segments: 40,
        camera_position: setup.camera.eye.to_array(),
        camera_yaw: setup.camera.yaw,
        camera_pitch: setup.camera.pitch,
        cluster_error_threshold: 1.0,
        tick_dt: 1.0 / 60.0,
        sun: SunDefaults {
            sun_color: "#ffffff".into(),
            sun_intensity: 0.0,
            sun_position: [0.0, 90.0, 30.0],
            ambient_intensity: 0.0,
        },
        emission_intensity: setup.emission_intensity,
    };

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../worlds")
        .join(setup.world);
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world)
        .unwrap_or_else(|e| panic!("[teacher] load {}: {e}", setup.world));
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params)
        .expect("[teacher] scene");

    // Perfect-mirror reflectance override (IRON): tris flagged metallic≈1 +
    // roughness≤MIRROR_ROUGHNESS get albedo = (R,R,R) so throughput *= R each
    // delta bounce — 0.999^1000 ≈ 0.37 stays visibly alive.
    let mut tris: Vec<LeafTriangle> = scene.leaf_triangles();
    let mut mirror_tris = 0u64;
    for t in tris.iter_mut() {
        if t.metallic >= 0.999 && t.roughness <= 1.0e-3 {
            t.albedo = [mirror_r, mirror_r, mirror_r];
            mirror_tris += 1;
        }
    }
    eprintln!(
        "[teacher] scene='{}' tris={} mirror_tris={} (R={mirror_r}) emissive_scale={}",
        scene_name, tris.len(), mirror_tris, setup.emission_intensity
    );

    let bvh = Bvh::build(&tris, &BvhParams::default());

    // Merge into the persisted accumulation (sum in xyz, count in w).
    let (mut combined, prior_frames) = load_state(&state, w, h);
    let integ = IntegratorParams {
        spp,
        max_bounces: setup.max_bounces,
        rr_start,
        seed: seed_base ^ prior_frames.wrapping_mul(0x9e3779b9),
        eps: 1.0e-3,
    };
    eprintln!(
        "[teacher] {w}x{h} spp={spp} frames_this={frames} max_bounces={} rr_start={rr_start} prior_frames={prior_frames}",
        setup.max_bounces
    );

    let t0 = Instant::now();
    let chunk = trace_headless(
        &device, &queue, &bvh, &setup.camera, &scene.sun,
        scene.sky_top, scene.sky_horizon, w, h, frames, &integ, None,
    );
    let secs = t0.elapsed().as_secs_f64();

    for (c, a) in combined.iter_mut().zip(chunk.iter()) {
        c[0] += a[0];
        c[1] += a[1];
        c[2] += a[2];
        c[3] += a[3];
    }
    let total_frames = prior_frames + frames;
    save_state(&state, w, h, total_frames, spp, &combined);

    let mean = resolve(&combined);
    write_png(&mean, w, h, exposure, &out);

    let total_spp = total_frames * spp;
    println!(
        "[teacher] scene={scene_name} out={} | frames_this={frames} ({secs:.1}s) | TOTAL frames={total_frames} spp={total_spp} | max_bounces={} rr_start={rr_start} mirror_R={mirror_r} mirror_tris={mirror_tris}",
        out.display(), setup.max_bounces
    );
}
