//! RITE VIII-1 — THE DREAM-DENOISER: forge-time training. Generates
//! noisy-1spp / converged-N-frame reference pairs from OUR integrator across
//! a documented set of poses of the naruko realm, trains the per-pixel MLP
//! (`scrying_glass::denoiser`) on a TRAIN split, evaluates per-frame RMSE on
//! a held-out VALIDATION split (whole poses, never pixels — the bound must
//! be per-frame), and writes:
//!
//!   - packages/scrying-glass/data/denoiser-weights-v1.bin   (the weights)
//!   - packages/scrying-glass/data/denoiser-weights-v1.provenance.json
//!     (dataset hash, config, train/val RMSE table, the DERIVED pinned
//!     bound = worst validation frame's RMSE, sha256 of the weights)
//!
//! DATASET SCOPE (proposal OPEN 10 — documented honestly, not hidden): five
//! poses of the merged naruko realm, all static geometry (no ticking — the
//! realm's leaf triangles as authored, matching `viii0_truth`'s scaffolding
//! choice to keep the dataset trivially reproducible from (seed, coords)):
//!   - "front" — the law front pose (`naruko_params` camera, the SAME pose
//!     `perf_audit`/`viii0_truth` use).
//!   - "wide"  — the composed-coexist three-quarter sea-side shot.
//!   - "orbit_+20"/"orbit_-20"/"orbit_+40" — three DERIVED orbit views: the
//!     front eye rotated by the given yaw (degrees) around the front pose's
//!     look-at point, same radius, added for viewpoint diversity beyond the
//!     two authored poses (a per-pixel MLP needs varied surfaces/angles/
//!     lighting incidence to generalize, not just varied camera FRAMING of
//!     the same two shots).
//!
//! TRAIN = {front, wide, orbit_+20}. VALIDATION = {orbit_-20, orbit_+40}
//! (whole, held-out poses — never seen during training). This is a small,
//! honestly-scoped set (one realm, five static views) — a prime-Guardian
//! ruling on broader validation-set composition is OPEN 10 in the proposal;
//! this atom ships the smallest honest set that lets the derived-bound
//! machinery run for real.
//!
//! Run:  cargo run -p scrying-glass --release --example viii1_train
//!       GAIA_VIII1_EPOCHS=200 cargo run -p scrying-glass --release --example viii1_train

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::{
    Adam, Mlp, MlpConfig, TrainingPixel, denoise_image, serialize_weights, sha256_hex, train_epoch,
};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

/// Forge-time weight-init PRNG seed (deterministic starting point — see
/// `Mlp::new_random` docs; training itself is NOT promised bit-reproducible,
/// proposal OPEN 4). A plain constant, not a magic number hidden inline.
const INIT_SEED: u64 = 0x0005_eed1;

/// Naruko authoring dials — the SAME front pose `perf_audit.rs`/
/// `viii0_truth.rs` render from (reused verbatim, not reinvented).
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

fn camera_at(eye: [f32; 3], look_at: [f32; 3], fov_deg: f32) -> Camera {
    let f = (GVec3::from_array(look_at) - GVec3::from_array(eye)).normalize();
    Camera {
        eye: GVec3::from_array(eye),
        yaw: (-f.x).atan2(-f.z),
        pitch: f.y.asin(),
        fov_y_radians: fov_deg.to_radians(),
        near: 0.1,
        far: 4_000.0,
    }
}

/// A derived orbit view: rotate `eye` by `yaw_deg` around Y about `pivot`,
/// keeping the same radius and height, looking back at `pivot`.
fn orbit_camera(eye: [f32; 3], pivot: [f32; 3], yaw_deg: f32, fov_deg: f32) -> Camera {
    let rel = GVec3::from_array(eye) - GVec3::from_array(pivot);
    let angle = yaw_deg.to_radians();
    let (s, c) = angle.sin_cos();
    let rotated = GVec3::new(rel.x * c + rel.z * s, rel.y, -rel.x * s + rel.z * c);
    let new_eye = GVec3::from_array(pivot) + rotated;
    camera_at(new_eye.to_array(), pivot, fov_deg)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// One rendered pose's full dataset: current-frame AOVs + noisy + reference,
/// all at the SAME (w, h).
struct PoseFrame {
    name: &'static str,
    albedo: Vec<GVec3>,
    normal: Vec<GVec3>,
    depth: Vec<f32>,
    noisy: Vec<GVec3>,
    reference: Vec<GVec3>,
}

/// Reference frame count for TRAINING pairs — smaller than `viii0_truth`'s
/// proof-quality 512 (argued there against noise floor honestly); this
/// dataset only needs a good-enough-to-teach-the-shape target within a
/// forge-time budget a builder actually runs. DERIVED the same way (1/sqrt
/// falloff): 128 frames × spp 2 = 256 samples/pixel, deep enough that
/// (per `viii0_truth`'s own printed convergence evidence at the SAME scene)
/// residual noise is well below the noisy-vs-reference gap the denoiser is
/// asked to close.
fn default_ref_frames() -> u32 {
    128
}

/// Dataset resolution — small enough that CPU per-pixel MLP training (no
/// GPU involved once radiance/AOV buffers are read back) finishes in a
/// reasonable forge-time budget across 5 poses × ~256 spp reference frames.
fn default_dataset_wh() -> (u32, u32) {
    (96, 64)
}

#[allow(clippy::too_many_arguments)]
fn render_pose(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &Camera,
    scene: &RenderScene,
    name: &'static str,
    w: u32,
    h: u32,
    ref_frames: u32,
) -> PoseFrame {
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

    let ref_params = IntegratorParams::default();
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
        &ref_params,
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

    let hit_frac =
        albedo.iter().filter(|a| a.length_squared() > 0.0).count() as f32 / albedo.len() as f32;
    let mean_depth = depth.iter().sum::<f32>() / depth.len() as f32;
    let max_depth = depth.iter().cloned().fold(0.0f32, f32::max);
    eprintln!(
        "[viii1-train] rendered pose '{name}' ({w}x{h}, ref_frames={ref_frames}) hit_frac={hit_frac:.3} mean_depth={mean_depth:.2} max_depth={max_depth:.2}"
    );
    PoseFrame {
        name,
        albedo,
        normal,
        depth,
        noisy,
        reference,
    }
}

fn to_training_pixels(pose: &PoseFrame) -> Vec<TrainingPixel> {
    (0..pose.noisy.len())
        .map(|i| TrainingPixel {
            noisy_radiance: pose.noisy[i],
            albedo: pose.albedo[i],
            normal: pose.normal[i],
            depth: pose.depth[i],
            reference_radiance: pose.reference[i],
        })
        .collect()
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[viii1-train] no GPU adapter on this host — cannot forge the training set");
    };

    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    eprintln!(
        "[viii1-train] naruko: {} static leaf tris",
        scene.leaf_triangles().len()
    );

    let (w, h) = default_dataset_wh();
    let ref_frames = env_u32("GAIA_VIII1_REF_FRAMES", default_ref_frames());

    let front_camera = Camera {
        eye: GVec3::from_array(params.camera_position),
        yaw: params.camera_yaw,
        pitch: params.camera_pitch,
        fov_y_radians: params.fov_y_degrees.to_radians(),
        near: params.near,
        far: params.far,
    };
    let front_pivot = [0.0, 2.0, 0.0];
    let wide_camera = camera_at([-4.5, 8.5, 33.0], [-5.5, 2.0, 15.5], 60.0);

    // Fixed dataset scope — see module docs.
    let poses: Vec<(&'static str, Camera)> = vec![
        ("front", front_camera),
        ("wide", wide_camera),
        (
            "orbit_+20",
            orbit_camera(
                params.camera_position,
                front_pivot,
                20.0,
                params.fov_y_degrees,
            ),
        ),
        (
            "orbit_-20",
            orbit_camera(
                params.camera_position,
                front_pivot,
                -20.0,
                params.fov_y_degrees,
            ),
        ),
        (
            "orbit_+40",
            orbit_camera(
                params.camera_position,
                front_pivot,
                40.0,
                params.fov_y_degrees,
            ),
        ),
    ];
    let train_names = ["front", "wide", "orbit_+20"];
    let val_names = ["orbit_-20", "orbit_+40"];

    let frames: Vec<PoseFrame> = poses
        .iter()
        .map(|(name, cam)| render_pose(&device, &queue, &bvh, cam, &scene, name, w, h, ref_frames))
        .collect();

    let train_pixels: Vec<TrainingPixel> = frames
        .iter()
        .filter(|f| train_names.contains(&f.name))
        .flat_map(to_training_pixels)
        .collect();
    eprintln!(
        "[viii1-train] train split: {:?} ({} pixels)",
        train_names,
        train_pixels.len()
    );

    // ── train ────────────────────────────────────────────────────────────
    let config = MlpConfig::default();
    let mut mlp = Mlp::new_random(config, INIT_SEED);
    let lr = env_f32("GAIA_VIII1_LR", 0.001);
    let epochs = env_u32("GAIA_VIII1_EPOCHS", 120);
    let batch_size = env_u32("GAIA_VIII1_BATCH", 64) as usize;
    let mut adam = Adam::new(&mlp, lr, 0.9, 0.999, 1e-8);

    for epoch in 0..epochs {
        let loss = train_epoch(&mut mlp, &mut adam, &train_pixels, batch_size);
        if epoch % 10 == 0 || epoch + 1 == epochs {
            println!("[viii1-train] epoch {epoch}/{epochs} train_mse(output-space)={loss:.6}");
        }
    }

    // ── validate: PER-FRAME RMSE, whole poses only ──────────────────────
    println!("\n[viii1-train] === VALIDATION (per-frame RMSE, radiance space) ===");
    println!(
        "{:<12} {:>14} {:>14} {:>10}",
        "pose", "noisy_rmse", "denoised_rmse", "beats?"
    );
    let mut worst_val_rmse = 0.0f64;
    let mut val_rows = Vec::new();
    for f in frames.iter().filter(|f| val_names.contains(&f.name)) {
        let denoised = denoise_image(&mlp, &f.noisy, &f.albedo, &f.normal, &f.depth);
        let noisy_rmse = rmse(&f.noisy, &f.reference);
        let denoised_rmse = rmse(&denoised, &f.reference);
        let beats = denoised_rmse < noisy_rmse;
        println!(
            "{:<12} {:>14.6} {:>14.6} {:>10}",
            f.name,
            noisy_rmse,
            denoised_rmse,
            if beats { "yes" } else { "NO" }
        );
        worst_val_rmse = worst_val_rmse.max(denoised_rmse);
        val_rows.push((f.name, noisy_rmse, denoised_rmse, beats));
    }

    println!("\n[viii1-train] === TRAIN (per-frame RMSE, radiance space, informational) ===");
    let mut train_rows = Vec::new();
    for f in frames.iter().filter(|f| train_names.contains(&f.name)) {
        let denoised = denoise_image(&mlp, &f.noisy, &f.albedo, &f.normal, &f.depth);
        let noisy_rmse = rmse(&f.noisy, &f.reference);
        let denoised_rmse = rmse(&denoised, &f.reference);
        println!(
            "{:<12} {:>14.6} {:>14.6}",
            f.name, noisy_rmse, denoised_rmse
        );
        train_rows.push((f.name, noisy_rmse, denoised_rmse));
    }

    let all_beat = val_rows.iter().all(|(_, _, _, b)| *b);
    println!(
        "\n[viii1-train] denoised strictly beats noisy on EVERY validation frame: {}",
        if all_beat { "YES" } else { "NO — WALL" }
    );
    println!("[viii1-train] PINNED BOUND (worst validation frame RMSE) = {worst_val_rmse:.6}");

    // ── serialize + provenance ──────────────────────────────────────────
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    let weights_bytes = serialize_weights(&mlp);
    let weights_path = data_dir.join("denoiser-weights-v1.bin");
    std::fs::write(&weights_path, &weights_bytes).unwrap();
    let weights_sha256 = sha256_hex(&weights_bytes);
    println!(
        "[viii1-train] wrote {} ({} bytes)",
        weights_path.display(),
        weights_bytes.len()
    );
    println!("[viii1-train] weights sha256 = {weights_sha256}");

    let provenance = serde_json::json!({
        "artifact": "denoiser-weights-v1.bin",
        "weights_sha256": weights_sha256,
        "architecture": {
            "input_features": scrying_glass::denoiser::INPUT_FEATURES,
            "output_channels": scrying_glass::denoiser::OUTPUT_CHANNELS,
            "hidden_layers": config.hidden_layers,
            "hidden_width": config.hidden_width,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "adam",
            "init_seed": INIT_SEED,
        },
        "dataset": {
            "realm": "naruko",
            "resolution": [w, h],
            "reference_frames": ref_frames,
            "reference_spp": IntegratorParams::default().spp,
            "seed": IntegratorParams::default().seed,
            "poses_all": poses.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            "poses_train": train_names,
            "poses_validation": val_names,
            "pixels_train": train_pixels.len(),
        },
        "metrics": {
            "train_per_frame_rmse": train_rows.iter().map(|(n, noisy, den)| serde_json::json!({
                "pose": n, "noisy_rmse": noisy, "denoised_rmse": den,
            })).collect::<Vec<_>>(),
            "validation_per_frame_rmse": val_rows.iter().map(|(n, noisy, den, beats)| serde_json::json!({
                "pose": n, "noisy_rmse": noisy, "denoised_rmse": den, "denoised_beats_noisy": beats,
            })).collect::<Vec<_>>(),
        },
        "pinned_bound": {
            "description": "worst validation-frame RMSE (denoised vs reference), derived at train time, never chosen in the ordeal",
            "value": worst_val_rmse,
        },
    });
    let provenance_path = data_dir.join("denoiser-weights-v1.provenance.json");
    std::fs::write(
        &provenance_path,
        serde_json::to_string_pretty(&provenance).unwrap(),
    )
    .unwrap();
    println!("[viii1-train] wrote {}", provenance_path.display());

    if !all_beat {
        eprintln!(
            "[viii1-train] WALL: denoised did not strictly beat noisy on every validation frame — see table above"
        );
    }
}
