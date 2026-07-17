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
//! Pose/scene definitions (dataset scope, proposal OPEN 10) live in
//! `scrying_glass::denoiser_dataset` — the ONE shared source this file and
//! `tests/viii1_ordeals.rs` both consume, so the two can never silently
//! drift apart (adversary finding A5, night-2 review).
//!
//! Run:  cargo run -p scrying-glass --release --example viii1_train
//!       GAIA_VIII1_EPOCHS=200 cargo run -p scrying-glass --release --example viii1_train

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::{
    Adam, Mlp, MlpConfig, TrainingPixel, denoise_image, serialize_weights, sha256_hex, train_epoch,
};
use scrying_glass::denoiser_dataset::{
    DATASET_HEIGHT, DATASET_REF_FRAMES, DATASET_WIDTH, TRAIN_POSE_NAMES, VALIDATION_POSE_NAMES,
    law_poses, naruko_params,
};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, RenderScene};

/// Forge-time weight-init PRNG seed (deterministic starting point — see
/// `Mlp::new_random` docs; training itself is NOT promised bit-reproducible,
/// proposal OPEN 4). A plain constant, not a magic number hidden inline.
const INIT_SEED: u64 = 0x0005_eed1;

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

/// Content digest of the generated training PAIRS (not the weights) —
/// hashes every train-split pixel's five f32x3-or-f32 fields in fixed
/// (pose, pixel-index, field) order, so retraining from an unchanged
/// dataset reproduces the SAME digest (adversary finding A7).
fn hash_training_pairs(pixels: &[TrainingPixel]) -> String {
    let mut bytes = Vec::with_capacity(pixels.len() * (3 + 3 + 3 + 1 + 3) * 4);
    for px in pixels {
        for v in [px.noisy_radiance, px.albedo, px.normal] {
            bytes.extend_from_slice(&v.x.to_le_bytes());
            bytes.extend_from_slice(&v.y.to_le_bytes());
            bytes.extend_from_slice(&v.z.to_le_bytes());
        }
        bytes.extend_from_slice(&px.depth.to_le_bytes());
        let r = px.reference_radiance;
        bytes.extend_from_slice(&r.x.to_le_bytes());
        bytes.extend_from_slice(&r.y.to_le_bytes());
        bytes.extend_from_slice(&r.z.to_le_bytes());
    }
    sha256_hex(&bytes)
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

    let (w, h) = (DATASET_WIDTH, DATASET_HEIGHT);
    let ref_frames = env_u32("GAIA_VIII1_REF_FRAMES", DATASET_REF_FRAMES);

    // Fixed dataset scope — see `scrying_glass::denoiser_dataset` module docs.
    let poses = law_poses(&params);
    let train_names = TRAIN_POSE_NAMES;
    let val_names = VALIDATION_POSE_NAMES;

    let frames: Vec<PoseFrame> = poses
        .iter()
        .map(|(name, cam)| render_pose(&device, &queue, &bvh, cam, &scene, name, w, h, ref_frames))
        .collect();

    let train_pixels: Vec<TrainingPixel> = frames
        .iter()
        .filter(|f| train_names.contains(&f.name))
        .flat_map(to_training_pixels)
        .collect();
    let dataset_pairs_sha256 = hash_training_pairs(&train_pixels);
    eprintln!(
        "[viii1-train] train split: {:?} ({} pixels, pairs sha256={})",
        train_names,
        train_pixels.len(),
        dataset_pairs_sha256
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
            // (A7) sha256 of the generated train-split pixel PAIRS (noisy,
            // albedo, normal, depth, reference — fixed pose/pixel/field
            // order), NOT the weights. Recorded starting this run; if an
            // older provenance file lacks this field, it was "not recorded
            // for v1" — an honest gap, not a silent retro-claim.
            "training_pairs_sha256": dataset_pairs_sha256,
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
