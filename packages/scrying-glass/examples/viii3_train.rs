//! RITE VIII-3 — THE UPSCALER: forge-time training. Generates, per pose:
//!   - a LOW-resolution noisy 1-spp traced frame (the internal render),
//!   - the TARGET-resolution AOVs (albedo/normal/depth — cheap, full-res),
//!   - the TARGET-resolution converged N-frame reference (the truth),
//! across the SAME naruko poses/split the denoiser uses (shared
//! `upscaler_dataset`), trains the per-pixel residual-over-bilinear MLP
//! (`scrying_glass::upscaler`) on the TRAIN split, evaluates NEURAL vs NAIVE
//! BILINEAR per held-out VALIDATION frame (whole poses, never pixels — the
//! bound must be per-frame), and writes:
//!
//!   - packages/scrying-glass/data/upscaler-weights-v1.bin
//!   - packages/scrying-glass/data/upscaler-weights-v1.provenance.json
//!     (dataset hash, config, scale, per-frame bilinear/neural RMSE table,
//!     the DERIVED pinned bound = worst validation frame's NEURAL RMSE, and
//!     the beats-bilinear margin per pose, sha256 of the weights).
//!
//! Run:  cargo run -p scrying-glass --release --example viii3_train
//!       GAIA_VIII3_EPOCHS=400 cargo run -p scrying-glass --release --example viii3_train

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, RenderScene};
use scrying_glass::upscaler::{
    Adam, Mlp, TrainingFrame, UpscaleConfig, bilinear_upsample, serialize_weights, train_epoch,
    upscale_image, weights_sha256,
};
use scrying_glass::upscaler_dataset::{
    DATASET_REF_FRAMES, TRAIN_POSE_NAMES, UPSCALE_SCALE, VALIDATION_POSE_NAMES, dataset_dims,
    law_poses, naruko_params,
};

/// Forge-time weight-init PRNG seed (deterministic starting point — see
/// `Mlp::new_bilinear_start`; the last layer is zeroed regardless, so the
/// starting net IS bilinear whatever the seed).
const INIT_SEED: u64 = 0x0005_ca1e;

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

struct PoseData {
    name: &'static str,
    frame: TrainingFrame,
}

#[allow(clippy::too_many_arguments)]
fn render_pose(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &Camera,
    scene: &RenderScene,
    name: &'static str,
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
    ref_frames: u32,
) -> PoseData {
    // LOW-resolution noisy 1-spp internal frame — the only expensive light
    // traced, at internal resolution.
    let noisy_params = IntegratorParams {
        spp: 1,
        ..IntegratorParams::default()
    };
    let low_noisy = resolve(&trace_headless(
        device,
        queue,
        bvh,
        camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        low_w,
        low_h,
        1,
        &noisy_params,
        None,
    ));

    // TARGET-resolution converged reference (the truth we upscale toward).
    let reference = resolve(&trace_headless(
        device,
        queue,
        bvh,
        camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        target_w,
        target_h,
        ref_frames,
        &IntegratorParams::default(),
        None,
    ));

    // TARGET-resolution AOVs (cheap geometry-only pass, full native res).
    let raw_aov = trace_headless_aov(
        device,
        queue,
        bvh,
        camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        target_w,
        target_h,
    );
    let (hi_albedo, hi_normal, hi_depth) = split_aov(&raw_aov);

    let hit_frac = hi_albedo.iter().filter(|a| a.length_squared() > 0.0).count() as f32
        / hi_albedo.len() as f32;
    eprintln!(
        "[viii3-train] pose '{name}' low={low_w}x{low_h} target={target_w}x{target_h} ref_frames={ref_frames} hi_hit_frac={hit_frac:.3}"
    );

    PoseData {
        name,
        frame: TrainingFrame {
            low_radiance: low_noisy,
            low_w,
            low_h,
            hi_albedo,
            hi_normal,
            hi_depth,
            reference,
            target_w,
            target_h,
        },
    }
}

/// Content digest of the generated training PAIRS (train split only) — hashes
/// every low pixel + every high AOV/reference pixel in fixed (pose, buffer,
/// index, channel) order, so retraining from an unchanged dataset reproduces
/// the SAME digest (VIII-1's A7 pattern).
fn hash_training_frames(frames: &[&TrainingFrame]) -> String {
    let mut bytes = Vec::new();
    let push_v = |bytes: &mut Vec<u8>, v: &GVec3| {
        bytes.extend_from_slice(&v.x.to_le_bytes());
        bytes.extend_from_slice(&v.y.to_le_bytes());
        bytes.extend_from_slice(&v.z.to_le_bytes());
    };
    for fr in frames {
        for v in &fr.low_radiance {
            push_v(&mut bytes, v);
        }
        for v in &fr.hi_albedo {
            push_v(&mut bytes, v);
        }
        for v in &fr.hi_normal {
            push_v(&mut bytes, v);
        }
        for d in &fr.hi_depth {
            bytes.extend_from_slice(&d.to_le_bytes());
        }
        for v in &fr.reference {
            push_v(&mut bytes, v);
        }
    }
    weights_sha256(&bytes)
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[viii3-train] no GPU adapter on this host — cannot forge the training set");
    };

    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    eprintln!(
        "[viii3-train] naruko: {} static leaf tris, scale={UPSCALE_SCALE}",
        scene.leaf_triangles().len()
    );

    let (low_w, low_h, target_w, target_h) = dataset_dims();
    let ref_frames = env_u32("GAIA_VIII3_REF_FRAMES", DATASET_REF_FRAMES);

    let poses = law_poses(&params);
    let data: Vec<PoseData> = poses
        .iter()
        .map(|(name, cam)| {
            render_pose(
                &device, &queue, &bvh, cam, &scene, name, low_w, low_h, target_w, target_h,
                ref_frames,
            )
        })
        .collect();

    let train_frames: Vec<&TrainingFrame> = data
        .iter()
        .filter(|d| TRAIN_POSE_NAMES.contains(&d.name))
        .map(|d| &d.frame)
        .collect();
    let train_owned: Vec<TrainingFrame> = data
        .iter()
        .filter(|d| TRAIN_POSE_NAMES.contains(&d.name))
        .map(|d| clone_frame(&d.frame))
        .collect();
    let dataset_pairs_sha256 = hash_training_frames(&train_frames);
    eprintln!(
        "[viii3-train] train split: {:?} ({} frames, pairs sha256={})",
        TRAIN_POSE_NAMES,
        train_frames.len(),
        dataset_pairs_sha256
    );

    // ── train ────────────────────────────────────────────────────────────
    let config = UpscaleConfig::default();
    let mut mlp = Mlp::new_bilinear_start(config, INIT_SEED);
    let lr = env_f32("GAIA_VIII3_LR", 0.001);
    let epochs = env_u32("GAIA_VIII3_EPOCHS", 300);
    let batch_size = env_u32("GAIA_VIII3_BATCH", 128) as usize;
    let mut adam = Adam::new(&mlp, lr, 0.9, 0.999, 1e-8);

    for epoch in 0..epochs {
        let loss = train_epoch(&mut mlp, &mut adam, &train_owned, batch_size);
        if epoch % 20 == 0 || epoch + 1 == epochs {
            println!("[viii3-train] epoch {epoch}/{epochs} train_mse(residual-space)={loss:.6}");
        }
    }

    // ── validate: NEURAL vs NAIVE BILINEAR, PER FRAME, whole poses only ──
    println!("\n[viii3-train] === VALIDATION (per-frame RMSE vs truth, radiance space) ===");
    println!(
        "{:<12} {:>14} {:>14} {:>10} {:>12}",
        "pose", "bilinear_rmse", "neural_rmse", "beats?", "margin"
    );
    let mut worst_neural_rmse = 0.0f64;
    let mut val_rows = Vec::new();
    for d in data.iter().filter(|d| VALIDATION_POSE_NAMES.contains(&d.name)) {
        let f = &d.frame;
        let bilinear = bilinear_upsample(&f.low_radiance, f.low_w, f.low_h, f.target_w, f.target_h);
        let neural = upscale_image(
            &mlp,
            &f.low_radiance,
            f.low_w,
            f.low_h,
            &f.hi_albedo,
            &f.hi_normal,
            &f.hi_depth,
            f.target_w,
            f.target_h,
        );
        let bilinear_rmse = rmse(&bilinear, &f.reference);
        let neural_rmse = rmse(&neural, &f.reference);
        let beats = neural_rmse < bilinear_rmse;
        let margin = bilinear_rmse - neural_rmse;
        println!(
            "{:<12} {:>14.6} {:>14.6} {:>10} {:>12.6}",
            d.name,
            bilinear_rmse,
            neural_rmse,
            if beats { "yes" } else { "NO" },
            margin
        );
        worst_neural_rmse = worst_neural_rmse.max(neural_rmse);
        val_rows.push((d.name, bilinear_rmse, neural_rmse, beats, margin));
    }

    println!("\n[viii3-train] === TRAIN (per-frame RMSE, informational) ===");
    let mut train_rows = Vec::new();
    for d in data.iter().filter(|d| TRAIN_POSE_NAMES.contains(&d.name)) {
        let f = &d.frame;
        let bilinear = bilinear_upsample(&f.low_radiance, f.low_w, f.low_h, f.target_w, f.target_h);
        let neural = upscale_image(
            &mlp,
            &f.low_radiance,
            f.low_w,
            f.low_h,
            &f.hi_albedo,
            &f.hi_normal,
            &f.hi_depth,
            f.target_w,
            f.target_h,
        );
        let bilinear_rmse = rmse(&bilinear, &f.reference);
        let neural_rmse = rmse(&neural, &f.reference);
        println!(
            "{:<12} bilinear={bilinear_rmse:.6} neural={neural_rmse:.6}",
            d.name
        );
        train_rows.push((d.name, bilinear_rmse, neural_rmse));
    }

    let all_beat = val_rows.iter().all(|(_, _, _, b, _)| *b);
    println!(
        "\n[viii3-train] neural strictly beats naive bilinear on EVERY validation frame: {}",
        if all_beat { "YES" } else { "NO — WALL" }
    );
    println!("[viii3-train] PINNED BOUND (worst validation-frame NEURAL RMSE) = {worst_neural_rmse:.6}");

    // ── serialize + provenance ──────────────────────────────────────────
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    let weights_bytes = serialize_weights(&mlp);
    let weights_path = data_dir.join("upscaler-weights-v1.bin");
    std::fs::write(&weights_path, &weights_bytes).unwrap();
    let weights_hash = weights_sha256(&weights_bytes);
    println!(
        "[viii3-train] wrote {} ({} bytes), sha256={weights_hash}",
        weights_path.display(),
        weights_bytes.len()
    );

    let provenance = serde_json::json!({
        "artifact": "upscaler-weights-v1.bin",
        "weights_sha256": weights_hash,
        "architecture": {
            "kind": "per-pixel residual-over-bilinear MLP (current-frame only)",
            "input_features": scrying_glass::upscaler::INPUT_FEATURES,
            "output_channels": scrying_glass::upscaler::OUTPUT_CHANNELS,
            "hidden_layers": config.hidden_layers,
            "hidden_width": config.hidden_width,
        },
        "scale": UPSCALE_SCALE,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "adam",
            "init_seed": INIT_SEED,
        },
        "dataset": {
            "realm": "naruko",
            "low_resolution": [low_w, low_h],
            "target_resolution": [target_w, target_h],
            "reference_frames": ref_frames,
            "poses_all": poses.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            "poses_train": TRAIN_POSE_NAMES,
            "poses_validation": VALIDATION_POSE_NAMES,
            "training_pairs_sha256": dataset_pairs_sha256,
        },
        "metrics": {
            "train_per_frame_rmse": train_rows.iter().map(|(n, bil, neu)| serde_json::json!({
                "pose": n, "bilinear_rmse": bil, "neural_rmse": neu,
            })).collect::<Vec<_>>(),
            "validation_per_frame_rmse": val_rows.iter().map(|(n, bil, neu, beats, margin)| serde_json::json!({
                "pose": n, "bilinear_rmse": bil, "neural_rmse": neu,
                "neural_beats_bilinear": beats, "beats_bilinear_margin": margin,
            })).collect::<Vec<_>>(),
        },
        "pinned_bound": {
            "description": "worst validation-frame NEURAL RMSE (neural upscale vs truth reference), derived at train time, never chosen in the ordeal",
            "value": worst_neural_rmse,
        },
    });
    let provenance_path = data_dir.join("upscaler-weights-v1.provenance.json");
    std::fs::write(
        &provenance_path,
        serde_json::to_string_pretty(&provenance).unwrap(),
    )
    .unwrap();
    println!("[viii3-train] wrote {}", provenance_path.display());

    if !all_beat {
        eprintln!(
            "[viii3-train] WALL: neural did not strictly beat naive bilinear on every validation frame — see table above"
        );
    }
}

/// Deep-clone a `TrainingFrame` (no `Clone` derive on the public type — kept
/// minimal; the train harness needs both a borrowed view for hashing and an
/// owned vec for the epoch loop).
fn clone_frame(f: &TrainingFrame) -> TrainingFrame {
    TrainingFrame {
        low_radiance: f.low_radiance.clone(),
        low_w: f.low_w,
        low_h: f.low_h,
        hi_albedo: f.hi_albedo.clone(),
        hi_normal: f.hi_normal.clone(),
        hi_depth: f.hi_depth.clone(),
        reference: f.reference.clone(),
        target_w: f.target_w,
        target_h: f.target_h,
    }
}
