//! R-DIRECT v3 — MEMORY (N2). Retrain the direct-render MLP with a widened
//! input (27 = v2's 23 + reprojected previous demod-log radiance (3) + validity
//! (1)) and train the RECURRENCE UNROLLED: at stillness the net's own previous
//! output feeds back as input, so it learns to AVERAGE across frames — killing
//! the spp=1 sparkle dots the current-frame net cannot close. Frame 0 (and any
//! disoccluded pixel) is the validity=0 case, which is v2's exact single-frame
//! task, so the same weights denoise a fresh frame AND accumulate a held one.
//!
//! Training doctrine ("unroll the algorithm"): for each still pose we render K
//! fresh-seed 1-spp frames (the dots move frame to frame — the variance to
//! average away) and one long-accumulation teacher. Per sampled pixel we run
//! the K-step recurrence — prev_dl starts zero (validity 0), then each step's
//! output becomes the next step's history input (validity 1) — backprop at
//! EVERY step vs the teacher (truncated BPTT, prev treated as a fixed input).
//! data = f(seed): fresh pixel subset AND fresh render seeds each epoch.
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_train_v3
//!   GAIA_V3_EPOCHS, GAIA_V3_STILL (K), GAIA_V3_SUBSAMPLE, GAIA_V3_W/H, GAIA_V3_CKPT

use std::path::Path;
use std::time::Instant;

use glam::{Vec2, Vec3 as GVec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::rdirect::{
    Adam, HIST_FEATURES, Mlp, RdirectConfig, accumulate_backward, adam_apply, deserialize_weights,
    hist_features, pixel_features, serialize_weights, target_demod_log, weights_sha256, zero_grads,
    OUTPUT_CHANNELS,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

const INIT_SEED: u64 = 0x00d1_5eed_0003;

fn env_u32(n: &str, d: u32) -> u32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f32(n: &str, d: f32) -> f32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// One still pose: K fresh-seed low-radiance frames, a native G-buffer, teacher.
struct Pose {
    lows: Vec<Vec<GVec3>>, // K low-radiance frames
    albedo: Vec<GVec3>,
    normal: Vec<GVec3>,
    depth: Vec<f32>,
    teacher: Vec<GVec3>,
    low_w: u32,
    low_h: u32,
    tw: u32,
    th: u32,
}

#[allow(clippy::too_many_arguments)]
fn render_pose(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_tris: &[LeafTriangle],
    scene: &RenderScene,
    cam: &Camera,
    k: u32,
    low_w: u32,
    low_h: u32,
    tw: u32,
    th: u32,
    ref_frames: u32,
) -> Pose {
    let bvh = Bvh::build(base_tris, &BvhParams::default());
    let lows: Vec<Vec<GVec3>> = (0..k)
        .map(|f| {
            let np = IntegratorParams { spp: 1, seed: 0x7abc + f * 131 + 5, ..IntegratorParams::default() };
            resolve(&trace_headless(
                device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h,
                1, &np, None,
            ))
        })
        .collect();
    let (albedo, normal, depth) = split_aov(&trace_headless_aov(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th,
    ));
    let teacher = resolve(&trace_headless(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
        &IntegratorParams::default(), None,
    ));
    Pose { lows, albedo, normal, depth, teacher, low_w, low_h, tw, th }
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[v3] no GPU");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();

    let tw = env_u32("GAIA_V3_W", 480);
    let th = env_u32("GAIA_V3_H", 360);
    let low_w = tw / 2;
    let low_h = th / 2;
    let k = env_u32("GAIA_V3_STILL", 4); // unroll depth
    let ref_frames = env_u32("GAIA_V3_REF", 96);
    let epochs = env_u32("GAIA_V3_EPOCHS", 120);
    let subsample = env_u32("GAIA_V3_SUBSAMPLE", 6000) as usize; // px/pose/epoch
    let batch = env_u32("GAIA_V3_BATCH", 64) as usize;
    let lr0 = env_f32("GAIA_V3_LR", 0.002);
    let ckpt_every = env_u32("GAIA_V3_CKPT", 20);

    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| all.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();
    let train_cams = [("front", find("front")), ("wide", find("wide")), ("orbit_+20", find("orbit_+20"))];

    let t_render = Instant::now();
    let poses: Vec<Pose> = train_cams
        .iter()
        .map(|(_, c)| render_pose(&device, &queue, &base_tris, &scene, c, k, low_w, low_h, tw, th, ref_frames))
        .collect();
    eprintln!("[v3] rendered {} poses (K={k}, {tw}x{th}, teacher {ref_frames}) in {:.1}s",
        poses.len(), t_render.elapsed().as_secs_f64());

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let wpath = data_dir.join("rdirect-weights-v3.bin");
    let config = RdirectConfig {
        hidden_layers: env_u32("GAIA_V3_LAYERS", 5) as usize,
        hidden_width: env_u32("GAIA_V3_WIDTH", 64) as usize,
    };
    let resume = matches!(std::env::var("GAIA_V3_RESUME").as_deref(), Ok("1" | "true"));
    let mut mlp = if resume && wpath.exists() {
        let m = deserialize_weights(&std::fs::read(&wpath).unwrap()).expect("resume v3");
        eprintln!("[v3] RESUMED from {}", wpath.display());
        m
    } else {
        Mlp::new_random_with_input(config, HIST_FEATURES, INIT_SEED)
    };
    assert_eq!(mlp.layer_dims()[0].0 as usize, HIST_FEATURES, "v3 net must be 27-input");
    let epoch_start = env_u32("GAIA_V3_EPOCH_START", 0);
    let epoch_total = env_u32("GAIA_V3_EPOCH_TOTAL", epochs).max(1);
    let mut adam = Adam::new(&mlp, lr0, 0.9, 0.999, 1e-8);
    eprintln!("[v3] arch {:?} in={HIST_FEATURES} macs/px={} — training {epochs} epochs, subsample {subsample}px/pose",
        config, mlp.macs());

    let n_px: Vec<usize> = poses.iter().map(|p| (p.tw * p.th) as usize).collect();
    let mut rng = Rng(INIT_SEED ^ 0xF00D ^ ((epoch_start as u64) << 21));
    let t_train = Instant::now();

    for epoch in 0..epochs {
        let frac = (epoch_start + epoch) as f32 / epoch_total as f32;
        adam.set_lr(lr0 / (1.0 + 2.0 * frac));
        let mut epoch_loss = 0.0f64;
        let mut n_steps = 0u64;

        // gather a fresh subsampled pixel list (data = f(seed)), then unroll.
        let mut samples: Vec<(usize, usize)> = Vec::new(); // (pose, pixel)
        for (pi, np) in n_px.iter().enumerate() {
            for _ in 0..subsample {
                samples.push((pi, (rng.next() as usize) % np));
            }
        }
        // deterministic shuffle by fresh draw order is already random; batch it.
        let mut bstart = 0usize;
        while bstart < samples.len() {
            let bend = (bstart + batch).min(samples.len());
            let (mut wg, mut bg) = zero_grads(&mlp);
            let blen = (bend - bstart) as f32;
            for &(pi, px) in &samples[bstart..bend] {
                let p = &poses[pi];
                let tx = (px as u32) % p.tw;
                let ty = (px as u32) / p.tw;
                let albedo = p.albedo[px];
                let target: [f32; OUTPUT_CHANNELS] = target_demod_log(p.teacher[px], albedo);
                // unroll the still recurrence: prev output feeds back (identity
                // reproject at stillness), backprop every step.
                let mut prev_dl = [0.0f32; 3];
                let mut valid = 0.0f32;
                for step in 0..k as usize {
                    let base = pixel_features(
                        &p.lows[step], p.low_w, p.low_h, p.tw, p.th, tx, ty, albedo, p.normal[px],
                        p.depth[px], Vec2::ZERO,
                    );
                    let feat = hist_features(&base, prev_dl, valid);
                    // scale: mean over batch AND over steps (each step contributes)
                    accumulate_backward(&mlp, &feat, &target, &mut wg, &mut bg, 1.0 / (blen * k as f32));
                    let out = mlp.forward(&feat);
                    for c in 0..OUTPUT_CHANNELS {
                        let d = (out[c] - target[c]) as f64;
                        epoch_loss += d * d;
                    }
                    n_steps += 1;
                    prev_dl = out;
                    valid = 1.0;
                }
            }
            adam_apply(&mut adam, &mut mlp, &wg, &bg);
            bstart = bend;
        }

        if epoch % 10 == 0 || epoch + 1 == epochs {
            println!(
                "[v3] epoch {}/{} loss(out-space)={:.6} ({:.1}s)",
                epoch_start + epoch, epoch_total,
                epoch_loss / (n_steps as f64 * OUTPUT_CHANNELS as f64).max(1.0),
                t_train.elapsed().as_secs_f64()
            );
        }
        if (epoch + 1) % ckpt_every == 0 || epoch + 1 == epochs {
            std::fs::write(&wpath, serialize_weights(&mlp)).unwrap();
            eprintln!("[v3] checkpoint @ {} → {}", epoch + 1, wpath.display());
        }
    }
    eprintln!("[v3] training done in {:.1}s", t_train.elapsed().as_secs_f64());

    std::fs::write(&wpath, serialize_weights(&mlp)).unwrap();
    let wsha = weights_sha256(&mlp);
    println!("[v3] wrote {} sha256={wsha}", wpath.display());
    let prov = serde_json::json!({
        "artifact": "rdirect-weights-v3.bin",
        "weights_sha256": wsha,
        "supersedes": "rdirect-weights-v2.bin",
        "architecture": {
            "kind": "N2 recurrent direct-render MLP — v2 base + reprojected prev demod-log radiance (3) + validity (1)",
            "input_features": HIST_FEATURES,
            "output_channels": OUTPUT_CHANNELS,
            "hidden_layers": config.hidden_layers,
            "hidden_width": config.hidden_width,
            "macs_per_pixel": mlp.macs(),
        },
        "training": {
            "epochs": epochs, "unroll_steps": k, "batch": batch, "lr0": lr0,
            "subsample_px_per_pose_per_epoch": subsample, "ref_frames": ref_frames,
            "recurrence": "still unroll (identity reproject), prev output fed as history input, validity 1 after step 0",
            "note": "data=f(seed); fresh 1-spp render seeds per frame carry the variance to average"
        },
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [tw, th],
            "train": ["front", "wide", "orbit_+20"] },
        "gate": "presents ONLY if real_image_ordeal writes a PASS stamp beside this file",
    });
    std::fs::write(data_dir.join("rdirect-weights-v3.provenance.json"),
        serde_json::to_string_pretty(&prov).unwrap()).unwrap();
    println!("[v3] wrote provenance. NEXT: run `real_image_ordeal` to earn the stamp.");
}
