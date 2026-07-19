//! R-DIRECT v4 — THE FIREFLY LOSS (N3). Same arch/recurrence as v3 (27-in,
//! 5×64, still-unroll BPTT) but the training loss gains a SPATIAL FIREFLY
//! CLAMP term. v3 read as the real image everywhere except isolated
//! emissive-edge fireflies (sparkle 345/Mpx vs bar 40) — a STABLE spatial
//! bias the temporal recurrence cannot average away. The fix is spatial:
//!
//!   LOSS = MSE(out, teacher) + ff_w · Σ_c relu(out_c − cap_c)²
//!
//! where cap_c is the TEACHER's local-neighbourhood max (k×k) in the net's
//! own output space (demod-log) plus a small margin. If the net is BRIGHTER
//! than anything the teacher shows nearby (a firefly over a dark neighbour-
//! hood), the excess is crushed; genuine bright edges (high cap) are free.
//! Differentiable, one extra forward-free delta per unroll step, cheap at
//! 320-res. `ff_w` and the k×k window are IRON params. Ordeal thresholds are
//! NOT touched.
//!
//! Monitors the ORDEAL's own sparkle metric on a HELD-OUT val pose each check
//! (rendered once, inference-only) and early-stops when under bar with margin.
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_train_v4
//!   GAIA_V4_EPOCHS, GAIA_V4_STILL (K), GAIA_V4_SUBSAMPLE, GAIA_V4_W/H, GAIA_V4_CKPT,
//!   GAIA_V4_FF (firefly weight), GAIA_V4_FFK (cap window radius), GAIA_V4_FFMARGIN,
//!   GAIA_V4_SPARK_STOP (early-stop sparkle target), GAIA_V4_MONITOR (epochs/check)

use std::path::Path;
use std::time::Instant;

use glam::{Vec2, Vec3 as GVec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::rdirect::{
    Adam, HIST_FEATURES, Mlp, RdirectConfig, accumulate_backward_firefly, adam_apply,
    deserialize_weights, hist_features, pixel_features, serialize_weights, target_demod_log,
    weights_sha256, zero_grads, OUTPUT_CHANNELS,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

const INIT_SEED: u64 = 0x00d1_5eed_0004;

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

/// One still pose: K fresh-seed low-radiance frames, a native G-buffer, teacher,
/// plus the precomputed teacher demod-log target field and its k×k neighbourhood
/// cap (both in the net's OUTPUT space) for the firefly term.
struct Pose {
    lows: Vec<Vec<GVec3>>,
    albedo: Vec<GVec3>,
    normal: Vec<GVec3>,
    depth: Vec<f32>,
    teacher: Vec<GVec3>,
    teacher_dl: Vec<[f32; OUTPUT_CHANNELS]>, // per-pixel demod-log target
    cap: Vec<[f32; OUTPUT_CHANNELS]>,        // teacher k×k max + margin
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
    cap_radius: i32,
    cap_margin: f32,
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
    // teacher target field (demod-log) + k×k neighbourhood cap (per channel max).
    let n = (tw * th) as usize;
    let teacher_dl: Vec<[f32; OUTPUT_CHANNELS]> =
        (0..n).map(|px| target_demod_log(teacher[px], albedo[px])).collect();
    let mut cap = vec![[f32::NEG_INFINITY; OUTPUT_CHANNELS]; n];
    for y in 0..th as i32 {
        for x in 0..tw as i32 {
            let mut m = [f32::NEG_INFINITY; OUTPUT_CHANNELS];
            for dy in -cap_radius..=cap_radius {
                let ny = y + dy;
                if ny < 0 || ny >= th as i32 {
                    continue;
                }
                for dx in -cap_radius..=cap_radius {
                    let nx = x + dx;
                    if nx < 0 || nx >= tw as i32 {
                        continue;
                    }
                    let t = teacher_dl[(ny as u32 * tw + nx as u32) as usize];
                    for c in 0..OUTPUT_CHANNELS {
                        if t[c] > m[c] {
                            m[c] = t[c];
                        }
                    }
                }
            }
            let idx = (y as u32 * tw + x as u32) as usize;
            for c in 0..OUTPUT_CHANNELS {
                cap[idx][c] = m[c] + cap_margin;
            }
        }
    }
    Pose { lows, albedo, normal, depth, teacher, teacher_dl, cap, low_w, low_h, tw, th }
}

fn lum(c: GVec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// The ORDEAL's exact sparkle metric: isolated bright dots the net invented
/// over the converged teacher, per megapixel. Used as a training monitor.
fn sparkle_resid_per_mpx(net: &[GVec3], teacher: &[GVec3], w: u32, h: u32) -> f64 {
    const SPARK_DELTA: f32 = 0.15;
    let idx = |x: i32, y: i32| (y as usize) * w as usize + x as usize;
    let err = |x: i32, y: i32| lum(net[idx(x, y)]) - lum(teacher[idx(x, y)]);
    let mut count = 0u64;
    for y in 1..h as i32 - 1 {
        for x in 1..w as i32 - 1 {
            let e = err(x, y);
            if e <= SPARK_DELTA {
                continue;
            }
            let mut is_peak = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if (dx != 0 || dy != 0) && err(x + dx, y + dy) >= e {
                        is_peak = false;
                    }
                }
            }
            if is_peak {
                count += 1;
            }
        }
    }
    (count as f64) * 1.0e6 / (w as f64 * h as f64)
}

/// Settle the still recurrence on a pose (identity reproject at stillness),
/// return the linear-RGB settled frame. Inference-only monitor.
fn settle_still(mlp: &Mlp, p: &Pose, k: u32) -> Vec<GVec3> {
    let n = (p.tw * p.th) as usize;
    let mut out = vec![GVec3::ZERO; n];
    for ty in 0..p.th {
        for tx in 0..p.tw {
            let px = (ty * p.tw + tx) as usize;
            let albedo = p.albedo[px];
            let mut prev_dl = [0.0f32; 3];
            let mut valid = 0.0f32;
            let mut dl = [0.0f32; OUTPUT_CHANNELS];
            for step in 0..k as usize {
                let base = pixel_features(
                    &p.lows[step.min(p.lows.len() - 1)], p.low_w, p.low_h, p.tw, p.th, tx, ty,
                    albedo, p.normal[px], p.depth[px], Vec2::ZERO,
                );
                let feat = hist_features(&base, prev_dl, valid);
                dl = mlp.forward(&feat);
                prev_dl = dl;
                valid = 1.0;
            }
            let div = if albedo.length_squared() > 1e-8 { albedo + GVec3::splat(1e-3) } else { GVec3::ONE };
            let expm1 = GVec3::new(dl[0].exp() - 1.0, dl[1].exp() - 1.0, dl[2].exp() - 1.0);
            out[px] = GVec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0)) * div;
        }
    }
    out
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[v4] no GPU");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();

    let tw = env_u32("GAIA_V4_W", 480);
    let th = env_u32("GAIA_V4_H", 360);
    let low_w = tw / 2;
    let low_h = th / 2;
    let k = env_u32("GAIA_V4_STILL", 4);
    let ref_frames = env_u32("GAIA_V4_REF", 96);
    let epochs = env_u32("GAIA_V4_EPOCHS", 120);
    let subsample = env_u32("GAIA_V4_SUBSAMPLE", 6000) as usize;
    let batch = env_u32("GAIA_V4_BATCH", 64) as usize;
    let lr0 = env_f32("GAIA_V4_LR", 0.002);
    let _ckpt_every = env_u32("GAIA_V4_CKPT", 20);
    // ── IRON firefly params ──
    let ff_w = env_f32("GAIA_V4_FF", 6.0); // firefly clamp weight
    let cap_radius = env_u32("GAIA_V4_FFK", 2) as i32; // k×k window = (2r+1)²
    let cap_margin = env_f32("GAIA_V4_FFMARGIN", 0.05); // demod-log slack over teacher nbhd max
    let spark_stop = env_f32("GAIA_V4_SPARK_STOP", 30.0); // early-stop under this (bar 40)
    let monitor_every = env_u32("GAIA_V4_MONITOR", 10);

    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| all.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();
    let train_cams = [("front", find("front")), ("wide", find("wide")), ("orbit_+20", find("orbit_+20"))];

    let t_render = Instant::now();
    let poses: Vec<Pose> = train_cams
        .iter()
        .map(|(_, c)| render_pose(&device, &queue, &base_tris, &scene, c, k, low_w, low_h, tw, th, ref_frames, cap_radius, cap_margin))
        .collect();
    // held-out val pose (NOT in the train set) for the sparkle monitor.
    let val_pose = render_pose(&device, &queue, &base_tris, &scene, &find("orbit_-20"), k, low_w, low_h, tw, th, ref_frames, cap_radius, cap_margin);
    eprintln!("[v4] rendered {}+1 poses (K={k}, {tw}x{th}, teacher {ref_frames}, cap {}×{} margin {cap_margin}, ff_w {ff_w}) in {:.1}s",
        poses.len(), 2 * cap_radius + 1, 2 * cap_radius + 1, t_render.elapsed().as_secs_f64());

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let wpath = data_dir.join("rdirect-weights-v4.bin");
    let config = RdirectConfig {
        hidden_layers: env_u32("GAIA_V4_LAYERS", 5) as usize,
        hidden_width: env_u32("GAIA_V4_WIDTH", 64) as usize,
    };
    // warm-start from v3 (same arch) if present — inherits the good image, then
    // the firefly term files off the dots. Set GAIA_V4_SCRATCH=1 to disable.
    let v3path = data_dir.join("rdirect-weights-v3.bin");
    let scratch = matches!(std::env::var("GAIA_V4_SCRATCH").as_deref(), Ok("1" | "true"));
    let resume = matches!(std::env::var("GAIA_V4_RESUME").as_deref(), Ok("1" | "true"));
    let mut mlp = if resume && wpath.exists() {
        let m = deserialize_weights(&std::fs::read(&wpath).unwrap()).expect("resume v4");
        eprintln!("[v4] RESUMED from {}", wpath.display());
        m
    } else if !scratch && v3path.exists() {
        let m = deserialize_weights(&std::fs::read(&v3path).unwrap()).expect("warm-start v3");
        eprintln!("[v4] WARM-STARTED from v3 {}", v3path.display());
        m
    } else {
        Mlp::new_random_with_input(config, HIST_FEATURES, INIT_SEED)
    };
    assert_eq!(mlp.layer_dims()[0].0 as usize, HIST_FEATURES, "v4 net must be 27-input");
    let epoch_start = env_u32("GAIA_V4_EPOCH_START", 0);
    let epoch_total = env_u32("GAIA_V4_EPOCH_TOTAL", epochs).max(1);
    let mut adam = Adam::new(&mlp, lr0, 0.9, 0.999, 1e-8);
    eprintln!("[v4] arch {:?} in={HIST_FEATURES} macs/px={} — training {epochs} epochs, subsample {subsample}px/pose",
        config, mlp.macs());

    // baseline monitor before any step (warm-start read).
    {
        let net = settle_still(&mlp, &val_pose, k);
        let sp = sparkle_resid_per_mpx(&net, &val_pose.teacher, tw, th);
        eprintln!("[v4] MONITOR epoch -1 (start): val sparkle {sp:.1}/Mpx");
    }

    let n_px: Vec<usize> = poses.iter().map(|p| (p.tw * p.th) as usize).collect();
    let mut rng = Rng(INIT_SEED ^ 0xF00D ^ ((epoch_start as u64) << 21));
    let t_train = Instant::now();
    let mut best_sp = f64::INFINITY;
    // KEEP THE BEST net by held-out sparkle (MSE refit re-sharpens dots as it
    // trains, so the LATEST net is not the cleanest — the artifact is the best).
    let mut best_bytes: Vec<u8> = serialize_weights(&mlp);

    for epoch in 0..epochs {
        let frac = (epoch_start + epoch) as f32 / epoch_total as f32;
        adam.set_lr(lr0 / (1.0 + 2.0 * frac));
        let mut epoch_mse = 0.0f64;
        let mut epoch_ff = 0.0f64;
        let mut n_steps = 0u64;

        let mut samples: Vec<(usize, usize)> = Vec::new();
        for (pi, np) in n_px.iter().enumerate() {
            for _ in 0..subsample {
                samples.push((pi, (rng.next() as usize) % np));
            }
        }
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
                let target = p.teacher_dl[px];
                let cap = p.cap[px];
                let mut prev_dl = [0.0f32; 3];
                let mut valid = 0.0f32;
                for step in 0..k as usize {
                    let base = pixel_features(
                        &p.lows[step], p.low_w, p.low_h, p.tw, p.th, tx, ty, albedo, p.normal[px],
                        p.depth[px], Vec2::ZERO,
                    );
                    let feat = hist_features(&base, prev_dl, valid);
                    let out = mlp.forward(&feat);
                    let (mse, ff) = accumulate_backward_firefly(
                        &mlp, &feat, &out, &target, &cap, ff_w, &mut wg, &mut bg,
                        1.0 / (blen * k as f32),
                    );
                    epoch_mse += mse;
                    epoch_ff += ff;
                    n_steps += 1;
                    prev_dl = out;
                    valid = 1.0;
                }
            }
            adam_apply(&mut adam, &mut mlp, &wg, &bg);
            bstart = bend;
        }

        if epoch % 10 == 0 || epoch + 1 == epochs {
            let denom = (n_steps as f64 * OUTPUT_CHANNELS as f64).max(1.0);
            println!(
                "[v4] epoch {}/{} mse={:.6} ff={:.6} ({:.1}s)",
                epoch_start + epoch, epoch_total, epoch_mse / denom, epoch_ff / denom,
                t_train.elapsed().as_secs_f64()
            );
        }

        // ── sparkle monitor on held-out val pose ──
        if (epoch + 1) % monitor_every == 0 || epoch + 1 == epochs {
            let net = settle_still(&mlp, &val_pose, k);
            let sp = sparkle_resid_per_mpx(&net, &val_pose.teacher, tw, th);
            let better = sp < best_sp;
            if better {
                best_sp = sp;
                best_bytes = serialize_weights(&mlp);
                std::fs::write(&wpath, &best_bytes).unwrap(); // best-so-far is the checkpoint
            }
            eprintln!("[v4] MONITOR epoch {}: val sparkle {sp:.1}/Mpx (bar 40, stop<{spark_stop}){}",
                epoch_start + epoch, if better { " *BEST→saved" } else { "" });
            if sp < spark_stop as f64 {
                eprintln!("[v4] EARLY STOP @ epoch {}: val sparkle {sp:.1} < {spark_stop} (best net saved)",
                    epoch_start + epoch);
                break;
            }
        }
    }
    eprintln!("[v4] training done in {:.1}s (best val sparkle {best_sp:.1})", t_train.elapsed().as_secs_f64());

    // THE ARTIFACT IS THE BEST NET (by held-out sparkle), not the last epoch.
    std::fs::write(&wpath, &best_bytes).unwrap();
    let mlp = deserialize_weights(&best_bytes).expect("reload best");
    let wsha = weights_sha256(&mlp);
    println!("[v4] wrote {} sha256={wsha}", wpath.display());
    let prov = serde_json::json!({
        "artifact": "rdirect-weights-v4.bin",
        "weights_sha256": wsha,
        "supersedes": "rdirect-weights-v3.bin",
        "architecture": {
            "kind": "N3 recurrent direct-render MLP — v3 arch + SPATIAL FIREFLY LOSS",
            "input_features": HIST_FEATURES,
            "output_channels": OUTPUT_CHANNELS,
            "hidden_layers": config.hidden_layers,
            "hidden_width": config.hidden_width,
            "macs_per_pixel": mlp.macs(),
        },
        "training": {
            "epochs": epochs, "unroll_steps": k, "batch": batch, "lr0": lr0,
            "subsample_px_per_pose_per_epoch": subsample, "ref_frames": ref_frames,
            "firefly_loss": {
                "weight_ff": ff_w, "cap_window": 2 * cap_radius + 1, "cap_margin": cap_margin,
                "form": "MSE + ff_w·Σ_c relu(out_c − teacher_kxk_max_c − margin)² in demod-log space",
                "note": "penalises isolated bright outliers over dark teacher neighbourhoods; bright edges (high cap) free"
            },
            "warm_start": if !scratch && v3path.exists() { "v3" } else { "scratch" },
            "monitor": "ordeal sparkle metric on held-out val pose orbit_-20, early-stop under bar with margin",
        },
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [tw, th],
            "train": ["front", "wide", "orbit_+20"], "val": ["orbit_-20"] },
        "gate": "presents ONLY if real_image_ordeal writes a PASS stamp beside this file",
    });
    std::fs::write(data_dir.join("rdirect-weights-v4.provenance.json"),
        serde_json::to_string_pretty(&prov).unwrap()).unwrap();
    println!("[v4] wrote provenance. NEXT: run `real_image_ordeal` (GAIA_ORDEAL_WEIGHTS=v4) to earn the stamp.");
}
