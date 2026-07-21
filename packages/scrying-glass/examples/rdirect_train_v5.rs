//! R-DIRECT v5 — THE TEACHER-GATED FIREFLY LOSS (N4, the Pareto escape).
//! Same arch/recurrence as v3 (27-in, 5×64, still-unroll BPTT). N3's v4 proved
//! a SPATIAL firefly clamp kills the invented dots (sparkle 345→39/Mpx PASS) but
//! at ff_w=15 it STRANGLED the real cyan waterline neon into a broken smear
//! (resid 0.0325→0.051 FAIL). Diagnosis on record: a scalar clamp cannot tell an
//! invented dot from a pixel bordering a REAL emissive — both sit above the dark
//! local cap. N4 fixes that with a TEACHER GATE:
//!
//!   LOSS = MSE(out, teacher) + gate · ff_w · Σ_c relu(out_c − cap_c)²
//!
//! where cap_c is the TEACHER's k×k neighbourhood max (demod-log) + margin, AND
//! `gate` is 1.0 ONLY where the teacher's k×k neighbourhood is genuinely DARK
//! (its max luminance below a scene-adaptive percentile ceiling) and 0.0 where
//! the teacher itself is bright (real neon / lit windows / the cyan waterline).
//! Where the teacher is bright plain MSE rules — the net renders the real light
//! exactly, no clamp can smear it. Where the teacher is dark an invented firefly
//! over the cap is crushed. `ff_w` and the percentile are the IRON params.
//! Ordeal thresholds are NOT touched. Warm-starts from v3 (sparkle-fail but
//! SHARP) — never from v4 (its smear bias is baked in).
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_train_v5
//!   GAIA_V5_EPOCHS, GAIA_V5_STILL (K), GAIA_V5_SUBSAMPLE, GAIA_V5_W/H, GAIA_V5_REF,
//!   GAIA_V5_FF (firefly weight), GAIA_V5_FFK (cap window radius), GAIA_V5_FFMARGIN,
//!   GAIA_V5_DARKPCT (teacher-luminance percentile: neighbourhoods brighter than
//!     this are gated OFF), GAIA_V5_SPARK_TGT / GAIA_V5_RESID_GATE, GAIA_V5_MONITOR

use std::path::Path;
use std::time::Instant;

use glam::{Vec2, Vec3 as GVec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::rdirect::{
    Adam, HIST_FEATURES, Mlp, RdirectConfig, OUTPUT_CHANNELS, accumulate_backward_firefly_gated,
    adam_apply, deserialize_weights, hist_features, pixel_features, serialize_weights,
    target_demod_log, weights_sha256, zero_grads,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

const INIT_SEED: u64 = 0x00d1_5eed_0005;

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

fn lum(c: GVec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// One still pose: K fresh-seed low-radiance frames, a native G-buffer, teacher,
/// the teacher demod-log target field, its k×k neighbourhood cap (net output
/// space), AND the N4 per-pixel DARK GATE (1.0 where the teacher neighbourhood
/// is genuinely dark, 0.0 where a real emissive lives nearby).
struct Pose {
    lows: Vec<Vec<GVec3>>,
    albedo: Vec<GVec3>,
    normal: Vec<GVec3>,
    depth: Vec<f32>,
    teacher: Vec<GVec3>,
    teacher_dl: Vec<[f32; OUTPUT_CHANNELS]>,
    cap: Vec<[f32; OUTPUT_CHANNELS]>,
    gate: Vec<f32>, // N4: firefly-penalty gate (1 dark, 0 near real light)
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
    dark_pct: f32,
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
    let n = (tw * th) as usize;
    let teacher_dl: Vec<[f32; OUTPUT_CHANNELS]> =
        (0..n).map(|px| target_demod_log(teacher[px], albedo[px])).collect();

    // scene-adaptive DARK CEILING: the dark_pct-th percentile of teacher
    // luminance across the pose. Neighbourhoods whose max luminance exceeds this
    // hold a real emissive → gate OFF (plain MSE). This is the IRON percentile.
    let mut lums: Vec<f32> = teacher.iter().map(|c| lum(*c)).collect();
    lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dark_ceiling = lums[((dark_pct.clamp(0.0, 1.0) * (n as f32 - 1.0)) as usize).min(n - 1)];

    let mut cap = vec![[f32::NEG_INFINITY; OUTPUT_CHANNELS]; n];
    let mut gate = vec![0.0f32; n];
    for y in 0..th as i32 {
        for x in 0..tw as i32 {
            let mut m = [f32::NEG_INFINITY; OUTPUT_CHANNELS];
            let mut lmax = 0.0f32; // teacher neighbourhood max luminance (for the gate)
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
                    let nidx = (ny as u32 * tw + nx as u32) as usize;
                    let t = teacher_dl[nidx];
                    for c in 0..OUTPUT_CHANNELS {
                        if t[c] > m[c] {
                            m[c] = t[c];
                        }
                    }
                    let l = lum(teacher[nidx]);
                    if l > lmax {
                        lmax = l;
                    }
                }
            }
            let idx = (y as u32 * tw + x as u32) as usize;
            for c in 0..OUTPUT_CHANNELS {
                cap[idx][c] = m[c] + cap_margin;
            }
            // GATE: dark neighbourhood (no real light) → penalise fireflies.
            gate[idx] = if lmax < dark_ceiling { 1.0 } else { 0.0 };
        }
    }
    let gated = gate.iter().filter(|g| **g > 0.5).count();
    eprintln!("[v5]   pose dark_ceiling={dark_ceiling:.4} (pct {dark_pct}) gate ON for {}/{} px ({:.0}%)",
        gated, n, 100.0 * gated as f32 / n as f32);
    Pose { lows, albedo, normal, depth, teacher, teacher_dl, cap, gate, low_w, low_h, tw, th }
}

fn rmse_lin(net: &[GVec3], teacher: &[GVec3]) -> f64 {
    let mut s = 0.0f64;
    for (a, b) in net.iter().zip(teacher) {
        let d = *a - *b;
        s += (d.x * d.x + d.y * d.y + d.z * d.z) as f64;
    }
    (s / (net.len() as f64 * 3.0)).sqrt()
}

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
        panic!("[v5] no GPU");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();

    let tw = env_u32("GAIA_V5_W", 480);
    let th = env_u32("GAIA_V5_H", 360);
    let low_w = tw / 2;
    let low_h = th / 2;
    let k = env_u32("GAIA_V5_STILL", 4);
    let ref_frames = env_u32("GAIA_V5_REF", 96);
    let epochs = env_u32("GAIA_V5_EPOCHS", 120);
    let subsample = env_u32("GAIA_V5_SUBSAMPLE", 6000) as usize;
    let batch = env_u32("GAIA_V5_BATCH", 64) as usize;
    let lr0 = env_f32("GAIA_V5_LR", 0.002);
    // ── IRON teacher-gated firefly params ──
    let ff_w = env_f32("GAIA_V5_FF", 15.0); // firefly clamp weight (N3's proven-lethal weight)
    let cap_radius = env_u32("GAIA_V5_FFK", 2) as i32; // k×k window = (2r+1)²
    let cap_margin = env_f32("GAIA_V5_FFMARGIN", 0.05); // demod-log slack over the local max
    let dark_pct = env_f32("GAIA_V5_DARKPCT", 0.80); // teacher-lum percentile: brighter → gate OFF
    let spark_target = env_f32("GAIA_V5_SPARK_TGT", 16.0);
    let resid_gate = env_f32("GAIA_V5_RESID_GATE", 0.036);
    let monitor_every = env_u32("GAIA_V5_MONITOR", 10);

    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| all.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();
    let train_cams = [("front", find("front")), ("wide", find("wide")), ("orbit_+20", find("orbit_+20"))];

    let t_render = Instant::now();
    let poses: Vec<Pose> = train_cams
        .iter()
        .map(|(_, c)| render_pose(&device, &queue, &base_tris, &scene, c, k, low_w, low_h, tw, th, ref_frames, cap_radius, cap_margin, dark_pct))
        .collect();
    let val_pose = render_pose(&device, &queue, &base_tris, &scene, &find("orbit_-20"), k, low_w, low_h, tw, th, ref_frames, cap_radius, cap_margin, dark_pct);
    eprintln!("[v5] rendered {}+1 poses (K={k}, {tw}x{th}, teacher {ref_frames}, cap {}×{} margin {cap_margin}, ff_w {ff_w}, dark_pct {dark_pct}) in {:.1}s",
        poses.len(), 2 * cap_radius + 1, 2 * cap_radius + 1, t_render.elapsed().as_secs_f64());

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let wpath = data_dir.join("rdirect-weights-v5.bin");
    let config = RdirectConfig {
        hidden_layers: env_u32("GAIA_V5_LAYERS", 5) as usize,
        hidden_width: env_u32("GAIA_V5_WIDTH", 64) as usize,
    };
    // warm-start from v3 (sparkle-fail but SHARP) — NEVER v4 (its smear is baked).
    let v3path = data_dir.join("rdirect-weights-v3.bin");
    let scratch = matches!(std::env::var("GAIA_V5_SCRATCH").as_deref(), Ok("1" | "true"));
    let resume = matches!(std::env::var("GAIA_V5_RESUME").as_deref(), Ok("1" | "true"));
    let mut mlp = if resume && wpath.exists() {
        let m = deserialize_weights(&std::fs::read(&wpath).unwrap()).expect("resume v5");
        eprintln!("[v5] RESUMED from {}", wpath.display());
        m
    } else if !scratch && v3path.exists() {
        let m = deserialize_weights(&std::fs::read(&v3path).unwrap()).expect("warm-start v3");
        eprintln!("[v5] WARM-STARTED from v3 {}", v3path.display());
        m
    } else {
        Mlp::new_random_with_input(config, HIST_FEATURES, INIT_SEED)
    };
    assert_eq!(mlp.layer_dims()[0].0 as usize, HIST_FEATURES, "v5 net must be 27-input");
    let mut adam = Adam::new(&mlp, lr0, 0.9, 0.999, 1e-8);
    eprintln!("[v5] arch {:?} in={HIST_FEATURES} macs/px={} — training {epochs} epochs, subsample {subsample}px/pose",
        config, mlp.macs());

    {
        let net = settle_still(&mlp, &val_pose, k);
        let sp = sparkle_resid_per_mpx(&net, &val_pose.teacher, tw, th);
        let rs = rmse_lin(&net, &val_pose.teacher);
        eprintln!("[v5] MONITOR epoch -1 (start=v3): val sparkle {sp:.1}/Mpx resid {rs:.4}");
    }

    let n_px: Vec<usize> = poses.iter().map(|p| (p.tw * p.th) as usize).collect();
    let mut rng = Rng(INIT_SEED ^ 0xF00D);
    let t_train = Instant::now();
    // Keep the BEST net by a COMBINED held-out criterion: among nets whose
    // sparkle clears the target, the LOWEST resid; if none clear it yet, the
    // lowest sparkle.
    let mut best_bytes: Vec<u8> = serialize_weights(&mlp);
    let mut best_sp = f64::INFINITY;
    let mut best_resid = f64::INFINITY;
    let mut have_pass = false;

    for epoch in 0..epochs {
        let frac = epoch as f32 / epochs as f32;
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
                let gate = p.gate[px];
                let mut prev_dl = [0.0f32; 3];
                let mut valid = 0.0f32;
                for step in 0..k as usize {
                    let base = pixel_features(
                        &p.lows[step], p.low_w, p.low_h, p.tw, p.th, tx, ty, albedo, p.normal[px],
                        p.depth[px], Vec2::ZERO,
                    );
                    let feat = hist_features(&base, prev_dl, valid);
                    let out = mlp.forward(&feat);
                    let (mse, ff) = accumulate_backward_firefly_gated(
                        &mlp, &feat, &out, &target, &cap, gate, ff_w, &mut wg, &mut bg,
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
                "[v5] epoch {}/{} mse={:.6} ff={:.6} ({:.1}s)",
                epoch, epochs, epoch_mse / denom, epoch_ff / denom,
                t_train.elapsed().as_secs_f64()
            );
        }

        if (epoch + 1) % monitor_every == 0 || epoch + 1 == epochs {
            let net = settle_still(&mlp, &val_pose, k);
            let sp = sparkle_resid_per_mpx(&net, &val_pose.teacher, tw, th);
            let rs = rmse_lin(&net, &val_pose.teacher);
            let passes = sp < spark_target as f64 && rs < resid_gate as f64;
            let mut better = false;
            if passes {
                if !have_pass || rs < best_resid {
                    have_pass = true;
                    best_resid = rs;
                    better = true;
                }
            } else if !have_pass && sp < best_sp {
                best_sp = sp;
                better = true;
            }
            if better {
                best_bytes = serialize_weights(&mlp);
                std::fs::write(&wpath, &best_bytes).unwrap();
            }
            eprintln!("[v5] MONITOR epoch {}: val sparkle {sp:.1}/Mpx resid {rs:.4} (tgt sp<{spark_target} resid<{resid_gate}){}",
                epoch, if better { " *BEST→saved" } else { "" });
        }
    }
    eprintln!("[v5] training done in {:.1}s (best combined: pass={have_pass} resid {best_resid:.4} / fallback sparkle {best_sp:.1})",
        t_train.elapsed().as_secs_f64());

    std::fs::write(&wpath, &best_bytes).unwrap();
    let mlp = deserialize_weights(&best_bytes).expect("reload best");
    let wsha = weights_sha256(&mlp);
    println!("[v5] wrote {} sha256={wsha}", wpath.display());
    let prov = serde_json::json!({
        "artifact": "rdirect-weights-v5.bin",
        "weights_sha256": wsha,
        "supersedes": "rdirect-weights-v4.bin",
        "architecture": {
            "kind": "N4 recurrent direct-render MLP — v3 arch + TEACHER-GATED SPATIAL FIREFLY LOSS",
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
                "dark_percentile": dark_pct,
                "form": "MSE + gate·ff_w·Σ_c relu(out_c − teacher_kxk_max_c − margin)² in demod-log space",
                "gate": "1 where teacher k×k neighbourhood max-luminance < dark_pct-percentile ceiling, else 0",
                "note": "penalises invented dots ONLY over dark teacher neighbourhoods; real neon/windows are plain MSE (gate=0) — fixes N3 v4's over-clamp of the cyan waterline"
            },
            "warm_start": if !scratch && v3path.exists() { "v3 (sharp, sparkle-fail)" } else { "scratch" },
            "monitor": "ordeal sparkle + resid on held-out val pose orbit_-20, keep lowest-resid sparkle-passing net",
        },
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [tw, th],
            "train": ["front", "wide", "orbit_+20"], "val": ["orbit_-20"] },
        "gate": "presents ONLY if real_image_ordeal writes a PASS stamp beside this file",
    });
    std::fs::write(data_dir.join("rdirect-weights-v5.provenance.json"),
        serde_json::to_string_pretty(&prov).unwrap()).unwrap();
    println!("[v5] wrote provenance. NEXT: run `real_image_ordeal` (GAIA_ORDEAL_WEIGHTS=v5) to earn the stamp.");
}
