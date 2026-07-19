//! R-DIRECT v7 — THE E/D SPLIT AT THE TARGET (the design-note escape).
//! N5 (v6) split the INPUT into E (direct/specular-chain, sharp) and D
//! (post-diffuse-bounce, firefly source) but still supervised a single
//! combined output against the RAW exact teacher with PLAIN MSE — the net
//! had every incentive to chase D's per-pixel 1-spp variance because the
//! target itself carried that variance. Result: sparkle FAIL (253.9/Mpx vs
//! bar 40) at a passing resid. v3-v6 all lived on one sparkle<->resid Pareto
//! front trying to fix this with OUTPUT-side scalar/gated luminance caps
//! (N3/N4) — banned from retry.
//!
//! v7 escapes structurally at the TARGET, not the output: the SAME converged
//! split render that feeds the input taps (trace_headless_split at ref_frames)
//! also builds the training TARGET — E kept EXACT (real light stays sharp,
//! zero smoothing, D's ~0.88% energy share means E carries the frame), D
//! spatially box-blurred BEFORE being added back to form the supervised
//! total. The net still emits ONE combined demod-log radiance (byte-identical
//! shape to v6: 39-in / 3-out, so the live GPU path and the ordeal's loader
//! are untouched) — but it is now trained toward a signal that has had its
//! variance channel smoothed at the SOURCE, so amplifying 1-spp D-input noise
//! no longer reduces loss. No cap, no gate, no firefly weight: PLAIN MSE
//! against the split-smoothed target. This is signal engineering, not a
//! penalty term.
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_train_v7
//!   GAIA_V7_EPOCHS, GAIA_V7_STILL (K), GAIA_V7_SUBSAMPLE, GAIA_V7_W/H,
//!   GAIA_V7_REF, GAIA_V7_BLUR (D box-blur radius, default 2),
//!   GAIA_V7_SPARK_TGT, GAIA_V7_RESID_GATE, GAIA_V7_MONITOR, GAIA_V7_WALL

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use glam::{Vec2, Vec3 as GVec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, headless_device, trace_headless_split,
};
use scrying_glass::rdirect::{
    Adam, HIST_FEATURES_SPLIT, Mlp, RdirectConfig, OUTPUT_CHANNELS,
    accumulate_backward_two_head_slice, adam_apply, deserialize_weights, hist_features_split,
    pixel_features_split, serialize_weights, target_demod_log, weights_sha256, zero_grads,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

const INIT_SEED: u64 = 0x00d1_5eed_0007;

fn env_u32(n: &str, d: u32) -> u32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f32(n: &str, d: f32) -> f32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_i32(n: &str, d: i32) -> i32 {
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

/// Separable-free box blur (small radius, correctness over speed — trainer
/// runs once). CURRENT-FRAME ONLY (single image in, single image out).
fn box_blur(img: &[GVec3], w: u32, h: u32, radius: i32) -> Vec<GVec3> {
    if radius <= 0 {
        return img.to_vec();
    }
    let mut out = vec![GVec3::ZERO; img.len()];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut sum = GVec3::ZERO;
            let mut cnt = 0.0f32;
            for dy in -radius..=radius {
                let ny = y + dy;
                if ny < 0 || ny >= h as i32 {
                    continue;
                }
                for dx in -radius..=radius {
                    let nx = x + dx;
                    if nx < 0 || nx >= w as i32 {
                        continue;
                    }
                    sum += img[(ny as u32 * w + nx as u32) as usize];
                    cnt += 1.0;
                }
            }
            out[(y as u32 * w + x as u32) as usize] = sum / cnt.max(1.0);
        }
    }
    out
}

/// One still pose: K fresh-seed SPLIT low-radiance frames (E, D) at 1spp for
/// the net's input taps, a native G-buffer, the converged EXACT split
/// teacher (E_full, D_full @ ref_frames — their sum is the exact reference
/// used for every metric), and the SMOOTHED-TARGET demod-log (E_full exact +
/// D_full box-blurred) that trains the net.
struct Pose {
    lows_e: Vec<Vec<GVec3>>,
    lows_d: Vec<Vec<GVec3>>,
    albedo: Vec<GVec3>,
    normal: Vec<GVec3>,
    teacher: Vec<GVec3>,       // EXACT (E_full + D_full), metrics only
    target_e: Vec<[f32; 3]>,  // E head target: SHARP (exact E_full), demod-log
    target_d: Vec<[f32; 3]>,  // D head target: SMOOTHED (blur(D_full)), demod-log
    low_w: u32,
    low_h: u32,
    tw: u32,
    th: u32,
    depth_field: Vec<f32>,
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
    blur_radius: i32,
) -> Pose {
    let bvh = Bvh::build(base_tris, &BvhParams::default());
    // SAME pose seeds as v5/v6 (0x7abc + f*131 + 5) — only the target changes.
    let mut lows_e = Vec::with_capacity(k as usize);
    let mut lows_d = Vec::with_capacity(k as usize);
    for f in 0..k {
        let np = IntegratorParams { spp: 1, seed: 0x7abc + f * 131 + 5, ..IntegratorParams::default() };
        let (e, d) = trace_headless_split(
            device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np,
        );
        lows_e.push(e);
        lows_d.push(d);
    }
    let integrator_aov = scrying_glass::integrator::trace_headless_aov(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th,
    );
    let (albedo, normal, depth) = scrying_glass::integrator::split_aov(&integrator_aov);

    // The ONE converged split render: same source feeds the exact metric
    // teacher (E_full+D_full) AND the smoothed training target (E_full +
    // blur(D_full)) — no cross-call discrepancy, single source of truth.
    let (e_full, d_full) = trace_headless_split(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
        &IntegratorParams::default(),
    );
    let d_blurred = box_blur(&d_full, tw, th, blur_radius);

    let n = (tw * th) as usize;
    let teacher: Vec<GVec3> = (0..n).map(|i| e_full[i] + d_full[i]).collect();
    // TWO-HEADED TARGET (the structural fix): E kept EXACT/sharp, D smoothed
    // at the source, each demod-logged against the SAME albedo divisor as
    // the split input taps (completes the input/target symmetry) — no
    // single shared target for the net to trade sharpness for variance on.
    let target_e: Vec<[f32; 3]> = (0..n).map(|px| target_demod_log(e_full[px], albedo[px])).collect();
    let target_d: Vec<[f32; 3]> = (0..n).map(|px| target_demod_log(d_blurred[px], albedo[px])).collect();

    Pose {
        lows_e, lows_d, albedo, normal, teacher, target_e, target_d,
        low_w, low_h, tw, th, depth_field: depth,
    }
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

/// Settle the recurrent split net over K still frames (feeds its own prev out).
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
                let s = step.min(p.lows_e.len() - 1);
                let base = pixel_features_split(
                    &p.lows_e[s], &p.lows_d[s], p.low_w, p.low_h, p.tw, p.th, tx, ty, albedo,
                    p.normal[px], p.depth_field[px], Vec2::ZERO,
                );
                let feat = hist_features_split(&base, prev_dl, valid);
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
        panic!("[v7] no GPU");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();

    let tw = env_u32("GAIA_V7_W", 480);
    let th = env_u32("GAIA_V7_H", 360);
    let low_w = tw / 2;
    let low_h = th / 2;
    let k = env_u32("GAIA_V7_STILL", 4);
    let ref_frames = env_u32("GAIA_V7_REF", 96);
    let epochs = env_u32("GAIA_V7_EPOCHS", 120);
    let subsample = env_u32("GAIA_V7_SUBSAMPLE", 6000) as usize;
    let batch = env_u32("GAIA_V7_BATCH", 64) as usize;
    let lr0 = env_f32("GAIA_V7_LR", 0.002);
    let blur_radius = env_i32("GAIA_V7_BLUR", 2);
    let spark_target = env_f32("GAIA_V7_SPARK_TGT", 16.0);
    let resid_gate = env_f32("GAIA_V7_RESID_GATE", 0.036);
    let monitor_every = env_u32("GAIA_V7_MONITOR", 10);
    let wall_budget = env_f32("GAIA_V7_WALL", 660.0); // seconds — stop early, keep best

    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| all.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();
    let train_cams = [("front", find("front")), ("wide", find("wide")), ("orbit_+20", find("orbit_+20"))];

    let t_render = Instant::now();
    let poses: Vec<Pose> = train_cams
        .iter()
        .map(|(_, c)| render_pose(&device, &queue, &base_tris, &scene, c, k, low_w, low_h, tw, th, ref_frames, blur_radius))
        .collect();
    let val_pose = render_pose(&device, &queue, &base_tris, &scene, &find("orbit_-20"), k, low_w, low_h, tw, th, ref_frames, blur_radius);
    eprintln!("[v7] rendered {}+1 SPLIT poses (K={k}, {tw}x{th}, teacher {ref_frames}, D-blur r={blur_radius}) in {:.1}s",
        poses.len(), t_render.elapsed().as_secs_f64());

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let wpath = data_dir.join("rdirect-weights-v7.bin");
    let config = RdirectConfig {
        hidden_layers: env_u32("GAIA_V7_LAYERS", 5) as usize,
        hidden_width: env_u32("GAIA_V7_WIDTH", 64) as usize,
    };
    // TWO-HEAD net: 39-in / 6-out (E_dl(3) + D_dl(3)). GAIA_V7_RESUME=1 with
    // an existing checkpoint either resumes a two-head net directly, or
    // WARM-STARTS the body from a prior 3-out v7 checkpoint (new D-head
    // columns small-random init) — see `Mlp::warm_start_two_head`.
    let resume = matches!(std::env::var("GAIA_V7_RESUME").as_deref(), Ok("1" | "true"));
    let mut mlp = if resume && wpath.exists() {
        let base = deserialize_weights(&std::fs::read(&wpath).unwrap()).expect("resume v7");
        let out_dim = base.layer_dims().last().unwrap().1 as usize;
        if out_dim == 6 {
            eprintln!("[v7] RESUMED two-head net from {}", wpath.display());
            base
        } else {
            eprintln!("[v7] WARM-START two-head (body copied, D-head small-random) from 3-out {}", wpath.display());
            Mlp::warm_start_two_head(&base, INIT_SEED ^ 0xBEEF)
        }
    } else {
        Mlp::new_random_with_shape(config, HIST_FEATURES_SPLIT, 6, INIT_SEED)
    };
    assert_eq!(mlp.layer_dims()[0].0 as usize, HIST_FEATURES_SPLIT, "v7 net must be 39-input");
    assert_eq!(mlp.layer_dims().last().unwrap().1 as usize, 6, "v7 net must be two-head 6-output");
    let mut adam = Adam::new(&mlp, lr0, 0.9, 0.999, 1e-8);
    eprintln!("[v7] arch {:?} in={HIST_FEATURES_SPLIT} out=6 (two-head) macs/px={} — {epochs} epochs (wall<={wall_budget}s), subsample {subsample}px/pose, split MSE: E vs sharp, D vs smoothed",
        config, mlp.macs());

    {
        let net = settle_still(&mlp, &val_pose, k);
        let sp = sparkle_resid_per_mpx(&net, &val_pose.teacher, tw, th);
        let rs = rmse_lin(&net, &val_pose.teacher);
        eprintln!("[v7] MONITOR epoch -1 (fresh init): val sparkle {sp:.1}/Mpx resid {rs:.4}");
    }

    let n_px: Vec<usize> = poses.iter().map(|p| (p.tw * p.th) as usize).collect();
    let mut rng = Rng(INIT_SEED ^ 0xF00D);
    let t_train = Instant::now();
    let mut best_bytes: Vec<u8> = serialize_weights(&mlp);
    // (B) bar-normalized checkpoint criterion: score = max(sparkle/40,
    // resid/0.035), save on new MIN. Replaces the old resid-primary rule
    // that let a passing-resid/exploding-sparkle epoch overwrite a strictly
    // better (lower on BOTH bars) checkpoint — the exact failure that lost
    // the 46.3-sparkle state to a 92.6-sparkle one in the interrupted run.
    let mut best_score = f64::INFINITY;
    let mut best_sp_log = 0.0f64;
    let mut best_rs_log = 0.0f64;

    'train: for epoch in 0..epochs {
        if t_train.elapsed().as_secs_f64() > wall_budget as f64 {
            eprintln!("[v7] WALL budget {wall_budget}s reached at epoch {epoch} — stopping, keeping best");
            break;
        }
        let frac = epoch as f32 / epochs as f32;
        adam.set_lr(lr0 / (1.0 + 2.0 * frac));
        let mut epoch_mse = 0.0f64;
        let mut n_steps = 0u64;

        let mut samples: Vec<(usize, usize)> = Vec::new();
        for (pi, np) in n_px.iter().enumerate() {
            for _ in 0..subsample {
                samples.push((pi, (rng.next() as usize) % np));
            }
        }
        let mut bstart = 0usize;
        while bstart < samples.len() {
            // (C) WALL BUG FIX: check the budget INSIDE the epoch too — a
            // single epoch (K * subsample * 3 poses steps) can itself run
            // long; the old check only fired at epoch boundaries and let a
            // run go 3x over wall, silent, before this fix.
            if t_train.elapsed().as_secs_f64() > wall_budget as f64 {
                eprintln!("[v7] WALL budget {wall_budget}s reached MID-epoch {epoch} (batch {bstart}/{}) — stopping, keeping best", samples.len());
                std::io::stderr().flush().ok();
                break 'train;
            }
            let bend = (bstart + batch).min(samples.len());
            let (mut wg, mut bg) = zero_grads(&mlp);
            let blen = (bend - bstart) as f32;
            for &(pi, px) in &samples[bstart..bend] {
                let p = &poses[pi];
                let tx = (px as u32) % p.tw;
                let ty = (px as u32) / p.tw;
                let albedo = p.albedo[px];
                let target_e = p.target_e[px]; // sharp E teacher
                let target_d = p.target_d[px]; // smoothed D teacher
                let mut prev_dl = [0.0f32; 3];
                let mut valid = 0.0f32;
                for step in 0..k as usize {
                    let base = pixel_features_split(
                        &p.lows_e[step], &p.lows_d[step], p.low_w, p.low_h, p.tw, p.th, tx, ty,
                        albedo, p.normal[px], p.depth_field[px], Vec2::ZERO,
                    );
                    let feat = hist_features_split(&base, prev_dl, valid);
                    let raw = mlp.forward_full(&feat); // [E_dl(3), D_dl(3)]
                    let (mse_e, mse_d) = accumulate_backward_two_head_slice(
                        &mlp, &feat, &raw, &target_e, &target_d, &mut wg, &mut bg,
                        1.0 / (blen * k as f32),
                    );
                    epoch_mse += mse_e + mse_d;
                    n_steps += 1;
                    // recurrent feedback = the PRESENTED (E+D) demod-log —
                    // undo each head, sum in LINEAR space, re-encode (same
                    // math `Mlp::forward` does internally for a 6-out net;
                    // inlined here via `target_demod_log` to avoid a second
                    // full forward pass through the net every step).
                    let div = if albedo.length_squared() > 1e-8 { albedo + GVec3::splat(1e-3) } else { GVec3::ONE };
                    let e_expm1 = GVec3::new((raw[0].exp() - 1.0).max(0.0), (raw[1].exp() - 1.0).max(0.0), (raw[2].exp() - 1.0).max(0.0));
                    let d_expm1 = GVec3::new((raw[3].exp() - 1.0).max(0.0), (raw[4].exp() - 1.0).max(0.0), (raw[5].exp() - 1.0).max(0.0));
                    let combined_lin = e_expm1 * div + d_expm1 * div;
                    prev_dl = target_demod_log(combined_lin, albedo);
                    valid = 1.0;
                }
            }
            adam_apply(&mut adam, &mut mlp, &wg, &bg);
            bstart = bend;
        }

        if epoch % 10 == 0 || epoch + 1 == epochs {
            let denom = (n_steps as f64 * 6.0).max(1.0);
            println!("[v7] epoch {}/{} mse={:.6} ({:.1}s)",
                epoch, epochs, epoch_mse / denom, t_train.elapsed().as_secs_f64());
            std::io::stdout().flush().ok();
        }

        if (epoch + 1) % monitor_every == 0 || epoch + 1 == epochs {
            let net = settle_still(&mlp, &val_pose, k);
            let sp = sparkle_resid_per_mpx(&net, &val_pose.teacher, tw, th);
            let rs = rmse_lin(&net, &val_pose.teacher);
            let passes = sp < spark_target as f64 && rs < resid_gate as f64;
            let score = (sp / 40.0).max(rs / 0.035);
            let better = score < best_score;
            if better {
                best_score = score;
                best_sp_log = sp;
                best_rs_log = rs;
                best_bytes = serialize_weights(&mlp);
                std::fs::write(&wpath, &best_bytes).unwrap();
            }
            eprintln!("[v7] MONITOR epoch {}: val sparkle {sp:.1}/Mpx resid {rs:.4} score={score:.3}{} (tgt sp<{spark_target} resid<{resid_gate}){}",
                epoch, if passes { " PASS" } else { "" }, if better { " *BEST->saved" } else { "" });
            std::io::stderr().flush().ok();
        }
    }
    eprintln!("[v7] training done in {:.1}s (best score={best_score:.3} sparkle {best_sp_log:.1} resid {best_rs_log:.4})",
        t_train.elapsed().as_secs_f64());
    std::io::stderr().flush().ok();

    std::fs::write(&wpath, &best_bytes).unwrap();
    let mlp = deserialize_weights(&best_bytes).expect("reload best");
    let wsha = weights_sha256(&mlp);
    println!("[v7] wrote {} sha256={wsha}", wpath.display());
    let prov = serde_json::json!({
        "artifact": "rdirect-weights-v7.bin",
        "weights_sha256": wsha,
        "supersedes": "rdirect-weights-v6.bin",
        "architecture": {
            "kind": "N7 TWO-HEAD, structural fix — recurrent direct-render MLP, split radiance input (E+D), TWO separate output heads (E_dl, D_dl) each supervised against its own teacher (E sharp/exact, D box-blurred), summed at present time",
            "input_features": HIST_FEATURES_SPLIT,
            "output_channels": 6,
            "output_heads": "E_dl[0..3] vs sharp E teacher, D_dl[3..6] vs smoothed D teacher; Mlp::forward() head-sums them back to one 3-channel demod-log for every existing caller (ordeal/live-loader unchanged)",
            "hidden_layers": config.hidden_layers,
            "hidden_width": config.hidden_width,
            "macs_per_pixel": mlp.macs(),
        },
        "training": {
            "epochs": epochs, "unroll_steps": k, "batch": batch, "lr0": lr0,
            "subsample_px_per_pose_per_epoch": subsample, "ref_frames": ref_frames,
            "init": if resume { "WARM-START two-head (body copied from prior v7 3-out checkpoint if needed, D-head columns small-random) or direct two-head resume" } else { "FRESH two-head (39-in/6-out)" },
            "loss": "TWO separate MSE terms summed: E head vs sharp E teacher, D head vs smoothed D teacher (no cap, no gate, no firefly weight, no shared target)",
            "target_construction": "target_e = demod_log(E_full exact, ref_frames); target_d = demod_log(box_blur(D_full, radius)) — separate per-head targets, not a single combined smoothed target (v7's first attempt traded E sharpness for D variance under one target; this cannot recur since each head only ever sees its own teacher)",
            "best_checkpoint_criterion": "bar-normalized score = max(sparkle/40, resid/0.035), save on new MIN (replaces resid-primary rule that could overwrite a strictly-better checkpoint)",
            "d_blur_radius": blur_radius,
            "split": "E = radiance via zero-or-more specular/low-roughness bounces (SPEC_CHAIN_MAX_ROUGHNESS 0.25); D = radiance after a diffuse/rough scatter (~0.88% of frame energy)",
        },
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [tw, th],
            "train": ["front", "wide", "orbit_+20"], "val": ["orbit_-20"] },
        "gate": "presents ONLY if real_image_ordeal writes a PASS stamp beside this file",
    });
    std::fs::write(data_dir.join("rdirect-weights-v7.provenance.json"),
        serde_json::to_string_pretty(&prov).unwrap()).unwrap();
    println!("[v7] wrote provenance. NEXT: real_image_ordeal (GAIA_ORDEAL_WEIGHTS=v7) to earn the stamp.");
}
