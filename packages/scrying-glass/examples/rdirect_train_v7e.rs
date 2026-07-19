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
//! total. The net still emits ONE combined demod-log radiance (39-in / 3-out)
//! — but it is now trained toward a signal that has had its variance channel
//! smoothed at the SOURCE, so amplifying 1-spp D-input noise no longer
//! reduces loss. No cap, no gate, no firefly weight: PLAIN MSE against the
//! split-smoothed target. This is signal engineering, not a penalty term.
//!
//! v7d — THE CONVERGENCE CURE (2026-07-19). v7c tried a TWO-HEAD net
//! (separate E/D output heads, each vs its own teacher) to kill the
//! sparkle<->resid seesaw structurally instead of chasing it with more
//! epochs — FAILED, made the seesaw WORSE (resid 0.0314 but sparkle 225.7 at
//! epoch 29, still climbing). Two-head is BANNED from retry (recorded, see
//! scratch/v7c-train.log). The v7-resume run (plain continued training from
//! the 636c8743 3-out checkpoint) showed the SAME seesaw signature: sparkle
//! climbs monotonically epoch over epoch (46.3 -> 92.6 -> 156.2 -> 243.1)
//! while resid creeps down — a noise-ball outlier growing under an
//! unconstrained late-training LR, exactly the failure mode
//! magic-crystal-nrc/docs/perf/2026-07-17-nrc-spike-verdict.md diagnosed and
//! cured with two levers: (1) a MUCH lower starting LR with harmonic decay
//! (prevents the optimizer from ever re-opening the noise-ball), (2)
//! Polyak/EMA weight averaging (the checkpoint that gets measured/saved is
//! a slow-moving average of the optimizer's actual weights, which cannot
//! itself develop a sharp outlier the way the raw SGD/Adam iterate can).
//! v7d applies BOTH here, fine-tuning FROM the best-ever 3-out checkpoint
//! (636c8743, sparkle 46.3/resid 0.0361) — NOT two-head, NOT output
//! penalties (still banned). A CROSS-RUN score floor (best-checkpoint =
//! bar-normalized score max(sparkle/40, resid/0.035), saved only on a NEW
//! MINIMUM that beats the starting checkpoint's own score) means this run
//! can never regress the best state on disk even if it never beats it.
//!
//! v7e — THE EVIDENCE CLAMP, TRAINABLE (2026-07-19, evidence-clamp round).
//! v7d's cure (lower LR + EMA + score floor) fixed the noise-ball-outlier
//! SHAPE but left the underlying cause untouched: autopsy abbcb64 found the
//! net systematically overshoots genuinely-bright E-structure by 1.15x-4.1x
//! at sparkle outliers (COPIED-dominant — the smoothed TARGET itself is
//! locally bright there, the net just over-amplifies past it). v7e adds a
//! CLAMP AT THE ACT (architecture, not a loss penalty — no cap/gate/firefly
//! weight, still banned): `presented_dl = min(out_dl, ceiling_dl)` where
//! `ceiling_dl = log_demod(gamma * local_max(evidence))`, evidence = the SAME
//! E+D taps the net's OWN input reads (temporal-mean over the K settle steps
//! seen so far + spatial 3x3 max — see rdirect::EvidenceAccum's doc for why
//! max-across-time alone is a dead end), gamma DERIVED in
//! scratch/v7e-gamma-derive.log (non-outlier p99.9 ratio ~1.5, outlier
//! median ~1.6-2.0 — GAMMA=1.5, see rdirect::EVIDENCE_CLAMP_GAMMA_DEFAULT).
//! Backprop through the clamp: gradient 0 on a clamped channel (the loss can
//! no longer reward pushing the net past the ceiling), identity otherwise —
//! same plain MSE, only the presented ACT changes. Resumes from the restored
//! 636c8743 checkpoint exactly like v7d (same LR/EMA/score-floor cure, now
//! WITH the clamp active during fine-tune too, so overshoot cannot regrow).
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_train_v7e
//!   Same envs as rdirect_train_v7 (v7d cure), PLUS GAIA_V7_CLAMP_GAMMA
//!   (evidence-clamp gamma override, default derived 1.5).
//!   GAIA_V7_EPOCHS, GAIA_V7_STILL (K), GAIA_V7_SUBSAMPLE, GAIA_V7_W/H,
//!   GAIA_V7_REF, GAIA_V7_BLUR (D box-blur radius, default 2),
//!   GAIA_V7_SPARK_TGT, GAIA_V7_RESID_GATE, GAIA_V7_MONITOR, GAIA_V7_WALL,
//!   GAIA_V7_RESUME (1 = fine-tune from existing rdirect-weights-v7.bin),
//!   GAIA_V7_LR (default 0.3x the prior baseline lr, harmonic-decayed),
//!   GAIA_V7_EMA (Polyak/EMA decay, default 0.999),
//!   GAIA_V7_TWOHEAD (DISABLED — the two-head net is a recorded FAIL, this
//!   flag exists only to fail loudly if something still sets it).

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
    EvidenceAccum, accumulate_backward_clamped_slice, adam_apply, clamp_evidence_lin,
    deserialize_weights, evidence_clamp_gamma, evidence_ceiling_demod_log,
    evidence_composite_frame, hist_features_split, pixel_features_split, serialize_weights,
    target_demod_log, weights_sha256, zero_grads,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

const INIT_SEED: u64 = 0x00d1_5eed_0007;
/// The prior (v7/v7-resume) baseline lr0 — the CURE starts at ~0.3x this.
const PRIOR_LR: f32 = 0.002;
/// Bar-normalized score of the best-ever checkpoint (636c8743, sparkle
/// 46.3/Mpx resid 0.0361: max(46.3/40, 0.0361/0.035) = 1.1575). The
/// cross-run rule: this run may only overwrite that checkpoint by beating
/// this floor, and every subsequent save must beat the PREVIOUS best — a
/// worse run can never clobber the best state again.
const START_SCORE_FLOOR: f64 = 1.157;

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
    teacher: Vec<GVec3>,             // EXACT (E_full + D_full), metrics only
    target_dl: Vec<[f32; OUTPUT_CHANNELS]>, // SMOOTHED (E_full + blur(D_full)), trains the net
    low_w: u32,
    low_h: u32,
    tw: u32,
    th: u32,
    depth_field: Vec<f32>,
    // v7e EVIDENCE CLAMP: per-settle-step ceiling, ALREADY in demod-log space
    // (evidence_ceiling_demod_log) at every pixel — ceiling_dl_steps[step][px].
    // Precomputed once per pose (lows_e/lows_d are fixed after render_pose;
    // ceiling doesn't depend on epoch/weights), reused every epoch.
    ceiling_dl_steps: Vec<Vec<[f32; OUTPUT_CHANNELS]>>,
    // RAW (gamma-free) local_max_evidence in LINEAR space at the LAST settle
    // step's accumulation (for settle_still's final clamp via
    // clamp_evidence_lin, matching the ordeal/CPU-reference act exactly).
    evidence_lin_last: Vec<GVec3>,
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
    let smoothed: Vec<GVec3> = (0..n).map(|i| e_full[i] + d_blurred[i]).collect();
    let target_dl: Vec<[f32; OUTPUT_CHANNELS]> =
        (0..n).map(|px| target_demod_log(smoothed[px], albedo[px])).collect();

    // v7e EVIDENCE CLAMP ceiling, precomputed once (lows_e/lows_d are fixed
    // for this pose): temporal-mean accumulate across the K settle taps +
    // spatial 3x3 max (EvidenceAccum), snapshotted after EVERY step so the
    // training loop's per-step loss sees the SAME ceiling the net's own
    // history has actually accumulated by that step.
    let gamma = evidence_clamp_gamma();
    let mut accum = EvidenceAccum::new(tw, th);
    let mut ceiling_dl_steps: Vec<Vec<[f32; OUTPUT_CHANNELS]>> = Vec::with_capacity(k as usize);
    let mut evidence_lin_last: Vec<GVec3> = Vec::new();
    for step in 0..k as usize {
        let frame_composite = evidence_composite_frame(&lows_e[step], &lows_d[step], low_w, low_h, tw, th);
        accum.push(&frame_composite);
        let local_max = accum.ceiling(); // raw evidence, gamma NOT applied yet
        let ceiling_dl: Vec<[f32; OUTPUT_CHANNELS]> =
            (0..n).map(|px| evidence_ceiling_demod_log(local_max[px], gamma, albedo[px])).collect();
        ceiling_dl_steps.push(ceiling_dl);
        evidence_lin_last = local_max;
    }

    Pose {
        lows_e, lows_d, albedo, normal, teacher, target_dl,
        low_w, low_h, tw, th, depth_field: depth,
        ceiling_dl_steps, evidence_lin_last,
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
            let net_lin = GVec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0)) * div;
            // v7e: clamp the PRESENTED (settled) frame exactly like the
            // ordeal/CPU-reference act — same evidence source, same gamma.
            out[px] = clamp_evidence_lin(net_lin, p.evidence_lin_last[px], evidence_clamp_gamma());
        }
    }
    out
}

fn main() {
    let two_head = matches!(std::env::var("GAIA_V7_TWOHEAD").as_deref(), Ok("1" | "true"));
    if two_head {
        panic!("[v7] GAIA_V7_TWOHEAD is disabled: the two-head net was tried and FAILED \
            (resid 0.0314 but sparkle 225.7/Mpx @ep29, worse seesaw than the 3-out baseline) \
            — see scratch/v7c-train.log and commit 90c4292. This trainer is 3-out only now.");
    }

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
    let lr0 = env_f32("GAIA_V7_LR", PRIOR_LR * 0.3); // THE CURE (a): ~0.3x prior lr
    let ema_decay = env_f32("GAIA_V7_EMA", 0.999); // THE CURE (b): Polyak/EMA
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
    std::io::stderr().flush().ok();

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let wpath = data_dir.join("rdirect-weights-v7.bin");
    let config = RdirectConfig {
        hidden_layers: env_u32("GAIA_V7_LAYERS", 5) as usize,
        hidden_width: env_u32("GAIA_V7_WIDTH", 64) as usize,
    };
    // FRESH init (same shape as v6: 39-in / 3-out - target engineering only),
    // unless GAIA_V7_RESUME=1 and a v7 checkpoint already exists (fine-tune
    // — the CURE round resumes from the restored 636c8743 best-ever state).
    let resume = matches!(std::env::var("GAIA_V7_RESUME").as_deref(), Ok("1" | "true"));
    let mut mlp = if resume && wpath.exists() {
        let m = deserialize_weights(&std::fs::read(&wpath).unwrap()).expect("resume v7");
        assert_eq!(m.layer_dims().last().unwrap().1 as usize, 3, "v7 CURE fine-tune expects a 3-out checkpoint");
        eprintln!("[v7] RESUMED (fine-tune) from {} sha256={}", wpath.display(), weights_sha256(&m));
        m
    } else {
        Mlp::new_random_with_input(config, HIST_FEATURES_SPLIT, INIT_SEED)
    };
    assert_eq!(mlp.layer_dims()[0].0 as usize, HIST_FEATURES_SPLIT, "v7 net must be 39-input");
    let mut adam = Adam::new(&mlp, lr0, 0.9, 0.999, 1e-8);
    // THE CURE (b): shadow net tracks a Polyak/EMA of the live weights;
    // MONITOR and the saved checkpoint always evaluate the EMA net, not the
    // raw (possibly noise-ball-outlier) Adam iterate.
    let mut ema_mlp = mlp.clone();
    eprintln!("[v7] arch {:?} in={HIST_FEATURES_SPLIT} macs/px={} — {epochs} epochs (wall<={wall_budget}s), subsample {subsample}px/pose, lr0={lr0} (harmonic decay), ema={ema_decay}, PLAIN MSE vs split-smoothed target",
        config, mlp.macs());
    std::io::stderr().flush().ok();

    {
        let net = settle_still(&ema_mlp, &val_pose, k);
        let sp = sparkle_resid_per_mpx(&net, &val_pose.teacher, tw, th);
        let rs = rmse_lin(&net, &val_pose.teacher);
        let score = (sp / 40.0).max(rs / 0.035);
        eprintln!("[v7] MONITOR epoch -1 (fresh/resumed, EMA=live): val sparkle {sp:.1}/Mpx resid {rs:.4} score={score:.3} (start floor {START_SCORE_FLOOR})");
        std::io::stderr().flush().ok();
    }

    let n_px: Vec<usize> = poses.iter().map(|p| (p.tw * p.th) as usize).collect();
    let mut rng = Rng(INIT_SEED ^ 0xF00D);
    let t_train = Instant::now();
    let mut best_bytes: Vec<u8> = serialize_weights(&ema_mlp);
    // THE CURE (c): bar-normalized checkpoint criterion, score = max
    // (sparkle/40, resid/0.035), save on new MIN — and CROSS-RUN: this run
    // may only overwrite the starting checkpoint by beating
    // START_SCORE_FLOOR (the 636c8743 state's own score); every later save
    // must then beat that, so a worse run can never clobber the best again.
    let mut best_score = START_SCORE_FLOOR;
    let mut best_sp_log = 0.0f64;
    let mut best_rs_log = 0.0f64;

    'train: for epoch in 0..epochs {
        if t_train.elapsed().as_secs_f64() > wall_budget as f64 {
            eprintln!("[v7] WALL budget {wall_budget}s reached at epoch {epoch} — stopping, keeping best");
            break;
        }
        let frac = epoch as f32 / epochs as f32;
        adam.set_lr(lr0 / (1.0 + 2.0 * frac)); // THE CURE (a): harmonic decay from the lowered lr0
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
            // THE CURE (d) / WALL BUG FIX: check the budget INSIDE the epoch
            // too — a single epoch (K * subsample * 3 poses steps) can
            // itself run long; an epoch-boundary-only check let a run go 3x
            // over wall, silent, before this fix.
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
                let target = p.target_dl[px]; // the SMOOTHED target — this is the escape
                let mut prev_dl = [0.0f32; 3];
                let mut valid = 0.0f32;
                for step in 0..k as usize {
                    let base = pixel_features_split(
                        &p.lows_e[step], &p.lows_d[step], p.low_w, p.low_h, p.tw, p.th, tx, ty,
                        albedo, p.normal[px], p.depth_field[px], Vec2::ZERO,
                    );
                    let feat = hist_features_split(&base, prev_dl, valid);
                    let out = mlp.forward(&feat);
                    // v7e: backprop plain MSE(presented, target) through the
                    // ARCHITECTURAL clamp (this step's own precomputed
                    // ceiling) — NOT a loss penalty (no cap/gate/ff_w), the
                    // ACT itself changes, structural per the CURE round.
                    let ceiling_dl = &p.ceiling_dl_steps[step][px];
                    let mse = accumulate_backward_clamped_slice(
                        &mlp, &feat, &out, ceiling_dl, &target, &mut wg, &mut bg,
                        1.0 / (blen * k as f32),
                    );
                    epoch_mse += mse;
                    n_steps += 1;
                    prev_dl = out;
                    valid = 1.0;
                }
            }
            adam_apply(&mut adam, &mut mlp, &wg, &bg);
            ema_mlp.ema_update(&mlp, ema_decay); // THE CURE (b): shadow update per optimizer step
            bstart = bend;
        }

        if epoch % 10 == 0 || epoch + 1 == epochs {
            let denom = (n_steps as f64 * OUTPUT_CHANNELS as f64).max(1.0);
            println!("[v7] epoch {}/{} mse={:.6} lr={:.6} ({:.1}s)",
                epoch, epochs, epoch_mse / denom, adam.lr(), t_train.elapsed().as_secs_f64());
            std::io::stdout().flush().ok();
        }

        if (epoch + 1) % monitor_every == 0 || epoch + 1 == epochs {
            // THE CURE (b): monitor evaluates the EMA weights, not raw `mlp`.
            let net = settle_still(&ema_mlp, &val_pose, k);
            let sp = sparkle_resid_per_mpx(&net, &val_pose.teacher, tw, th);
            let rs = rmse_lin(&net, &val_pose.teacher);
            let passes = sp < spark_target as f64 && rs < resid_gate as f64;
            let score = (sp / 40.0).max(rs / 0.035);
            let better = score < best_score;
            if better {
                best_score = score;
                best_sp_log = sp;
                best_rs_log = rs;
                best_bytes = serialize_weights(&ema_mlp);
                std::fs::write(&wpath, &best_bytes).unwrap();
            }
            eprintln!("[v7] MONITOR epoch {}: val sparkle {sp:.1}/Mpx resid {rs:.4} score={score:.3}{} (tgt sp<{spark_target} resid<{resid_gate}){}",
                epoch, if passes { " PASS" } else { "" }, if better { " *BEST->saved" } else { "" });
            std::io::stderr().flush().ok();
        }
    }
    eprintln!("[v7] training done in {:.1}s (best score={best_score:.3} sparkle {best_sp_log:.1} resid {best_rs_log:.4}, started at floor {START_SCORE_FLOOR})",
        t_train.elapsed().as_secs_f64());
    std::io::stderr().flush().ok();

    std::fs::write(&wpath, &best_bytes).unwrap();
    let best_mlp = deserialize_weights(&best_bytes).expect("reload best");
    let wsha = weights_sha256(&best_mlp);
    println!("[v7] wrote {} sha256={wsha}", wpath.display());
    std::io::stdout().flush().ok();
    let prov = serde_json::json!({
        "artifact": "rdirect-weights-v7.bin",
        "weights_sha256": wsha,
        "supersedes": "rdirect-weights-v6.bin",
        "architecture": {
            "kind": "N6 SIGNED EVIDENCE, TARGET-SMOOTHED — recurrent direct-render MLP, split radiance input (E+D), single combined output trained against an E-exact/D-blurred target",
            "input_features": HIST_FEATURES_SPLIT,
            "output_channels": OUTPUT_CHANNELS,
            "hidden_layers": config.hidden_layers,
            "hidden_width": config.hidden_width,
            "macs_per_pixel": best_mlp.macs(),
        },
        "training": {
            "epochs": epochs, "unroll_steps": k, "batch": batch, "lr0": lr0,
            "subsample_px_per_pose_per_epoch": subsample, "ref_frames": ref_frames,
            "init": if resume { "CURE fine-tune: resumed from restored 636c8743 3-out checkpoint" } else { "FRESH (same shape as v6, 39-in/3-out)" },
            "loss": "PLAIN MSE vs split-smoothed target, backprop through the v7e evidence clamp (no cap/gate/firefly-weight loss term, no two-head)",
            "evidence_clamp_gamma": evidence_clamp_gamma(),
            "target_construction": "target = E_full (exact, ref_frames) + box_blur(D_full, radius) — E kept sharp, D smoothed at the SOURCE before demod-log; escapes the sparkle<->resid Pareto front structurally instead of penalizing the output",
            "cure": "harmonic lr-decay from ~0.3x prior lr + Polyak/EMA (decay 0.999, monitor+checkpoint evaluate EMA weights) + cross-run bar-normalized score floor (max(sparkle/40, resid/0.035), only overwrites the starting 636c8743 checkpoint if strictly better than its own score 1.157, then monotonically tightens) — the same recipe that fixed the noise-ball-outlier signature in magic-crystal-nrc/docs/perf/2026-07-17-nrc-spike-verdict.md. Two-head (v7c) is BANNED, tried and FAILED (worse seesaw).",
            "ema_decay": ema_decay,
            "best_checkpoint_criterion": "bar-normalized score = max(sparkle/40, resid/0.035), save on new MIN, floor 1.157",
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
    std::io::stdout().flush().ok();
}
