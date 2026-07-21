//! R-DIRECT v8c — THE DOCTRINE ROUND (2026-07-20, first lane under the
//! 07-20 enforcement clause, NEURAL.md §TRAINING DOCTRINE). Evolves v8
//! (`rdirect_train_v8.rs`) by implementing all three sealed tiers instead
//! of tier-3-only distillation:
//!
//! TIER 1 — BORN AS THE ESTIMATOR. `Mlp::evidence_mean_init_split` (new,
//!   `src/rdirect.rs`) replaces He-random FRESH init: the net's weights are
//!   CONSTRUCTED, not learned, so that before a single gradient step its
//!   output is EXACTLY the classical evidence-averaging estimator (box mean
//!   of the 4 E-taps + box mean of the 4 D-taps the net's own input already
//!   carries — no teacher, no rendering beyond the ordinary evidence pass).
//!   `main` runs an ORDEAL-ASSERT right after construction: forward() on
//!   real rendered features vs a hand-computed box mean (must match to
//!   float precision — a construction identity, not a training outcome),
//!   then the ordinary MONITOR pass (val sparkle/resid vs the teacher) so
//!   the "sparkle ~0 by construction" claim is a measured, printed number,
//!   not an assertion of faith.
//!
//! TIER 2 — EQUATION SIGNAL (noise2noise). v8/v8b/A0-A3 all trained the net
//!   against a TEACHER-derived target (E_full[ref_frames] + blur(D_full)) —
//!   tier-3 distillation, doctrine-violating per the 07-20 LASHES entry.
//!   v8c instead traces TWO INDEPENDENT sparse evidence draws per pose per
//!   step, same integrator, same spp, DIFFERENT seeds: draw A (unchanged —
//!   the net's own low-res input taps, `low_e`/`low_d`) and draw B (NEW — a
//!   second independent trace at NATIVE resolution, never fed to the net,
//!   seed offset 0xB222 vs draw A's 0x7abc). The training LABEL is draw B's
//!   raw per-pixel radiance (demod-log, unfiltered — no ref_frames average,
//!   no box blur: filtering the label would reintroduce a biased target and
//!   defeat the whole point). Because A and B are independent draws of the
//!   SAME underlying radiance distribution, E[draw B] = the true converged
//!   radiance — minimizing MSE(net(A-features), draw-B-label) is, in
//!   expectation over the noise, the SAME minimizer as MSE(net, truth),
//!   with NO teacher render in the loss at all. This directly attacks the
//!   gamma-sweep's -38.7% highlight under-render (a genuine undertraining
//!   gap, not a clamp artifact) by giving the net an UNBIASED highlight
//!   signal on every pixel every epoch, instead of a spatially-biased
//!   oversampling hack (tier-2's honest answer to what A2 tried dishonestly
//!   — see below). The teacher (E_full[ref_frames], converged) DEMOTES to
//!   VALIDATOR ONLY: `run_monitor` still measures sparkle/resid/highlight_
//!   ratio against it every epoch (the score floor and the eventual ordeal
//!   both stay teacher-referenced — that is what "validator" means), it
//!   just never appears in `accumulate_backward_clamped_slice`'s target
//!   argument.
//!
//! TIER 3 (kept, demoted to validator/structure, NOT the training signal):
//!   - MOVING-CAMERA HISTORY (`scratch/v8-ablate-A1.log`: proven INNOCENT —
//!     epoch-for-epoch near-identical to the A0 v7e-parity baseline through
//!     epoch 24, no detonation). `render_pose_seq` still drifts camera yaw
//!     across the K unroll steps (`GAIA_V8_PANSTEP`, default 0.004 rad/step)
//!     and `reproject_prev` still runs the real `CamPose::reproject` +
//!     depth/normal guard + `sky_history_reject()` during training, exactly
//!     as v8 shipped it.
//!   - MIRROR POSE, spp-4 evidence (`scratch/v8-ablate-A3.log`: proven
//!     INNOCENT — actually the single BEST-behaved of the four ablation
//!     arms, lowest val sparkle creep of A0/A1/A3 by epoch 24). Kept
//!     unconditionally in the training pose set (no ablation on/off switch
//!     — the doctrine round is not re-litigating a settled verdict).
//!   - EMA-SOURCED HISTORY CHAIN (`cf8bd7b`): `history_forward` still reads
//!     `ema_mlp`, not the raw actively-optimized `mlp`, as its once-per-
//!     epoch recurrent-feedback source (v7d's noise-ball-isolation
//!     principle, applied one level deeper — kept even though it did NOT
//!     cure the v8/v8b runaway alone; that runaway is now understood to be
//!     A2's highlight-biased sampling, not the history source).
//!   - GAMMA=1.5 EVIDENCE CLAMP (`evidence_clamp_gamma`, IRON-derived,
//!     v7e): `presented = min(net_linear, gamma * local_max(evidence))`,
//!     unchanged act, still built from draw A's own evidence composite
//!     (`EvidenceAccum`, temporal-mean then 3x3 spatial max-pool) — a
//!     structural ceiling on the OUTPUT, orthogonal to what target trains
//!     against, so tier 2's loss-target swap does not touch it.
//!   - ALL ENV PARAMS governing the above kept at the SAME names/defaults:
//!     `GAIA_V7_SKY_HISTORY`, `GAIA_V8_PANSTEP`, `GAIA_V8_MIRROR_SPP`,
//!     `GAIA_V8_EMA`, `GAIA_V7_CLAMP_GAMMA`, `GAIA_V8_W/H/STILL/REF/EPOCHS/
//!     SUBSAMPLE/BATCH/LR/SPARK_TGT/RESID_GATE/WALL/MONITOR/LAYERS/WIDTH`.
//!
//! DROPPED ENTIRELY: HIGHLIGHT-TARGETED SAMPLING (`GAIA_V8_HIGHLIGHT_FRAC`/
//!   `_PCTL`, the per-epoch bright-pixel oversampling pool). Convicted
//!   detonator: `scratch/v8-ablate-A2.log` is the ONLY one of the four 25-
//!   epoch ablation arms that reproduces the v8/v8b runaway signature (val
//!   sparkle 0.0→651.0/Mpx by epoch 24, climbing from epoch ~4 onward,
//!   >300/Mpx well before epoch 10) while A0 (all three off, v7e-parity
//!   baseline) stays at val sparkle <=117.5/Mpx over the identical 25
//!   epochs and A1/A3 track A0 almost exactly. No highlight pool, no
//!   biased subsample, no `GAIA_V8_HIGHLIGHT_*` env vars in this file at
//!   all — tier 2's unbiased noise2noise signal is the doctrine-compliant
//!   replacement for what A2 was trying (and failing, dangerously) to buy.
//!   `box_blur`/`GAIA_V8_BLUR` also dropped (v8's smoothed-target escape,
//!   meaningless once the target is an unfiltered independent draw — a
//!   deliberate, disclosed deviation from "keep all env params", scoped
//!   narrowly to the one param whose entire referent (the teacher-blur
//!   target construction) tier 2 replaces).
//!
//! Training laws unchanged: FRESH init only (= the tier-1 construction,
//! not a resume path), corner-crawl lr 1e-4, monitor + print score EVERY
//! epoch, cross-run bar-normalized score floor (tag-scoped — this trainer's
//! default tag is "v8c", NOT "v8", so it can never read or clobber the
//! real v8 checkpoint/floor), checkpoints via the same score-floor gate,
//! detach for the real 200-epoch run.
//!
//! Run: cargo run --release -j2 --example rdirect_train_v8c
//!   GAIA_V7_SKY_HISTORY=reject   (mandate: sky-reject semantics ACTIVE.)

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use glam::{Vec2, Vec3 as GVec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, headless_device, trace_headless_aov, trace_headless_split,
};
use scrying_glass::rdirect::{
    Adam, CamPose, HIST_FEATURES_SPLIT, HistFrameSplit, Mlp, OUTPUT_CHANNELS, RdirectConfig,
    EvidenceAccum, accumulate_backward_clamped_slice, adam_apply, bilinear_vec3,
    deserialize_weights, direct_render_sequence_hist_split, evidence_clamp_gamma,
    evidence_ceiling_demod_log, evidence_composite_frame, hist_features_split,
    pixel_features_split, serialize_weights, sky_history_reject, target_demod_log,
    weights_sha256, zero_grads,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

const DEPTH_TOL: f32 = 0.05; // same reprojection guard the ordeal uses
const NORMAL_THRESH: f32 = 0.85;
// TIER 2: independent seed families for draw A (net input evidence, kept
// byte-identical to v8's own seed) and draw B (the noise2noise LABEL, new
// this round — must never collide with A's sequence).
const DRAW_A_SEED_BASE: u32 = 0x7abc;
const DRAW_B_SEED_BASE: u32 = 0xB222;

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

fn cam_pose(cam: &Camera, w: u32, h: u32) -> CamPose {
    let (right, up, forward) = cam.basis();
    CamPose { eye: cam.eye, right, up, forward, half_tan: (cam.fov_y_radians * 0.5).tan(), aspect: w as f32 / h as f32 }
}

/// One step of a moving-camera pose sequence.
struct Step {
    low_e: Vec<GVec3>,   // draw A (net input evidence, low-res, 1/mirror_spp)
    low_d: Vec<GVec3>,
    albedo: Vec<GVec3>,
    normal: Vec<GVec3>,
    depth: Vec<f32>,
    teacher: Vec<GVec3>,                     // E_full[ref_frames]+D_full — VALIDATOR ONLY (tier 2: never in the loss)
    target_dl: Vec<[f32; OUTPUT_CHANNELS]>,  // TIER 2: draw B's raw per-pixel radiance, demod-log — the noise2noise LABEL
    ceiling_dl: Vec<[f32; OUTPUT_CHANNELS]>, // v7e evidence clamp ceiling, from draw A's own evidence composite
}

/// A full K-step camera sequence for one pose. `pan_step=0` degenerates to a
/// still camera (SNAP_EPS) — same code path serves still validation and
/// moving training (kept from v8, proven innocent — A1).
struct PoseSeq {
    steps: Vec<Step>,
    cams: Vec<CamPose>,
    low_w: u32,
    low_h: u32,
    tw: u32,
    th: u32,
}

#[allow(clippy::too_many_arguments)]
fn render_pose_seq(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_tris: &[LeafTriangle],
    scene: &RenderScene,
    base_cam: &Camera,
    k: u32,
    low_w: u32,
    low_h: u32,
    tw: u32,
    th: u32,
    ref_frames: u32,
    pan_step: f32,
    evidence_spp: u32,
) -> PoseSeq {
    let bvh = Bvh::build(base_tris, &BvhParams::default());
    let gamma = evidence_clamp_gamma();
    let mut accum = EvidenceAccum::new(tw, th);
    let mut steps = Vec::with_capacity(k as usize);
    let mut cams = Vec::with_capacity(k as usize);
    let n = (tw * th) as usize;
    for step in 0..k {
        let mut cam = *base_cam;
        cam.yaw += pan_step * step as f32;
        cams.push(cam_pose(&cam, tw, th));

        // TIER 2 — DRAW A: the net's own low-res evidence input (byte-
        // identical seed family to v8, so its statistics are unchanged).
        let np_a = IntegratorParams { spp: evidence_spp, seed: DRAW_A_SEED_BASE + step * 131 + 5, ..IntegratorParams::default() };
        let (low_e, low_d) = trace_headless_split(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np_a,
        );
        let (albedo, normal, depth) = scrying_glass::integrator::split_aov(&trace_headless_aov(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th,
        ));

        // TIER 2 — DRAW B: an INDEPENDENT sparse trace, native resolution,
        // SAME spp as draw A, DIFFERENT seed family — never fed to the net,
        // this is purely the noise2noise training LABEL. Unfiltered (no
        // blur, no accumulation): filtering here would bias the label away
        // from an unbiased per-pixel radiance estimate.
        let np_b = IntegratorParams { spp: evidence_spp, seed: DRAW_B_SEED_BASE + step * 257 + 11, ..IntegratorParams::default() };
        let (e_b, d_b) = trace_headless_split(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, 1, &np_b,
        );
        let target_dl: Vec<[f32; OUTPUT_CHANNELS]> = (0..n)
            .map(|px| target_demod_log(e_b[px] + d_b[px], albedo[px]))
            .collect();

        // Teacher: converged reference, VALIDATOR ONLY (never in the loss;
        // `run_monitor` reads it, `accumulate_backward_clamped_slice` never
        // sees it in this file).
        let (e_full, d_full) = trace_headless_split(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
            &IntegratorParams::default(),
        );
        let teacher: Vec<GVec3> = (0..n).map(|i| e_full[i] + d_full[i]).collect();

        // v7e evidence clamp ceiling — built from draw A's own evidence
        // composite (unchanged act; orthogonal to what the loss trains
        // against).
        let frame_composite = evidence_composite_frame(&low_e, &low_d, low_w, low_h, tw, th);
        accum.push(&frame_composite);
        let local_max = accum.ceiling();
        let ceiling_dl: Vec<[f32; OUTPUT_CHANNELS]> =
            (0..n).map(|px| evidence_ceiling_demod_log(local_max[px], gamma, albedo[px])).collect();

        steps.push(Step { low_e, low_d, albedo, normal, depth, teacher, target_dl, ceiling_dl });
    }
    PoseSeq { steps, cams, low_w, low_h, tw, th }
}

/// Reproject step `s-1`'s real net output into step `s`'s screen. Byte-for-
/// byte the same accept/reject rule `direct_render_sequence_hist_split`
/// runs — unchanged from v8 (A1: proven innocent).
#[allow(clippy::too_many_arguments)]
fn reproject_prev(
    cur_cam: &CamPose,
    cur_depth: f32,
    cur_normal: GVec3,
    tx: u32,
    ty: u32,
    tw: u32,
    th: u32,
    prev_cam: &CamPose,
    prev_out_dl: &[GVec3],
    prev_depth: &[f32],
    prev_normal: &[GVec3],
    pw: u32,
    ph: u32,
    sky_reject: bool,
) -> ([f32; 3], f32) {
    let is_miss = cur_depth <= 0.0;
    let dir = cur_cam.ray_dir(tx, ty, tw, th);
    let dist = if is_miss { 1.0e5 } else { cur_depth };
    let world = cur_cam.eye + dir * dist;
    match prev_cam.reproject(world, pw, ph) {
        None => ([0.0; 3], 0.0),
        Some((fx, fy)) => {
            let ipx = fx.round().clamp(0.0, (pw - 1) as f32) as usize;
            let ipy = fy.round().clamp(0.0, (ph - 1) as f32) as usize;
            let pj = ipy * pw as usize + ipx;
            let prev_d = prev_depth[pj];
            let prev_miss = prev_d <= 0.0;
            let ok = if is_miss {
                prev_miss && !sky_reject
            } else if prev_miss {
                false
            } else {
                let dist_prev = (world - prev_cam.eye).length();
                let depth_ok = (dist_prev - prev_d).abs() <= DEPTH_TOL * dist_prev.max(1e-4);
                let normal_ok = cur_normal.dot(prev_normal[pj]) >= NORMAL_THRESH;
                depth_ok && normal_ok
            };
            if ok {
                let s = bilinear_vec3(prev_out_dl, fx, fy, pw, ph);
                ([s.x, s.y, s.z], 1.0)
            } else {
                ([0.0; 3], 0.0)
            }
        }
    }
}

/// Precompute the FULL native-res out_dl chain for every step of a pose
/// sequence using the EMA weights snapshotted at epoch start — kept from
/// v8/cf8bd7b (EMA-sourced history chain is a KEEP item; the v8/v8b
/// runaway is now attributed to A2's highlight sampling, not this).
fn history_forward(mlp: &Mlp, seq: &PoseSeq, sky_reject: bool) -> Vec<Vec<GVec3>> {
    let (tw, th) = (seq.tw, seq.th);
    let n = (tw * th) as usize;
    let mut chain: Vec<Vec<GVec3>> = Vec::with_capacity(seq.steps.len());
    for (s, step) in seq.steps.iter().enumerate() {
        let mut out = vec![GVec3::ZERO; n];
        for ty in 0..th {
            for tx in 0..tw {
                let px = (ty * tw + tx) as usize;
                let albedo = step.albedo[px];
                let base = pixel_features_split(
                    &step.low_e, &step.low_d, seq.low_w, seq.low_h, tw, th, tx, ty, albedo,
                    step.normal[px], step.depth[px], Vec2::ZERO,
                );
                let (prev_dl, valid) = if s == 0 {
                    ([0.0f32; 3], 0.0f32)
                } else {
                    let prev_step = &seq.steps[s - 1];
                    reproject_prev(
                        &seq.cams[s], step.depth[px], step.normal[px], tx, ty, tw, th,
                        &seq.cams[s - 1], &chain[s - 1], &prev_step.depth, &prev_step.normal, tw, th,
                        sky_reject,
                    )
                };
                let feat = hist_features_split(&base, prev_dl, valid);
                let dl = mlp.forward(&feat);
                out[px] = GVec3::new(dl[0], dl[1], dl[2]);
            }
        }
        chain.push(out);
    }
    chain
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

/// (diagnostic, teacher-referenced, VALIDATOR ONLY) mean net/teacher
/// luminance ratio over the pose's own brightest `pctl` teacher pixels —
/// reports the gamma-sweep's -38.7% highlight under-render metric without
/// feeding it back into sampling (that mechanism is DROPPED, A2 convicted).
fn highlight_ratio(net: &[GVec3], teacher: &[GVec3], pctl: f32) -> f64 {
    let mut order: Vec<usize> = (0..teacher.len()).collect();
    order.sort_by(|&a, &b| lum(teacher[b]).partial_cmp(&lum(teacher[a])).unwrap());
    let n = ((teacher.len() as f32 * pctl).ceil() as usize).max(1).min(teacher.len());
    let mut net_sum = 0.0f64;
    let mut teacher_sum = 0.0f64;
    for &i in &order[..n] {
        net_sum += lum(net[i]) as f64;
        teacher_sum += lum(teacher[i]) as f64;
    }
    if teacher_sum > 1e-9 { net_sum / teacher_sum } else { 1.0 }
}

/// Settle a pose sequence through the net exactly like the ordeal's own act.
fn settle(mlp: &Mlp, seq: &PoseSeq) -> (Vec<GVec3>, Vec<GVec3>) {
    let hf: Vec<HistFrameSplit> = seq
        .steps
        .iter()
        .zip(&seq.cams)
        .map(|(s, c)| HistFrameSplit {
            low_e: &s.low_e, low_d: &s.low_d, low_w: seq.low_w, low_h: seq.low_h,
            hi_albedo: &s.albedo, hi_normal: &s.normal, hi_depth: &s.depth,
            target_w: seq.tw, target_h: seq.th, cam: *c,
        })
        .collect();
    let outs = direct_render_sequence_hist_split(mlp, &hf, DEPTH_TOL, NORMAL_THRESH);
    (outs.last().unwrap().clone(), seq.steps.last().unwrap().teacher.clone())
}

#[allow(clippy::too_many_arguments)]
fn run_monitor(
    run_tag: &str,
    tag: &str,
    ema_mlp: &Mlp,
    val_seq: &PoseSeq,
    mirror_seq: &PoseSeq,
    highlight_pctl: f32,
    spark_target: f32,
    resid_gate: f32,
    best_score: &mut f64,
    best_bytes: &mut Vec<u8>,
    wpath: &Path,
) -> (f64, f64) {
    let (net, teacher) = settle(ema_mlp, val_seq);
    let sp = sparkle_resid_per_mpx(&net, &teacher, val_seq.tw, val_seq.th);
    let rs = rmse_lin(&net, &teacher);
    let hl = highlight_ratio(&net, &teacher, highlight_pctl);
    let (mnet, mteacher) = settle(ema_mlp, mirror_seq);
    let msp = sparkle_resid_per_mpx(&mnet, &mteacher, mirror_seq.tw, mirror_seq.th);
    let mrs = rmse_lin(&mnet, &mteacher);
    let mhl = highlight_ratio(&mnet, &mteacher, highlight_pctl);
    let passes = sp < spark_target as f64 && rs < resid_gate as f64;
    let score = (sp / 40.0).max(rs / 0.035);
    let better = score < *best_score;
    if better {
        *best_score = score;
        *best_bytes = serialize_weights(ema_mlp);
        std::fs::write(wpath, &*best_bytes).unwrap();
    }
    eprintln!(
        "[{run_tag}] MONITOR {tag}: val sparkle {sp:.1}/Mpx resid {rs:.4} highlight_ratio {hl:.3} score={score:.3}{} | mirror sparkle {msp:.1}/Mpx resid {mrs:.4} highlight_ratio {mhl:.3} (tgt sp<{spark_target} resid<{resid_gate}){}",
        if passes { " PASS" } else { "" }, if better { " *BEST->saved" } else { "" },
    );
    std::io::stderr().flush().ok();
    (sp, rs)
}

fn main() {
    let run_tag = std::env::var("GAIA_V8_TAG").unwrap_or_else(|_| "v8c".to_string());
    let sky_reject = sky_history_reject();
    eprintln!("[{run_tag}] GAIA_V7_SKY_HISTORY reject={sky_reject} — mandate expects true (set GAIA_V7_SKY_HISTORY=reject)");
    if !sky_reject {
        eprintln!("[{run_tag}] WARNING: sky-reject semantics NOT active — set GAIA_V7_SKY_HISTORY=reject per the mandate.");
    }

    let Some((device, queue)) = headless_device() else {
        panic!("[{run_tag}] no GPU");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();

    let tw = env_u32("GAIA_V8_W", 384);
    let th = env_u32("GAIA_V8_H", 288);
    let low_w = tw / 2;
    let low_h = th / 2;
    let k = env_u32("GAIA_V8_STILL", 3);
    let ref_frames = env_u32("GAIA_V8_REF", 64);
    let epochs = env_u32("GAIA_V8_EPOCHS", 200);
    let subsample = env_u32("GAIA_V8_SUBSAMPLE", 5000) as usize;
    let batch = env_u32("GAIA_V8_BATCH", 64) as usize;
    let lr0 = env_f32("GAIA_V8_LR", 1.0e-4); // TRAINING LAW: corner-crawl lr 1e-4
    let ema_decay = env_f32("GAIA_V8_EMA", 0.999);
    let pan_step = env_f32("GAIA_V8_PANSTEP", 0.004); // matches GAIA_ORDEAL_PANSTEP
    let mirror_spp = env_u32("GAIA_V8_MIRROR_SPP", 4);
    let highlight_pctl = env_f32("GAIA_V8_HIGHLIGHT_PCTL", 0.05); // diagnostic only now
    let spark_target = env_f32("GAIA_V8_SPARK_TGT", 16.0);
    let resid_gate = env_f32("GAIA_V8_RESID_GATE", 0.035); // the IRON bar itself
    let monitor_every = env_u32("GAIA_V8_MONITOR", 1); // TRAINING LAW: every epoch
    let wall_budget = env_f32("GAIA_V8_WALL", 10_800.0); // 3h default — detach for real runs

    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| all.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();
    let mirror_cam = scrying_glass::denoiser_dataset::mirror_camera();
    // Mirror pose KEPT unconditionally (A3: proven innocent, best-behaved of
    // the three ablation arms) — no on/off switch, the doctrine round is not
    // re-litigating a settled verdict.
    let train_cams: Vec<(&str, Camera, u32)> = vec![
        ("front", find("front"), 1),
        ("wide", find("wide"), 1),
        ("orbit_+20", find("orbit_+20"), 1),
        ("mirror", mirror_cam, mirror_spp),
    ];
    eprintln!(
        "[{run_tag}] training poses (mirror always on, A3-innocent): {:?}",
        train_cams.iter().map(|(n, ..)| *n).collect::<Vec<_>>()
    );

    let t_render = Instant::now();
    let poses: Vec<PoseSeq> = train_cams
        .iter()
        .map(|(_, c, spp)| render_pose_seq(&device, &queue, &base_tris, &scene, c, k, low_w, low_h, tw, th, ref_frames, pan_step, *spp))
        .collect();
    // Validation pose: STILL (pan_step=0, matching the ordeal's own STILL
    // test exactly).
    let val_seq = render_pose_seq(&device, &queue, &base_tris, &scene, &find("orbit_-20"), k, low_w, low_h, tw, th, ref_frames, 0.0, 1);
    // Mirror monitor pose: STILL too (diagnostic only, not gating).
    let mirror_val_seq = render_pose_seq(&device, &queue, &base_tris, &scene, &mirror_cam, k, low_w, low_h, tw, th, ref_frames, 0.0, mirror_spp);
    eprintln!(
        "[{run_tag}] rendered {}+2 MOVING/STILL pose sequences (K={k} steps, {tw}x{th}, teacher {ref_frames} [VALIDATOR ONLY], pan_step={pan_step}, mirror_spp={mirror_spp}, TIER2 draw_B seed base=0x{DRAW_B_SEED_BASE:x}) in {:.1}s",
        poses.len(), t_render.elapsed().as_secs_f64()
    );
    std::io::stderr().flush().ok();

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let wpath = data_dir.join(format!("rdirect-weights-{run_tag}.bin"));
    let config = RdirectConfig {
        hidden_layers: env_u32("GAIA_V8_LAYERS", 5) as usize,
        hidden_width: env_u32("GAIA_V8_WIDTH", 64) as usize,
    };

    // ── TIER 1 — BORN AS THE ESTIMATOR ──────────────────────────────────
    // FRESH init IS this analytic construction now (training law "fresh
    // init" = "the tier-1 construction" per the mandate) — no He-random
    // weights, no pretrain loop.
    let mut mlp = Mlp::evidence_mean_init_split(config);
    assert_eq!(mlp.layer_dims()[0].0 as usize, HIST_FEATURES_SPLIT, "v8c net must be 39-input");

    // ORDEAL-ASSERT: output == evidence-mean, by construction, checked
    // against REAL rendered features (not synthetic) before any gradient
    // step. Sampled corners+center of every training pose's first step.
    {
        let mut max_abs = 0.0f32;
        let mut mean_abs = 0.0f64;
        let mut n_checked = 0usize;
        for seq in &poses {
            let step0 = &seq.steps[0];
            let n_px = (seq.tw * seq.th) as usize;
            for &px in &[0usize, n_px / 2, n_px - 1] {
                let tx = (px as u32) % seq.tw;
                let ty = (px as u32) / seq.tw;
                let albedo = step0.albedo[px];
                let base = pixel_features_split(
                    &step0.low_e, &step0.low_d, seq.low_w, seq.low_h, seq.tw, seq.th, tx, ty,
                    albedo, step0.normal[px], step0.depth[px], Vec2::ZERO,
                );
                let feat = hist_features_split(&base, [0.0; 3], 0.0);
                let net_out = mlp.forward(&feat);
                let e_mean = [
                    (base[0] + base[3] + base[6] + base[9]) / 4.0,
                    (base[1] + base[4] + base[7] + base[10]) / 4.0,
                    (base[2] + base[5] + base[8] + base[11]) / 4.0,
                ];
                let d_mean = [
                    (base[12] + base[15] + base[18] + base[21]) / 4.0,
                    (base[13] + base[16] + base[19] + base[22]) / 4.0,
                    (base[14] + base[17] + base[20] + base[23]) / 4.0,
                ];
                for c in 0..3 {
                    let expect = e_mean[c] + d_mean[c];
                    let diff = (net_out[c] - expect).abs();
                    max_abs = max_abs.max(diff);
                    mean_abs += diff as f64;
                    n_checked += 1;
                }
            }
        }
        mean_abs /= n_checked.max(1) as f64;
        eprintln!(
            "[{run_tag}] TIER1 ORDEAL-ASSERT: output==evidence-mean(box-mean of 4 E-taps + 4 D-taps), BY CONSTRUCTION, on {n_checked} real-render channel samples: max|diff|={max_abs:.8} mean|diff|={mean_abs:.8}"
        );
        std::io::stderr().flush().ok();
        assert!(max_abs < 1.0e-4, "[{run_tag}] TIER1 construction FAILED: net output does not copy the evidence-mean feature (max diff {max_abs})");
        eprintln!("[{run_tag}] TIER1 construction VERIFIED — output IS the classical evidence-averaging estimator before epoch 0.");
    }

    let mut adam = Adam::new(&mlp, lr0, 0.9, 0.999, 1e-8);
    let mut ema_mlp = mlp.clone();
    eprintln!(
        "[{run_tag}] arch {:?} in={HIST_FEATURES_SPLIT} macs/px={} — {epochs} epochs (wall<={wall_budget}s), subsample {subsample}px/pose (uniform, NO highlight bias — A2 convicted), lr0={lr0} (mild harmonic decay), ema={ema_decay}, TIER2 loss = plain MSE vs independent draw-B radiance (noise2noise, teacher NEVER in the loss)",
        config, mlp.macs(),
    );
    std::io::stderr().flush().ok();

    // Cross-run bar-normalized score floor: beat the EXISTING checkpoint FOR
    // THIS TAG (v8c's own floor file — tag-scoped, can never read/clobber
    // the real v8 checkpoint's floor).
    let mut best_score: f64 = if wpath.exists() {
        if let Some(prior) = std::fs::read(&wpath).ok().and_then(|b| deserialize_weights(&b)) {
            let (net, teacher) = settle(&prior, &val_seq);
            let sp = sparkle_resid_per_mpx(&net, &teacher, val_seq.tw, val_seq.th);
            let rs = rmse_lin(&net, &teacher);
            let s = (sp / 40.0).max(rs / 0.035);
            eprintln!("[{run_tag}] cross-run floor: existing checkpoint score={s:.3} (sparkle {sp:.1} resid {rs:.4}) — this run must beat it to overwrite");
            s
        } else {
            f64::INFINITY
        }
    } else {
        f64::INFINITY
    };
    let mut best_bytes: Vec<u8> = serialize_weights(&ema_mlp);

    // ── TIER 1 MONITOR: val sparkle/resid BEFORE epoch 0, measured (not
    // asserted) against the teacher validator — the printed "sparkle ~0 by
    // construction, resid already near the classical-estimator floor"
    // claim.
    {
        let (sp, rs) = run_monitor(&run_tag, "epoch -1 (TIER1 fresh, BORN AS ESTIMATOR)", &ema_mlp, &val_seq, &mirror_val_seq, highlight_pctl, spark_target, resid_gate, &mut best_score, &mut best_bytes, &wpath);
        eprintln!("[{run_tag}] TIER1 fresh-init baseline (vs teacher validator): sparkle {sp:.1} resid {rs:.4} (floor {best_score:.3})");
    }

    let n_px: Vec<usize> = poses.iter().map(|p| (p.tw * p.th) as usize).collect();
    let mut rng = Rng(0xd15e_ed00_08f0_0dc0);
    let t_train = Instant::now();

    'train: for epoch in 0..epochs {
        if t_train.elapsed().as_secs_f64() > wall_budget as f64 {
            eprintln!("[{run_tag}] WALL budget {wall_budget}s reached at epoch {epoch} — stopping, keeping best");
            break;
        }
        let frac = epoch as f32 / epochs as f32;
        adam.set_lr(lr0 / (1.0 + 1.0 * frac)); // mild harmonic decay from the corner-crawl lr

        // EMA-sourced history precompute (kept, cf8bd7b) — once per epoch.
        let t_hist = Instant::now();
        let history: Vec<Vec<Vec<GVec3>>> = poses.iter().map(|seq| history_forward(&ema_mlp, seq, sky_reject)).collect();
        let hist_ms = t_hist.elapsed().as_secs_f64() * 1000.0;

        let mut epoch_mse = 0.0f64;
        let mut n_steps = 0u64;

        // TIER 2: sample (pose, pixel) pairs UNIFORMLY — no highlight pool,
        // no biased oversampling (A2 convicted, dropped entirely).
        let mut samples: Vec<(usize, usize)> = Vec::with_capacity(subsample * n_px.len());
        for (pi, np) in n_px.iter().enumerate() {
            for _ in 0..subsample {
                samples.push((pi, (rng.next() as usize) % np));
            }
        }
        for i in (1..samples.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            samples.swap(i, j);
        }

        let mut bstart = 0usize;
        while bstart < samples.len() {
            if t_train.elapsed().as_secs_f64() > wall_budget as f64 {
                eprintln!("[{run_tag}] WALL budget {wall_budget}s reached MID-epoch {epoch} (batch {bstart}/{}) — stopping, keeping best", samples.len());
                std::io::stderr().flush().ok();
                break 'train;
            }
            let bend = (bstart + batch).min(samples.len());
            let (mut wg, mut bg) = zero_grads(&mlp);
            let blen = (bend - bstart) as f32;
            for &(pi, px) in &samples[bstart..bend] {
                let seq = &poses[pi];
                let tx = (px as u32) % seq.tw;
                let ty = (px as u32) / seq.tw;
                for step in 0..k as usize {
                    let cur = &seq.steps[step];
                    let albedo = cur.albedo[px];
                    let base = pixel_features_split(
                        &cur.low_e, &cur.low_d, seq.low_w, seq.low_h, seq.tw, seq.th, tx, ty,
                        albedo, cur.normal[px], cur.depth[px], Vec2::ZERO,
                    );
                    let (prev_dl, valid) = if step == 0 {
                        ([0.0f32; 3], 0.0f32)
                    } else {
                        let prev_step = &seq.steps[step - 1];
                        reproject_prev(
                            &seq.cams[step], cur.depth[px], cur.normal[px], tx, ty, seq.tw, seq.th,
                            &seq.cams[step - 1], &history[pi][step - 1], &prev_step.depth, &prev_step.normal,
                            seq.tw, seq.th, sky_reject,
                        )
                    };
                    let feat = hist_features_split(&base, prev_dl, valid);
                    let out = mlp.forward(&feat);
                    let ceiling_dl = &cur.ceiling_dl[px];
                    // TIER 2: target is draw B's independent radiance
                    // sample — the teacher NEVER appears here.
                    let target = &cur.target_dl[px];
                    let mse = accumulate_backward_clamped_slice(
                        &mlp, &feat, &out, ceiling_dl, target, &mut wg, &mut bg, 1.0 / (blen * k as f32),
                    );
                    epoch_mse += mse;
                    n_steps += 1;
                }
            }
            adam_apply(&mut adam, &mut mlp, &wg, &bg);
            ema_mlp.ema_update(&mlp, ema_decay);
            bstart = bend;
        }

        let denom = (n_steps as f64 * OUTPUT_CHANNELS as f64).max(1.0);
        println!(
            "[{run_tag}] epoch {}/{} n2n_mse={:.6} lr={:.6} hist_ms={hist_ms:.0} ({:.1}s)",
            epoch, epochs, epoch_mse / denom, adam.lr(), t_train.elapsed().as_secs_f64()
        );
        std::io::stdout().flush().ok();

        if (epoch + 1) % monitor_every == 0 || epoch + 1 == epochs {
            run_monitor(&run_tag, &format!("epoch {epoch}"), &ema_mlp, &val_seq, &mirror_val_seq, highlight_pctl, spark_target, resid_gate, &mut best_score, &mut best_bytes, &wpath);
        }
    }
    eprintln!("[{run_tag}] training done in {:.1}s (best score={best_score:.3})", t_train.elapsed().as_secs_f64());
    std::io::stderr().flush().ok();

    std::fs::write(&wpath, &best_bytes).unwrap();
    let best_mlp = deserialize_weights(&best_bytes).expect("reload best");
    let wsha = weights_sha256(&best_mlp);
    println!("[{run_tag}] wrote {} sha256={wsha}", wpath.display());
    std::io::stdout().flush().ok();
    let prov = serde_json::json!({
        "artifact": format!("rdirect-weights-{run_tag}.bin"),
        "ablation_tag": run_tag,
        "weights_sha256": wsha,
        "supersedes": "rdirect-weights-v8.bin",
        "doctrine_concordance": "TIER1(estimator-init)+TIER2(noise2noise, teacher=validator-only)+TIER3(structure: moving-camera history + mirror pose + EMA history-source + gamma=1.5 clamp, kept as validator/structural, NOT the training signal) — first lane under the 07-20 enforcement clause",
        "architecture": {
            "kind": "N7 MOVING-HISTORY + SPECULAR-EVIDENCE + TIER1-ESTIMATOR-INIT + TIER2-NOISE2NOISE — recurrent direct-render MLP, split radiance input (E+D), 39-in/3-out, evidence-clamped act",
            "input_features": HIST_FEATURES_SPLIT,
            "output_channels": OUTPUT_CHANNELS,
            "hidden_layers": config.hidden_layers,
            "hidden_width": config.hidden_width,
            "macs_per_pixel": best_mlp.macs(),
        },
        "training": {
            "epochs": epochs, "unroll_steps": k, "batch": batch, "lr0": lr0,
            "subsample_px_per_pose_per_epoch": subsample, "ref_frames_validator_only": ref_frames,
            "init": "TIER1 evidence_mean_init_split — analytic, output==box-mean(E taps)+box-mean(D taps) BY CONSTRUCTION, verified at runtime before epoch 0",
            "loss": "TIER2 noise2noise: plain MSE(net(draw_A features), draw_B radiance), backprop through the v7e evidence clamp — teacher NEVER in the loss, validator only",
            "evidence_clamp_gamma": evidence_clamp_gamma(),
            "sky_history_reject_active": sky_reject,
            "moving_camera_settle": { "pan_step_rad_per_step": pan_step, "note": "TIER3 kept, A1-innocent" },
            "mirror_pose": { "camera": "denoiser_dataset::mirror_camera", "evidence_spp": mirror_spp, "note": "TIER3 kept unconditionally, A3-innocent (best-behaved arm)" },
            "highlight_sampling": "DROPPED ENTIRELY — A2 convicted detonator (scratch/v8-ablate-A2.log)",
            "history_precompute": "once per epoch, sourced from ema_mlp (kept, cf8bd7b)",
        },
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [tw, th],
            "train": train_cams.iter().map(|(n, ..)| *n).collect::<Vec<_>>(), "val": ["orbit_-20 (still)"] },
        "gate": "presents ONLY if real_image_ordeal writes a PASS stamp beside this file (run with GAIA_V7_SKY_HISTORY=reject to match training semantics)",
    });
    std::fs::write(data_dir.join(format!("rdirect-weights-{run_tag}.provenance.json")), serde_json::to_string_pretty(&prov).unwrap()).unwrap();
    println!("[{run_tag}] wrote provenance ({}). tag={run_tag} weights={} — NOT the real v8 checkpoint unless tag==\"v8\".", data_dir.join(format!("rdirect-weights-{run_tag}.provenance.json")).display(), wpath.display());
    std::io::stdout().flush().ok();
}
