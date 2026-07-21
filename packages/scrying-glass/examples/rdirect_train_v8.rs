//! R-DIRECT v8 — MOVING-CAMERA SETTLE + SPECULAR EVIDENCE + HIGHLIGHT TARGET.
//! Smallest-diff evolution of v7e (evidence-clamp trainer). Three structural
//! changes, all requested by HANDOFF.md §07-20 EVE:
//!
//! (a) MOVING-CAMERA HISTORY (GHOST AUTOPSY root fix, training side). v7/v7e
//!     trained the recurrent unroll with IDENTITY self-feedback: `prev_dl`
//!     at step s+1 was always THIS SAME PIXEL's own step-s output — no
//!     `CamPose::reproject` ever ran during training, only at eval
//!     (`direct_render_sequence_hist_split`, the ordeal's own act). Ghost
//!     autopsy found sky ghosting is BOTH a structural gate bug (fixed,
//!     `GAIA_V7_SKY_HISTORY=reject`) AND a training-distribution bug: the
//!     net never saw a `valid=1` sample that wasn't identity feedback, so
//!     it has no learned reason to discount a genuinely-reprojected (and
//!     therefore slightly different) history sample. v8's settle loop now
//!     drifts the camera yaw across the K unroll steps (`pan_step`,
//!     default = the ordeal's own `GAIA_ORDEAL_PANSTEP` 0.004 rad/frame)
//!     and reprojects the PREVIOUS step's real net output through
//!     `CamPose::reproject` + the depth/normal guard + `sky_history_reject()`
//!     — the exact same act `direct_render_sequence_hist_split` runs at
//!     eval, just also exercised during training. `pan_step=0` degenerates
//!     EXACTLY to a still camera (SNAP_EPS, room 5, makes static
//!     self-reprojection exact) — so the validation pose (still, matching
//!     the ordeal's own STILL test) needs no separate code path; one
//!     function serves both.
//!
//! (b) SPECULAR EVIDENCE + MIRROR POSES (MIRROR AUTOPSY fix direction #1).
//!     Autopsy verdict: `naruko_show_chrome`'s GGX lobe at roughness 0.02 is
//!     near-delta — a curved mirror's 2x2 low-res bilinear tap neighbourhood
//!     is 4 near-independent 1-spp draws of UNRELATED reflected points, not
//!     a coherent patch; the net (trained on diffuse-heavy data) falls back
//!     to a flat gray blob. Fix direction #1: raise trace spp specifically
//!     for curved-mirror poses so the low-res evidence taps become locally
//!     coherent. v8 adds a `mirror` training pose (`denoiser_dataset::
//!     mirror_camera`, the `naruko_show_chrome`-framing `spawn_eye`) whose
//!     `low_e`/`low_d` evidence taps AND teacher/target render at
//!     `GAIA_V8_MIRROR_SPP` (default 4, vs 1 for every other pose) —
//!     "coherent specular evidence... in the teacher+evidence generation
//!     for mirror poses" per the mandate's own wording.
//!
//! (c) HIGHLIGHT-TARGETED SAMPLING (gamma-sweep verdict). γ stays 1.5 (IRON
//!     — the sweep proved only 1.5 holds the resid≤0.035 bar) and no
//!     cap/gate/firefly-weight loss term is added (still banned, v7c/N3/N4
//!     precedent) — PLAIN MSE, unchanged loss SHAPE. What changes is the
//!     per-epoch pixel SAMPLING distribution: `GAIA_V8_HIGHLIGHT_FRAC`
//!     (default 0.3) of each pose's per-epoch subsample is drawn from that
//!     pose's own top-luminance (brightest `GAIA_V8_HIGHLIGHT_PCTL`%, default
//!     top 5%) teacher pixels instead of uniformly at random — the gamma-
//!     sweep found the net under-renders highlights by -38.7% vs teacher at
//!     γ=1.5 (clamp-independent, a genuine undertraining gap, not a clamp
//!     artifact) — oversampling those pixels raises their gradient density
//!     without touching the loss function itself.
//!
//! FRESH init only (no resume) — the CURE round's own log: "fine-tune-from-
//! best ALWAYS regressed". Corner-crawl lr 1e-4 (flat-ish, mild harmonic
//! decay). Monitor + print score EVERY epoch. Cross-run bar-normalized score
//! floor: if `data/rdirect-weights-v8.bin` already exists (a resumed v8
//! round), its own score is the floor this run must beat to overwrite it;
//! otherwise any first save is accepted (score floor = +inf).
//!
//! Run: cargo run -p scrying-glass --release -j2 --example rdirect_train_v8
//!   GAIA_V7_SKY_HISTORY=reject   (mandate: sky-reject semantics ACTIVE in
//!                                 the settle loop — this is v8's act, not
//!                                 an optional env; set it when running.)
//!   GAIA_V8_W/H (default 384x288), GAIA_V8_STILL (K, default 3),
//!   GAIA_V8_REF (ref_frames, default 64), GAIA_V8_PANSTEP (rad/step,
//!   default 0.004 — matches the ordeal's own GAIA_ORDEAL_PANSTEP),
//!   GAIA_V8_MIRROR_SPP (default 4), GAIA_V8_BLUR (D box-blur radius,
//!   default 2), GAIA_V8_HIGHLIGHT_FRAC (default 0.3), GAIA_V8_HIGHLIGHT_PCTL
//!   (default 0.05 = top 5%), GAIA_V8_EPOCHS (default 200), GAIA_V8_SUBSAMPLE
//!   (px/pose/epoch, default 5000), GAIA_V8_BATCH (default 64), GAIA_V8_LR
//!   (default 1e-4), GAIA_V8_EMA (default 0.999), GAIA_V8_SPARK_TGT (default
//!   16.0), GAIA_V8_RESID_GATE (default 0.035 — the IRON bar itself),
//!   GAIA_V8_WALL (seconds, default 10800 = 3h — detach for real runs).
//!
//! ABLATION SWITCHES (2026-07-20, all default to CURRENT v8 behavior —
//!   omitting them reproduces the exact mandate trainer above):
//!   GAIA_V8_MIRROR_POSE (default 1; 0 = drop the "mirror" pose from the
//!   TRAINING pose set entirely — component (b) off; the diagnostic mirror
//!   monitor still renders/reports, it just no longer contributes gradient).
//!   GAIA_V8_TAG (default "v8") — every log line, the saved checkpoint
//!   (`data/rdirect-weights-<TAG>.bin`) and its provenance JSON are named
//!   from this tag, so ablation runs (GAIA_V8_TAG=a0, etc.) can NEVER
//!   overwrite the real v8 checkpoint or its cross-run score floor — each
//!   tag gets its own floor file and own weights path. GAIA_V8_PANSTEP=0
//!   already disables component (a) (still-camera history, degenerates via
//!   SNAP_EPS) and GAIA_V8_HIGHLIGHT_FRAC=0 already disables component (c)
//!   (pure uniform sampling) — both pre-existing, no new switch needed.

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

const INIT_SEED: u64 = 0x00d1_5eed_0008;
const DEPTH_TOL: f32 = 0.05; // same reprojection guard the ordeal uses
const NORMAL_THRESH: f32 = 0.85;

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
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn lum(c: GVec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

fn cam_pose(cam: &Camera, w: u32, h: u32) -> CamPose {
    let (right, up, forward) = cam.basis();
    CamPose { eye: cam.eye, right, up, forward, half_tan: (cam.fov_y_radians * 0.5).tan(), aspect: w as f32 / h as f32 }
}

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

/// One step of a moving-camera pose sequence: everything render_pose_seq
/// produces at that step's own camera.
struct Step {
    low_e: Vec<GVec3>,
    low_d: Vec<GVec3>,
    albedo: Vec<GVec3>,
    normal: Vec<GVec3>,
    depth: Vec<f32>,
    teacher: Vec<GVec3>,                     // EXACT (E_full+D_full) at THIS step's cam — metrics only
    target_dl: Vec<[f32; OUTPUT_CHANNELS]>,  // SMOOTHED (E_full + blur(D_full)) — trains the net
    ceiling_dl: Vec<[f32; OUTPUT_CHANNELS]>, // v7e evidence clamp ceiling, temporal-accumulated through this step
}

/// A full K-step camera sequence for one pose. `pan_step=0` degenerates to a
/// still camera (identical cam every step, SNAP_EPS makes the resulting
/// self-reprojection exact) — the SAME code path serves both "still
/// validation" and "moving training" (mandate a: one settle loop, real
/// reprojection, not a special-cased identity shortcut).
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
    blur_radius: i32,
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

        let np = IntegratorParams { spp: evidence_spp, seed: 0x7abc + step * 131 + 5, ..IntegratorParams::default() };
        let (low_e, low_d) = trace_headless_split(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np,
        );
        let (albedo, normal, depth) = scrying_glass::integrator::split_aov(&trace_headless_aov(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th,
        ));
        let (e_full, d_full) = trace_headless_split(
            device, queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
            &IntegratorParams::default(),
        );
        let d_blurred = box_blur(&d_full, tw, th, blur_radius);
        let teacher: Vec<GVec3> = (0..n).map(|i| e_full[i] + d_full[i]).collect();
        let smoothed: Vec<GVec3> = (0..n).map(|i| e_full[i] + d_blurred[i]).collect();
        let target_dl: Vec<[f32; OUTPUT_CHANNELS]> = (0..n).map(|px| target_demod_log(smoothed[px], albedo[px])).collect();

        let frame_composite = evidence_composite_frame(&low_e, &low_d, low_w, low_h, tw, th);
        accum.push(&frame_composite);
        let local_max = accum.ceiling();
        let ceiling_dl: Vec<[f32; OUTPUT_CHANNELS]> =
            (0..n).map(|px| evidence_ceiling_demod_log(local_max[px], gamma, albedo[px])).collect();

        steps.push(Step { low_e, low_d, albedo, normal, depth, teacher, target_dl, ceiling_dl });
    }
    PoseSeq { steps, cams, low_w, low_h, tw, th }
}

/// Reproject step `s-1`'s real net output (`prev_out_dl`, full native-res
/// image, EMA weights snapshotted at epoch start — see `history_forward`)
/// into step `s`'s screen at pixel (tx,ty). Byte-for-byte the same
/// accept/reject rule `direct_render_sequence_hist_split` runs (is_miss /
/// sky_reject / depth+normal guard), just standalone so it can be called
/// per-sampled-pixel during training without re-deriving the whole image.
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
/// sequence, using the EMA (Polyak-averaged) weights snapshotted at epoch
/// start, forward-only (no grad). This is the "target network" for the
/// epoch's training batches: batches within the epoch look up `prev_dl`
/// from this FROZEN chain (built once, before any of the epoch's Adam
/// steps) via real reprojection, exactly mirroring what
/// `direct_render_sequence_hist_split` does at eval — the only difference
/// is WHICH weights compute the live forward/backward pass for the sampled
/// pixel (the epoch's progressively-updated raw `mlp`) vs which weights
/// built the history it reprojects from (the smoothed `ema_mlp`, frozen at
/// epoch start). Sourcing the recurrent-feedback chain from the EMA rather
/// than the raw iterate is deliberate, not incidental: v7d's cure principle
/// ("the checkpoint that gets measured/saved is a slow-moving average...
/// which cannot itself develop a sharp outlier the way the raw SGD/Adam
/// iterate can") was previously applied only to monitor/checkpoint
/// selection; v8's diagnosed sparkle<->resid seesaw (epoch 2->21, sparkle
/// 45->795/Mpx while resid plateaus above bar) traced to this function
/// reading the RAW `mlp` as its once-per-epoch history source — any local
/// sparkle/overshoot in that epoch-start raw net got baked into the
/// `prev_dl` training-input feature for every later step of every pixel
/// that reprojects near it, a feedback path for exactly the observed
/// runaway signature. Reading `ema_mlp` here instead applies the same
/// noise-ball-isolation principle one level deeper: the recurrent input the
/// network trains against is now itself smoothed, matching what the
/// monitor/checkpoint already relied on. Standard truncated-BPTT-with-a-
/// target-chain; recomputing this on every SGD step would need a full-image
/// forward per batch, computationally infeasible for this trainer's
/// per-pixel MLP.
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

/// (c) Highlight recovery, measured directly: mean net/teacher luminance
/// ratio over the pose's own brightest `pctl` teacher pixels (last step).
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

/// Settle a pose sequence through the net exactly like the ordeal's own act
/// (`direct_render_sequence_hist_split`) and return the LAST step's
/// presented (clamped) image + its teacher.
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
    let run_tag = std::env::var("GAIA_V8_TAG").unwrap_or_else(|_| "v8".to_string());
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
    let blur_radius = env_u32("GAIA_V8_BLUR", 2) as i32;
    let pan_step = env_f32("GAIA_V8_PANSTEP", 0.004); // matches GAIA_ORDEAL_PANSTEP
    let mirror_spp = env_u32("GAIA_V8_MIRROR_SPP", 4);
    let mirror_pose_enabled = env_u32("GAIA_V8_MIRROR_POSE", 1) != 0;
    let highlight_frac = env_f32("GAIA_V8_HIGHLIGHT_FRAC", 0.3);
    let highlight_pctl = env_f32("GAIA_V8_HIGHLIGHT_PCTL", 0.05);
    let spark_target = env_f32("GAIA_V8_SPARK_TGT", 16.0);
    let resid_gate = env_f32("GAIA_V8_RESID_GATE", 0.035); // the IRON bar itself
    let monitor_every = env_u32("GAIA_V8_MONITOR", 1); // TRAINING LAW: every epoch
    let wall_budget = env_f32("GAIA_V8_WALL", 10_800.0); // 3h default — detach for real runs

    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| all.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();
    let mirror_cam = scrying_glass::denoiser_dataset::mirror_camera();
    let mut train_cams: Vec<(&str, Camera, u32)> = vec![
        ("front", find("front"), 1),
        ("wide", find("wide"), 1),
        ("orbit_+20", find("orbit_+20"), 1),
    ];
    if mirror_pose_enabled {
        train_cams.push(("mirror", mirror_cam, mirror_spp));
    }
    eprintln!(
        "[{run_tag}] mirror_pose_enabled={mirror_pose_enabled} — training poses: {:?}",
        train_cams.iter().map(|(n, ..)| *n).collect::<Vec<_>>()
    );

    let t_render = Instant::now();
    let poses: Vec<PoseSeq> = train_cams
        .iter()
        .map(|(_, c, spp)| render_pose_seq(&device, &queue, &base_tris, &scene, c, k, low_w, low_h, tw, th, ref_frames, blur_radius, pan_step, *spp))
        .collect();
    // Validation pose: STILL (pan_step=0, matching the ordeal's own STILL
    // test exactly — same camera every step, fresh 1-spp seed per step).
    let val_seq = render_pose_seq(&device, &queue, &base_tris, &scene, &find("orbit_-20"), k, low_w, low_h, tw, th, ref_frames, blur_radius, 0.0, 1);
    // Mirror monitor pose: STILL too (diagnostic only, not gating) — reuses
    // the mirror camera at pan_step=0 so its sparkle/resid/highlight numbers
    // are read the same way the ordeal reads any STILL pose.
    let mirror_val_seq = render_pose_seq(&device, &queue, &base_tris, &scene, &mirror_cam, k, low_w, low_h, tw, th, ref_frames, blur_radius, 0.0, mirror_spp);
    eprintln!(
        "[{run_tag}] rendered {}+2 MOVING/STILL pose sequences (K={k} steps, {tw}x{th}, teacher {ref_frames}, D-blur r={blur_radius}, pan_step={pan_step}, mirror_spp={mirror_spp}) in {:.1}s",
        poses.len(), t_render.elapsed().as_secs_f64()
    );
    std::io::stderr().flush().ok();

    // (c) highlight pixel pools: top `highlight_pctl` teacher-luminance
    // pixels at the LAST step of each training pose (the step the loss for
    // that step trains toward — every step has its own target, but the
    // settled/last-step highlights are what the ordeal's own resid_still
    // ultimately measures, so that's the pool oversampled).
    let highlight_pools: Vec<Vec<usize>> = poses
        .iter()
        .map(|p| {
            let teacher = &p.steps.last().unwrap().teacher;
            let mut order: Vec<usize> = (0..teacher.len()).collect();
            order.sort_by(|&a, &b| lum(teacher[b]).partial_cmp(&lum(teacher[a])).unwrap());
            let n = ((teacher.len() as f32 * highlight_pctl).ceil() as usize).max(1);
            order[..n].to_vec()
        })
        .collect();

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let wpath = data_dir.join(format!("rdirect-weights-{run_tag}.bin"));
    let config = RdirectConfig {
        hidden_layers: env_u32("GAIA_V8_LAYERS", 5) as usize,
        hidden_width: env_u32("GAIA_V8_WIDTH", 64) as usize,
    };

    // TRAINING LAW: FRESH init only — "fine-tune-from-best ALWAYS
    // regressed" (v7's own recorded lesson). No resume path in this trainer.
    let mut mlp = Mlp::new_random_with_input(config, HIST_FEATURES_SPLIT, INIT_SEED);
    assert_eq!(mlp.layer_dims()[0].0 as usize, HIST_FEATURES_SPLIT, "v8 net must be 39-input");
    let mut adam = Adam::new(&mlp, lr0, 0.9, 0.999, 1e-8);
    let mut ema_mlp = mlp.clone();
    eprintln!(
        "[{run_tag}] arch {:?} in={HIST_FEATURES_SPLIT} macs/px={} — {epochs} epochs (wall<={wall_budget}s), subsample {subsample}px/pose, lr0={lr0} (mild harmonic decay), ema={ema_decay}, highlight_frac={highlight_frac} (top {}%), PLAIN MSE vs split-smoothed target",
        config, mlp.macs(), highlight_pctl * 100.0,
    );
    std::io::stderr().flush().ok();

    // Cross-run bar-normalized score floor: beat the EXISTING checkpoint FOR
    // THIS TAG (if this is a resumed/second round of this tag's runs) or
    // accept any first save (no prior checkpoint for this tag on disk).
    // Tag-scoped by construction (wpath is per-tag) — an ablation run's
    // floor can never read or clobber the real v8 checkpoint's floor.
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

    {
        let (sp, rs) = run_monitor(&run_tag, "epoch -1 (fresh)", &ema_mlp, &val_seq, &mirror_val_seq, highlight_pctl, spark_target, resid_gate, &mut best_score, &mut best_bytes, &wpath);
        eprintln!("[{run_tag}] fresh-init baseline: sparkle {sp:.1} resid {rs:.4} (floor {best_score:.3})");
    }

    let n_px: Vec<usize> = poses.iter().map(|p| (p.tw * p.th) as usize).collect();
    let mut rng = Rng(INIT_SEED ^ 0xF00D);
    let t_train = Instant::now();

    'train: for epoch in 0..epochs {
        if t_train.elapsed().as_secs_f64() > wall_budget as f64 {
            eprintln!("[{run_tag}] WALL budget {wall_budget}s reached at epoch {epoch} — stopping, keeping best");
            break;
        }
        let frac = epoch as f32 / epochs as f32;
        adam.set_lr(lr0 / (1.0 + 1.0 * frac)); // mild harmonic decay from the corner-crawl lr

        // (a) Per-epoch history precompute: snapshot the full moving-camera
        // out_dl chain for every pose at the CURRENT (epoch-start) weights —
        // see history_forward's own doc for why this is once-per-epoch, not
        // once-per-batch.
        let t_hist = Instant::now();
        let history: Vec<Vec<Vec<GVec3>>> = poses.iter().map(|seq| history_forward(&ema_mlp, seq, sky_reject)).collect();
        let hist_ms = t_hist.elapsed().as_secs_f64() * 1000.0;

        let mut epoch_mse = 0.0f64;
        let mut n_steps = 0u64;

        // sample (pose, pixel) pairs: (1-highlight_frac) uniform random +
        // highlight_frac drawn from that pose's own bright-pixel pool.
        let mut samples: Vec<(usize, usize)> = Vec::new();
        for (pi, np) in n_px.iter().enumerate() {
            let n_hl = ((subsample as f32) * highlight_frac) as usize;
            let n_uniform = subsample - n_hl;
            for _ in 0..n_uniform {
                samples.push((pi, (rng.next() as usize) % np));
            }
            let pool = &highlight_pools[pi];
            for _ in 0..n_hl {
                let idx = (rng.next() as usize) % pool.len();
                samples.push((pi, pool[idx]));
            }
        }
        // shuffle so highlight/uniform samples interleave across batches
        for i in (1..samples.len()).rev() {
            let j = (rng.unit() * (i + 1) as f32) as usize % (i + 1);
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
            "[{run_tag}] epoch {}/{} mse={:.6} lr={:.6} hist_ms={hist_ms:.0} ({:.1}s)",
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
        "mirror_pose_enabled_in_training": mirror_pose_enabled,
        "weights_sha256": wsha,
        "supersedes": "rdirect-weights-v7.bin",
        "architecture": {
            "kind": "N7 MOVING-HISTORY + SPECULAR-EVIDENCE + HIGHLIGHT-SAMPLED — recurrent direct-render MLP, split radiance input (E+D), 39-in/3-out, evidence-clamped act",
            "input_features": HIST_FEATURES_SPLIT,
            "output_channels": OUTPUT_CHANNELS,
            "hidden_layers": config.hidden_layers,
            "hidden_width": config.hidden_width,
            "macs_per_pixel": best_mlp.macs(),
        },
        "training": {
            "epochs": epochs, "unroll_steps": k, "batch": batch, "lr0": lr0,
            "subsample_px_per_pose_per_epoch": subsample, "ref_frames": ref_frames,
            "init": "FRESH ONLY (39-in/3-out) — fine-tune-from-best banned, v7's own recorded regression",
            "loss": "PLAIN MSE vs split-smoothed target, backprop through the v7e evidence clamp (no cap/gate/firefly-weight loss term, no two-head — still banned)",
            "evidence_clamp_gamma": evidence_clamp_gamma(),
            "sky_history_reject_active": sky_reject,
            "moving_camera_settle": { "pan_step_rad_per_step": pan_step, "note": "real CamPose::reproject history feedback, NOT identity self-feedback; pan_step=0 (validation pose) degenerates exactly to a still camera via SNAP_EPS" },
            "mirror_pose": { "camera": "denoiser_dataset::mirror_camera (spawn_eye, frames naruko_show_chrome)", "evidence_spp": mirror_spp, "note": "coherent specular evidence — multiple GGX samples per low-res tap instead of 1, teacher+evidence generation both boosted" },
            "highlight_sampling": { "frac": highlight_frac, "top_pctl": highlight_pctl, "note": "SAMPLING-side oversampling of bright teacher pixels, not a loss-shape change — addresses the gamma-sweep's -38.7% highlight under-render at gamma 1.5" },
            "d_blur_radius": blur_radius,
            "target_construction": "target = E_full (exact, ref_frames) + box_blur(D_full, radius) — same v7e escape",
            "history_precompute": "once per epoch (target-network pattern): full-frame forward at epoch-start weights builds the reprojected-history chain batches read from; live weights update within the epoch as usual",
        },
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [tw, th],
            "train": train_cams.iter().map(|(n, ..)| *n).collect::<Vec<_>>(), "val": ["orbit_-20 (still)"] },
        "gate": "presents ONLY if real_image_ordeal writes a PASS stamp beside this file (run with GAIA_V7_SKY_HISTORY=reject to match training semantics)",
    });
    std::fs::write(data_dir.join(format!("rdirect-weights-{run_tag}.provenance.json")), serde_json::to_string_pretty(&prov).unwrap()).unwrap();
    println!("[{run_tag}] wrote provenance ({}). tag={run_tag} weights={} — NOT the real v8 checkpoint unless tag==\"v8\".", data_dir.join(format!("rdirect-weights-{run_tag}.provenance.json")).display(), wpath.display());
    std::io::stdout().flush().ok();
}
