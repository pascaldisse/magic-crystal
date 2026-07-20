// V7-LIVE LANE STAGE 1 — SPLIT FEATURE GATHER (E/D evidence, no history yet).
// Sibling of rdirect_gather.wgsl's `gather` (23-in, composite radiance);
// this entry (`gather_split`) mirrors CPU `rdirect::pixel_features_split`
// bit-for-bit (parity target: n0-gate1-shaped test, see
// scratch/v7-live-lane.md): E's 2×2 demod-log taps (12) + D's 2×2 demod-log
// taps (12) + subpixel offset (2) + hi-res albedo (3) + normal (3) +
// log depth (1) + screen-space motion (2) = 35 (`INPUT_FEATURES_SPLIT`).
// History (prev_dl + valid, → 39 `HIST_FEATURES_SPLIT`) is STAGE 2 — not
// written here.
//
// Inputs:
//   accum_ed : integrator.wgsl `integrate_split`'s trace-resolution output —
//              2 vec4 cells/px: [2i+0] = (E.rgb sum, sample count),
//                                [2i+1] = (D.rgb sum, sample count).
//   aov      : SAME native-resolution primary-hit AOVs as the 23-in gather
//              (rdirect_gather.wgsl) — 2 vec4 cells/px, unchanged layout.
//
// CURRENT-FRAME ONLY — no prior-frame read, no cross-frame state. BAN-SCOPED.

struct GatherU {
  // dims = (low_w, low_h, target_w, target_h)
  dims: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: GatherU;
@group(0) @binding(1) var<storage, read> accum_ed: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> aov: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> feats: array<f32>;

// Mirror rdirect.rs: ALBEDO_DEMOD_EPS and NO_HIT_ALBEDO_THRESHOLD_SQ.
const EPS: f32 = 1.0e-3;
const NO_HIT_SQ: f32 = 1.0e-8;
const FEATURES_SPLIT: u32 = 35u;

// rdirect.rs::low_coord — target index → continuous low-res coordinate.
fn low_coord(t: u32, low: u32, tgt: u32) -> f32 {
  return (f32(t) + 0.5) * f32(low) / f32(tgt) - 0.5;
}

@compute @workgroup_size(8, 8, 1)
fn gather_split(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tw = u.dims.z;
  let th = u.dims.w;
  if (gid.x >= tw || gid.y >= th) { return; }
  let tx = gid.x;
  let ty = gid.y;
  let lw = u.dims.x;
  let lh = u.dims.y;
  let i = ty * tw + tx;

  // Native-res primary-hit AOVs for this target pixel — SAME as the 23-in
  // gather (both E and D share this pixel's albedo demod divisor).
  let a = aov[2u * i + 0u];
  let b = aov[2u * i + 1u];
  let albedo = a.xyz;
  let depth = a.w;
  let normal = b.xyz;

  var divisor = vec3<f32>(1.0, 1.0, 1.0);
  if (dot(albedo, albedo) > NO_HIT_SQ) {
    divisor = albedo + vec3<f32>(EPS, EPS, EPS);
  }

  // bilinear_taps — SAME clamped 2×2 neighbourhood + subpixel offset the
  // 23-in gather uses (E and D reuse the SAME taps/dx/dy, per
  // pixel_features_split).
  let fx = low_coord(tx, lw, tw);
  let fy = low_coord(ty, lh, th);
  let x0 = floor(fx);
  let y0 = floor(fy);
  let dx = fx - x0;
  let dy = fy - y0;
  let x0i = min(u32(max(x0, 0.0)), lw - 1u);
  let x1i = min(u32(max(x0 + 1.0, 0.0)), lw - 1u);
  let y0i = min(u32(max(y0, 0.0)), lh - 1u);
  let y1i = min(u32(max(y0 + 1.0, 0.0)), lh - 1u);
  var taps = array<u32, 4>(
    y0i * lw + x0i,
    y0i * lw + x1i,
    y1i * lw + x0i,
    y1i * lw + x1i,
  );

  let base = i * FEATURES_SPLIT;
  var k = 0u;
  // E taps (accum_ed[2*tap + 0]).
  for (var t = 0u; t < 4u; t = t + 1u) {
    let cell = accum_ed[2u * taps[t] + 0u];
    let rad = cell.xyz / max(cell.w, 1.0);
    let d = rad / divisor;
    feats[base + k + 0u] = log(max(d.x, 0.0) + 1.0);
    feats[base + k + 1u] = log(max(d.y, 0.0) + 1.0);
    feats[base + k + 2u] = log(max(d.z, 0.0) + 1.0);
    k = k + 3u;
  }
  // D taps (accum_ed[2*tap + 1]).
  for (var t = 0u; t < 4u; t = t + 1u) {
    let cell = accum_ed[2u * taps[t] + 1u];
    let rad = cell.xyz / max(cell.w, 1.0);
    let d = rad / divisor;
    feats[base + k + 0u] = log(max(d.x, 0.0) + 1.0);
    feats[base + k + 1u] = log(max(d.y, 0.0) + 1.0);
    feats[base + k + 2u] = log(max(d.z, 0.0) + 1.0);
    k = k + 3u;
  }
  feats[base + k + 0u] = dx;
  feats[base + k + 1u] = dy;
  k = k + 2u;
  feats[base + k + 0u] = albedo.x;
  feats[base + k + 1u] = albedo.y;
  feats[base + k + 2u] = albedo.z;
  k = k + 3u;
  feats[base + k + 0u] = normal.x;
  feats[base + k + 1u] = normal.y;
  feats[base + k + 2u] = normal.z;
  k = k + 3u;
  feats[base + k] = log(max(depth, 0.0) + 1.0);
  k = k + 1u;
  // Screen-space motion — zeroed this wave (matches the CPU training-set
  // reference and the 23-in gather's own current wave scope).
  feats[base + k + 0u] = 0.0;
  feats[base + k + 1u] = 0.0;
}

// ─────────────────────────────────────────────────────────────────────────
// V7-LIVE LANE STAGE 2 — RECURRENT HISTORY (features 35-38, → 39-in
// `HIST_FEATURES_SPLIT`). Mirrors CPU `rdirect::hist_features_split` fed by
// `direct_render_sequence_hist_split`'s per-pixel reprojection block
// bit-for-bit: world point = this frame's camera ray * depth (or 1e5 on a
// miss, matching CPU's `is_miss` convention), reprojected into the PREVIOUS
// camera's screen via `CamPose::reproject` (pinhole, same sign convention),
// nearest-pixel depth+normal reject test against the previous frame's own
// AOV, and — only if accepted — a BILINEAR resample of the previous frame's
// net output (demod-log space) at the fractional reprojected coordinate.
// First frame / any reject ⇒ prev_dl = 0, valid = 0 (copy of the CPU rule).
// This entry is ADDITIVE: `gather_split` (35-in, Stage 1) above is untouched
// byte-for-byte; nothing here executes unless the caller builds
// `FeatureGatherHistSplit` (flag-gated the same as Stage 1, default OFF).

const FEATURES_HIST: u32 = 39u;
// SNAP_EPS — symmetric pixel-boundary snap (v7 seam closure, room 5).
// GPU vs CPU dot-product/FMA evaluation order can disagree by a few ULP on
// `sx`/`sy`; invisible almost everywhere, but a self-reprojecting static
// camera lands the fractional coord EXACTLY on an integer pixel boundary
// (tx=0/ty=0/w-1/h-1 reproject to fpx/fpy==0 or dim-1 algebraically), so the
// sub-ULP noise flips which side of that boundary the coord lands on, which
// the `is_miss` accept/reject test converts into a full valid=0/1 disagreement.
// Fix: snap fpx/fpy to the nearest integer whenever within SNAP_EPS of one,
// BEFORE the accept test — this removes the boundary ambiguity at its root
// (the fractional coordinate itself) instead of admitting a fuzzy accept
// window (room 4's REPROJ_EDGE_EPS, superseded: asymmetric — GPU-only — so it
// could flip pixels the OTHER side correctly rejected, and did, per
// scratch/v7-live-lane.md room 4's px=2976 A/B dump). Applied identically on
// BOTH sides (CamPose::reproject in rdirect.rs, cam_reproject here), so it
// cannot introduce a new GPU-vs-CPU asymmetry.
// Magnitude: observed ULP noise on `sx`/`sy` after the pinhole projection is
// ~1e-6 (this lane's own pan-sequence parity floor); a half-pixel is 0.5.
// 1e-3 sits ~1000x above the noise floor and ~500x below the half-pixel
// tie point — comfortably inside both margins, not tuned to any one probe's
// numbers.
const SNAP_EPS: f32 = 1.0e-3;

// One camera pose, GPU layout: eye_ht.xyz=eye, .w=half_tan;
// right_asp.xyz=right, .w=aspect; up.xyz=up; fwd.xyz=forward. Mirrors CPU
// `rdirect::CamPose` field-for-field (no reordering, no repacking games).
struct CamGpu {
  eye_ht: vec4<f32>,
  right_asp: vec4<f32>,
  up: vec4<f32>,
  fwd: vec4<f32>,
};

struct HistU {
  cur: CamGpu,
  prev: CamGpu,
  // params = (prev_w, prev_h, has_prev, depth_tol)
  params: vec4<f32>,
  // params2 = (normal_thresh, pad, pad, pad)
  params2: vec4<f32>,
};

@group(1) @binding(0) var<storage, read> prev_out_dl: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> prev_aov: array<vec4<f32>>;
@group(1) @binding(2) var<uniform> hu: HistU;

// CamPose::ray_dir — SAME pixel-centre primary ray the integrator/CPU use.
fn cam_ray_dir(cam: CamGpu, tx: u32, ty: u32, w: u32, h: u32) -> vec3<f32> {
  let cx = (2.0 * (f32(tx) + 0.5) / f32(w)) - 1.0;
  let cy = 1.0 - (2.0 * (f32(ty) + 0.5) / f32(h));
  let half_tan = cam.eye_ht.w;
  let aspect = cam.right_asp.w;
  let d = cam.fwd.xyz + cam.right_asp.xyz * cx * half_tan * aspect + cam.up.xyz * cy * half_tan;
  let len = length(d);
  if (len <= 1.0e-8) { return vec3<f32>(0.0, 0.0, 0.0); }
  return d / len;
}

// CamPose::reproject — returns (fpx, fpy, ok) with ok as 1.0/0.0 (WGSL has no
// Option); sign-for-sign against the CPU version, including the same
// behind-eye and off-screen rejections.
fn cam_reproject(cam: CamGpu, world: vec3<f32>, w: f32, h: f32) -> vec3<f32> {
  let rel = world - cam.eye_ht.xyz;
  let rz = dot(rel, cam.fwd.xyz);
  if (rz <= 1.0e-4) { return vec3<f32>(0.0, 0.0, 0.0); }
  let half_tan = cam.eye_ht.w;
  let aspect = cam.right_asp.w;
  let sx = dot(rel, cam.right_asp.xyz) / (rz * half_tan * aspect);
  let sy = dot(rel, cam.up.xyz) / (rz * half_tan);
  var fpx = (sx + 1.0) * 0.5 * w - 0.5;
  var fpy = (1.0 - sy) * 0.5 * h - 0.5;
  let snap_x = floor(fpx + 0.5);
  if (abs(fpx - snap_x) < SNAP_EPS) { fpx = snap_x; }
  let snap_y = floor(fpy + 0.5);
  if (abs(fpy - snap_y) < SNAP_EPS) { fpy = snap_y; }
  if (fpx < 0.0 || fpy < 0.0 || fpx > (w - 1.0) || fpy > (h - 1.0)) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  return vec3<f32>(fpx, fpy, 1.0);
}

// bilinear_vec3 — SAME clamped 4-tap resample as the CPU TAA-style history
// fetch, over the `prev_out_dl` buffer (xyz used, w unused/pad).
fn bilinear_prev_dl(fx: f32, fy: f32, w: u32, h: u32) -> vec3<f32> {
  let x0 = i32(floor(fx));
  let y0 = i32(floor(fy));
  let tx = fx - f32(x0);
  let ty = fy - f32(y0);
  let x0c = u32(clamp(x0, 0, i32(w) - 1));
  let x1c = u32(clamp(x0 + 1, 0, i32(w) - 1));
  let y0c = u32(clamp(y0, 0, i32(h) - 1));
  let y1c = u32(clamp(y0 + 1, 0, i32(h) - 1));
  let a = prev_out_dl[y0c * w + x0c].xyz;
  let b = prev_out_dl[y0c * w + x1c].xyz;
  let c = prev_out_dl[y1c * w + x0c].xyz;
  let d = prev_out_dl[y1c * w + x1c].xyz;
  let top = a * (1.0 - tx) + b * tx;
  let bot = c * (1.0 - tx) + d * tx;
  return top * (1.0 - ty) + bot * ty;
}

@compute @workgroup_size(8, 8, 1)
fn gather_hist_split(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tw = u.dims.z;
  let th = u.dims.w;
  if (gid.x >= tw || gid.y >= th) { return; }
  let tx = gid.x;
  let ty = gid.y;
  let lw = u.dims.x;
  let lh = u.dims.y;
  let i = ty * tw + tx;

  // Native-res primary-hit AOVs for this target pixel — SAME as `gather_split`.
  let a = aov[2u * i + 0u];
  let b = aov[2u * i + 1u];
  let albedo = a.xyz;
  let depth = a.w;
  let normal = b.xyz;

  var divisor = vec3<f32>(1.0, 1.0, 1.0);
  if (dot(albedo, albedo) > NO_HIT_SQ) {
    divisor = albedo + vec3<f32>(EPS, EPS, EPS);
  }

  let fxs = low_coord(tx, lw, tw);
  let fys = low_coord(ty, lh, th);
  let x0 = floor(fxs);
  let y0 = floor(fys);
  let dx = fxs - x0;
  let dy = fys - y0;
  let x0i = min(u32(max(x0, 0.0)), lw - 1u);
  let x1i = min(u32(max(x0 + 1.0, 0.0)), lw - 1u);
  let y0i = min(u32(max(y0, 0.0)), lh - 1u);
  let y1i = min(u32(max(y0 + 1.0, 0.0)), lh - 1u);
  var taps = array<u32, 4>(
    y0i * lw + x0i,
    y0i * lw + x1i,
    y1i * lw + x0i,
    y1i * lw + x1i,
  );

  let base = i * FEATURES_HIST;
  var k = 0u;
  for (var t = 0u; t < 4u; t = t + 1u) {
    let cell = accum_ed[2u * taps[t] + 0u];
    let rad = cell.xyz / max(cell.w, 1.0);
    let d = rad / divisor;
    feats[base + k + 0u] = log(max(d.x, 0.0) + 1.0);
    feats[base + k + 1u] = log(max(d.y, 0.0) + 1.0);
    feats[base + k + 2u] = log(max(d.z, 0.0) + 1.0);
    k = k + 3u;
  }
  for (var t = 0u; t < 4u; t = t + 1u) {
    let cell = accum_ed[2u * taps[t] + 1u];
    let rad = cell.xyz / max(cell.w, 1.0);
    let d = rad / divisor;
    feats[base + k + 0u] = log(max(d.x, 0.0) + 1.0);
    feats[base + k + 1u] = log(max(d.y, 0.0) + 1.0);
    feats[base + k + 2u] = log(max(d.z, 0.0) + 1.0);
    k = k + 3u;
  }
  feats[base + k + 0u] = dx;
  feats[base + k + 1u] = dy;
  k = k + 2u;
  feats[base + k + 0u] = albedo.x;
  feats[base + k + 1u] = albedo.y;
  feats[base + k + 2u] = albedo.z;
  k = k + 3u;
  feats[base + k + 0u] = normal.x;
  feats[base + k + 1u] = normal.y;
  feats[base + k + 2u] = normal.z;
  k = k + 3u;
  feats[base + k] = log(max(depth, 0.0) + 1.0);
  k = k + 1u;
  feats[base + k + 0u] = 0.0;
  feats[base + k + 1u] = 0.0;
  k = k + 2u;

  // ── HISTORY (idx 35-38) — copy of CPU direct_render_sequence_hist_split's
  // reprojection block, exactly. ──
  var prev_dl = vec3<f32>(0.0, 0.0, 0.0);
  var valid = 0.0;
  let has_prev = hu.params.z;
  if (has_prev > 0.5) {
    let prev_w = u32(hu.params.x);
    let prev_h = u32(hu.params.y);
    let depth_tol = hu.params.w;
    let normal_thresh = hu.params2.x;

    let is_miss = depth <= 0.0;
    let dir = cam_ray_dir(hu.cur, tx, ty, tw, th);
    var dist = depth;
    if (is_miss) { dist = 1.0e5; }
    let world = hu.cur.eye_ht.xyz + dir * dist;

    let rep = cam_reproject(hu.prev, world, f32(prev_w), f32(prev_h));
    if (rep.z > 0.5) {
      let fx = rep.x;
      let fy = rep.y;
      // fx/fy are always >=0 here (cam_reproject rejects fpx<0/fpy<0 above),
      // so floor(x+0.5) == Rust's f32::round() (half-away-from-zero). WGSL's
      // round() is half-to-even and disagreed with the CPU reference at .5
      // ties — this is the fix for the reprojection-guard validity mismatch.
      let ipx = u32(clamp(floor(fx + 0.5), 0.0, f32(prev_w) - 1.0));
      let ipy = u32(clamp(floor(fy + 0.5), 0.0, f32(prev_h) - 1.0));
      let pj = ipy * prev_w + ipx;
      let prev_depth = prev_aov[2u * pj + 0u].w;
      let prev_norm = prev_aov[2u * pj + 1u].xyz;
      let prev_miss = prev_depth <= 0.0;
      var ok = false;
      if (is_miss) {
        ok = prev_miss;
      } else if (prev_miss) {
        ok = false;
      } else {
        let dist_prev = length(world - hu.prev.eye_ht.xyz);
        let depth_ok = abs(dist_prev - prev_depth) <= depth_tol * max(dist_prev, 1.0e-4);
        let normal_ok = dot(normal, prev_norm) >= normal_thresh;
        ok = depth_ok && normal_ok;
      }
      if (ok) {
        prev_dl = bilinear_prev_dl(fx, fy, prev_w, prev_h);
        valid = 1.0;
      }
    }
  }
  feats[base + k + 0u] = prev_dl.x;
  feats[base + k + 1u] = prev_dl.y;
  feats[base + k + 2u] = prev_dl.z;
  feats[base + k + 3u] = valid;
}
