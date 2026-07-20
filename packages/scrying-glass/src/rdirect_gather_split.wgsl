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
