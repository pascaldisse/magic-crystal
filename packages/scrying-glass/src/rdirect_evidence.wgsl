// V7-LIVE LANE STAGE 3 — evidence-clamp GPU kernels (rdirect.rs semantics,
// commit c8b9ba6): presented = min(net_linear, gamma * local_max_3x3(
// temporal_mean(evidence_composite))). Three small compute entries, each on
// its OWN @group(0) binding range (0-2 / 3-5 / 6-8) so they coexist in one
// module without location collisions while staying independently
// pipeline-able (mirrors the house pattern: one file, several @compute fns,
// see rdirect_gather_split.wgsl's gather_split/gather_hist_split split).
//
//  evidence_accumulate — bilinear-upsamples this frame's low-res E+D 1-spp
//    radiance (integrator `accum_ed`, 2 vec4 sum+count cells/px) to native
//    res with the SAME 4-tap low_coord scheme every gather kernel uses, and
//    ADDS the composite (E_up + D_up) into a persistent native-res sum
//    buffer — the running numerator of rdirect.rs `EvidenceAccum`'s
//    temporal MEAN. `count` (plain frame counter) is tracked on the CPU and
//    folded into evidence_clamp_present below instead of a separate mean
//    buffer: max(sum/count) == max(sum)/count for a positive constant
//    count, so this is bit-exact against `EvidenceAccum::ceiling`, not an
//    approximation.
//  evidence_clamp_present — 3x3 (r=1) border-clamped spatial max of the sum
//    buffer (rdirect.rs `local_max_3x3`, applied to sum), scaled by
//    gamma/count, then `present = min(present, ceiling)` per channel
//    (rdirect.rs `clamp_evidence_lin`).
//  pack_out_dl3to4 — repacks the live net's tightly-packed `[n,3]` demod-log
//    output (rdirect_live.rs `output_buffer_set`) into the vec4-per-pixel
//    layout `rdirect_gather::HistoryBuffers::prev_out_dl` expects for next
//    frame's reprojection (Stage 2's ping-pong contract).

fn low_coord(t: u32, low: u32, tgt: u32) -> f32 {
  return (f32(t) + 0.5) * f32(low) / f32(tgt) - 0.5;
}

// ── evidence_accumulate: bindings 0-2 ──────────────────────────────────────
struct Dims4 { dims: vec4<u32> }; // (low_w, low_h, target_w, target_h)

@group(0) @binding(0) var<uniform> u_dims: Dims4;
@group(0) @binding(1) var<storage, read> accum_ed: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> evidence_sum: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn evidence_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tw = u_dims.dims.z;
  let th = u_dims.dims.w;
  if (gid.x >= tw || gid.y >= th) { return; }
  let tx = gid.x;
  let ty = gid.y;
  let lw = u_dims.dims.x;
  let lh = u_dims.dims.y;

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

  let e00c = accum_ed[2u * (y0i * lw + x0i) + 0u]; let e00 = e00c.xyz / max(e00c.w, 1.0);
  let e10c = accum_ed[2u * (y0i * lw + x1i) + 0u]; let e10 = e10c.xyz / max(e10c.w, 1.0);
  let e01c = accum_ed[2u * (y1i * lw + x0i) + 0u]; let e01 = e01c.xyz / max(e01c.w, 1.0);
  let e11c = accum_ed[2u * (y1i * lw + x1i) + 0u]; let e11 = e11c.xyz / max(e11c.w, 1.0);
  let e_top = e00 * (1.0 - dx) + e10 * dx;
  let e_bot = e01 * (1.0 - dx) + e11 * dx;
  let e_up = e_top * (1.0 - dy) + e_bot * dy;

  let d00c = accum_ed[2u * (y0i * lw + x0i) + 1u]; let d00 = d00c.xyz / max(d00c.w, 1.0);
  let d10c = accum_ed[2u * (y0i * lw + x1i) + 1u]; let d10 = d10c.xyz / max(d10c.w, 1.0);
  let d01c = accum_ed[2u * (y1i * lw + x0i) + 1u]; let d01 = d01c.xyz / max(d01c.w, 1.0);
  let d11c = accum_ed[2u * (y1i * lw + x1i) + 1u]; let d11 = d11c.xyz / max(d11c.w, 1.0);
  let d_top = d00 * (1.0 - dx) + d10 * dx;
  let d_bot = d01 * (1.0 - dx) + d11 * dx;
  let d_up = d_top * (1.0 - dy) + d_bot * dy;

  let composite = e_up + d_up;
  let i = ty * tw + tx;
  evidence_sum[i] = vec4<f32>(evidence_sum[i].xyz + composite, 0.0);
}

// ── evidence_clamp_present: bindings 3-5 ───────────────────────────────────
struct ClampU { params: vec4<f32> }; // (tw, th, count, gamma)

@group(0) @binding(3) var<uniform> u_clamp: ClampU;
@group(0) @binding(4) var<storage, read> sum_in: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> present: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn evidence_clamp_present(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u32(u_clamp.params.x);
  let h = u32(u_clamp.params.y);
  if (gid.x >= w || gid.y >= h) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  var m = vec3<f32>(0.0, 0.0, 0.0);
  for (var ddy = -1; ddy <= 1; ddy = ddy + 1) {
    let ny = clamp(y + ddy, 0, i32(h) - 1);
    for (var ddx = -1; ddx <= 1; ddx = ddx + 1) {
      let nx = clamp(x + ddx, 0, i32(w) - 1);
      m = max(m, sum_in[u32(ny) * w + u32(nx)].xyz);
    }
  }
  let count = max(u_clamp.params.z, 1.0);
  let gamma = u_clamp.params.w;
  let ceiling = max((gamma / count) * m, vec3<f32>(0.0, 0.0, 0.0));
  let i = gid.y * w + gid.x;
  let cur = present[i];
  present[i] = vec4<f32>(min(cur.xyz, ceiling), cur.w);
}

// ── pack_out_dl3to4: bindings 6-8 ──────────────────────────────────────────
struct PackU { n: vec4<u32> }; // n.x used, rest padding

@group(0) @binding(6) var<uniform> u_pack: PackU;
@group(0) @binding(7) var<storage, read> src3: array<f32>;
@group(0) @binding(8) var<storage, read_write> dst4: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn pack_out_dl3to4(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u_pack.n.x) { return; }
  dst4[i] = vec4<f32>(src3[i * 3u], src3[i * 3u + 1u], src3[i * 3u + 2u], 0.0);
}
