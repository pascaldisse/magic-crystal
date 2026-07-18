// R-DIRECT AT SPEED — the fp16 fused kernel (MODE A: f16 storage, f32
// accumulate). Numerically identical feature construction and the SAME fixed
// accumulation order as src/rdirect.rs::Mlp::forward, but the net weights are
// converted to f16 and a PREFIX of them is loaded once per workgroup into
// threadgroup memory (var<workgroup>), so the 64 threads read the hot layers
// from on-chip memory instead of streaming device storage per thread.
//
// MEMORY NOTE (measured finding): this net is ~18.4k weights ≈ 36 KB as f16,
// which EXCEEDS the 32 KB Metal threadgroup limit — the whole net does NOT
// fit on-chip (unlike the 13.8 KB VIII-3 upscaler). So we cache the first
// CACHE_WEIGHTS (16000 f16 = 32000 B < 32768) and stream the tail from device
// storage. The cache/stream branch is UNIFORM across the workgroup (the o,i
// loop counters are identical for every thread), so it costs no divergence.
//
// MODE A: weights + inter-layer activations stored f16; the dot-product
// accumulator stays f32 (the sound fp16 mode, fp16 verdict). THE BAN: every
// value is a function of THIS frame's buffers alone. BAN-SCOPED

enable f16;

const MAX_LAYERS: u32 = 16u;
const MAX_WIDTH: u32 = 64u;
// Threadgroup weight cache ceiling in f16 scalars: 16000·2 = 32000 B < 32768.
const CACHE_WEIGHTS: u32 = 16000u;

const RADIANCE_TAPS: u32 = 4u;
const ALBEDO_DEMOD_EPS: f32 = 1e-3;
const NO_HIT_ALBEDO_THRESHOLD_SQ: f32 = 1e-8;

struct RdirectU {
  dims: vec4<u32>,   // low_w, low_h, target_w, target_h
  info: vec4<u32>,   // layer_count, weight_count, _, _
  layers: array<vec4<u32>, 16>, // per layer: in_dim, out_dim, w_off, b_off
};

@group(0) @binding(0) var<uniform> u: RdirectU;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> low_radiance: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> hi_albedo: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> hi_normal: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> hi_depth: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> hi_motion: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out_img: array<vec4<f32>>;

// Prefix of the net cached in threadgroup memory as f16 (loaded once/workgroup).
var<workgroup> w_cache: array<f16, 16000>;

fn demod_divisor(a: vec3<f32>) -> vec3<f32> {
  if (dot(a, a) > NO_HIT_ALBEDO_THRESHOLD_SQ) {
    return a + vec3<f32>(ALBEDO_DEMOD_EPS, ALBEDO_DEMOD_EPS, ALBEDO_DEMOD_EPS);
  }
  return vec3<f32>(1.0, 1.0, 1.0);
}
fn log_demod(radiance: vec3<f32>, divisor: vec3<f32>) -> vec3<f32> {
  let d = radiance / divisor;
  return vec3<f32>(log(max(d.x, 0.0) + 1.0), log(max(d.y, 0.0) + 1.0), log(max(d.z, 0.0) + 1.0));
}
fn undo_log_demod(dl: vec3<f32>, divisor: vec3<f32>) -> vec3<f32> {
  let expm1 = vec3<f32>(exp(dl.x) - 1.0, exp(dl.y) - 1.0, exp(dl.z) - 1.0);
  let clamped = vec3<f32>(max(expm1.x, 0.0), max(expm1.y, 0.0), max(expm1.z, 0.0));
  return clamped * divisor;
}
fn low_coord(target_idx: u32, low_dim: u32, target_dim: u32) -> f32 {
  return (f32(target_idx) + 0.5) * f32(low_dim) / f32(target_dim) - 0.5;
}

// One weight value as f32, from the f16 threadgroup cache (prefix) or streamed
// from device storage (tail). The index is UNIFORM across the workgroup.
fn weight_at(idx: u32) -> f32 {
  if (idx < CACHE_WEIGHTS) {
    return f32(w_cache[idx]);
  }
  return weights[idx];
}

@compute @workgroup_size(8, 8, 1)
fn render(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lidx: u32,
) {
  // Cooperative threadgroup load: the 64 threads together copy the cached
  // prefix of the net into w_cache as f16, ONCE.
  let wc = min(u.info.y, CACHE_WEIGHTS);
  var wi = lidx;
  loop {
    if (wi >= wc) { break; }
    w_cache[wi] = f16(weights[wi]);
    wi = wi + 64u;
  }
  workgroupBarrier();

  let low_w = u.dims.x;
  let low_h = u.dims.y;
  let tw = u.dims.z;
  let th = u.dims.w;
  if (gid.x >= tw || gid.y >= th) { return; }
  let tx = gid.x;
  let ty = gid.y;
  let pixel = ty * tw + tx;

  let fx = low_coord(tx, low_w, tw);
  let fy = low_coord(ty, low_h, th);
  let x0f = floor(fx);
  let y0f = floor(fy);
  let dx = fx - x0f;
  let dy = fy - y0f;
  let x0i = min(u32(max(x0f, 0.0)), low_w - 1u);
  let x1i = min(u32(max(x0f + 1.0, 0.0)), low_w - 1u);
  let y0i = min(u32(max(y0f, 0.0)), low_h - 1u);
  let y1i = min(u32(max(y0f + 1.0, 0.0)), low_h - 1u);
  let c00 = low_radiance[y0i * low_w + x0i].xyz;
  let c10 = low_radiance[y0i * low_w + x1i].xyz;
  let c01 = low_radiance[y1i * low_w + x0i].xyz;
  let c11 = low_radiance[y1i * low_w + x1i].xyz;

  let alb = hi_albedo[pixel].xyz;
  let nrm = hi_normal[pixel].xyz;
  let depth = hi_depth[pixel].x;
  let mot = hi_motion[pixel].xy;
  let divisor = demod_divisor(alb);

  // Features (23), fp16-stored to match MODE A (inputs rounded to f16).
  var act: array<f16, 64>;
  let dl00 = log_demod(c00, divisor);
  let dl10 = log_demod(c10, divisor);
  let dl01 = log_demod(c01, divisor);
  let dl11 = log_demod(c11, divisor);
  act[0] = f16(dl00.x); act[1] = f16(dl00.y); act[2] = f16(dl00.z);
  act[3] = f16(dl10.x); act[4] = f16(dl10.y); act[5] = f16(dl10.z);
  act[6] = f16(dl01.x); act[7] = f16(dl01.y); act[8] = f16(dl01.z);
  act[9] = f16(dl11.x); act[10] = f16(dl11.y); act[11] = f16(dl11.z);
  act[12] = f16(dx);
  act[13] = f16(dy);
  act[14] = f16(alb.x); act[15] = f16(alb.y); act[16] = f16(alb.z);
  act[17] = f16(nrm.x); act[18] = f16(nrm.y); act[19] = f16(nrm.z);
  act[20] = f16(log(max(depth, 0.0) + 1.0));
  act[21] = f16(mot.x);
  act[22] = f16(mot.y);

  // Feed-forward, f16 storage + f32 ACCUMULATE (MODE A), fixed order.
  let lc = u.info.x;
  for (var li = 0u; li < lc; li = li + 1u) {
    let layer = u.layers[li];
    let in_dim = layer.x;
    let out_dim = layer.y;
    let w_off = layer.z;
    let b_off = layer.w;
    let is_last = (li + 1u == lc);
    var next: array<f16, 64>;
    for (var o = 0u; o < out_dim; o = o + 1u) {
      var sum = weight_at(b_off + o);
      let row = w_off + o * in_dim;
      for (var i = 0u; i < in_dim; i = i + 1u) {
        sum = sum + weight_at(row + i) * f32(act[i]);
      }
      if (is_last) {
        next[o] = f16(sum);
      } else {
        next[o] = f16(max(sum, 0.0));
      }
    }
    for (var k = 0u; k < out_dim; k = k + 1u) {
      act[k] = next[k];
    }
  }

  let out_dl = vec3<f32>(f32(act[0]), f32(act[1]), f32(act[2]));
  out_img[pixel] = vec4<f32>(undo_log_demod(out_dl, divisor), 0.0);
}
