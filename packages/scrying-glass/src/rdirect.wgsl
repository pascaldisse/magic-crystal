// R-DIRECT — THE NET IS THE RENDERER, ON THE GPU (f32 storage, parity anchor).
// The fused single-dispatch compute port of the CPU reference
// (src/rdirect.rs::direct_render_image). One MLP forward per TARGET pixel:
// per-pixel feature gather inline (2×2 low-res demod-log radiance taps +
// subpixel + hi-res albedo/normal/log-depth/motion = 23 features) → 5×64 ReLU
// → 3, ABSOLUTE demod-log output (no bilinear base — this net EMITS the image).
// Same fixed accumulation order as Mlp::forward, so it agrees with the CPU
// reference to an fp32-rounding tolerance (parity ordeal).
//
// THE BAN: every value is a function of THIS frame's buffers alone — the
// low-res radiance of THIS frame and THIS frame's target-res
// albedo/normal/depth/motion. No cross-frame anything. BAN-SCOPED

const MAX_LAYERS: u32 = 16u;
const MAX_WIDTH: u32 = 64u;

const RADIANCE_TAPS: u32 = 4u;
const ALBEDO_DEMOD_EPS: f32 = 1e-3;
const NO_HIT_ALBEDO_THRESHOLD_SQ: f32 = 1e-8;

struct RdirectU {
  // low_w, low_h, target_w, target_h
  dims: vec4<u32>,
  // layer_count, weight_count, _pad, _pad
  info: vec4<u32>,
  // per layer: x=in_dim, y=out_dim, z=weights offset, w=bias offset
  layers: array<vec4<u32>, 16>,
};

@group(0) @binding(0) var<uniform> u: RdirectU;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> low_radiance: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> hi_albedo: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> hi_normal: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> hi_depth: array<vec4<f32>>;
// hi_motion[i] = (motion.xy, _, _), one per TARGET pixel.
@group(0) @binding(6) var<storage, read> hi_motion: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out_img: array<vec4<f32>>;

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

@compute @workgroup_size(8, 8, 1)
fn render(@builtin(global_invocation_id) gid: vec3<u32>) {
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

  // pixel_features (src/rdirect.rs): 4 taps log-demod (12) + dx,dy (2) +
  // albedo (3) + normal (3) + log depth (1) + motion (2) = 23, in THIS order.
  var act: array<f32, 64>;
  let dl00 = log_demod(c00, divisor);
  let dl10 = log_demod(c10, divisor);
  let dl01 = log_demod(c01, divisor);
  let dl11 = log_demod(c11, divisor);
  act[0] = dl00.x; act[1] = dl00.y; act[2] = dl00.z;
  act[3] = dl10.x; act[4] = dl10.y; act[5] = dl10.z;
  act[6] = dl01.x; act[7] = dl01.y; act[8] = dl01.z;
  act[9] = dl11.x; act[10] = dl11.y; act[11] = dl11.z;
  act[12] = dx;
  act[13] = dy;
  act[14] = alb.x; act[15] = alb.y; act[16] = alb.z;
  act[17] = nrm.x; act[18] = nrm.y; act[19] = nrm.z;
  act[20] = log(max(depth, 0.0) + 1.0);
  act[21] = mot.x;
  act[22] = mot.y;

  // Feed-forward, ReLU hidden, LINEAR output. FIXED accumulation order,
  // identical to Mlp::forward: sum = bias; for i { sum += w[o*in+i]*act[i]; }.
  let lc = u.info.x;
  for (var li = 0u; li < lc; li = li + 1u) {
    let layer = u.layers[li];
    let in_dim = layer.x;
    let out_dim = layer.y;
    let w_off = layer.z;
    let b_off = layer.w;
    let is_last = (li + 1u == lc);
    var next: array<f32, 64>;
    for (var o = 0u; o < out_dim; o = o + 1u) {
      var sum = weights[b_off + o];
      let row = w_off + o * in_dim;
      for (var i = 0u; i < in_dim; i = i + 1u) {
        sum = sum + weights[row + i] * act[i];
      }
      if (is_last) {
        next[o] = sum;
      } else {
        next[o] = max(sum, 0.0);
      }
    }
    for (var k = 0u; k < out_dim; k = k + 1u) {
      act[k] = next[k];
    }
  }

  // ABSOLUTE output — the net emits the image directly (no bilinear base).
  let out_dl = vec3<f32>(act[0], act[1], act[2]);
  out_img[pixel] = vec4<f32>(undo_log_demod(out_dl, divisor), 0.0);
}
