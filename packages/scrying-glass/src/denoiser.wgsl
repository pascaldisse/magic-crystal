// RITE VIII-2 — THE DREAM AT SPEED: the GPU port of the VIII-1 CPU reference
// denoiser (src/denoiser.rs). A per-pixel MLP evaluated as a plain compute
// pass (wgpu has no tensor surface — Breda/NRC precedent, RENDER.md §8). Same
// hash-pinned weights, LOADED (uploaded) not re-derived; same feature
// engineering and same fixed accumulation order as the CPU forward pass, so
// the two agree to an fp32-rounding tolerance (parity ordeal, viii2_ordeals).
//
// THE BAN (this file carries the marker below so the VIII-0/1 grep-gate and
// the VIII-2 shader scan pick it up): every value here is a function of THIS
// pixel's current-frame buffers alone — noisy radiance, albedo, normal, depth.
// No accumulation across passes, no read of any earlier output, no parameter
// carrying a pass index or an earlier buffer. One dispatch is the complete,
// deterministic answer for this frame.
//
// BAN-SCOPED

// Fixed ceilings — the shipped net (4 hidden layers, width 32; input 10,
// output 3) fits under both; the live geometry is passed as data via the
// uniform's layer_count/dims, never assumed. MAX_WIDTH bounds the per-thread
// activation scratch. 64 measured FASTER than a tight 32 at 900x600 on this
// M1 (27 ms vs 53 ms — naga/Metal lays out the wider fixed array in a form
// the register allocator handles better here); the scratch cost is a known
// optimization surface for a fp16/subgroup follow-up (see denoiser_gpu docs).
const MAX_LAYERS: u32 = 16u;
const MAX_WIDTH: u32 = 64u;

const ALBEDO_DEMOD_EPS: f32 = 1e-3;
const NO_HIT_ALBEDO_THRESHOLD_SQ: f32 = 1e-8;

struct DenoiseU {
  // width, height, layer_count, _pad
  dims: vec4<u32>,
  // per layer: x=in_dim, y=out_dim, z=weights offset, w=bias offset
  // (both offsets index the flat `weights` array; see Mlp::flat_weights).
  layers: array<vec4<u32>, 16>,
};

@group(0) @binding(0) var<uniform> u: DenoiseU;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
// Current-frame input buffers, one vec4 per pixel:
//   noisy[i]  = (radiance.rgb, depth)
//   albedo[i] = (albedo.rgb, _)
//   normal[i] = (normal.xyz, _)
@group(0) @binding(2) var<storage, read> noisy: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> albedo: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> normal: array<vec4<f32>>;
// out_img[i] = (denoised.rgb, 0)
@group(0) @binding(5) var<storage, read_write> out_img: array<vec4<f32>>;

// The demodulation divisor, matching src/denoiser.rs::demod_divisor EXACTLY:
// a real hit (nonzero albedo) demodulates by albedo+eps; a primary miss (sky,
// albedo == 0) passes radiance through undivided (divisor 1) so the sky is not
// amplified ~1000x.
fn demod_divisor(a: vec3<f32>) -> vec3<f32> {
  if (dot(a, a) > NO_HIT_ALBEDO_THRESHOLD_SQ) {
    return a + vec3<f32>(ALBEDO_DEMOD_EPS, ALBEDO_DEMOD_EPS, ALBEDO_DEMOD_EPS);
  }
  return vec3<f32>(1.0, 1.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn denoise(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u.dims.x;
  let h = u.dims.y;
  if (gid.x >= w || gid.y >= h) { return; }
  let pixel = gid.y * w + gid.x;

  let rad = noisy[pixel].xyz;
  let depth = noisy[pixel].w;
  let alb = albedo[pixel].xyz;
  let nrm = normal[pixel].xyz;

  // pixel_features (src/denoiser.rs) — demodulated log radiance (3), albedo
  // (3), normal (3), log depth (1). ln(1 + max(x,0)) throughout.
  let demod = rad / demod_divisor(alb);
  var act: array<f32, 64>;
  act[0] = log(max(demod.x, 0.0) + 1.0);
  act[1] = log(max(demod.y, 0.0) + 1.0);
  act[2] = log(max(demod.z, 0.0) + 1.0);
  act[3] = alb.x;
  act[4] = alb.y;
  act[5] = alb.z;
  act[6] = nrm.x;
  act[7] = nrm.y;
  act[8] = nrm.z;
  act[9] = log(max(depth, 0.0) + 1.0);

  // Feed-forward, ReLU hidden, linear output. FIXED accumulation order,
  // identical to Mlp::forward: sum = bias; for i in 0..in_dim { sum +=
  // w[o*in_dim + i] * act[i]; }.
  let lc = u.dims.z;
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

  // undo_output_transform: expm1 then re-modulate by the SAME frame's albedo.
  let expm1 = vec3<f32>(exp(act[0]) - 1.0, exp(act[1]) - 1.0, exp(act[2]) - 1.0);
  let clamped = vec3<f32>(max(expm1.x, 0.0), max(expm1.y, 0.0), max(expm1.z, 0.0));
  let denoised = clamped * demod_divisor(alb);
  out_img[pixel] = vec4<f32>(denoised, 0.0);
}
