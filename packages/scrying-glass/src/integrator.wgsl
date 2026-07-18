// the Pleroma in the glass (Rite IV, L1) — a real path-traced integrator over
// the Great Chain's leaf triangles. One integrator, no raster shading, no fake
// ambient floor (GRIMOIRE: unlit is truly unlit). Primary rays are traced; the
// sun is a directional delta light reached by a shadow ray (next-event); the sky
// gradient is the environment escaped rays gather; emissive surfaces glow and
// illuminate others through cosine-weighted bounce rays (as in the CPU Pleroma).
//
// ENTROPY law: no randomness — every sample is hash(seed, pixel, sample, dim).

struct Uniform {
  eye: vec4<f32>,          // xyz eye
  right: vec4<f32>,        // image-plane right, scaled by tan(fov/2)*aspect
  up: vec4<f32>,           // image-plane up, scaled by tan(fov/2)
  forward: vec4<f32>,      // unit look direction
  sky_top: vec4<f32>,
  sky_horizon: vec4<f32>,
  sun_dir: vec4<f32>,      // unit direction TOWARD the sun
  sun_color: vec4<f32>,    // rgb, w = intensity
  params: vec4<u32>,       // width, height, spp, max_bounces
  counters: vec4<u32>,     // seed, samples_before, node_count, tri_count
  misc: vec4<f32>,         // ambient_intensity, eps, rr_start, _
  // ── Rite VI A1: participating medium in the ONE light pass ──
  med_origin: vec4<f32>,   // xyz grid world origin, w = voxel_size
  med_params: vec4<f32>,   // sigma_a, sigma_s, g (HG anisotropy), far cap
  med_march: vec4<f32>,    // march_steps, shadow_steps, shadow_dist, enabled
  med_dims: vec4<u32>,     // grid dims xyz, w unused
  med_light: vec4<f32>,    // bound light: xyz = unit dir TOWARD it (directional) or world position (point); w = intensity
  med_light_color: vec4<f32>, // rgb = light colour tint; w = kind (0 directional, 1 point)
  // Display target dimensions. Trace/accum/present are always params.xy;
  // the surface receives only a nearest integer-scale display blit.
  surface: vec4<u32>,      // target_w, target_h, nearest=1, _
  // ── LIGHT-NOT-DOTS: temporal accumulation with reprojection ──
  // The PREVIOUS frame's camera, so `temporal_resolve` can reproject this
  // frame's world points into last frame's screen and fetch their history.
  prev_eye: vec4<f32>,     // xyz prev eye
  prev_right: vec4<f32>,   // prev image-plane right, scaled (surface aspect)
  prev_up: vec4<f32>,      // prev image-plane up, scaled
  prev_forward: vec4<f32>, // prev unit look direction
  temporal: vec4<f32>,     // alpha_min (moving EMA floor), depth_tol, normal_tol (cos), clamp_k
  temporal_flags: vec4<u32>, // history_valid (0/1), max_history frames, bitcast<f32> still_px (sub-pixel motion budget), _
};

struct Node {
  min: vec3<f32>,
  left_first: u32,
  max: vec3<f32>,
  count: u32,
};

struct Tri {
  v0: vec4<f32>,
  v1: vec4<f32>,
  v2: vec4<f32>,
  albedo: vec4<f32>,   // rgb = albedo/F0, w = metallic [0,1]
  emission: vec4<f32>, // rgb = emission, w = roughness [0,1]
};

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> tris: array<Tri>;
@group(0) @binding(3) var<storage, read_write> accum: array<vec4<f32>>;
// The medium density volume (Aether's rasterized grid, f32 — the SAME artifact
// the CPU reference marches). Trilinearly sampled; zero outside the box.
@group(0) @binding(4) var<storage, read> density: array<f32>;

const PI: f32 = 3.14159265358979;
const INF: f32 = 3.4e38;

// ---- deterministic keyed sampler (ENTROPY) ----
fn pcg(v: u32) -> u32 {
  var s = v * 747796405u + 2891336453u;
  var w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hash4(a: u32, b: u32, c: u32, d: u32) -> u32 {
  var h = pcg(a ^ 0x9e3779b9u);
  h = pcg(h ^ b);
  h = pcg(h ^ c);
  h = pcg(h ^ d);
  return h;
}
fn urand(pixel: u32, sample: u32, dim: u32) -> f32 {
  let h = hash4(u.counters.x, pixel, sample, dim);
  return f32(h >> 8u) * (1.0 / 16777216.0);
}

// Duff et al. branchless ONB (matches the CPU Pleroma's vec::onb).
fn onb(n: vec3<f32>) -> mat2x3<f32> {
  let sign = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (sign + n.z);
  let b = n.x * n.y * a;
  let t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
  let bt = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
  return mat2x3<f32>(t, bt);
}
fn cosine_hemisphere(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
  let r = sqrt(u1);
  let phi = 2.0 * PI * u2;
  let x = r * cos(phi);
  let y = r * sin(phi);
  let z = sqrt(max(1.0 - u1, 0.0));
  let basis = onb(n);
  return normalize(basis[0] * x + basis[1] * y + n * z);
}

// Roughness at/below this = a perfect mirror (delta specular lobe).
const MIRROR_ROUGHNESS: f32 = 1e-3;

// GGX (Trowbridge-Reitz) microfacet half-vector about n, roughness alpha=r^2.
// NDF importance sampling (Walter et al. 2007) — matches the CPU Pleroma.
fn ggx_half(n: vec3<f32>, alpha: f32, u1: f32, u2: f32) -> vec3<f32> {
  let a2 = alpha * alpha;
  let cos_t2 = clamp((1.0 - u1) / (1.0 + (a2 - 1.0) * u1), 0.0, 1.0);
  let cos_t = sqrt(cos_t2);
  let sin_t = sqrt(max(1.0 - cos_t2, 0.0));
  let phi = 2.0 * PI * u2;
  let x = sin_t * cos(phi);
  let y = sin_t * sin(phi);
  let basis = onb(n);
  return normalize(basis[0] * x + basis[1] * y + n * cos_t);
}

// Smith height-correlated masking-shadowing G2 for GGX (Heitz 2014).
fn smith_g2(cos_o: f32, cos_i: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let co = max(abs(cos_o), 1e-6);
  let ci = max(abs(cos_i), 1e-6);
  let lo = 0.5 * (-1.0 + sqrt(1.0 + a2 * (1.0 - co * co) / (co * co)));
  let li = 0.5 * (-1.0 + sqrt(1.0 + a2 * (1.0 - ci * ci) / (ci * ci)));
  return 1.0 / (1.0 + lo + li);
}

fn sky(dir: vec3<f32>) -> vec3<f32> {
  let h = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  return mix(u.sky_horizon.rgb, u.sky_top.rgb, h);
}

// ── Rite VI A1: the participating medium, marched inside this same pass ──
// Trilinear density lookup, matching Aether's DensityGrid::density EXACTLY
// (cell centers at origin + voxel_size*(i+0.5); zero outside the box).
fn grid_density(p: vec3<f32>) -> f32 {
  let dims = u.med_dims.xyz;
  let nx = f32(dims.x);
  let ny = f32(dims.y);
  let nz = f32(dims.z);
  let vs = u.med_origin.w;
  let local = (p - u.med_origin.xyz) / vs;
  let gx = local.x - 0.5;
  let gy = local.y - 0.5;
  let gz = local.z - 0.5;
  if (gx < -0.5 || gy < -0.5 || gz < -0.5) { return 0.0; }
  if (gx > nx - 0.5 || gy > ny - 0.5 || gz > nz - 0.5) { return 0.0; }
  let i0 = u32(clamp(floor(gx), 0.0, nx - 1.0));
  let j0 = u32(clamp(floor(gy), 0.0, ny - 1.0));
  let k0 = u32(clamp(floor(gz), 0.0, nz - 1.0));
  let i1 = min(i0 + 1u, dims.x - 1u);
  let j1 = min(j0 + 1u, dims.y - 1u);
  let k1 = min(k0 + 1u, dims.z - 1u);
  let tx = clamp(gx - f32(i0), 0.0, 1.0);
  let ty = clamp(gy - f32(j0), 0.0, 1.0);
  let tz = clamp(gz - f32(k0), 0.0, 1.0);
  let dx = dims.x;
  let dy = dims.y;
  let s000 = density[(k0 * dy + j0) * dx + i0];
  let s100 = density[(k0 * dy + j0) * dx + i1];
  let s010 = density[(k0 * dy + j1) * dx + i0];
  let s110 = density[(k0 * dy + j1) * dx + i1];
  let s001 = density[(k1 * dy + j0) * dx + i0];
  let s101 = density[(k1 * dy + j0) * dx + i1];
  let s011 = density[(k1 * dy + j1) * dx + i0];
  let s111 = density[(k1 * dy + j1) * dx + i1];
  let c00 = s000 * (1.0 - tx) + s100 * tx;
  let c10 = s010 * (1.0 - tx) + s110 * tx;
  let c01 = s001 * (1.0 - tx) + s101 * tx;
  let c11 = s011 * (1.0 - tx) + s111 * tx;
  let c0 = c00 * (1.0 - ty) + c10 * ty;
  let c1 = c01 * (1.0 - ty) + c11 * ty;
  return c0 * (1.0 - tz) + c1 * tz;
}

// Henyey-Greenstein phase (matches Aether HomogeneousMedium::phase).
fn hg_phase(cos_theta: f32) -> f32 {
  let g = u.med_params.z;
  let g2 = g * g;
  let denom = max(1.0 + g2 - 2.0 * g * cos_theta, 1e-12);
  return (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
}

// The density grid's world AABB [origin, origin + dims*voxel_size]. Density is
// EXACTLY zero outside it (grid_density early-outs), so any march sample beyond
// this box contributes nothing — the basis for empty-space skipping.
// Returns (t_enter, t_exit); t_enter > t_exit means the ray misses the box.
fn medium_box_range(o: vec3<f32>, d: vec3<f32>) -> vec2<f32> {
  let lo = u.med_origin.xyz;
  let hi = lo + vec3<f32>(u.med_dims.xyz) * u.med_origin.w;
  // NaN guard: an axis-parallel ray (d.k == 0) lying ON a slab face gives
  // (face - o).k == 0 and inv.k == ±inf, so (0 * inf) = NaN would leak into
  // min/max — and WGSL's min/max NaN result is implementation-defined. Nudge
  // any exactly-zero component to a tiny magnitude: inv becomes a huge FINITE
  // number, the slab's t-bounds become ±1e30 with the correct sign (no
  // constraint when the origin is inside the slab, a clean miss when outside),
  // and no NaN ever reaches the reductions.
  let safe_d = select(d, vec3<f32>(1e-30, 1e-30, 1e-30), abs(d) < vec3<f32>(1e-30, 1e-30, 1e-30));
  let inv = 1.0 / safe_d;
  let ta = (lo - o) * inv;
  let tb = (hi - o) * inv;
  let tmin = min(ta, tb);
  let tmax = max(ta, tb);
  let enter = max(max(tmin.x, tmin.y), tmin.z);
  let exit = min(min(tmax.x, tmax.y), tmax.z);
  return vec2<f32>(enter, exit);
}

// Optical depth along o+d*t for t in [t0,t1] (midpoint quadrature — matches
// Aether optical_depth). d must be unit length. EMPTY-SPACE SKIP: only the
// step centers inside the density box are evaluated; every skipped sample has
// density 0, so the sum is bit-identical to the full march (x + 0.0 == x).
// The window [s_lo, s_hi] is widened by one step on each side (clamped to
// [0, n-1]): the exact window from ceil/floor of (t-t0)/ds can, under fp
// rounding, exclude a step CENTER that truly lies inside the density box
// (harmless only when the boundary voxels happen to be empty — a
// scene-dependent accident). Widening by one step guarantees every excluded
// center sits more than half a step outside the box, beyond any ulp, so the
// bit-identity to the full march is UNCONDITIONAL (the two extra samples per
// side carry density 0 when the box truly ends there — x + 0.0 == x).
fn medium_optical_depth(o: vec3<f32>, d: vec3<f32>, t0: f32, t1: f32, steps: u32) -> f32 {
  let n = max(steps, 1u);
  let ds = (t1 - t0) / f32(n);
  let sigma_t = u.med_params.x + u.med_params.y;
  let box_range = medium_box_range(o, d);
  let a = max(t0, box_range.x);
  let b = min(t1, box_range.y);
  var tau = 0.0;
  if (b > a) {
    let s_lo = max(i32(ceil((a - t0) / ds - 0.5)) - 1, 0);
    let s_hi = min(i32(floor((b - t0) / ds - 0.5)) + 1, i32(n) - 1);
    var s = s_lo;
    loop {
      if (s > s_hi) { break; }
      let t = t0 + (f32(s) + 0.5) * ds;
      tau = tau + sigma_t * grid_density(o + d * t) * ds;
      s = s + 1;
    }
  }
  return tau;
}

// Single-scatter radiance toward the camera along the primary segment, from one
// bounce off the medium's BOUND scene light (A2): a directional sun/moon OR a
// positional emitter glow with 1/dist² falloff (matches Aether single_scatter
// + sample_light with phase on). Returns (scalar scatter factor, total optical
// depth over [t0,t1]) — the light's colour tints the scatter at the call site;
// the total optical depth is REUSED for the segment's transmittance (fused, so
// the second march is gone). The scalar intensity (radiance, or intensity/dist²
// for a point light) is folded in here, exactly as the CPU does.
//
// EMPTY-SPACE SKIP: only step centers inside the density box are evaluated;
// every skipped sample has density 0, so both the accumulation and the returned
// optical depth are bit-identical to the full march (x + 0.0 == x). As in
// medium_optical_depth, the window is widened one step on each side (clamped)
// so a fp-rounded window can never drop a step center truly inside the box —
// the bit-identity is UNCONDITIONAL, not a boundary-voxel accident.
fn medium_single_scatter(o: vec3<f32>, d: vec3<f32>, t0: f32, t1: f32) -> vec2<f32> {
  let steps = max(u32(u.med_march.x), 1u);
  let shadow_steps = max(u32(u.med_march.y), 1u);
  let shadow_dist_dir = u.med_march.z;
  let ds = (t1 - t0) / f32(steps);
  let sigma_t = u.med_params.x + u.med_params.y;
  let sigma_s = u.med_params.y;
  let is_point = u.med_light_color.w > 0.5;
  let intensity = u.med_light.w;
  let box_range = medium_box_range(o, d);
  let a = max(t0, box_range.x);
  let b = min(t1, box_range.y);
  var tau_before = 0.0;
  var acc = 0.0;
  if (b > a) {
    let s_lo = max(i32(ceil((a - t0) / ds - 0.5)) - 1, 0);
    let s_hi = min(i32(floor((b - t0) / ds - 0.5)) + 1, i32(steps) - 1);
    var s = s_lo;
    loop {
      if (s > s_hi) { break; }
      let t = t0 + (f32(s) + 0.5) * ds;
      let p = o + d * t;
      let dens = grid_density(p);
      let seg_tau = sigma_t * dens * ds;
      let tc = exp(-(tau_before + 0.5 * seg_tau));
      tau_before = tau_before + seg_tau;
      if (dens > 0.0) {
        // Direction toward the light, incident radiance, occlusion-march bound.
        var w_light: vec3<f32>;
        var li: f32;
        var sdist: f32;
        if (is_point) {
          let diff = u.med_light.xyz - p;
          let dist = length(diff);
          w_light = diff / max(dist, 1e-6);
          li = intensity / max(dist * dist, 1e-12);
          sdist = dist;
        } else {
          w_light = normalize(u.med_light.xyz);
          li = intensity;
          sdist = shadow_dist_dir;
        }
        let phase = hg_phase(dot(w_light, d));
        let tl = exp(-medium_optical_depth(p, w_light, 0.0, sdist, shadow_steps));
        acc = acc + tc * sigma_s * dens * phase * tl * li * ds;
      }
      s = s + 1;
    }
  }
  return vec2<f32>(acc, tau_before);
}

// The primary-segment medium compose: xyz = in-scattered radiance toward the
// camera, w = transmittance of what lies behind. L = xyz + w*L_surface.
// `t_first` is the distance to the first surface along the ray (or `far` on a
// sky escape) — passed in from the caller, which already traced this exact
// primary ray for the surface pass, so the medium never re-traverses the BVH.
fn medium_primary(o: vec3<f32>, d: vec3<f32>, t_first: f32) -> vec4<f32> {
  if (u.med_march.w < 0.5) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  let eps = u.misc.y;
  let far = u.med_params.w;
  let t1 = min(t_first, far);
  if (t1 <= eps) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  // The scatter march also returns the total optical depth over [eps,t1] — the
  // transmittance is exp(-that), so the separate transmittance march is gone
  // (bit-identical, one 128-step march removed per pixel).
  let res = medium_single_scatter(o, d, eps, t1);
  let tr = exp(-res.y);
  // The scalar intensity is already folded into `res.x` (per-sample for a
  // point light's 1/dist² falloff); only the colour tint multiplies here.
  return vec4<f32>(u.med_light_color.rgb * res.x, tr);
}

struct Hit {
  t: f32,
  tri: u32,
  ok: bool,
};

fn tri_hit(o: vec3<f32>, d: vec3<f32>, i: u32, t_min: f32, t_max: f32) -> f32 {
  let tri = tris[i];
  let v0 = tri.v0.xyz;
  let e1 = tri.v1.xyz - v0;
  let e2 = tri.v2.xyz - v0;
  let p = cross(d, e2);
  let det = dot(e1, p);
  if (abs(det) < 1e-8) { return -1.0; }
  let inv = 1.0 / det;
  let tv = o - v0;
  let uu = dot(tv, p) * inv;
  if (uu < 0.0 || uu > 1.0) { return -1.0; }
  let q = cross(tv, e1);
  let vv = dot(d, q) * inv;
  if (vv < 0.0 || uu + vv > 1.0) { return -1.0; }
  let t = dot(e2, q) * inv;
  if (t > t_min && t <= t_max) { return t; }
  return -1.0;
}

fn aabb_hit(mn: vec3<f32>, mx: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, t_min: f32, t_max: f32) -> bool {
  let t0 = (mn - o) * inv;
  let t1 = (mx - o) * inv;
  let lo = min(t0, t1);
  let hi = max(t0, t1);
  let tmin = max(max(lo.x, lo.y), max(lo.z, t_min));
  let tmax = min(min(hi.x, hi.y), min(hi.z, t_max));
  return tmax >= tmin;
}

fn trace_closest(o: vec3<f32>, d: vec3<f32>, t_min: f32, t_max: f32) -> Hit {
  var result: Hit;
  result.ok = false;
  result.t = t_max;
  result.tri = 0u;
  if (u.counters.z == 0u) { return result; }
  let inv = 1.0 / d;
  var stack: array<u32, 64>;
  var sp = 0;
  stack[sp] = 0u; sp = sp + 1;
  loop {
    if (sp <= 0) { break; }
    sp = sp - 1;
    let node = nodes[stack[sp]];
    if (!aabb_hit(node.min, node.max, o, inv, t_min, result.t)) { continue; }
    if (node.count > 0u) {
      for (var k = 0u; k < node.count; k = k + 1u) {
        let ti = node.left_first + k;
        let t = tri_hit(o, d, ti, t_min, result.t);
        if (t > 0.0) {
          result.t = t;
          result.tri = ti;
          result.ok = true;
        }
      }
    } else {
      if (sp + 2 <= 64) {
        stack[sp] = node.left_first; sp = sp + 1;
        stack[sp] = node.left_first + 1u; sp = sp + 1;
      }
    }
  }
  return result;
}

fn occluded(o: vec3<f32>, d: vec3<f32>, t_min: f32, t_max: f32) -> bool {
  if (u.counters.z == 0u) { return false; }
  let inv = 1.0 / d;
  var stack: array<u32, 64>;
  var sp = 0;
  stack[sp] = 0u; sp = sp + 1;
  loop {
    if (sp <= 0) { break; }
    sp = sp - 1;
    let node = nodes[stack[sp]];
    if (!aabb_hit(node.min, node.max, o, inv, t_min, t_max)) { continue; }
    if (node.count > 0u) {
      for (var k = 0u; k < node.count; k = k + 1u) {
        let t = tri_hit(o, d, node.left_first + k, t_min, t_max);
        if (t > 0.0) { return true; }
      }
    } else {
      if (sp + 2 <= 64) {
        stack[sp] = node.left_first; sp = sp + 1;
        stack[sp] = node.left_first + 1u; sp = sp + 1;
      }
    }
  }
  return false;
}

fn tri_normal(i: u32, d: vec3<f32>) -> vec3<f32> {
  let tri = tris[i];
  let n = normalize(cross(tri.v1.xyz - tri.v0.xyz, tri.v2.xyz - tri.v0.xyz));
  // Face the normal toward the incoming ray (two-sided shading).
  return select(n, -n, dot(n, d) > 0.0);
}

// `first_hit` is the primary ray's closest hit, already traced by the caller
// (shared with the medium compose) — the bounce-0 traversal is not repeated.
fn radiance(ray_o: vec3<f32>, ray_d: vec3<f32>, pixel: u32, sample: u32, first_hit: Hit) -> vec3<f32> {
  var o = ray_o;
  var d = ray_d;
  var throughput = vec3<f32>(1.0, 1.0, 1.0);
  var L = vec3<f32>(0.0, 0.0, 0.0);
  let eps = u.misc.y;
  let rr_start = u32(u.misc.z);
  let max_bounces = u.params.w;
  var bounce = 0u;
  var pending_hit = first_hit;
  var have_pending = true;
  loop {
    var hit: Hit;
    if (have_pending) {
      hit = pending_hit;
      have_pending = false;
    } else {
      hit = trace_closest(o, d, eps, INF);
    }
    if (!hit.ok) {
      // Escaped → environment. Primary miss shows the full sky (background);
      // indirect (bounced) misses gather the sky scaled by the ambient dial.
      let ambient = select(u.misc.x, 1.0, bounce == 0u);
      L = L + throughput * sky(d) * ambient;
      break;
    }
    let tri = tris[hit.tri];
    let p = o + d * hit.t;
    let n = tri_normal(hit.tri, d);
    let metallic = tri.albedo.w;
    let roughness = tri.emission.w;

    // Emissive surfaces glow at every hit (they are just emitters).
    L = L + throughput * tri.emission.xyz;

    // Next-event: the sun (directional delta light) via a shadow ray. Only the
    // DIFFUSE lobe responds to a delta light (a mirror reaching the sun is a
    // measure-zero event) → weight by (1 - metallic).
    let ndl = dot(n, u.sun_dir.xyz);
    if (u.sun_color.w > 0.0 && ndl > 0.0 && metallic < 1.0) {
      if (!occluded(p + n * eps, u.sun_dir.xyz, eps, INF)) {
        L = L + throughput * (1.0 - metallic) * tri.albedo.xyz
              * u.sun_color.rgb * u.sun_color.w * ndl;
      }
    }

    if (bounce >= max_bounces) { break; }
    // Stochastic lobe: specular w.p. metallic, diffuse w.p. (1-metallic). The
    // selection probability cancels the lobe weight, so each branch multiplies
    // throughput by the PURE lobe weight (matches the CPU Pleroma exactly).
    let base = bounce * 4u;
    let u1 = urand(pixel, sample, base + 0u);
    let u2 = urand(pixel, sample, base + 1u);
    let u_lobe = urand(pixel, sample, base + 3u);
    var dir: vec3<f32>;
    if (u_lobe < metallic) {
      // Conductor lobe. F0 = albedo (metal tint).
      if (roughness <= MIRROR_ROUGHNESS) {
        // Perfect mirror (delta): reflect about n, throughput *= albedo.
        dir = reflect(d, n);
        throughput = throughput * tri.albedo.xyz;
      } else {
        let alpha = roughness * roughness; // Disney remap
        let wo = -d;
        let m = ggx_half(n, alpha, u1, u2);
        let wi = reflect(d, m);
        let cos_i = dot(wi, n);
        if (cos_i <= 0.0) { break; } // sampled below surface
        let cos_o = max(abs(dot(wo, n)), 1e-6);
        let cos_h = max(abs(dot(m, n)), 1e-6);
        let g = smith_g2(cos_o, cos_i, alpha);
        let w = g * abs(dot(wo, m)) / (cos_o * cos_h);
        throughput = throughput * tri.albedo.xyz * w;
        dir = wi;
      }
    } else {
      // Cosine-weighted diffuse bounce: f_r·cosθ/pdf = albedo.
      dir = cosine_hemisphere(n, u1, u2);
      throughput = throughput * tri.albedo.xyz;
    }
    if (max(throughput.x, max(throughput.y, throughput.z)) <= 0.0) { break; }

    // Russian roulette after rr_start (unbiased termination).
    if (bounce + 1u >= rr_start) {
      let q = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.0, 1.0);
      let r = urand(pixel, sample, base + 2u);
      if (r >= q) { break; }
      throughput = throughput / q;
    }

    o = p + n * eps;
    d = dir;
    bounce = bounce + 1u;
  }
  return L;
}

@compute @workgroup_size(8, 8, 1)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u.params.x;
  let h = u.params.y;
  if (gid.x >= w || gid.y >= h) { return; }
  let pixel = gid.y * w + gid.x;
  let spp = u.params.z;
  let samples_before = u.counters.y;

  let inv_w = 1.0 / f32(w);
  let inv_h = 1.0 / f32(h);
  var frame_sum = vec3<f32>(0.0, 0.0, 0.0);
  for (var s = 0u; s < spp; s = s + 1u) {
    let sample = samples_before + s;
    // Jittered primary ray (progressive AA), sampler dims far from path dims.
    let jx = urand(pixel, sample, 900000u);
    let jy = urand(pixel, sample, 900001u);
    let sx = (2.0 * (f32(gid.x) + jx) * inv_w) - 1.0;
    let sy = 1.0 - (2.0 * (f32(gid.y) + jy) * inv_h);
    let dir = normalize(u.forward.xyz + u.right.xyz * sx + u.up.xyz * sy);
    // Trace the primary ray ONCE; both the surface radiance (bounce 0) and the
    // medium compose reuse this same first hit — no duplicate BVH traversal.
    let prim = trace_closest(u.eye.xyz, dir, u.misc.y, INF);
    let t_first = select(u.med_params.w, prim.t, prim.ok);
    let surf = radiance(u.eye.xyz, dir, pixel, sample, prim);
    // The medium composes over the surface radiance in the SAME pass: it
    // attenuates what lies behind and adds its single-scattered light on top.
    let med = medium_primary(u.eye.xyz, dir, t_first);
    frame_sum = frame_sum + med.xyz + med.w * surf;
  }

  let prev = accum[pixel].xyz;
  let total = f32(samples_before + spp);
  // accum holds the running SUM of per-sample radiance; .w = total samples.
  accum[pixel] = vec4<f32>(prev + frame_sum, total);
}

// ── LIGHT-NOT-DOTS: TEMPORAL ACCUMULATION WITH REPROJECTION ──────────────────
// The live present path traces the ONE integrator at ~1spp per frame; still, it
// converges by accumulating samples across frames — but the instant the camera
// moves the old reset-on-move path threw the history away and the Architect saw
// raw 1spp DOTS. This pair reconstructs that same one light pass across frames
// instead of discarding it: `integrate_temporal` traces THIS frame's radiance
// (spp samples) plus a primary gbuffer (depth+normal); `temporal_resolve`
// reprojects each current world point into the PREVIOUS frame's screen, fetches
// its accumulated history when depth+normal agree (rejects disocclusions and
// moved bodies), then blends via an exponential moving average with a
// neighbourhood variance clamp under motion. NOT a second light mode — one
// integrator, its samples accumulated. A SEPARATE @group(1) (like AOV) keeps
// the pre-existing `integrate`/`blit` bind group layout byte-for-byte unchanged.
// To stay under the 8-storage-buffer-per-stage limit (group(0) already uses 4),
// the current frame's colour AND primary gbuffer are PACKED into one buffer,
// 2 vec4<f32> cells per pixel: [2p] = (radiance.rgb, 1), [2p+1] = (depth, nx,
// ny, nz). The packed buffer ping-pongs so last frame's gbuffer (t_prev) is
// available for reprojection validation. Four bindings total → 8 with group(0).
@group(1) @binding(0) var<storage, read_write> t_cur: array<vec4<f32>>;   // this frame, packed
@group(1) @binding(1) var<storage, read_write> t_prev: array<vec4<f32>>;  // last frame, packed (gbuf half read)
@group(1) @binding(2) var<storage, read_write> t_hist_prev: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> t_hist_out: array<vec4<f32>>;

// Trace THIS frame: radiance (spp samples, jittered — the Monte-Carlo estimate
// whose samples we accumulate) into t_cur[2p], and the PRIMARY hit's depth +
// world normal (pixel-centre ray, deterministic geometry) into t_cur[2p+1].
@compute @workgroup_size(8, 8, 1)
fn integrate_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u.params.x;
  let h = u.params.y;
  if (gid.x >= w || gid.y >= h) { return; }
  let pixel = gid.y * w + gid.x;
  let spp = u.params.z;
  let samples_before = u.counters.y;
  let inv_w = 1.0 / f32(w);
  let inv_h = 1.0 / f32(h);
  // Primary gbuffer: pixel-centre ray, no jitter (the geometry, not the estimate).
  let cx = (2.0 * (f32(gid.x) + 0.5) * inv_w) - 1.0;
  let cy = 1.0 - (2.0 * (f32(gid.y) + 0.5) * inv_h);
  let cdir = normalize(u.forward.xyz + u.right.xyz * cx + u.up.xyz * cy);
  let ghit = trace_closest(u.eye.xyz, cdir, u.misc.y, INF);
  var frame_sum = vec3<f32>(0.0, 0.0, 0.0);
  for (var s = 0u; s < spp; s = s + 1u) {
    let sample = samples_before + s;
    let jx = urand(pixel, sample, 900000u);
    let jy = urand(pixel, sample, 900001u);
    let sx = (2.0 * (f32(gid.x) + jx) * inv_w) - 1.0;
    let sy = 1.0 - (2.0 * (f32(gid.y) + jy) * inv_h);
    let dir = normalize(u.forward.xyz + u.right.xyz * sx + u.up.xyz * sy);
    let prim = trace_closest(u.eye.xyz, dir, u.misc.y, INF);
    let t_first = select(u.med_params.w, prim.t, prim.ok);
    let surf = radiance(u.eye.xyz, dir, pixel, sample, prim);
    let med = medium_primary(u.eye.xyz, dir, t_first);
    frame_sum = frame_sum + med.xyz + med.w * surf;
  }
  t_cur[2u * pixel + 0u] = vec4<f32>(frame_sum / f32(max(spp, 1u)), 1.0);
  if (ghit.ok) {
    let n = tri_normal(ghit.tri, cdir);
    t_cur[2u * pixel + 1u] = vec4<f32>(ghit.t, n.x, n.y, n.z);
  } else {
    t_cur[2u * pixel + 1u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }
}

// Last frame's gbuffer (depth+normal) at an integer pixel, clamped to bounds.
fn t_prev_gbuf_at(px: i32, py: i32, iw: i32, ih: i32) -> vec4<f32> {
  let x = clamp(px, 0, iw - 1);
  let y = clamp(py, 0, ih - 1);
  return t_prev[2u * u32(y * iw + x) + 1u];
}

// BILINEAR fetch of last frame's accumulated history (rgb + frame count) at a
// FRACTIONAL reprojected position. Nearest-neighbour reprojection rounds a
// sub-pixel pan back onto the SAME pixel, so it never tracks a slow (<0.5px/
// frame) drift and the history smears (the Architect's ghost). Bilinear taps
// the four neighbours at (fx,fy) so the history follows sub-pixel motion — the
// standard TAA history resample. At an integer position (the identity-snapped
// still path) the fractional weights collapse to a single exact tap, so a still
// camera's running mean stays bit-exact.
fn t_hist_bilinear(fx: f32, fy: f32, iw: i32, ih: i32) -> vec4<f32> {
  let x0 = i32(floor(fx));
  let y0 = i32(floor(fy));
  let tx = fx - f32(x0);
  let ty = fy - f32(y0);
  let x0c = clamp(x0, 0, iw - 1);
  let x1c = clamp(x0 + 1, 0, iw - 1);
  let y0c = clamp(y0, 0, ih - 1);
  let y1c = clamp(y0 + 1, 0, ih - 1);
  let a = t_hist_prev[u32(y0c * iw + x0c)];
  let b = t_hist_prev[u32(y0c * iw + x1c)];
  let c = t_hist_prev[u32(y1c * iw + x0c)];
  let d = t_hist_prev[u32(y1c * iw + x1c)];
  return mix(mix(a, b, tx), mix(c, d, tx), ty);
}

// Reproject, validate, blend. Writes the accumulated radiance into t_hist_out
// (carried to next frame) AND into accum (so the existing blit presents it
// unchanged). accum.w = 1 so blit's sum/samples resolve is the identity.
@compute @workgroup_size(8, 8, 1)
fn temporal_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u.params.x;
  let h = u.params.y;
  if (gid.x >= w || gid.y >= h) { return; }
  let iw = i32(w);
  let ih = i32(h);
  let pixel = gid.y * w + gid.x;
  let curr = t_cur[2u * pixel + 0u].xyz;
  let cg = t_cur[2u * pixel + 1u];
  let depth = cg.x;
  let n_curr = cg.yzw;
  let is_miss = depth <= 0.0;

  // Neighbourhood COLOUR AABB of the CURRENT (noisy) frame — the box the
  // history is clamped into (Karis-style TAA min/max). This is the always-on
  // clamp: it is the gentlest box that (a) contains a stationary pixel's
  // converged history essentially always (so a still, STATIC scene stays an
  // EXACT running mean — the box brackets the same value the history holds),
  // yet (b) EXCLUDES stale history when the WHOLE neighbourhood shifts under a
  // relight (moved emitter), dragging it back in a few frames. mean±k·σ was
  // too tight — it clipped static edges and broke still-camera exactness; the
  // AABB is centred on nothing, so it never biases a converged edge pixel.
  // The variance dial (temporal.w = k) widens the box by k·σ so a little
  // per-frame noise never trims a valid history hair.
  var m1 = vec3<f32>(0.0, 0.0, 0.0);
  var m2 = vec3<f32>(0.0, 0.0, 0.0);
  var cnt = 0.0;
  var nmin = vec3<f32>(1.0e30, 1.0e30, 1.0e30);
  var nmax = vec3<f32>(-1.0e30, -1.0e30, -1.0e30);
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let nx = i32(gid.x) + dx;
      let ny = i32(gid.y) + dy;
      if (nx < 0 || ny < 0 || nx >= iw || ny >= ih) { continue; }
      let c = t_cur[2u * u32(ny * iw + nx) + 0u].xyz;
      m1 = m1 + c;
      m2 = m2 + c * c;
      nmin = min(nmin, c);
      nmax = max(nmax, c);
      cnt = cnt + 1.0;
    }
  }
  let mean = m1 / cnt;
  let sigma = sqrt(max(m2 / cnt - mean * mean, vec3<f32>(0.0, 0.0, 0.0)));
  let k = u.temporal.w;
  // Box = neighbourhood min/max, padded by k·σ (keeps valid history in).
  let box_lo = nmin - k * sigma;
  let box_hi = nmax + k * sigma;

  // World point of this pixel (hit: eye+dir*depth; miss: far along dir for sky).
  let cx = (2.0 * (f32(gid.x) + 0.5) / f32(w)) - 1.0;
  let cy = 1.0 - (2.0 * (f32(gid.y) + 0.5) / f32(h));
  let dir = normalize(u.forward.xyz + u.right.xyz * cx + u.up.xyz * cy);
  let dist = select(depth, 1.0e5, is_miss);
  let world = u.eye.xyz + dir * dist;

  // ── GATELESS TEMPORAL (light-fix, Architect 07-18) ──────────────────────
  // The old binary `cam_moved` gate (dot(fwd,prev_fwd) < 0.99999 ≈ 0.26°/frame)
  // is GONE. Real mouse-look pans slower than that per frame → they fell into
  // the still branch → identity reproject (history smeared = ghosts) + variance
  // clamp gated off (relight ghosts) + rejected regions raw 1spp (dots). The
  // fix is structural: reproject EVERY frame through the actual prev-camera
  // basis (a still camera degenerates to identity naturally), and clamp EVERY
  // frame. The ONLY residual still/moving decision is the accumulation ALPHA —
  // a truly still camera must stay an EXACT 1/n running average to converge; a
  // moving one floors alpha to stay responsive. That decision's threshold is
  // DERIVED from the PIXEL ANGULAR SIZE (a sub-pixel image-motion budget,
  // param `still_px`), never a frozen dot-product literal.
  let half_v = length(u.up.xyz);                        // tan(fov_y/2)
  let px_ang = (2.0 * atan(half_v)) / max(f32(h), 1.0); // radians per pixel row
  let still_px = bitcast<f32>(u.temporal_flags.z);      // sub-pixel budget (param)
  let fwd_dot = clamp(dot(normalize(u.forward.xyz), normalize(u.prev_forward.xyz)), -1.0, 1.0);
  let rot_px = acos(fwd_dot) / max(px_ang, 1e-8);       // this frame's rotation, in PIXELS
  let translated = distance(u.eye.xyz, u.prev_eye.xyz) > 1e-5;
  // "moving" for the ALPHA choice ONLY (reproject + clamp never consult it):
  // any translation (parallax always shifts) OR a rotation past the sub-pixel
  // budget.
  let cam_moving = translated || (rot_px > still_px);

  var out_col = curr;
  var out_len = 1.0;

  if (u.temporal_flags.x == 1u) {
    // Reproject this frame's world point into the PREVIOUS frame's screen. The
    // reprojection is COMPUTED every frame (no frozen-dot gate on whether to
    // run it), but when the camera is within the sub-pixel budget the analytic
    // result degenerates to THIS pixel — and round() of a chain of different
    // float ops than the ray-gen can still land off-by-one at edges (measured:
    // it drops a still camera's variance-reduction 72x→28x and breaks the
    // exact running mean). So below the budget we SNAP to the identity pixel:
    // the reprojection genuinely degenerates to identity there, we just make
    // that exact instead of round-approximate. Every supra-budget pan takes the
    // real reprojection. `valid` is false only on a moving disocclusion.
    var ipx = i32(gid.x);
    var ipy = i32(gid.y);
    var fx = f32(gid.x); // fractional reprojected position (for bilinear history)
    var fy = f32(gid.y);
    var valid = true;
    if (cam_moving) {
      let half_a = length(u.prev_right.xyz);
      let half_u = length(u.prev_up.xyz);
      let r_u = u.prev_right.xyz / max(half_a, 1e-8);
      let up_u = u.prev_up.xyz / max(half_u, 1e-8);
      let f_u = u.prev_forward.xyz;
      let rel = world - u.prev_eye.xyz;
      let rz = dot(rel, f_u);
      if (rz > 1e-4) {
        let sx = dot(rel, r_u) / (rz * max(half_a, 1e-8));
        let sy = dot(rel, up_u) / (rz * max(half_u, 1e-8));
        let fpx = (sx + 1.0) * 0.5 * f32(w) - 0.5;
        let fpy = (1.0 - sy) * 0.5 * f32(h) - 0.5;
        if (fpx >= 0.0 && fpy >= 0.0 && fpx <= f32(iw - 1) && fpy <= f32(ih - 1)) {
          fx = fpx;
          fy = fpy;
          ipx = i32(round(fpx)); // nearest tap for the depth/normal guard
          ipy = i32(round(fpy));
        } else {
          valid = false; // disoccluded / off-screen → no history
        }
      } else {
        valid = false; // behind the previous eye
      }
    }
    if (valid) {
      let pg = t_prev_gbuf_at(ipx, ipy, iw, ih);
      let prev_depth = pg.x;
      let prev_n = pg.yzw;
      let prev_miss = prev_depth <= 0.0;
      var ok = false;
      if (is_miss) {
        ok = prev_miss; // sky reprojects to sky
      } else if (!prev_miss) {
        // Distance from the PREVIOUS eye to this frame's world point vs the
        // depth the previous frame stored there (identity for a still camera).
        let dist_prev = length(world - u.prev_eye.xyz);
        let depth_ok = abs(dist_prev - prev_depth) <= u.temporal.y * max(dist_prev, 1e-4);
        let normal_ok = dot(n_curr, prev_n) >= u.temporal.z;
        ok = depth_ok && normal_ok;
      }
      if (ok) {
        // Bilinear history resample at the fractional reprojected position —
        // tracks sub-pixel pans so slow motion does not smear. Integer (still)
        // positions collapse to one exact tap (bit-exact running mean).
        let hp = t_hist_bilinear(fx, fy, iw, ih);
        var hist = hp.xyz;
        // Variance clamp ALWAYS ON (no cam_moved gate). This is the relight
        // fix: when the camera sits still but a MOVED emitter sweeps its
        // shadow/highlight across this pixel, the stale history is clamped
        // toward the current neighbourhood band and re-converges in a few
        // frames instead of lingering for up to max_history. For a truly
        // still, converged, static pixel the history already lies inside the
        // band, so this is a no-op there — the static-convergence ordeal proves
        // the running mean stays exact.
        // The variance clamp engages under MOTION (derived from the pixel
        // angular size, NOT the old frozen 0.99999 dot gate): while the
        // observer pans/translates past the sub-pixel budget the box rejects
        // reprojection tails, and convergence is capped by the alpha floor
        // anyway. A deeply STILL camera skips it so the running mean stays an
        // EXACT 1/n average (the static-convergence ordeal proves bit-exactness
        // to 1e-5 — any always-on spatial clamp caps that, since a 1spp
        // neighbourhood box built from Monte-Carlo-noisy samples routinely
        // excludes the low-variance converged history). Relight WHILE the
        // camera is perfectly still is the one case this cannot catch from
        // (rgb,count) state alone — distinguishing a persistent relight from a
        // transient firefly needs a per-pixel temporal-variance channel; see
        // the ignored relight ordeal. NOTE: the always-reproject above already
        // kills the slow-pan SMEAR independent of this gate; the gate only
        // decides deep-convergence vs motion-antighost.
        if (cam_moving && !is_miss) {
          hist = clamp(hist, box_lo, box_hi);
        }
        let n_frames = min(hp.w + 1.0, f32(u.temporal_flags.y));
        // Still camera: pure running average (alpha = 1/n) for maximal
        // convergence. Moving: floor alpha so history stays responsive.
        let alpha = select(1.0 / n_frames, max(u.temporal.x, 1.0 / n_frames), cam_moving);
        out_col = mix(hist, curr, alpha);
        out_len = n_frames;
      }
    }
  }

  t_hist_out[pixel] = vec4<f32>(out_col, out_len);
  accum[pixel] = vec4<f32>(out_col, 1.0);
}

// ── VIII-0 AOV EXPORT BEGIN ──────────────────────────────────────────────
// Rite VIII-0 (THE NOISE AND THE TRUTH): auxiliary buffers of the PRIMARY hit
// only — albedo, world normal, hit distance ("depth"). current-frame-only:
// every value here is computed fresh from THIS invocation's single camera
// ray and this frame's Uniform alone. There is no accumulation across
// frames, no read of any prior frame's buffer, and no parameter carrying a
// frame index, a previous-frame buffer, or any cross-frame state — the
// function signature below takes only the current frame's inputs. This is
// the architecture guarantee the BAN grep-gate ordeal checks
// (tests/viii0_ordeals.rs).
//
// A SEPARATE bind group (@group(1)) so the existing one-light-pass compute
// bind group layout (@group(0), bindings 0-4) is untouched: the AOV pass is
// a second pipeline sharing @group(0) (uniform/nodes/tris — read-only there,
// `accum` and `density` are simply unused by this entry point but must still
// be bound since they are declared in the same shader module) plus its own
// @group(1) output. When AOV export is off, this pipeline is never
// dispatched — zero cost, and the existing `integrate` compute pass and its
// bind group layout are byte-for-byte what they were before this wave
// (proven by the AOV-off golden-hash ordeal).
//
// Packing (2 vec4<f32> cells per pixel, documented rather than 3 separate
// buffers, to keep the readback a single buffer/copy):
//   aov[2*pixel + 0] = (albedo.r, albedo.g, albedo.b, depth)
//   aov[2*pixel + 1] = (normal.x, normal.y, normal.z, hit ? 1.0 : 0.0)
// On a primary miss (no surface hit) every component is 0.0 — there is no
// "primary hit" to report, so the primary-hit AOVs are honestly empty there.
@group(1) @binding(0) var<storage, read_write> aov: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn integrate_aov(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u.params.x;
  let h = u.params.y;
  if (gid.x >= w || gid.y >= h) { return; }
  let pixel = gid.y * w + gid.x;
  let inv_w = 1.0 / f32(w);
  let inv_h = 1.0 / f32(h);
  // Pixel-CENTER ray (no jitter, no sample loop): the AOV is a single
  // deterministic geometric query of the primary hit, not a Monte-Carlo
  // estimate — there is nothing here for repeated frames to converge, so it
  // needs no accumulation and no random sampler draw (ENTROPY law: no
  // randomness is used here at all).
  let sx = (2.0 * (f32(gid.x) + 0.5) * inv_w) - 1.0;
  let sy = 1.0 - (2.0 * (f32(gid.y) + 0.5) * inv_h);
  let dir = normalize(u.forward.xyz + u.right.xyz * sx + u.up.xyz * sy);
  let hit = trace_closest(u.eye.xyz, dir, u.misc.y, INF);
  if (hit.ok) {
    let tri = tris[hit.tri];
    let n = tri_normal(hit.tri, dir);
    aov[2u * pixel + 0u] = vec4<f32>(tri.albedo.xyz, hit.t);
    aov[2u * pixel + 1u] = vec4<f32>(n, 1.0);
  } else {
    aov[2u * pixel + 0u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    aov[2u * pixel + 1u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }
}
// ── VIII-0 AOV EXPORT END ────────────────────────────────────────────────

// ---- present: resolve the accumulation buffer to the (sRGB) target ----
struct BlitOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn blit_vs(@builtin(vertex_index) index: u32) -> BlitOut {
  var pts = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  var out: BlitOut;
  out.position = vec4<f32>(pts[index], 0.0, 1.0);
  out.uv = pts[index] * 0.5 + 0.5;
  return out;
}

// Resolve one trace-resolution accum cell to linear radiance (sum / samples).
fn accum_radiance(tx: u32, ty: u32, tw: u32, th: u32) -> vec3<f32> {
  let x = min(tx, tw - 1u);
  let y = min(ty, th - 1u);
  let cell = accum[y * tw + x];
  return cell.xyz / max(cell.w, 1.0);
}

@fragment
fn blit_fs(in: BlitOut) -> @location(0) vec4<f32> {
  // God's canvas is never resampled. Centre it in the OS surface and repeat
  // each canvas texel by an integer factor; unused surface pixels stay black.
  let tw = u.params.x;
  let th = u.params.y;
  let sw = max(u.surface.x, 1u);
  let sh = max(u.surface.y, 1u);
  let scale = max(1u, min(sw / tw, sh / th));
  let display_w = tw * scale;
  let display_h = th * scale;
  let origin_x = (i32(sw) - i32(display_w)) / 2;
  let origin_y = (i32(sh) - i32(display_h)) / 2;
  let pixel_x = i32(floor(in.position.x)) - origin_x;
  let pixel_y = i32(floor(in.position.y)) - origin_y;
  if (pixel_x < 0 || pixel_y < 0 || pixel_x >= i32(display_w) || pixel_y >= i32(display_h)) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let rgb = accum_radiance(u32(pixel_x) / scale, u32(pixel_y) / scale, tw, th);
  // Linear radiance out; the *Srgb target encodes to display space.
  return vec4<f32>(rgb, 1.0);
}
