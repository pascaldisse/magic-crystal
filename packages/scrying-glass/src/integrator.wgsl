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

// Optical depth along o+d*t for t in [t0,t1] (midpoint quadrature — matches
// Aether optical_depth). d must be unit length.
fn medium_optical_depth(o: vec3<f32>, d: vec3<f32>, t0: f32, t1: f32, steps: u32) -> f32 {
  let n = max(steps, 1u);
  let ds = (t1 - t0) / f32(n);
  let sigma_t = u.med_params.x + u.med_params.y;
  var tau = 0.0;
  for (var s = 0u; s < n; s = s + 1u) {
    let t = t0 + (f32(s) + 0.5) * ds;
    tau = tau + sigma_t * grid_density(o + d * t) * ds;
  }
  return tau;
}

// Single-scatter radiance toward the camera along the primary segment, from one
// bounce off the medium's BOUND scene light (A2): a directional sun/moon OR a
// positional emitter glow with 1/dist² falloff (matches Aether single_scatter
// + sample_light with phase on). Returns the SCALAR scatter factor (the light's
// colour tints it at the call site); the scalar intensity (radiance, or
// intensity/dist² for a point light) is folded in here, exactly as the CPU does.
fn medium_single_scatter(o: vec3<f32>, d: vec3<f32>, t0: f32, t1: f32) -> f32 {
  let steps = max(u32(u.med_march.x), 1u);
  let shadow_steps = max(u32(u.med_march.y), 1u);
  let shadow_dist_dir = u.med_march.z;
  let ds = (t1 - t0) / f32(steps);
  let sigma_t = u.med_params.x + u.med_params.y;
  let sigma_s = u.med_params.y;
  let is_point = u.med_light_color.w > 0.5;
  let intensity = u.med_light.w;
  var tau_before = 0.0;
  var acc = 0.0;
  for (var s = 0u; s < steps; s = s + 1u) {
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
  }
  return acc;
}

// The primary-segment medium compose: xyz = in-scattered radiance toward the
// camera, w = transmittance of what lies behind. L = xyz + w*L_surface.
fn medium_primary(o: vec3<f32>, d: vec3<f32>) -> vec4<f32> {
  if (u.med_march.w < 0.5) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  let eps = u.misc.y;
  let far = u.med_params.w;
  let hit = trace_closest(o, d, eps, INF);
  let t_first = select(far, hit.t, hit.ok);
  let t1 = min(t_first, far);
  if (t1 <= eps) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  let scatter = medium_single_scatter(o, d, eps, t1);
  let tr = exp(-medium_optical_depth(o, d, eps, t1, u32(u.med_march.x)));
  // The scalar intensity is already folded into `scatter` (per-sample for a
  // point light's 1/dist² falloff); only the colour tint multiplies here.
  return vec4<f32>(u.med_light_color.rgb * scatter, tr);
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

fn radiance(ray_o: vec3<f32>, ray_d: vec3<f32>, pixel: u32, sample: u32) -> vec3<f32> {
  var o = ray_o;
  var d = ray_d;
  var throughput = vec3<f32>(1.0, 1.0, 1.0);
  var L = vec3<f32>(0.0, 0.0, 0.0);
  let eps = u.misc.y;
  let rr_start = u32(u.misc.z);
  let max_bounces = u.params.w;
  var bounce = 0u;
  loop {
    let hit = trace_closest(o, d, eps, INF);
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
    let surf = radiance(u.eye.xyz, dir, pixel, sample);
    // The medium composes over the surface radiance in the SAME pass: it
    // attenuates what lies behind and adds its single-scattered light on top.
    let med = medium_primary(u.eye.xyz, dir);
    frame_sum = frame_sum + med.xyz + med.w * surf;
  }

  let prev = accum[pixel].xyz;
  let total = f32(samples_before + spp);
  // accum holds the running SUM of per-sample radiance; .w = total samples.
  accum[pixel] = vec4<f32>(prev + frame_sum, total);
}

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

@fragment
fn blit_fs(in: BlitOut) -> @location(0) vec4<f32> {
  let w = u.params.x;
  let h = u.params.y;
  let x = min(u32(in.position.x), w - 1u);
  let y = min(u32(in.position.y), h - 1u);
  let cell = accum[y * w + x];
  let samples = max(cell.w, 1.0);
  // Linear radiance out; the *Srgb target encodes to display space.
  return vec4<f32>(cell.xyz / samples, 1.0);
}
