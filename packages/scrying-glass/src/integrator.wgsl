// Lumen Naturae in the glass (Rite IV, L1) — a real path-traced integrator over
// the Great Chain's leaf triangles. One integrator, no raster shading, no fake
// ambient floor (GRIMOIRE: unlit is truly unlit). Primary rays are traced; the
// sun is a directional delta light reached by a shadow ray (next-event); the sky
// gradient is the environment escaped rays gather; emissive surfaces glow and
// illuminate others through cosine-weighted bounce rays (as in the CPU Lumen).
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
  albedo: vec4<f32>,
  emission: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> tris: array<Tri>;
@group(0) @binding(3) var<storage, read_write> accum: array<vec4<f32>>;

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

// Duff et al. branchless ONB (matches the CPU Lumen's vec::onb).
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

fn sky(dir: vec3<f32>) -> vec3<f32> {
  let h = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  return mix(u.sky_horizon.rgb, u.sky_top.rgb, h);
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

    // Emissive surfaces glow at every hit (they are just emitters).
    L = L + throughput * tri.emission.xyz;

    // Next-event: the sun (directional delta light) via a shadow ray.
    let ndl = dot(n, u.sun_dir.xyz);
    if (u.sun_color.w > 0.0 && ndl > 0.0) {
      if (!occluded(p + n * eps, u.sun_dir.xyz, eps, INF)) {
        L = L + throughput * tri.albedo.xyz * u.sun_color.rgb * u.sun_color.w * ndl;
      }
    }

    if (bounce >= max_bounces) { break; }
    // Cosine-weighted bounce: f_r·cosθ/pdf = albedo (the multiply IS the albedo).
    let base = bounce * 3u;
    let u1 = urand(pixel, sample, base + 0u);
    let u2 = urand(pixel, sample, base + 1u);
    let dir = cosine_hemisphere(n, u1, u2);
    throughput = throughput * tri.albedo.xyz;
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
    frame_sum = frame_sum + radiance(u.eye.xyz, dir, pixel, sample);
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
