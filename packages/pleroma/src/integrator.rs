//! Integrator — unidirectional path tracing. The one true light (RENDER.md:
//! one integrator, no fallbacks). Lights, sky, emissive materials are all
//! ONE thing: emitters. Reflections/GI/shadows are not features, just paths.
//!
//! Estimator (per path):
//!   L = Σ_k  β_k · Le(x_k)          β_0 = 1
//!   β_{k+1} = β_k · f_r·cosθ / pdf
//! For a lambertian surface with cosine-weighted sampling:
//!   f_r = albedo/π,  pdf = cosθ/π  ⇒  f_r·cosθ/pdf = albedo
//! so each bounce multiplies throughput by exactly the albedo — this is the
//! energy law the ORDEALS pin down (albedo 1 loses nothing, 0.5 halves).
//!
//! Russian roulette (after `rr_start` bounces): survive with probability
//! p = clamp(max_channel(β)); on survival β /= p (unbiased). Termination is
//! the ONLY thing RR changes — never the expected value.
//!
//! NO ambient, NO background light: a ray that escapes returns black
//! (GRIMOIRE: unlit is truly unlit).

use crate::geometry::Ray;
use crate::sampler::{
    cosine_hemisphere, ggx_half, reflect, smith_g2, uniform, DIMS_PER_BOUNCE, DIM_HEMI_U1,
    DIM_HEMI_U2, DIM_LOBE, DIM_RR,
};
use crate::scene::Scene;
use crate::vec::Vec3;

/// Roughness at or below this is treated as a PERFECT MIRROR (delta specular
/// lobe) — below it the GGX lobe would be a numerically degenerate spike.
pub const MIRROR_ROUGHNESS: f64 = 1e-3;

/// Integrator parameters. All dials, no hardcoding — these are the
/// documented DEFAULTS (used by `Params::default`).
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Hard path-length cap (safety net above russian roulette).
    pub max_bounces: u32,
    /// Samples per pixel.
    pub spp: u32,
    /// Bounce index at which russian roulette begins (earlier bounces always
    /// survive, so near-field GI keeps its energy).
    pub rr_start: u32,
    /// Ray self-intersection epsilon (offset along the normal).
    pub eps: f64,
    /// Master seed — the entropy origin; same seed ⇒ same buffer.
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            max_bounces: 32,
            spp: 64,
            rr_start: 4,
            eps: 1e-4,
            seed: 0x5eed,
        }
    }
}

/// Trace ONE path for pixel `pixel`, sample `sample`, returning its radiance
/// estimate. Public so ordeals can drive it directly (measure a point's
/// reflected radiance without a camera).
///
/// When the scene carries a participating medium, the primary camera segment is
/// marched inside THIS pass (no separate volumetric pass): the surface/emitter
/// radiance behind the medium is attenuated by its transmittance and the
/// medium's single-scattered light is added on top (Rite VI A1).
pub fn radiance(scene: &Scene, primary: Ray, pixel: u64, sample: u64, p: &Params) -> Vec3 {
    let surface = surface_radiance(scene, primary, pixel, sample, p);
    match &scene.medium {
        None => surface,
        Some(medium) => {
            // Distance to the first surface bounds the primary march (else the
            // medium's own far cap does — godrays in empty sky).
            let t_first = scene
                .hit(&primary, p.eps, f64::INFINITY)
                .map(|(h, _)| h.t)
                .unwrap_or(medium.far);
            let (inscatter, tr) = medium.primary(primary.origin, primary.dir, p.eps, t_first);
            inscatter + tr * surface
        }
    }
}

/// The surface-only radiance estimate (the L0/L1 path) — the medium composes
/// over this in [`radiance`].
fn surface_radiance(scene: &Scene, primary: Ray, pixel: u64, sample: u64, p: &Params) -> Vec3 {
    let mut ray = primary;
    let mut throughput = Vec3::ONE;
    let mut acc = Vec3::ZERO;

    let mut bounce = 0u32;
    // A ray that finds no surface escapes → black (no ambient).
    while let Some((h, mat)) = scene.hit(&ray, p.eps, f64::INFINITY) {
        // Emission at every hit (emitters are just surfaces).
        acc = acc + throughput.hadamard(mat.emission);

        if bounce >= p.max_bounces {
            break;
        }

        // BSDF bounce. Pure emitters have albedo 0 → path dies here.
        let base = bounce as u64 * DIMS_PER_BOUNCE;
        let u1 = uniform(p.seed, pixel, sample, bounce as u64, base + DIM_HEMI_U1);
        let u2 = uniform(p.seed, pixel, sample, bounce as u64, base + DIM_HEMI_U2);
        let u_lobe = uniform(p.seed, pixel, sample, bounce as u64, base + DIM_LOBE);

        // Stochastic lobe selection. Selecting the specular lobe with
        // probability `metallic` and the diffuse lobe with probability
        // (1-metallic) makes the metallic weight cancel the selection
        // probability exactly, so each branch's throughput multiply is the
        // PURE lobe weight (no variance-blowing 1/p factor).
        let dir;
        if u_lobe < mat.metallic {
            // ── Specular (conductor) lobe ──────────────────────────────────
            let wo = -ray.dir; // toward the viewer/incoming path
            if mat.roughness <= MIRROR_ROUGHNESS {
                // Perfect mirror (delta BRDF): reflect about the normal,
                // throughput ×= albedo (exact, energy-conserving).
                dir = reflect(ray.dir, h.normal);
                throughput = throughput.hadamard(mat.albedo);
            } else {
                let alpha = mat.roughness * mat.roughness; // Disney remap
                let m = ggx_half(h.normal, alpha, u1, u2);
                let wi = reflect(ray.dir, m);
                let cos_i = wi.dot(h.normal);
                if cos_i <= 0.0 {
                    break; // sampled below the surface → path dies
                }
                let cos_o = wo.dot(h.normal).abs().max(1e-6);
                let cos_h = m.dot(h.normal).abs().max(1e-6);
                // BSDF-sampling weight = F · G2 · |ωo·m| / (|ωo·n|·|m·n|),
                // F = albedo (metal tint). Energy-conserving (Walter 2007).
                let g = smith_g2(cos_o, cos_i, alpha);
                let w = (g * wo.dot(m).abs()) / (cos_o * cos_h);
                throughput = throughput.hadamard(mat.albedo) * w;
                dir = wi;
            }
        } else {
            // ── Diffuse (lambertian) lobe ──────────────────────────────────
            let (d, _pdf) = cosine_hemisphere(h.normal, u1, u2);
            // f_r·cosθ/pdf = albedo (cosine sampling); the multiply IS albedo.
            throughput = throughput.hadamard(mat.albedo);
            dir = d;
        }

        if throughput.max_component() <= 0.0 {
            break;
        }

        // Russian roulette after rr_start.
        if bounce + 1 >= p.rr_start {
            let q = throughput.max_component().clamp(0.0, 1.0);
            let r = uniform(p.seed, pixel, sample, bounce as u64, base + DIM_RR);
            if r >= q {
                break;
            }
            throughput = throughput / q;
        }

        ray = Ray::new(h.point + h.normal * p.eps, dir);
        bounce += 1;
    }
    acc
}

/// Average `spp` independent paths through `primary`. Convenience for the
/// point-radiance ordeals.
pub fn estimate(scene: &Scene, primary: Ray, pixel: u64, p: &Params) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for s in 0..p.spp {
        sum = sum + radiance(scene, primary, pixel, s as u64, p);
    }
    sum / p.spp as f64
}
