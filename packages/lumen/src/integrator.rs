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
    cosine_hemisphere, uniform, DIMS_PER_BOUNCE, DIM_HEMI_U1, DIM_HEMI_U2, DIM_RR,
};
use crate::scene::Scene;
use crate::vec::Vec3;

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
pub fn radiance(scene: &Scene, primary: Ray, pixel: u64, sample: u64, p: &Params) -> Vec3 {
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

        // Lambertian bounce. Pure emitters have albedo 0 → path dies here.
        let base = bounce as u64 * DIMS_PER_BOUNCE;
        let u1 = uniform(p.seed, pixel, sample, bounce as u64, base + DIM_HEMI_U1);
        let u2 = uniform(p.seed, pixel, sample, bounce as u64, base + DIM_HEMI_U2);
        let (dir, _pdf) = cosine_hemisphere(h.normal, u1, u2);
        // f_r·cosθ/pdf = albedo (cosine sampling); the multiply IS the albedo.
        throughput = throughput.hadamard(mat.albedo);

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
