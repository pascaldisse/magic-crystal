//! Transport reference — ray-march transmittance (Beer-Lambert) and a
//! single-scatter estimator against one light.
//!
//! This is the CPU ground truth the GPU path tracer (Rite VI) must converge
//! to: transport is the law, the intersector/march is the implementation.
//! Everything is deterministic — a fixed march over a deterministic field
//! yields byte-identical results (ENTROPY law).

use crate::medium::HomogeneousMedium;
use crate::sources::Density;
use crate::vec::Vec3;

/// A light illuminating the medium.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Light {
    /// A light infinitely far away. `to_light` is a unit direction *toward*
    /// the light; `radiance` is the (distance-independent) incident radiance.
    Directional {
        /// Unit direction from any scene point toward the light.
        to_light: Vec3,
        /// Incident radiance.
        radiance: f64,
    },
    /// A point light. Incident radiance falls off as `intensity / dist²`.
    Point {
        /// World-space position.
        position: Vec3,
        /// Radiant intensity.
        intensity: f64,
    },
}

/// Ray-march the **transmittance** `T = exp(−∫ sigma_t · density ds)` along
/// `origin + dir · t` for `t ∈ [t0, t1]` (Beer-Lambert). `dir` must be unit
/// length; `steps ≥ 1`. Midpoint quadrature — exact for constant density.
pub fn transmittance<D: Density>(
    medium: &HomogeneousMedium,
    field: &D,
    origin: Vec3,
    dir: Vec3,
    t0: f64,
    t1: f64,
    steps: usize,
) -> f64 {
    (-optical_depth(medium, field, origin, dir, t0, t1, steps)).exp()
}

/// The optical depth `τ = ∫ sigma_t · density ds` along the ray. `T = exp(−τ)`.
pub fn optical_depth<D: Density>(
    medium: &HomogeneousMedium,
    field: &D,
    origin: Vec3,
    dir: Vec3,
    t0: f64,
    t1: f64,
    steps: usize,
) -> f64 {
    let steps = steps.max(1);
    let ds = (t1 - t0) / steps as f64;
    let sigma_t = medium.sigma_t();
    let mut tau = 0.0;
    for s in 0..steps {
        let t = t0 + (s as f64 + 0.5) * ds;
        let p = origin + dir * t;
        tau += sigma_t * field.density(p) * ds;
    }
    tau
}

/// Direction toward the light, incident radiance, and distance-to-source at a
/// world point.
fn sample_light(light: &Light, p: Vec3, shadow_dist: f64) -> (Vec3, f64, f64) {
    match *light {
        Light::Directional { to_light, radiance } => (to_light.normalize(), radiance, shadow_dist),
        Light::Point {
            position,
            intensity,
        } => {
            let diff = position - p;
            let dist = diff.length();
            let dir = if dist > 0.0 { diff / dist } else { Vec3::ZERO };
            let li = if dist > 0.0 {
                intensity / (dist * dist)
            } else {
                0.0
            };
            (dir, li, dist)
        }
    }
}

/// Shared camera-ray march. `phase_on` selects the single-scatter estimator
/// (phase-weighted radiance toward the camera) vs the in-scattered energy
/// accounting (phase integrated to 1 → replaced by unity).
#[allow(clippy::too_many_arguments)]
fn march_scatter<D: Density>(
    medium: &HomogeneousMedium,
    field: &D,
    cam_origin: Vec3,
    cam_dir: Vec3,
    t0: f64,
    t1: f64,
    steps: usize,
    light: &Light,
    shadow_dist: f64,
    shadow_steps: usize,
    phase_on: bool,
) -> f64 {
    let steps = steps.max(1);
    let ds = (t1 - t0) / steps as f64;
    let sigma_t = medium.sigma_t();
    let sigma_s = medium.sigma_s;
    let mut tau_before = 0.0; // camera-ray optical depth up to segment start
    let mut acc = 0.0;
    for s in 0..steps {
        let t = t0 + (s as f64 + 0.5) * ds;
        let p = cam_origin + cam_dir * t;
        let dens = field.density(p);
        let seg_tau = sigma_t * dens * ds;
        // Transmittance from the camera to this segment midpoint.
        let tc = (-(tau_before + 0.5 * seg_tau)).exp();
        tau_before += seg_tau;
        if dens > 0.0 {
            let (w_light, li, dist) = sample_light(light, p, shadow_dist);
            let tl = transmittance(medium, field, p, w_light, 0.0, dist, shadow_steps);
            let phase = if phase_on {
                // Incoming photon travels along −w_light; outgoing (toward the
                // camera) travels along −cam_dir. cosθ = (−w)·(−cam) = w·cam.
                medium.phase(w_light.dot(cam_dir))
            } else {
                1.0
            };
            acc += tc * sigma_s * dens * phase * tl * li * ds;
        }
    }
    acc
}

/// **Single-scatter estimator**: radiance scattered toward the camera along
/// `cam_origin + cam_dir · t` for `t ∈ [t0, t1]`, from one bounce off `light`.
///
/// `shadow_dist` bounds the occlusion march toward a directional light (for a
/// point light the true source distance is used). Deterministic.
#[allow(clippy::too_many_arguments)]
pub fn single_scatter<D: Density>(
    medium: &HomogeneousMedium,
    field: &D,
    cam_origin: Vec3,
    cam_dir: Vec3,
    t0: f64,
    t1: f64,
    steps: usize,
    light: &Light,
    shadow_dist: f64,
    shadow_steps: usize,
) -> f64 {
    march_scatter(
        medium,
        field,
        cam_origin,
        cam_dir,
        t0,
        t1,
        steps,
        light,
        shadow_dist,
        shadow_steps,
        true,
    )
}

/// **In-scattered energy** along the camera ray — the single-scatter integrand
/// with the phase function integrated over all outgoing directions (`∮ p dω =
/// 1`, so phase → 1). This is the total energy the beam sheds into first-order
/// scattering; the energy ordeal asserts it never exceeds the extinction
/// budget `1 − T`.
#[allow(clippy::too_many_arguments)]
pub fn in_scattered_energy<D: Density>(
    medium: &HomogeneousMedium,
    field: &D,
    cam_origin: Vec3,
    cam_dir: Vec3,
    t0: f64,
    t1: f64,
    steps: usize,
    light: &Light,
    shadow_dist: f64,
    shadow_steps: usize,
) -> f64 {
    march_scatter(
        medium,
        field,
        cam_origin,
        cam_dir,
        t0,
        t1,
        steps,
        light,
        shadow_dist,
        shadow_steps,
        false,
    )
}
