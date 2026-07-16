//! Material — lambertian diffuse + a metallic microfacet specular lobe +
//! emission. NO ambient term. Unlit is black (GRIMOIRE law: Darkness = truly
//! unlit; no fake ambient EVER). A surface contributes light ONLY through its
//! emitter or through paths that reach an emitter — never a constant floor.
//!
//! ## The surface model (L2 — documented, one integrator, no alternatives)
//! A surface is a stochastic blend of two lobes selected per bounce:
//!   f = (1 - metallic)·f_diffuse  +  metallic·f_specular
//! - `f_diffuse`  = lambertian `albedo/π` (the L0/L1 surface, unchanged).
//! - `f_specular` = a GGX microfacet conductor whose Fresnel reflectance F0 is
//!   the material's `albedo` (a metal's colour IS its reflectance — no
//!   dielectric angular Fresnel in L2; metals only). At `roughness == 0` the
//!   specular lobe collapses to a PERFECT MIRROR (a delta BRDF: reflect about
//!   the normal, throughput ×= albedo — exact, energy-conserving).
//!
//! Defaults `metallic = 0`, `roughness = 1` reproduce the pure lambertian
//! surface exactly (the specular lobe is never selected), so every L0/L1
//! ordeal still closes byte-for-byte.
//!
//! ENERGY: the mirror lobe (`roughness 0`) multiplies throughput by exactly
//! `albedo` — albedo 1 loses nothing (the white furnace still closes), albedo
//! 0.5 halves per bounce. The rough GGX lobe uses the Smith masking-shadowing
//! weight (Walter et al. 2007), which conserves energy for single scattering
//! (never gains; loses a little to un-modelled multiple scattering — a
//! one-sided bound, tested in the energy ordeal).

use crate::vec::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// Diffuse reflectance (dielectric) OR specular reflectance F0 (metal), in
    /// `[0,1]` per channel. Lambertian BRDF = albedo/π; metal mirror tint = albedo.
    pub albedo: Vec3,
    /// Emitted radiance (linear). ZERO for a non-emitter. This is the ONLY
    /// source of light in the world — there is no ambient.
    pub emission: Vec3,
    /// Metallic `[0,1]`: 0 = pure lambertian diffuse, 1 = pure conductor
    /// specular. Selects the lobe probability per bounce.
    pub metallic: f64,
    /// Microfacet roughness `[0,1]`: 1 = lambertian-broad, 0 = perfect mirror.
    /// GGX α = roughness² (Disney remap).
    pub roughness: f64,
}

impl Material {
    /// A pure lambertian surface (metallic 0, roughness 1 — the L0/L1 default).
    pub fn lambertian(albedo: Vec3) -> Material {
        Material {
            albedo,
            emission: Vec3::ZERO,
            metallic: 0.0,
            roughness: 1.0,
        }
    }

    /// A pure emitter (color × intensity), no reflection.
    pub fn emissive(emission: Vec3) -> Material {
        Material {
            albedo: Vec3::ZERO,
            emission,
            metallic: 0.0,
            roughness: 1.0,
        }
    }

    /// A metal (conductor): `reflectance` is the specular tint (F0), `roughness`
    /// the microfacet spread. `roughness 0` ⇒ a perfect mirror.
    pub fn metal(reflectance: Vec3, roughness: f64) -> Material {
        Material {
            albedo: reflectance,
            emission: Vec3::ZERO,
            metallic: 1.0,
            roughness: roughness.clamp(0.0, 1.0),
        }
    }

    /// Full constructor (all four dials); clamps metallic/roughness to `[0,1]`.
    pub fn new(albedo: Vec3, emission: Vec3, metallic: f64, roughness: f64) -> Material {
        Material {
            albedo,
            emission,
            metallic: metallic.clamp(0.0, 1.0),
            roughness: roughness.clamp(0.0, 1.0),
        }
    }

    pub fn is_emitter(&self) -> bool {
        self.emission.max_component() > 0.0
    }
}
