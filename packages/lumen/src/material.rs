//! Material — lambertian albedo + emissive. NO ambient term. Unlit is
//! black (GRIMOIRE law: Darkness = truly unlit; no fake ambient EVER). A
//! surface contributes light ONLY through its emitter or through paths that
//! reach an emitter — never a constant floor.

use crate::vec::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// Diffuse reflectance in `[0,1]` per channel. BRDF = albedo / π.
    pub albedo: Vec3,
    /// Emitted radiance (linear). ZERO for a non-emitter. This is the ONLY
    /// source of light in the world — there is no ambient.
    pub emission: Vec3,
}

impl Material {
    pub fn lambertian(albedo: Vec3) -> Material {
        Material {
            albedo,
            emission: Vec3::ZERO,
        }
    }

    /// A pure emitter (color × intensity), no reflection.
    pub fn emissive(emission: Vec3) -> Material {
        Material {
            albedo: Vec3::ZERO,
            emission,
        }
    }

    pub fn is_emitter(&self) -> bool {
        self.emission.max_component() > 0.0
    }
}
