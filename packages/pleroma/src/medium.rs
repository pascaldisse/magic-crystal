//! Participating medium in the ONE light pass (Rite VI, A1).
//!
//! The Aether crate carries the transport law (Beer-Lambert transmittance +
//! a single-scatter estimator against Henyey-Greenstein phase); this module
//! BINDS that law into the Pleroma's path integrator so steam, smoke and cloud
//! are lit by the same traced light as every surface — NO separate volumetric
//! pass, NO modes (DREAMFORGE VOLUMETRIC LAW). It is the CPU reference the GPU
//! port (`scrying-glass`) is measured against.
//!
//! Scope of A1: single scattering along the PRIMARY camera segment (eye → first
//! surface, or a far cap for escaped rays). The medium in-scatters light toward
//! the camera and attenuates whatever lies behind it (equivalent exchange: what
//! the beam sheds into scattering it removes from the transmitted radiance).
//! Marching the medium along bounce segments too is a later rite.

use crate::vec::Vec3;
use aether::{single_scatter, transmittance, vec3 as avec3, DensityGrid, HomogeneousMedium, Light};

/// A directional light the medium scatters toward (the sun / moon). Scalar
/// Aether transport is tinted per-channel by `radiance` (steam is grey, so the
/// colour rides on the light, not the medium).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalSun {
    /// Unit direction from any point TOWARD the light.
    pub to_light: Vec3,
    /// Incident radiance, per colour channel.
    pub radiance: Vec3,
}

/// A participating medium the integrator marches inside the light pass: the
/// optical coefficients + HG phase ([`HomogeneousMedium`]), a density volume
/// ([`DensityGrid`], the Aether f16-convertible grid — the SAME artifact the
/// GPU uploads), and the light it scatters toward.
#[derive(Clone, Debug, PartialEq)]
pub struct Medium {
    /// Absorption/scattering coefficients + Henyey-Greenstein anisotropy.
    pub optics: HomogeneousMedium,
    /// The density field (rasterized volume) the ray marches through.
    pub grid: DensityGrid,
    /// The directional light single scattering gathers.
    pub sun: DirectionalSun,
    /// Camera-ray march step count over the primary segment.
    pub march_steps: usize,
    /// Shadow-ray march step count toward the light (self-shadowing).
    pub shadow_steps: usize,
    /// Bound on the occlusion march toward the (directional) light.
    pub shadow_dist: f64,
    /// Far cap for the primary march when the camera ray escapes to the sky
    /// (no surface to bound the segment). A dial, never hardcoded at the site.
    pub far: f64,
}

impl Medium {
    /// March the primary camera segment `origin + dir·t` for `t ∈ [t0, t_first]`
    /// (`t_first` = distance to the first surface, or [`Medium::far`] on
    /// escape). Returns `(in_scattered_radiance, transmittance)`: the radiance
    /// the medium adds toward the camera, and the fraction of what lies behind
    /// it that survives. Both feed the integrator's compose:
    /// `L = in_scatter + transmittance · L_behind`.
    pub fn primary(&self, origin: Vec3, dir: Vec3, t0: f64, t_first: f64) -> (Vec3, f64) {
        let o = avec3(origin.x, origin.y, origin.z);
        let d = avec3(dir.x, dir.y, dir.z).normalize();
        let to_light = avec3(
            self.sun.to_light.x,
            self.sun.to_light.y,
            self.sun.to_light.z,
        );
        let t1 = t_first.min(self.far);
        if t1 <= t0 {
            return (Vec3::ZERO, 1.0);
        }
        // Scalar single-scatter (unit light) — the colour rides on the sun.
        let light = Light::Directional {
            to_light,
            radiance: 1.0,
        };
        let scatter = single_scatter(
            &self.optics,
            &self.grid,
            o,
            d,
            t0,
            t1,
            self.march_steps,
            &light,
            self.shadow_dist,
            self.shadow_steps,
        );
        let tr = transmittance(&self.optics, &self.grid, o, d, t0, t1, self.march_steps);
        let inscatter = self.sun.radiance * scatter;
        (inscatter, tr)
    }
}
