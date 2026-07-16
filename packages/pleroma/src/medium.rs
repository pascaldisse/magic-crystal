//! Participating medium in the ONE light pass (Rite VI, A1 → A2).
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
//!
//! A2 — TRUE BINDING: the light the medium scatters toward is no longer a free
//! parameter (first_light's ghost). [`MediumLight`] holds a light the caller
//! read from a REAL realm entity: an emissive glow (a positional [`MediumLight::Point`],
//! radiance falling off as 1/dist² like the stall's lantern) or the directional
//! sun/moon ([`MediumLight::Directional`]). The medium never invents a source —
//! its position and colour come from the world.

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

/// The scene light the medium's single scattering gathers — BOUND to a real
/// realm entity (A2), never invented. Both variants split colour (tint,
/// multiplied outside the scalar march) from the scalar `intensity` the Aether
/// transport folds into the accumulation, so the GPU port composes identically:
/// `in_scatter = colour · scalar_scatter(intensity)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MediumLight {
    /// A directional light (sun / moon): parallel rays from `to_light`, no
    /// distance falloff. `intensity` is the incident radiance scale.
    Directional {
        /// Unit direction from any point TOWARD the light.
        to_light: Vec3,
        /// Linear-rgb tint (the light's colour), multiplied outside the march.
        color: Vec3,
        /// Incident radiance scale folded into the scalar scatter.
        intensity: f64,
    },
    /// A positional glow (an emissive entity — the stall's lantern): world
    /// `position`, radiance falling off as `intensity / dist²` (matches Aether
    /// [`Light::Point`]). This is how a real emitter lights the steam around it.
    Point {
        /// World-space position of the emitter (read from the realm entity).
        position: Vec3,
        /// Linear-rgb tint (the emitter's colour), multiplied outside the march.
        color: Vec3,
        /// Radiant intensity (W/sr) folded into the scalar scatter via 1/dist².
        intensity: f64,
    },
}

impl MediumLight {
    /// Split into the Aether transport light (scalar `intensity` inside) and the
    /// colour tint (applied outside the scalar scatter).
    fn split(&self) -> (Light, Vec3) {
        match *self {
            MediumLight::Directional {
                to_light,
                color,
                intensity,
            } => (
                Light::Directional {
                    to_light: avec3(to_light.x, to_light.y, to_light.z),
                    radiance: intensity,
                },
                color,
            ),
            MediumLight::Point {
                position,
                color,
                intensity,
            } => (
                Light::Point {
                    position: avec3(position.x, position.y, position.z),
                    intensity,
                },
                color,
            ),
        }
    }
}

impl DirectionalSun {
    /// Adapt the legacy directional sun (colour × unit radiance) into the bound
    /// [`MediumLight::Directional`]: the tint is `radiance`, the scalar
    /// intensity is 1 (the old convention — colour carried the whole radiance).
    pub fn as_light(&self) -> MediumLight {
        MediumLight::Directional {
            to_light: self.to_light,
            color: self.radiance,
            intensity: 1.0,
        }
    }
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
    /// The scene light single scattering gathers — bound to a realm entity.
    pub light: MediumLight,
    /// Camera-ray march step count over the primary segment.
    pub march_steps: usize,
    /// Shadow-ray march step count toward the light (self-shadowing).
    pub shadow_steps: usize,
    /// Bound on the occlusion march toward a DIRECTIONAL light (a point light
    /// uses its true source distance instead).
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
        let t1 = t_first.min(self.far);
        if t1 <= t0 {
            return (Vec3::ZERO, 1.0);
        }
        // Scalar single-scatter (the scalar intensity rides inside the light,
        // the colour tints it outside) — the GPU port composes identically.
        let (light, color) = self.light.split();
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
        let inscatter = color * scatter;
        (inscatter, tr)
    }
}
