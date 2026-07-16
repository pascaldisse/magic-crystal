//! The homogeneous participating medium — the optical law a density field is
//! measured in.
//!
//! Every cloud, plume of steam, tongue of fire and wisp of smoke is
//! participating media in the one traced light (DREAMFORGE VOLUMETRIC LAW —
//! 2D billboards are forbidden vocabulary). This struct carries the medium's
//! constant optical coefficients; the spatially-varying part is the density
//! field ([`crate::Density`]), which scales `sigma_t` per world point.

/// Absorption + scattering coefficients and the Henyey-Greenstein anisotropy.
///
/// Coefficients are per-unit-density, per-world-unit of path length (units of
/// `1/length`). The extinction a ray actually sees at a point is
/// `sigma_t * density(point)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HomogeneousMedium {
    /// Absorption coefficient `sigma_a` (≥ 0). Light removed, not redirected.
    pub sigma_a: f64,
    /// Scattering coefficient `sigma_s` (≥ 0). Light redirected by the phase
    /// function.
    pub sigma_s: f64,
    /// Henyey-Greenstein anisotropy `g` in `(-1, 1)`. `0` isotropic, `> 0`
    /// forward-scattering (clouds ~0.6–0.85), `< 0` back-scattering.
    pub g: f64,
}

impl Default for HomogeneousMedium {
    /// A neutral grey scattering medium: unit scattering, no absorption,
    /// isotropic. Documented starting point — nothing here is hardcoded into
    /// the transport, callers set their own.
    fn default() -> Self {
        HomogeneousMedium {
            sigma_a: 0.0,
            sigma_s: 1.0,
            g: 0.0,
        }
    }
}

impl HomogeneousMedium {
    /// Construct from the three coefficients.
    pub fn new(sigma_a: f64, sigma_s: f64, g: f64) -> Self {
        HomogeneousMedium {
            sigma_a,
            sigma_s,
            g,
        }
    }

    /// Extinction coefficient `sigma_t = sigma_a + sigma_s` (per unit density).
    #[inline]
    pub fn sigma_t(&self) -> f64 {
        self.sigma_a + self.sigma_s
    }

    /// Single-scattering albedo `sigma_s / sigma_t` in `[0, 1]`. Returns `0`
    /// for a vacuum (`sigma_t == 0`).
    #[inline]
    pub fn albedo(&self) -> f64 {
        let st = self.sigma_t();
        if st == 0.0 {
            0.0
        } else {
            self.sigma_s / st
        }
    }

    /// The Henyey-Greenstein phase function evaluated at scattering-angle
    /// cosine `cos_theta` (the cosine between the *incoming* photon travel
    /// direction and the *outgoing* scattered travel direction).
    ///
    /// Normalized so that `∮ phase dω = 1` over the unit sphere:
    /// `p(μ) = (1/4π) · (1 − g²) / (1 + g² − 2gμ)^{3/2}`.
    #[inline]
    pub fn phase(&self, cos_theta: f64) -> f64 {
        let g = self.g;
        let g2 = g * g;
        let denom = 1.0 + g2 - 2.0 * g * cos_theta;
        // denom > 0 for g in (-1,1) and cos in [-1,1]; guard the pole anyway.
        let denom = denom.max(1e-12);
        (1.0 - g2) / (4.0 * std::f64::consts::PI * denom.powf(1.5))
    }
}
