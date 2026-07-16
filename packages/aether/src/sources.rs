//! Procedural density sources — analytic fields the grid rasterizes.
//!
//! No randomness anywhere (ENTROPY law): all "noise" is value noise sampled
//! from [`crate::hash`], a pure function of `(seed, lattice cell, octave)`.
//! Two rasterizations with the same seed are byte-identical.
//!
//! Presets ([`SteamColumn`], [`CloudPuff`]) are **param sets** — their fields
//! have documented [`Default`] values, and nothing about them is hardcoded
//! into the transport; a caller may override any field.

use crate::hash::uniform;
use crate::vec::Vec3;

/// A scalar density field over world space. Implemented by analytic sources
/// and by [`crate::DensityGrid`] (via trilinear sampling), so transport code
/// marches either through one generic path.
pub trait Density {
    /// Density at a world point (≥ 0; `0` is vacuum).
    fn density(&self, p: Vec3) -> f64;
}

/// A soft sphere: full `peak` density at the center, smoothstep falloff to `0`
/// at `radius`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereFalloff {
    /// Sphere center in world space.
    pub center: Vec3,
    /// Radius at which density reaches zero.
    pub radius: f64,
    /// Peak density at the center.
    pub peak: f64,
}

impl Default for SphereFalloff {
    /// Unit sphere at the origin, peak density `1`.
    fn default() -> Self {
        SphereFalloff {
            center: Vec3::ZERO,
            radius: 1.0,
            peak: 1.0,
        }
    }
}

impl Density for SphereFalloff {
    fn density(&self, p: Vec3) -> f64 {
        let d = (p - self.center).length();
        if d >= self.radius {
            return 0.0;
        }
        let t = 1.0 - d / self.radius; // 1 at center, 0 at rim
                                       // Smoothstep for a soft, C¹ edge.
        self.peak * t * t * (3.0 - 2.0 * t)
    }
}

/// Deterministic value noise at a world point, folded from integer lattice
/// corners via [`uniform`] and trilinearly interpolated. Pure function of
/// `(seed, cell, octave)`.
fn value_noise(seed: u64, p: Vec3, octave: u64) -> f64 {
    // Bias into positive lattice space so `floor` keys stay well-defined for
    // negative coordinates.
    let bias = 1024.0;
    let x = p.x + bias;
    let y = p.y + bias;
    let z = p.z + bias;
    let (xi, yi, zi) = (x.floor(), y.floor(), z.floor());
    let (fx, fy, fz) = (x - xi, y - yi, z - zi);

    let corner = |dx: u64, dy: u64, dz: u64| -> f64 {
        let cx = xi as i64 as u64 + dx;
        let cy = yi as i64 as u64 + dy;
        let cz = zi as i64 as u64 + dz;
        uniform(seed, cx, cy, cz, octave)
    };

    // Smoothstep the interpolants for a C¹ field.
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let sz = fz * fz * (3.0 - 2.0 * fz);

    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;

    let c000 = corner(0, 0, 0);
    let c100 = corner(1, 0, 0);
    let c010 = corner(0, 1, 0);
    let c110 = corner(1, 1, 0);
    let c001 = corner(0, 0, 1);
    let c101 = corner(1, 0, 1);
    let c011 = corner(0, 1, 1);
    let c111 = corner(1, 1, 1);

    let x00 = lerp(c000, c100, sx);
    let x10 = lerp(c010, c110, sx);
    let x01 = lerp(c001, c101, sx);
    let x11 = lerp(c011, c111, sx);
    let y0 = lerp(x00, x10, sy);
    let y1 = lerp(x01, x11, sy);
    lerp(y0, y1, sz)
}

/// A fractal stack of `value_noise` octaves (fBm). Deterministic in `seed`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseStack {
    /// Sampler seed.
    pub seed: u64,
    /// Spatial frequency of the first octave (1 / world-unit).
    pub base_frequency: f64,
    /// Number of octaves summed.
    pub octaves: u32,
    /// Frequency multiplier per octave.
    pub lacunarity: f64,
    /// Amplitude multiplier per octave.
    pub gain: f64,
    /// Overall amplitude scale of the summed field.
    pub amplitude: f64,
}

impl Default for NoiseStack {
    /// A gentle 4-octave stack, unit amplitude, seed `0`.
    fn default() -> Self {
        NoiseStack {
            seed: 0,
            base_frequency: 1.0,
            octaves: 4,
            lacunarity: 2.0,
            gain: 0.5,
            amplitude: 1.0,
        }
    }
}

impl NoiseStack {
    /// Sample the fBm at `p`, normalized to `[0, 1]` (before `amplitude`).
    pub fn sample(&self, p: Vec3) -> f64 {
        let mut freq = self.base_frequency;
        let mut amp = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for o in 0..self.octaves as u64 {
            sum += amp * value_noise(self.seed, p * freq, o);
            norm += amp;
            freq *= self.lacunarity;
            amp *= self.gain;
        }
        let unit = if norm > 0.0 { sum / norm } else { 0.0 };
        self.amplitude * unit
    }
}

impl Density for NoiseStack {
    fn density(&self, p: Vec3) -> f64 {
        self.sample(p).max(0.0)
    }
}

/// **Preset (param set)** — a rising column of steam.
///
/// A vertical cylinder of density that fades to nothing at its radius and top,
/// perturbed by an fBm stack that grows with height (a plume widening and
/// breaking up as it rises). All fields are documented defaults; override any.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SteamColumn {
    /// World-space center of the column base.
    pub base: Vec3,
    /// Column height (density fades to zero at `base.y + height`).
    pub height: f64,
    /// Column radius at the base.
    pub radius: f64,
    /// Peak density at the base center.
    pub peak: f64,
    /// fBm perturbation applied to the density.
    pub noise: NoiseStack,
    /// How strongly the noise erodes the column `[0, 1]`.
    pub turbulence: f64,
}

impl Default for SteamColumn {
    /// A 4 m plume, 0.5 m base radius, moderate turbulence, seed `1`.
    fn default() -> Self {
        SteamColumn {
            base: Vec3::ZERO,
            height: 4.0,
            radius: 0.5,
            peak: 1.0,
            noise: NoiseStack {
                seed: 1,
                base_frequency: 1.5,
                octaves: 4,
                lacunarity: 2.0,
                gain: 0.5,
                amplitude: 1.0,
            },
            turbulence: 0.6,
        }
    }
}

impl Density for SteamColumn {
    fn density(&self, p: Vec3) -> f64 {
        let rel = p - self.base;
        let h = rel.y;
        if h < 0.0 || h >= self.height {
            return 0.0;
        }
        let frac_h = h / self.height; // 0 base, 1 top
                                      // Plume widens with height.
        let r_here = self.radius * (0.6 + 0.8 * frac_h);
        let r = (rel.x * rel.x + rel.z * rel.z).sqrt();
        if r >= r_here {
            return 0.0;
        }
        // Radial smoothstep falloff.
        let tr = 1.0 - r / r_here;
        let radial = tr * tr * (3.0 - 2.0 * tr);
        // Vertical fade: rises then thins near the top.
        let vertical = (1.0 - frac_h).max(0.0);
        let base_density = self.peak * radial * vertical;
        // Turbulent erosion, stronger higher up.
        let n = self.noise.density(p);
        let erosion = 1.0 - self.turbulence * frac_h * (1.0 - n);
        (base_density * erosion).max(0.0)
    }
}

/// **Preset (param set)** — a single cloud puff.
///
/// A soft sphere ([`SphereFalloff`]) whose surface is eroded by an fBm stack —
/// the cauliflower boundary of a cumulus tuft. Documented defaults; override
/// any field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudPuff {
    /// Core soft sphere.
    pub core: SphereFalloff,
    /// fBm perturbation of the surface.
    pub noise: NoiseStack,
    /// How deeply the noise erodes the core `[0, 1]`.
    pub erosion: f64,
}

impl Default for CloudPuff {
    /// A 2 m puff at the origin, moderate erosion, seed `2`.
    fn default() -> Self {
        CloudPuff {
            core: SphereFalloff {
                center: Vec3::ZERO,
                radius: 2.0,
                peak: 1.0,
            },
            noise: NoiseStack {
                seed: 2,
                base_frequency: 1.2,
                octaves: 5,
                lacunarity: 2.0,
                gain: 0.5,
                amplitude: 1.0,
            },
            erosion: 0.5,
        }
    }
}

impl Density for CloudPuff {
    fn density(&self, p: Vec3) -> f64 {
        let core = self.core.density(p);
        if core <= 0.0 {
            return 0.0;
        }
        let n = self.noise.density(p);
        // Erode the core: subtract eroded noise, keep positive.
        (core - self.erosion * (1.0 - n)).max(0.0)
    }
}

/// A constant density everywhere — the homogeneous field the analytic-slab and
/// grid-agreement ordeals are measured against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constant(pub f64);

impl Density for Constant {
    fn density(&self, _p: Vec3) -> f64 {
        self.0
    }
}
