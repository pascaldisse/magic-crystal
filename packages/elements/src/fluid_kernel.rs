//! POSITION-BASED FLUIDS — the SPH smoothing kernels and the density-constraint
//! dials (Macklin & Müller, *Position Based Fluids*, SIGGRAPH 2013).
//!
//! The XPBD-native fluid: incompressibility is ONE constraint per particle,
//!
//!     C_i(p) = ρ_i / ρ₀ − 1 ,   ρ_i = Σ_j m_j W(‖p_i − p_j‖, h) ,
//!
//! solved by the SAME positional-projection machinery as every distance bond
//! (a new constraint TYPE beside bonds/contacts — not a parallel system). The
//! two kernels are the paper's (after Müller 2003 "Particle-Based Fluid
//! Simulation"): POLY6 for the density estimate (smooth, its zero gradient at
//! r=0 keeps the density sum well-behaved) and SPIKY for the gradient (a
//! non-vanishing gradient near r=0 so close particles feel real repulsion —
//! poly6's gradient collapses there and lets particles clump).
//!
//! Every constant here is DERIVED from the support radius `h` by the kernels'
//! own normalisation integrals (∫W dV = 1 over the ball of radius `h`) — none
//! is a plucked literal (the never-hardcode law). The user dials are `h`, the
//! rest density `ρ₀` (calibrated, see the solver), and the three artificial-
//! pressure parameters, each a documented default.

use crate::math::Vec3;
use std::f64::consts::PI;

/// The Position-Based-Fluids dials — every field a parameter with a documented
/// default (never-hardcode). The rest density `ρ₀` is NOT here: it is DERIVED
/// per pool by the solver's [`crate::Solver::calibrate_fluid_rest_density`]
/// from the actual spawn packing, so `C_i = 0` holds at rest by construction
/// rather than being asserted against a plucked number.
#[derive(Clone, Copy, Debug)]
pub struct FluidConfig {
    /// The SMOOTHING RADIUS `h` (metres) — the kernel support. A particle's
    /// density sums only neighbours within `h`; the neighbour grid's cell is
    /// this exact value (derived, [`crate::pointgrid::PointGrid::cell_size`]).
    /// Set by [`crate::Solver::spawn_fluid_box`] to a multiple of the spawn
    /// spacing (default there: `3×spacing`, enough neighbours for a smooth
    /// density estimate — the paper's regime).
    pub h: f64,
    /// The DERIVED rest density `ρ₀` (SPH units: Σ m·W at full packing) —
    /// filled by [`crate::Solver::calibrate_fluid_rest_density`], NOT authored.
    /// The physical water density lives in the particle MASS; this is the
    /// self-consistent SPH value that makes an interior particle report
    /// `C_i = 0` at the spawn lattice. `0.0` until calibrated.
    pub rest_density: f64,
    /// CFM relaxation FRACTION (Macklin eq. 11, `ε`) expressed RELATIVE to a
    /// full neighbourhood's `Σ|∇C|²` — never a bare units² literal. The
    /// absolute `ε = cfm_relax × (interior Σ|∇C|²)` is DERIVED at calibration
    /// (stored in `cfm_epsilon`), so it scales with the pool's own gradient
    /// magnitude instead of being plucked. Default `1e-4` (the paper's regime:
    /// a small softening that only matters for the near-singular few-neighbour
    /// case). `0.0` = no relaxation (a lone particle can then blow up).
    pub cfm_relax: f64,
    /// The DERIVED absolute CFM ε `= cfm_relax × interior Σ|∇C|²`, filled by
    /// [`crate::Solver::calibrate_fluid_rest_density`]. `0.0` until calibrated.
    pub cfm_epsilon: f64,
    /// Artificial-pressure strength `k` (Macklin eq. 13) — the tensile-
    /// instability corrector that keeps particles from clumping into clusters
    /// under negative pressure and yields a slight surface cohesion (reads as
    /// water, not soup). Default `0.1` (the paper's value). `0.0` disables it.
    pub tensile_k: f64,
    /// Artificial-pressure exponent `n` (Macklin eq. 13). Default `4` (paper).
    pub tensile_n: f64,
    /// Δq for the artificial pressure, as a FRACTION of `h` (Macklin: `Δq` in
    /// `[0.1h, 0.3h]`). Default `0.2` → `Δq = 0.2h`. The reference kernel value
    /// `W(Δq)` the corrector normalises against.
    pub tensile_dq_frac: f64,
}

impl Default for FluidConfig {
    fn default() -> Self {
        FluidConfig {
            h: 0.0,           // set by spawn_fluid_box
            rest_density: 0.0, // set by calibrate_fluid_rest_density
            cfm_relax: 1.0e-4,
            cfm_epsilon: 0.0, // set by calibrate_fluid_rest_density
            tensile_k: 0.1,
            tensile_n: 4.0,
            tensile_dq_frac: 0.2,
        }
    }
}

/// The POLY6 density kernel `W(r, h) = 315/(64π h⁹) (h² − r²)³` for
/// `0 ≤ r ≤ h`, else `0`. The `315/(64π h⁹)` factor is the 3-D normalisation
/// (∫ W dV = 1 over the ball) — derived, not plucked.
#[inline]
pub fn poly6(r: f64, h: f64) -> f64 {
    if r < 0.0 || r > h {
        return 0.0;
    }
    let h2 = h * h;
    let r2 = r * r;
    let d = h2 - r2;
    let coeff = 315.0 / (64.0 * PI * h.powi(9));
    coeff * d * d * d
}

/// The SPIKY gradient kernel `∇W(r_vec, h) = −45/(π h⁶) (h − r)² · r_vec/r`
/// for `0 < r ≤ h`, else `0`. Points from `j` toward `i` when `r_vec =
/// p_i − p_j` (the repulsive direction). The `45/(π h⁶)` factor is the
/// spiky kernel's gradient normalisation — derived. At `r = 0` the direction
/// is undefined and the gradient is taken `0` (coincident particles exert no
/// directional force; the density sum still counts them via poly6).
#[inline]
pub fn spiky_grad(r_vec: Vec3, h: f64) -> Vec3 {
    let r = r_vec.length();
    if r <= 0.0 || r > h {
        return Vec3::ZERO;
    }
    let coeff = -45.0 / (PI * h.powi(6));
    let mag = coeff * (h - r) * (h - r);
    r_vec.scale(mag / r)
}
