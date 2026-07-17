//! POSITION-BASED FLUIDS — the SPH smoothing kernels and the density-constraint
//! dials (Macklin & Müller, *Position Based Fluids*, SIGGRAPH 2013).
//!
//! The XPBD-native fluid: incompressibility is ONE constraint per particle,
//!
//! ```text
//!     C_i(p) = ρ_i / ρ₀ − 1 ,   ρ_i = Σ_j m_j W(‖p_i − p_j‖, h) ,
//! ```
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
    /// JACOBI SOR under-relaxation on the per-particle position correction Δp
    /// (Macklin §4, "Algorithm 1" applies the density correction with a
    /// relaxation because ALL particles project simultaneously — the pairwise
    /// Newton step is applied from BOTH ends at once, so the full step
    /// over-corrects and, with a stiff many-neighbour kernel, diverges). Δp is
    /// scaled by `relax` before being applied; `1.0` = the raw (unstable)
    /// Jacobi step, `→0` = frozen. Default `0.1` — measured stability edge for
    /// the default pool (`h = 3×spacing`, `≈63` neighbours): `relax ≤ 0.15`
    /// stays bounded, `≥ 0.25` diverges in one step, so `0.1` sits safely inside
    /// the contractive regime. With the Small-Steps loop (`iterations = 1`, `4`
    /// substeps) the constraint still converges to a hydrostatic column with
    /// ≤6% peak compression. Not a physical constant — a numerical relaxation,
    /// hence a dial.
    pub relax: f64,
    /// DENSITY-SOLVER ITERATIONS per substep (Macklin §Algorithm 1: the
    /// neighbour set is found ONCE per substep, then the density constraint is
    /// projected `solver_iterations` times, each pass recomputing λ and Δp
    /// from the CURRENT positions). One SOR-relaxed Jacobi projection barely
    /// nudges a stiff many-neighbour column, so a lone pass leaves the surface
    /// domed (reads as jelly); iterating the projection lets hydrostatic
    /// pressure equalise and the free surface settle FLAT (reads as water).
    /// The paper uses `2..=4`. Default `4` — with `relax = 0.1` the effective
    /// per-substep correction (`1 − (1 − relax)^iters ≈ 0.34`) stays inside the
    /// contractive regime (measured: no divergence) while flattening the dome
    /// an order of magnitude versus one pass. Not a physical constant — a
    /// convergence dial, hence a dial. `0` clamps to `1`.
    pub solver_iterations: usize,
    /// UNILATERAL (compression-only) density constraint. A LIQUID free surface
    /// resists being COMPRESSED (`ρ > ρ₀`, `C > 0`) but exerts no cohesion when
    /// stretched (`ρ < ρ₀`, `C < 0`) — water is not a solid membrane. Because
    /// `ρ₀` is calibrated to the FULLEST (interior) packing, nearly every
    /// particle sits at `C < 0`; letting the bilateral constraint pull those
    /// together makes the whole pool cohere into a rounded heap (jelly/dome).
    /// With this `true` (default) the correction clamps `C_i → max(0, C_i)`, so
    /// the solver only ever pushes apart an over-dense region — gravity then
    /// settles the column to a FLAT hydrostatic surface (reads as water).
    /// `false` recovers Macklin's bilateral constraint (cohesive surface
    /// tension, needs the artificial-pressure term to avoid clustering). Not a
    /// magnitude — a constraint-sidedness law, hence a bool dial.
    pub compression_only: bool,
    /// XSPH VISCOSITY blend fraction `c` (Macklin §5 / Algorithm 1 step 5 —
    /// the velocity post-filter the density projection alone LACKS). After the
    /// positional density solve each fluid particle's velocity is nudged toward
    /// its poly6-weighted neighbourhood mean:
    /// `v_i ← v_i + c·(⟨v⟩_i − v_i)`, `⟨v⟩_i = Σ_j W_poly6(r_ij) v_j / Σ_j W`.
    /// WITHOUT this, a UNILATERAL (compression-only) constraint cannot settle:
    /// a decompression push becomes outward velocity via the PBD position→
    /// velocity read-back, and once the region reaches `C ≤ 0` the one-sided
    /// constraint switches off with NOTHING to absorb that velocity — the fluid
    /// coasts apart forever (measured: a compressed cube in free space expands
    /// at a fixed 1.1 m/s indefinitely) or churns against the floor. XSPH is
    /// the momentum-diffusion term that removes exactly this coasting kick, so
    /// the pool damps to a FLAT hydrostatic rest. `c` is a dimensionless blend
    /// fraction in `[0,1]` (`0` = the raw undamped scheme, `1` = replace each
    /// velocity by the neighbour mean each substep). Default `0.10` — measured
    /// as the smallest blend that damps the default pool's compression-only
    /// churn to a settled surface within ~1 s while leaving a dropped crate's
    /// splash visibly dynamic (not molasses). A physical fluid's viscosity is
    /// this term's continuum limit, but here it is a numerical damping dial,
    /// hence a documented default. Applied to fluid particles only, Jacobi
    /// (old velocities in, new out), index-ordered → determinism preserved.
    pub viscosity_c: f64,
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
            relax: 0.1,
            solver_iterations: 4,
            compression_only: true,
            viscosity_c: 0.10,
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
