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
    /// ROUND-9 — HYDROSTATIC-GRADIENT FACTOR. Multiplies the packing density
    /// measured by [`crate::Solver::calibrate_fluid_rest_density`] before it
    /// becomes `rest_density`, dimensionless in `(0,1]`. `1.0` sets ρ₀ = the
    /// MAX (fullest) spawn-lattice packing, so at rest every particle reads
    /// `C_i ≤ 0` and the `compression_only` constraint is inert — the fluid is
    /// PRESSURELESS in the bulk (the round-8 buoyancy gap: no gradient, nothing
    /// lifts a submerged body). A factor `< 1` places ρ₀ BELOW packing, so the
    /// settled lattice would be genuinely OVER-dense (`C > 0`) and, since a
    /// column under gravity compresses MORE with depth, `C` — hence the density
    /// pressure `λ` — would RISE with depth: a real hydrostatic gradient.
    ///
    /// ROUND-9 MEASURED VERDICT (`fluid_profile_probe`, kept): this lever DOES
    /// NOT deliver buoyancy; the default stays `1.0`. Two proven reasons:
    ///   1. With a FREE SURFACE + `compression_only`, the pool simply EXPANDS
    ///      (surface rose ~0.185→0.204 m as factor 1.0→0.85) to relieve the
    ///      over-density, so the settled column reads UNDER-dense in EVERY
    ///      depth bin at every factor — no sustained `C>0`, so `λ≈0` at rest,
    ///      so no gradient. Pressure is the multiplier `λ`, not density;
    ///      `compression_only` clamps `C→0` hence `λ→0` for a fluid at rest.
    ///   2. The discrimination sweep converged EVERY density (200–2000 kg/m³)
    ///      AND every release height to the SAME equilibrium depth (~0.165 m)
    ///      — zero mass discrimination. A density-2000 box (twice water, must
    ///      sink) rested at the same depth as a density-200 cork: the "rise"
    ///      is a geometric displacement artifact of the Akinci push, not
    ///      Archimedes.
    /// A factor `< 1` also breaks the by-construction `C≤0`-at-spawn rest gate.
    /// The true remaining lever is a real `λ` FIELD at depth — CONTAINER-
    /// boundary Akinci particles (so confined bottom fluid stops reading
    /// boundary-deficient and develops `λ`) plus a confined over-density; a
    /// larger effort, escalated not faked. Documented, measured infrastructure.
    pub rest_density_factor: f64,
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
    /// RETIRED (round-8) — the artificial-pressure/tensile-instability
    /// corrector `s_corr` (Macklin eq. 13). The solver NO LONGER READS these
    /// three fields: `solve_fluid` computes no `s_corr` term. They survive as
    /// inert config only so the round-6/7 diagnostic probes
    /// (`fluid_unit_probe`, `fluid_clamp_probe`, `fluid_probe`,
    /// `fluid_flatness`) still compile; setting them has ZERO runtime effect.
    ///
    /// The full round-6/7 verdict: `s_corr>0` DETONATES under sustained
    /// hydrostatic compression even correctly per-pair gated (a compressed
    /// column keeps its particles gate-live tick after tick, so the repulsion
    /// compounds), while `s_corr=0` leaves the SPH density estimate blind to a
    /// real-space particle COLLAPSE (mean NN distance measured 70-90% below
    /// spawn spacing, pairs coincident) — s_corr traded one failure for an
    /// invisible one. The round-8 CURE replaces it with a genuine collision-
    /// style pairwise MINIMUM-SEPARATION resolved through the solver's OWN
    /// contact machinery (`min_sep_factor`/`min_separation` below), decoupled
    /// from the SPH density feedback loop entirely. s_corr is dead.
    pub tensile_k: f64,
    /// RETIRED — see `tensile_k`. Inert.
    pub tensile_n: f64,
    /// RETIRED — see `tensile_k`. Inert.
    pub tensile_dq_frac: f64,
    /// MINIMUM-SEPARATION contact floor `r_min`, as a FRACTION of the spawn
    /// spacing (`r_min = min_sep_factor × spacing`, derived at
    /// [`crate::Solver::spawn_fluid_box`] into `min_separation` — never a bare
    /// metre literal). This is the round-8 CURE for the tensile collapse: any
    /// two fluid particles closer than `r_min` become a CONTACT in the SAME
    /// per-substep contact solve every rigid body uses
    /// ([`crate::Solver::solve_fluid_contacts`]) — a collision-style hard
    /// floor on pairwise separation, entirely independent of the SPH density
    /// estimate that s_corr destabilised. Default `0.85`: the spawn lattice's
    /// nearest neighbour sits at exactly `spacing`, so a floor at `0.85 ×
    /// spacing` leaves the rest lattice force-free (no jitter, no added
    /// overdensity) while catching any genuine collapse long before the
    /// SPH-invisible clustering can start. `0.0` disables the floor (recovers
    /// the round-7 collapse — used by the sabotage ordeal to prove the
    /// min-separation gate is non-vacuous).
    pub min_sep_factor: f64,
    /// The DERIVED absolute minimum separation `r_min` (metres) `=
    /// min_sep_factor × spacing`, filled by
    /// [`crate::Solver::spawn_fluid_box`] from the actual spawn spacing. `0.0`
    /// until a fluid box is spawned (the contact pass then costs zero).
    pub min_separation: f64,
    /// Restitution for fluid–fluid minimum-separation contacts (a
    /// dimensionless bounce fraction in `[0,1]`). Default `0.0`: a fluid
    /// particle collision is inelastic — the floor kills inward normal
    /// velocity and adds no bounce (any residual momentum is diffused by XSPH
    /// viscosity). Tangential friction on these contacts rides the pool's
    /// shared [`crate::collision::ContactMaterial`], not a second dial.
    pub contact_restitution: f64,
    /// FLUID↔SOLID two-way pressure coupling strength (round-9), dimensionless
    /// in `[0,1]`. A submerged rigid/bonded body's particles act as Akinci
    /// (2012) BOUNDARY particles: each contributes `ψ_b = ρ₀·V_b` (with the
    /// Akinci volume `V_b = 1/Σ_{b'}W(r_bb')`, self-calibrated from the
    /// boundary's own packing — never a bare literal) to the SPH density of
    /// every nearby fluid particle. That raises the fluid pressure `λ` against
    /// the solid (fluid cannot enter the body), and the SAME position
    /// correction is mirrored back onto the solid particle scaled by this
    /// factor — so the depth-increasing hydrostatic `λ` integrated over the
    /// body surface becomes a NET BUOYANT force (Archimedes), the round-8
    /// escalation (a light body would not rise through the pressure-blind
    /// contact-only coupling). `1.0` = full two-way reaction; `0.0` disables
    /// the coupling entirely (the solid is invisible to the fluid density —
    /// the round-8 contact-only behaviour, used by the sabotage probe to prove
    /// this gate is non-vacuous). Costs zero when no rigid/bonded body exists.
    pub solid_coupling: f64,
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
            rest_density_factor: 1.0, // round-9: ρ₀ = max packing (round-8 exact). A factor < 1 was MEASURED insufficient for buoyancy (see solver::solve_fluid round-9 note + fluid_profile_probe) and breaks the by-construction C≤0 rest gate; kept as investigated infrastructure, NOT the cure.
            cfm_relax: 1.0e-4,
            cfm_epsilon: 0.0, // set by calibrate_fluid_rest_density
            tensile_k: 0.0,
            tensile_n: 4.0,
            tensile_dq_frac: 0.2,
            min_sep_factor: 0.85,
            min_separation: 0.0, // set by spawn_fluid_box
            contact_restitution: 0.0,
            solid_coupling: 1.0,
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
