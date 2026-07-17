//! P-SCALE — THE BUILDING FALLS. A parameterized bonded multi-storey
//! structure built ON the solver (no new physics, no hardcoded magnitude —
//! every dial is a [`BuildingSpec`] field with a documented default), plus
//! the authored "hand" that knocks it down. Shared by the measurement
//! example (`examples/pscale_measure.rs`) and the ordeals
//! (`tests/pscale_ordeals.rs`) so both drive the EXACT same worldline.
//!
//! A building here is a solid bonded LATTICE (one [`Solver::spawn_bonded_box`]
//! call — columns, floors and walls are the lattice's own +x/+y/+z bonds)
//! resting on the ground plane, with its BASE LAYER anchored (`inv_mass = 0`,
//! the physics-recon "anchors" problem, solved the only way the solver
//! honours: a zero-inverse-mass particle is immovable and every pass already
//! respects it). Mass is DERIVED (`density × volume`, `spawn_bonded_box`'s
//! own law). The collapse is driven by a lateral impulse on the upper
//! storeys (support-shear: the anchored base cannot follow, so the base
//! bonds carry the strife that tears them) — deterministic, the tick index
//! is the only clock.

use crate::collision::{Collider, ContactMaterial};
use crate::math::Vec3;
use crate::solver::{Solver, SolverConfig};

/// The building's dials — every one a parameter with a documented default
/// (never-hardcode law). Particle count `N = lattice.0 × lattice.1 ×
/// lattice.2`; pick `lattice` to land on the target order of magnitude.
#[derive(Clone, Copy, Debug)]
pub struct BuildingSpec {
    /// Footprint full extents in metres `(x, z)`. Default `(8, 8)`.
    pub footprint: (f64, f64),
    /// Total height in metres (multi-storey). Default `24` (~8 storeys at 3 m).
    pub height: f64,
    /// The particle lattice `(nx, ny, nz)` — the resolution dial that sets N.
    /// Default `(8, 16, 8)` = 1024 particles.
    pub lattice: (usize, usize, usize),
    /// Material density kg/m³ (mass derived from it). Default `2000`
    /// (masonry/concrete-adjacent).
    pub density: f64,
    /// Per-bond love on `[0, 1]` — the fragility dial. Default `0.05`: an
    /// AUTHORED fragile override (like `worlds/naruko-vi2`'s `0.02` crate),
    /// chosen so the structure actually shatters into rubble under its own
    /// collapse rather than holding as one rocking block. A real
    /// concrete-density `default_bond_love` (~0.74) would not break — that
    /// is a material-essence call, not this proof's concern. `0.5` holds the
    /// settled structure's static self-weight (see [`settle`]) yet tears
    /// under the topple's dynamic shear.
    pub love: f64,
    /// Bond compliance (XPBD inverse stiffness, m/N). Default `1e-7`
    /// (near-rigid — a stiff structure that holds its 24 m height without
    /// accordion-squashing under its own weight, matching the `body` sigil
    /// default). Its high static strife (≤3.1e6 across the profiled scales)
    /// still sits comfortably below the collapse's dynamic peak (≥2.0e7 at
    /// the default topple speed) — a clean ≥6.5× gap [`fracture_threshold`]
    /// sits inside (a SOFT bond would lower strife but visibly compress the
    /// tower, which is worse for a believable building than a high strife
    /// number the threshold is simply scaled to).
    pub compliance: f64,
    /// Each particle's contact thickness in metres. Default `0.06`.
    pub contact_radius: f64,
    /// Fracture threshold (strife-to-love conversion). Default `1.8e7` — the
    /// break bar is `love × threshold = 9e6`, DERIVED to sit inside the
    /// measured gap between the standing tower's rest-window strife ceiling
    /// (≤5.35e6 across the profiled scales N ∈ {1024, 2000, 3456}, over the
    /// metastable 60-tick plateau) and the collapse's dynamic peak (≥1.66e7
    /// at the default topple speed 30 m/s): the structure bears its own
    /// weight while standing, yet the topple's shear/impact tears it, at
    /// every scale. (Measured with `PSCALE_PEAK=1` on the measurement
    /// example, 40-tick settle.) The absolute magnitude is large because
    /// near-rigid XPBD bonds report large constraint force
    /// (strife ~ lambda/dt_sub²) — the threshold is simply scaled to that
    /// regime; only the RATIO to static strife carries physical meaning.
    pub fracture_threshold: f64,
    /// Substeps per tick — the 60 FPS regime uses `8` (the solver default),
    /// which is what this lane measures against the 16.667 ms budget.
    pub substeps: usize,
    /// Whether the base layer (lowest y) is anchored to the ground
    /// (`inv_mass = 0`). Default `true`. `false` makes a free-standing block
    /// that pancakes straight down (used by no ordeal, kept for completeness).
    pub base_anchored: bool,
}

impl Default for BuildingSpec {
    fn default() -> Self {
        BuildingSpec {
            footprint: (8.0, 8.0),
            height: 24.0,
            lattice: (8, 16, 8),
            density: 2000.0,
            love: 0.5,
            compliance: 1.0e-7,
            contact_radius: 0.06,
            fracture_threshold: 1.8e7,
            substeps: 8,
            base_anchored: true,
        }
    }
}

impl BuildingSpec {
    /// The particle count this spec produces (`nx × ny × nz`).
    pub fn particle_count(&self) -> usize {
        self.lattice.0 * self.lattice.1 * self.lattice.2
    }
}

/// A standing building wired into a fresh solver: the solver, the building's
/// whole particle set (lattice order), and the y-index of the base layer's
/// particles that were anchored (empty when `base_anchored` is false).
pub struct Building {
    pub solver: Solver,
    pub whole: Vec<usize>,
    pub anchored: Vec<usize>,
    pub spec: BuildingSpec,
}

/// Erect the building on a ground plane, at rest, base anchored. The lattice
/// is centred over the origin with its base sitting on `y = 0`; the top rises
/// to `height`. Deterministic: no randomness, the same spec always yields the
/// same solver.
pub fn erect(spec: BuildingSpec) -> Building {
    let cfg = SolverConfig {
        dt: 1.0 / 60.0,
        substeps: spec.substeps,
        fracture_threshold: spec.fracture_threshold,
        ..SolverConfig::default()
    };
    let mut solver = Solver::new(cfg);
    // A wide ground plane under the whole footprint + collapse spread.
    solver.collider = Some(Collider::ground_plane(0.0, 200.0, ContactMaterial::default()));

    let dims = Vec3::new(spec.footprint.0, spec.height, spec.footprint.1);
    // Base on y = 0: centre y = height/2.
    let center = Vec3::new(0.0, spec.height * 0.5, 0.0);
    let whole = solver.spawn_bonded_box(
        center,
        dims,
        spec.lattice,
        spec.density,
        spec.love,
        spec.compliance,
        spec.contact_radius,
    );

    // Anchor the base layer: `spawn_bonded_box` fills in (ix, iy, iz) order,
    // so a particle's lattice y-index is `(flat / nz) % ny` — but simplest
    // and order-independent is to anchor every particle whose world y is at
    // the lattice's lowest row. The lowest row sits at world y = 0 (origin =
    // center - dims/2, iy = 0). Anchor by lattice iy == 0, recovered from the
    // spawn order directly.
    let (nx, ny, nz) = spec.lattice;
    let mut anchored = Vec::new();
    if spec.base_anchored {
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let flat = ix * (ny * nz) + iy * nz + iz;
                    if iy == 0 {
                        let p = whole[flat];
                        solver.particles.inv_mass[p] = 0.0;
                        anchored.push(p);
                    }
                }
            }
        }
    }

    Building {
        solver,
        whole,
        anchored,
        spec,
    }
}

/// METASTABILITY (honest gap, measured): a tall anchored bonded tower under
/// this XPBD solver (8 substeps, 1 iteration) holds its height rock-solid for
/// ~1.3 s (~80 ticks) then slowly buckles/creeps downward — Euler-style
/// instability of a slender heavy column that the single-iteration Gauss-
/// Seidel constraint solve cannot resist indefinitely (more iterations do NOT
/// fix it — verified). The proof scenario settles to the STABLE PLATEAU
/// (`~40` ticks, top still ≈23.5 m of 24 m) and topples from there, before
/// the creep matters; the "rest stays at rest" ordeal checks a BOUNDED
/// window over that plateau, not indefinitely. Holding sustained static
/// equilibrium of tall stacks is exactly the kind of long-horizon stability
/// a learned corrector could add (see the report).
///
/// Settle the freshly-erected building to its standing plateau with fracture
/// DISARMED, then re-arm it. A near-rigid lattice (compliance ~1e-7) spikes
/// enormous transient constraint force in its first substeps as gravity and
/// the stiff bonds fight to equilibrium — a numerical startup shock, not real
/// structural load, that would spuriously tear a few bonds before the
/// building has ever borne its own steady weight. Real structures are built
/// AT REST; this reproduces that honestly: step `ticks` with the threshold at
/// infinity (no bond can break no matter the transient strife), then restore
/// the authored threshold so the standing structure — and everything after —
/// fractures only under genuine load. Deterministic; leaves the solver tick
/// at `ticks` (the caller measures the collapse from there). Returns the
/// number of fragments after settling (should be 1 for a stable structure —
/// the "rest stays at rest" ordeal's witness).
pub fn settle(building: &mut Building, ticks: u64) -> usize {
    let armed = building.solver.config.fracture_threshold;
    building.solver.config.fracture_threshold = f64::INFINITY;
    for _ in 0..ticks {
        building.solver.step();
    }
    building.solver.config.fracture_threshold = armed;
    building.solver.fragment_components(&building.whole).len()
}

/// The authored HAND that knocks it down: a uniform lateral velocity impulse
/// (`+x`) applied ONCE to every NON-anchored particle in the UPPER portion of
/// the building (at or above `fraction` of its height). The anchored base
/// cannot follow, so the base bonds carry the shear strife that tears them —
/// a support-shear collapse, not a staged effect. `speed` and `fraction` are
/// authored scenario magnitudes (the "op is the hand" law), never
/// solver-invented.
pub fn topple(building: &mut Building, speed: f64, fraction: f64) {
    let spec = building.spec;
    let (nx, ny, nz) = spec.lattice;
    let threshold_iy = (fraction * (ny.saturating_sub(1)) as f64).floor() as usize;
    let dv = Vec3::new(speed, 0.0, 0.0);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                if iy < threshold_iy {
                    continue;
                }
                let flat = ix * (ny * nz) + iy * nz + iz;
                let p = building.whole[flat];
                if building.solver.particles.inv_mass[p] != 0.0 {
                    building.solver.particles.vel[p] = building.solver.particles.vel[p] + dv;
                }
            }
        }
    }
}
