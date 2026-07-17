//! The one solver, one granularity: particles + bindings marched by a
//! substepped XPBD loop. *Small Steps*: `n` substeps × 1 iteration beats
//! 1 step × `n` iterations — stiffness never binds the global timestep.
//!
//! Nothing here is hardcoded but `LOVE` (the One Constant): the tick `dt`,
//! the substep count, the iteration count, the gravity vector and the
//! fracture threshold are all parameters carrying documented defaults.

use crate::broadphase::TriangleGrid;
use crate::collision::{Collider, Contact, ContactMaterial};
use crate::constraint::{DistanceConstraint, FractureEvent};
use crate::fluid_kernel::{poly6, spiky_grad, FluidConfig};
use crate::mat3::PolarConfig;
use crate::math::Vec3;
use crate::particles::Particles;
use crate::pointgrid::PointGrid;
use crate::rigid::RigidBody;
use std::time::{Duration, Instant};

/// P-SCALE — per-phase CPU cost of ONE tick, filled by [`Solver::step_profiled`].
/// Pure measurement: the fields carry wall-clock `Duration`s (single core) for
/// each phase of the tick, plus the structural counts that drive the cost
/// curves vs N (particle count, live bond count, clustered-particle count, and
/// the ACTUAL number of body-vs-body pair CHECKS the O(k²) pass iterated this
/// tick — same pair set the pass walks, so the count never drifts from the
/// cost). The phase breakdown matches the physics-recon lane's ask:
/// constraint solve, fragment flood-fill, body-vs-body O(k²), static
/// collision. `step_profiled` reproduces `step` BIT-FOR-BIT (guarded by
/// `ordeal_step_profiled_matches_step` in `tests/pscale_ordeals.rs`) — the
/// timers wrap the SAME private phase methods `step` calls, in the same
/// order; nothing here changes solver semantics.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseProfile {
    /// Symplectic-Euler prediction under gravity (`integrate`), summed over substeps.
    pub integrate: Duration,
    /// The XPBD compliant distance-bond solve (`solve_distance`) — "constraint solve".
    pub solve_distance: Duration,
    /// Rigid shape-matching (`solve_shape_matching`) — ~0 for a pure bonded building.
    pub shape_matching: Duration,
    /// FLUID — the Position-Based-Fluids density-constraint pass
    /// (`solve_fluid`): neighbour grid build + λ solve + Δp apply, per substep.
    /// ~0 for a world with no fluid.
    pub solve_fluid: Duration,
    /// FLUID — the number of fluid particles processed this tick.
    pub fluid_particles: usize,
    /// Particle-vs-static-world-triangle collision (`solve_collision_normal`)
    /// — the "collision broad/narrow" pass against the ground/anchors.
    pub collision_static: Duration,
    /// Body-vs-body / fragment-vs-fragment collision (`solve_body_collisions`)
    /// — the O(k²) pass over clustered particles. The phase the lane expects
    /// to explode with N.
    pub collision_body: Duration,
    /// The per-tick flood-fill (`particle_cluster_lookup` → `fragment_components`)
    /// — "fragment flood-fill", O(particles + bonds) over each bonded group.
    pub cluster_floodfill: Duration,
    /// The remaining velocity-level passes (read-back, friction, restitution,
    /// strife tally) — grouped, not a lane-named phase but recorded so the
    /// phase sum equals the whole tick.
    pub velocity_passes: Duration,
    /// Tearing bonds whose strife overcame love (`fracture_pass`).
    pub fracture_pass: Duration,
    /// The whole tick, wall to wall (≥ the sum of the phases — the small
    /// remainder is bookkeeping between phases).
    pub total: Duration,
    /// Live particle count this tick.
    pub particles: usize,
    /// Live distance-bond count this tick (falls as bonds fracture).
    pub bonds: usize,
    /// Particles owned by SOME cluster (rigid or bonded) this tick — the
    /// population the O(k²) body pass scans.
    pub clustered_particles: usize,
    /// The number of body-vs-body pair CHECKS the O(k²) pass performed this
    /// tick: `(k choose 2) × iterations × substeps` — the honest cost driver.
    pub body_pair_checks: u64,
}

/// The world's dials. Defaults are declared here, once — every field is a
/// parameter (never-hardcode law).
#[derive(Clone, Copy, Debug)]
pub struct SolverConfig {
    /// The fixed tick length, in seconds. Default `1/60`. Render interpolates
    /// between poses; the solver never sees a variable dt.
    pub dt: f64,
    /// Substeps per tick — the primary quality dial. Default `8`.
    pub substeps: usize,
    /// Constraint iterations per substep. Default `1` (the Small-Steps
    /// regime: prefer substeps over iterations).
    pub iterations: usize,
    /// The gravity all matter falls under. Default `(0, -9.81, 0)`.
    pub gravity: Vec3,
    /// The fracture threshold: a bond tears when its accumulated strife
    /// exceeds `love × threshold`. Default `1.0e4` (N, at unit love).
    pub fracture_threshold: f64,
    /// The world seed — the root of all deterministic jitter (ENTROPY).
    /// Default `0`. No value drawn from it is random; all is `hash(seed, …)`.
    pub seed: u64,
    /// Polar-decomposition dials for rigid shape matching (P2). Default
    /// [`PolarConfig::default`].
    pub polar: PolarConfig,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            dt: 1.0 / 60.0,
            substeps: 8,
            iterations: 1,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            fracture_threshold: 1.0e4,
            seed: 0,
            polar: PolarConfig::default(),
        }
    }
}

/// The Elements' solver: a body of particles, the bindings between them, and
/// the clock (`tick`, the entropy coordinate). Advance it one fixed tick at a
/// time with [`Solver::step`].
#[derive(Clone, Debug)]
pub struct Solver {
    pub config: SolverConfig,
    pub particles: Particles,
    pub constraints: Vec<DistanceConstraint>,
    /// The rigid (and deformable) bodies — particle clusters held by a
    /// shape-matching constraint (P2). Solved after the distance bindings
    /// each substep, in index order.
    pub rigids: Vec<RigidBody>,
    /// The static world the particles strike, if any (P2). `None` = the free
    /// void of P1 (bonds only, nothing to hit).
    pub collider: Option<Collider>,
    /// The entropy coordinate — the tick index, the x-axis of this worldline.
    pub tick: u64,
    /// The journal of fractures written this run (append-only).
    pub fractures: Vec<FractureEvent>,
    /// VI-2 — every bonded lattice's ORIGINAL (whole, pre-fracture) particle
    /// set, one entry per `spawn_bonded_box` call, in spawn order. Used only
    /// by `particle_cluster_lookup` to find each bonded body's LIVE fragments
    /// (via `fragment_components`) so post-break shards collide against each
    /// other and against rigids (`solve_body_collisions`, generalized) —
    /// never touched by anything else, so an ordinary loose `DistanceConstraint`
    /// chain/rope/cloth built by hand (not through `spawn_bonded_box`) is
    /// unaffected and keeps costing exactly zero in that pass, as before.
    pub bonded_groups: Vec<Vec<usize>>,
    /// P-SCALE — the conservative static broadphase over `collider`'s triangle
    /// soup. Cached and rebuilt only when the collider changes (detected by
    /// [`TriangleGrid::fingerprint`]); `None` until first built or when no
    /// collider is installed. Purely an acceleration structure — it changes
    /// WHICH triangles `solve_collision_normal` visits, never the physics.
    collision_grid: Option<TriangleGrid>,
    /// P-SCALE — whether `solve_collision_normal` uses the broadphase (`true`,
    /// production) or the brute-force particle-vs-EVERY-triangle sweep
    /// (`false`). The brute path is retained ONLY as the byte-identical
    /// reference the exactness ordeals check against; flip it with
    /// [`Solver::set_collision_broadphase`].
    broadphase_enabled: bool,
    /// P-SCALE — test-only guard: when `true` (debug builds), every particle's
    /// query also narrow-phases the PRUNED triangles at its pre-resolution
    /// position and asserts each yields zero contact — proving the broadphase
    /// never drops a real contact. Off in production (adds an O(tris) audit).
    /// Flip with [`Solver::set_broadphase_audit`].
    broadphase_audit: bool,
    /// FLUID — the Position-Based-Fluids dials, `Some` once a fluid pool has
    /// been spawned ([`Solver::spawn_fluid_box`]) and its rest density
    /// calibrated. `None` = no fluid (the density pass costs exactly zero,
    /// `solve_fluid` returns immediately). One shared config for every fluid
    /// particle in the world (one substance per world, for now).
    pub fluid: Option<FluidConfig>,
    /// FLUID — the indices of the fluid particles, ASCENDING (spawn order).
    /// They carry a real mass and radius like any other, fall under gravity,
    /// and strike the static collider (pool walls) — but belong to no
    /// rigid/bonded cluster; their mutual incompressibility is the density
    /// constraint, not pairwise collision. Empty = no fluid.
    pub fluid_particles: Vec<usize>,
}

/// A collision cluster: particles in the SAME cluster never collide with
/// each other in `Solver::solve_body_collisions` (they're held together by
/// their OWN internal constraint — shape matching or surviving bonds);
/// particles in DIFFERENT clusters do. `Rigid` = a shape-matched body's index
/// in `rigids`. `Bonded` = one LIVE connected component of a
/// `spawn_bonded_box` lattice's surviving bond graph — while whole this is
/// exactly one cluster per body (no self-collision, as before VI-2); the
/// tick a fracture splits it, `fragment_components` reports 2+ components
/// and each becomes its own cluster, so shards immediately collide against
/// each other instead of free-falling through one another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClusterId {
    Rigid(usize),
    Bonded(usize),
    /// FLUID — ALL fluid particles share this one id. In `solve_body_collisions`
    /// two fluid particles are therefore SAME-cluster and skip the pairwise
    /// contact (their incompressibility is the density constraint, not a
    /// collision), while a fluid particle vs a rigid/bonded particle is a
    /// DIFFERENT-cluster pair and DOES contact — that pairwise push is the
    /// two-way fluid↔solid coupling (buoyancy, splash displacement).
    Fluid,
}

impl Solver {
    pub fn new(config: SolverConfig) -> Self {
        Solver {
            config,
            particles: Particles::new(),
            constraints: Vec::new(),
            rigids: Vec::new(),
            collider: None,
            tick: 0,
            fractures: Vec::new(),
            bonded_groups: Vec::new(),
            collision_grid: None,
            broadphase_enabled: true,
            broadphase_audit: false,
            fluid: None,
            fluid_particles: Vec::new(),
        }
    }

    /// P-SCALE — enable (default) or disable the collision broadphase. Disabled
    /// = the brute-force particle-vs-every-triangle reference sweep; used by
    /// the exactness ordeals to prove the two paths are byte-identical.
    pub fn set_collision_broadphase(&mut self, on: bool) {
        self.broadphase_enabled = on;
    }

    /// P-SCALE — enable the debug-only pruned-pair zero-contact audit (see
    /// [`Solver::broadphase_audit`]). No effect in release builds.
    pub fn set_broadphase_audit(&mut self, on: bool) {
        self.broadphase_audit = on;
    }

    /// P-SCALE — the live collision-grid stats for evidence/derivation:
    /// `(triangle_count, (nx, ny, nz), cell_size_m)`, or `None` when no grid
    /// is built (broadphase off or no collider). `ensure_collision_grid` must
    /// have run (it runs at each `step`).
    pub fn collision_grid_stats(&self) -> Option<(usize, (usize, usize, usize), f64)> {
        self.collision_grid
            .as_ref()
            .map(|g| (g.triangle_count, g.resolution(), g.cell_size()))
    }

    /// P-SCALE — force the collision grid to be (re)built now for the current
    /// collider (test/measurement helper; `step` does this itself).
    pub fn build_collision_grid(&mut self) {
        self.ensure_collision_grid();
    }

    /// P-SCALE — the max STATIC contact reach: the largest particle collision
    /// radius plus the collider's contact margin. The per-substep travel slack
    /// is added at query time, not here. Used to derive the grid cell size.
    fn max_contact_reach(&self) -> f64 {
        let margin = self
            .collider
            .as_ref()
            .map(|c| c.material.contact_margin)
            .unwrap_or_else(|| ContactMaterial::default().contact_margin);
        let mut r: f64 = 0.0;
        for &rad in &self.particles.radius {
            r = r.max(rad);
        }
        r + margin
    }

    /// P-SCALE — ensure `collision_grid` matches the current collider. Rebuilt
    /// only when the triangle soup's fingerprint changes (colliders are static
    /// per scene); called once per tick before the substep loop.
    fn ensure_collision_grid(&mut self) {
        if !self.broadphase_enabled {
            return;
        }
        match &self.collider {
            None => self.collision_grid = None,
            Some(c) => {
                let fp = TriangleGrid::fingerprint(&c.triangles);
                let stale = match &self.collision_grid {
                    Some(g) => g.fingerprint != fp || g.triangle_count != c.triangles.len(),
                    None => true,
                };
                if stale {
                    let reach = self.max_contact_reach();
                    let cell = TriangleGrid::derive_cell_size(&c.triangles, reach);
                    self.collision_grid = Some(TriangleGrid::build(&c.triangles, cell, fp));
                }
            }
        }
    }

    /// Spawn a solid rigid BOX centred at `center`, of full extents `dims`,
    /// discretized into a `counts` (nx, ny, nz) particle lattice. Mass is
    /// DERIVED, never authored: total = `density × volume`, split evenly
    /// across the lattice (per-particle `density × cell_volume`). Each
    /// particle gets a collision `radius` (its contact thickness). Returns
    /// the new body's index in `rigids`.
    pub fn spawn_rigid_box(
        &mut self,
        center: Vec3,
        dims: Vec3,
        counts: (usize, usize, usize),
        density: f64,
        stiffness: f64,
        radius: f64,
    ) -> usize {
        let (nx, ny, nz) = (counts.0.max(1), counts.1.max(1), counts.2.max(1));
        let n_total = nx * ny * nz;
        let volume = dims.x * dims.y * dims.z;
        let particle_mass = density * volume / n_total as f64;
        // Lattice spacing: particles centred within each cell so the cluster
        // fills `dims` symmetrically about `center`.
        let step = Vec3::new(
            if nx > 1 {
                dims.x / (nx - 1) as f64
            } else {
                0.0
            },
            if ny > 1 {
                dims.y / (ny - 1) as f64
            } else {
                0.0
            },
            if nz > 1 {
                dims.z / (nz - 1) as f64
            } else {
                0.0
            },
        );
        let origin = center - dims.scale(0.5);
        let mut indices = Vec::with_capacity(n_total);
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let pos = origin
                        + Vec3::new(step.x * ix as f64, step.y * iy as f64, step.z * iz as f64);
                    let inv_mass = 1.0 / particle_mass;
                    indices.push(self.particles.add_with_radius(pos, inv_mass, radius));
                }
            }
        }
        let body = RigidBody::from_indices(&self.particles, indices, stiffness, self.config.polar);
        self.rigids.push(body);
        self.rigids.len() - 1
    }

    /// Spawn a solid rigid SPHERE centred at `center`, of the given `radius`,
    /// sampled on a `subdiv`³ lattice (every lattice point inside the sphere
    /// becomes a particle). Mass DERIVED: total = `density × (4/3)πr³`, split
    /// evenly across the retained particles. Each particle's collision extent
    /// is `particle_radius`. Returns the new body's index in `rigids`.
    pub fn spawn_rigid_sphere(
        &mut self,
        center: Vec3,
        radius: f64,
        subdiv: usize,
        density: f64,
        stiffness: f64,
        particle_radius: f64,
    ) -> usize {
        let n = subdiv.max(1);
        // Gather lattice points inside the sphere first (to count them).
        let mut points = Vec::new();
        let step = if n > 1 {
            2.0 * radius / (n - 1) as f64
        } else {
            0.0
        };
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let offset = Vec3::new(
                        -radius + step * ix as f64,
                        -radius + step * iy as f64,
                        -radius + step * iz as f64,
                    );
                    if offset.length() <= radius {
                        points.push(center + offset);
                    }
                }
            }
        }
        if points.is_empty() {
            points.push(center); // never empty — the centre always qualifies
        }
        let volume = 4.0 / 3.0 * std::f64::consts::PI * radius * radius * radius;
        let particle_mass = density * volume / points.len() as f64;
        let inv_mass = 1.0 / particle_mass;
        let indices: Vec<usize> = points
            .into_iter()
            .map(|p| self.particles.add_with_radius(p, inv_mass, particle_radius))
            .collect();
        let body = RigidBody::from_indices(&self.particles, indices, stiffness, self.config.polar);
        self.rigids.push(body);
        self.rigids.len() - 1
    }

    /// Spawn a BONDED box: a particle lattice like [`Solver::spawn_rigid_box`],
    /// but held together by nearest-neighbor [`DistanceConstraint`] bonds (one
    /// per lattice edge along +x/+y/+z) instead of a shape-matching
    /// constraint. A [`RigidBody`] carries no per-bond love/strife bookkeeping
    /// and so can never fracture (design note, VI-2); a bonded box's bonds
    /// each carry a real [`crate::constraint::Bond`] and CAN tear when strife
    /// exceeds love (`Solver::step`'s `fracture_pass` walks `self.constraints`
    /// exactly as it does for any other bond — no special-casing needed here).
    ///
    /// Mass is DERIVED exactly as `spawn_rigid_box`: total = `density × volume`,
    /// split evenly per particle. `love` is every bond's love in `[0, 1]` (use
    /// [`crate::constraint::default_bond_love`] to derive it from `density`
    /// rather than author one). `compliance` is the bonds' inverse stiffness
    /// (XPBD, `m/N`; `0.0` = rigid) — stiffness is expressed via compliance,
    /// never a separate 0..1 "rigidity" knob (VI-2 design note: a bonded body
    /// is not shape-matched, so `RigidBody`'s `stiffness` blend has no
    /// meaning here). `radius` is each particle's collision thickness.
    ///
    /// Returns the new particles' indices in LATTICE (ix, iy, iz) index order
    /// — the caller's handle for fragment bookkeeping after a break (the same
    /// order `spawn_rigid_box` uses internally, kept deterministic).
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_bonded_box(
        &mut self,
        center: Vec3,
        dims: Vec3,
        counts: (usize, usize, usize),
        density: f64,
        love: f64,
        compliance: f64,
        radius: f64,
    ) -> Vec<usize> {
        let (nx, ny, nz) = (counts.0.max(1), counts.1.max(1), counts.2.max(1));
        let n_total = nx * ny * nz;
        let volume = dims.x * dims.y * dims.z;
        let particle_mass = density * volume / n_total as f64;
        let step = Vec3::new(
            if nx > 1 {
                dims.x / (nx - 1) as f64
            } else {
                0.0
            },
            if ny > 1 {
                dims.y / (ny - 1) as f64
            } else {
                0.0
            },
            if nz > 1 {
                dims.z / (nz - 1) as f64
            } else {
                0.0
            },
        );
        let origin = center - dims.scale(0.5);
        // Lattice index -> particle index, filled in (ix, iy, iz) order — the
        // same triple-nested order spawn_rigid_box uses, so the two spawners
        // stay comparable and the caller can address a particle by its
        // lattice coordinate if it needs to.
        let mut lattice = vec![vec![vec![0usize; nz]; ny]; nx];
        let mut indices = Vec::with_capacity(n_total);
        for (ix, plane) in lattice.iter_mut().enumerate() {
            for (iy, row) in plane.iter_mut().enumerate() {
                for (iz, slot) in row.iter_mut().enumerate() {
                    let pos = origin
                        + Vec3::new(step.x * ix as f64, step.y * iy as f64, step.z * iz as f64);
                    let inv_mass = 1.0 / particle_mass;
                    let idx = self.particles.add_with_radius(pos, inv_mass, radius);
                    *slot = idx;
                    indices.push(idx);
                }
            }
        }
        // Nearest-neighbor bonds along each lattice axis. `rest` = the local
        // step length (each axis may be spaced differently for a non-cubic
        // `dims`/`counts`).
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let here = lattice[ix][iy][iz];
                    if ix + 1 < nx {
                        let there = lattice[ix + 1][iy][iz];
                        self.constraints.push(DistanceConstraint::new(
                            here, there, step.x, compliance, love,
                        ));
                    }
                    if iy + 1 < ny {
                        let there = lattice[ix][iy + 1][iz];
                        self.constraints.push(DistanceConstraint::new(
                            here, there, step.y, compliance, love,
                        ));
                    }
                    if iz + 1 < nz {
                        let there = lattice[ix][iy][iz + 1];
                        self.constraints.push(DistanceConstraint::new(
                            here, there, step.z, compliance, love,
                        ));
                    }
                }
            }
        }
        self.bonded_groups.push(indices.clone());
        indices
    }

    /// FLUID — spawn a box of FLUID particles on a cubic lattice of edge
    /// `spacing`, filling full extents `dims` centred at `center`. Each
    /// particle owns one lattice cell of fluid: its mass is DERIVED,
    /// `rest_density × spacing³` (the physical water mass of that cell), never
    /// authored — so a denser fluid is heavier per particle and a lighter one
    /// lighter, and the crate's float/sink follows the true mass ratio. `radius`
    /// is each particle's contact thickness against the pool walls and the
    /// crate. Installs (or reuses) the world's [`FluidConfig`] with the
    /// smoothing radius `h = h_factor × spacing` (default caller: `3×`, enough
    /// neighbours for a smooth density estimate). Returns the new particle
    /// indices (ascending). Call [`Solver::calibrate_fluid_rest_density`] ONCE
    /// after all fluid is spawned to fix the SPH rest density from the packing.
    pub fn spawn_fluid_box(
        &mut self,
        center: Vec3,
        dims: Vec3,
        spacing: f64,
        rest_density: f64,
        h_factor: f64,
        radius: f64,
    ) -> Vec<usize> {
        let spacing = spacing.max(f64::EPSILON);
        let nx = ((dims.x / spacing).floor() as i64 + 1).max(1) as usize;
        let ny = ((dims.y / spacing).floor() as i64 + 1).max(1) as usize;
        let nz = ((dims.z / spacing).floor() as i64 + 1).max(1) as usize;
        // Centre the actual lattice span (which may be < dims by up to one
        // spacing) about `center`.
        let span = Vec3::new(
            (nx - 1) as f64 * spacing,
            (ny - 1) as f64 * spacing,
            (nz - 1) as f64 * spacing,
        );
        let origin = center - span.scale(0.5);
        let particle_mass = rest_density * spacing * spacing * spacing;
        let inv_mass = 1.0 / particle_mass;
        let mut indices = Vec::with_capacity(nx * ny * nz);
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let pos = origin
                        + Vec3::new(
                            spacing * ix as f64,
                            spacing * iy as f64,
                            spacing * iz as f64,
                        );
                    let idx = self.particles.add_with_radius(pos, inv_mass, radius);
                    indices.push(idx);
                }
            }
        }
        if self.fluid.is_none() {
            self.fluid = Some(FluidConfig {
                h: h_factor * spacing,
                ..FluidConfig::default()
            });
        }
        self.fluid_particles.extend_from_slice(&indices);
        self.fluid_particles.sort_unstable();
        indices
    }

    /// FLUID — DERIVE the SPH rest density `ρ₀` (and the absolute CFM `ε`) from
    /// the CURRENT fluid packing, so an interior particle reports `C_i = 0` at
    /// rest by construction rather than against a plucked number. `ρ₀` is taken
    /// as the MAX density over the fluid particles at their present positions
    /// — the fullest (most interior) neighbourhood, i.e. the density of the
    /// spawn lattice at full packing. Every surface particle then reports
    /// `ρ_i < ρ₀` (`C_i < 0`, mild cohesion, balanced by the artificial
    /// pressure), interior `ρ_i ≈ ρ₀` (`C_i ≈ 0`), and only a compressed region
    /// (the crate pressing in, the hydrostatic base) reports `ρ_i > ρ₀` and
    /// pushes back — the incompressibility. The CFM `ε = cfm_relax × (interior
    /// Σ|∇C|²)` is derived from that same fullest particle. Call once, after
    /// all fluid is spawned and before stepping. No-op without fluid.
    pub fn calibrate_fluid_rest_density(&mut self) {
        let Some(mut cfg) = self.fluid else {
            return;
        };
        if self.fluid_particles.is_empty() {
            return;
        }
        let h = cfg.h;
        let cell = PointGrid::cell_size(h);
        let grid = PointGrid::build(&self.particles.pos, &self.fluid_particles, cell);
        let mut cand: Vec<u32> = Vec::new();
        let mut best_density = 0.0_f64;
        let mut best_sum_grad2 = 0.0_f64;
        for &i in &self.fluid_particles {
            let pi = self.particles.pos[i];
            grid.query_ball(pi, h, &mut cand);
            // ρ_i = Σ_j m_j W(r_ij) ; also the constraint denominator terms.
            let mut density = 0.0_f64;
            let mut grad_i = Vec3::ZERO; // Σ_j m_j ∇W_ij  (the k=i gradient, ×ρ₀ later)
            let mut sum_grad_j2 = 0.0_f64; // Σ_j |m_j ∇W_ij|²
            for &jc in &cand {
                let j = jc as usize;
                let mj = if self.particles.inv_mass[j] > 0.0 {
                    1.0 / self.particles.inv_mass[j]
                } else {
                    0.0
                };
                let r_vec = pi - self.particles.pos[j];
                density += mj * poly6(r_vec.length(), h);
                if j != i {
                    let g = spiky_grad(r_vec, h).scale(mj);
                    grad_i = grad_i + g;
                    sum_grad_j2 += g.dot(g);
                }
            }
            if density > best_density {
                best_density = density;
                // Σ_k|∇C_k|² with ρ₀ folded out (=1 here, scaled below): the
                // shape of the denominator at the fullest neighbourhood.
                best_sum_grad2 = grad_i.dot(grad_i) + sum_grad_j2;
            }
        }
        cfg.rest_density = best_density.max(f64::EPSILON);
        // ∇C = (1/ρ₀)×(mass-weighted ∇W); |∇C|² carries 1/ρ₀². The reference
        // denominator for ε is that full-neighbourhood Σ|∇C|².
        let ref_denom = best_sum_grad2 / (cfg.rest_density * cfg.rest_density);
        cfg.cfm_epsilon = cfg.cfm_relax * ref_denom;
        self.fluid = Some(cfg);
    }

    /// FLUID — the Position-Based-Fluids density-constraint pass (Macklin &
    /// Müller 2013): one positional projection per fluid particle that drives
    /// its SPH density to the rest density. Runs INSIDE the substep iteration
    /// loop like every other constraint solve — same particle arrays, same
    /// positional-correction covenant (velocity is read back from the position
    /// change afterwards). Two half-steps, each order-stable (ascending fluid
    /// index, ascending neighbour index) for byte-identical replay:
    ///
    ///   1. λ_i = −C_i / (Σ_k|∇_pk C_i|² + ε)  for every fluid particle, where
    ///      C_i = ρ_i/ρ₀ − 1 and ρ_i = Σ_j m_j W_poly6(r_ij, h).
    ///   2. Δp_i = (1/ρ₀) Σ_j (λ_i + λ_j + s_corr_ij) m_j ∇W_spiky(r_ij, h),
    ///      s_corr_ij = −k (W_poly6(r_ij)/W_poly6(Δq))ⁿ (artificial pressure).
    ///
    /// The neighbour set comes from the conservative [`PointGrid`] (cell = h),
    /// a byte-identical superset of the brute all-pairs scan. Returns the
    /// number of fluid particles processed (0 = no fluid, immediate return).
    fn solve_fluid(&mut self) -> usize {
        let Some(cfg) = self.fluid else {
            return 0;
        };
        if self.fluid_particles.is_empty() || cfg.rest_density <= 0.0 {
            return 0;
        }
        let h = cfg.h;
        let rho0 = cfg.rest_density;
        let inv_rho0 = 1.0 / rho0;
        let eps = cfg.cfm_epsilon;
        let w_dq = poly6(cfg.tensile_dq_frac * h, h).max(f64::EPSILON);
        let cell = PointGrid::cell_size(h);
        let grid = PointGrid::build(&self.particles.pos, &self.fluid_particles, cell);

        // Per-particle neighbour lists (ascending) captured once so both
        // half-steps read the SAME neighbours from the SAME positions.
        let fp = &self.fluid_particles;
        let mut neighbours: Vec<Vec<usize>> = Vec::with_capacity(fp.len());
        let mut cand: Vec<u32> = Vec::new();
        for &i in fp {
            grid.query_ball(self.particles.pos[i], h, &mut cand);
            neighbours.push(cand.iter().map(|&c| c as usize).collect());
        }

        let mass = |p: &Particles, j: usize| -> f64 {
            if p.inv_mass[j] > 0.0 {
                1.0 / p.inv_mass[j]
            } else {
                0.0
            }
        };

        // HALF-STEP 1 — λ per fluid particle.
        let mut lambda = vec![0.0_f64; fp.len()];
        for (a, &i) in fp.iter().enumerate() {
            let pi = self.particles.pos[i];
            let mut density = 0.0_f64;
            let mut grad_i = Vec3::ZERO;
            let mut sum_grad_j2 = 0.0_f64;
            for &j in &neighbours[a] {
                let mj = mass(&self.particles, j);
                let r_vec = pi - self.particles.pos[j];
                density += mj * poly6(r_vec.length(), h);
                if j != i {
                    let g = spiky_grad(r_vec, h).scale(mj * inv_rho0);
                    grad_i = grad_i + g;
                    sum_grad_j2 += g.dot(g);
                }
            }
            let c_i = density * inv_rho0 - 1.0;
            let denom = grad_i.dot(grad_i) + sum_grad_j2 + eps;
            lambda[a] = if denom > 0.0 { -c_i / denom } else { 0.0 };
        }
        // Map fluid index -> its slot in `fp`, for λ_j lookup (fp ascending).
        let slot = |j: usize| -> Option<usize> { fp.binary_search(&j).ok() };

        // HALF-STEP 2 — Δp per fluid particle, applied after all are computed
        // (Jacobi: every Δp reads the SAME pre-move positions/λ).
        let mut dp = vec![Vec3::ZERO; fp.len()];
        for (a, &i) in fp.iter().enumerate() {
            let pi = self.particles.pos[i];
            let li = lambda[a];
            let mut acc = Vec3::ZERO;
            for &j in &neighbours[a] {
                if j == i {
                    continue;
                }
                let mj = mass(&self.particles, j);
                let r_vec = pi - self.particles.pos[j];
                let lj = slot(j).map(|s| lambda[s]).unwrap_or(0.0);
                // Artificial pressure (tensile instability corrector).
                let s_corr = if cfg.tensile_k != 0.0 {
                    let ratio = poly6(r_vec.length(), h) / w_dq;
                    -cfg.tensile_k * ratio.powf(cfg.tensile_n)
                } else {
                    0.0
                };
                acc = acc + spiky_grad(r_vec, h).scale(mj * (li + lj + s_corr));
            }
            dp[a] = acc.scale(inv_rho0 * cfg.relax);
        }
        // Apply. Anchored fluid particles (none by default) stay put.
        for (a, &i) in fp.iter().enumerate() {
            if self.particles.inv_mass[i] != 0.0 {
                self.particles.pos[i] = self.particles.pos[i] + dp[a];
            }
        }
        fp.len()
    }

    /// Flood-fill connected components of `particles` over the SURVIVING bond
    /// graph (i.e. call this AFTER `fracture_pass` has removed torn bonds —
    /// `Solver::step` already does, so this reads `self.constraints` as it
    /// stands post-step). CPU-side, deterministic, index-ordered: components
    /// are discovered by BFS starting from the lowest unvisited particle
    /// index in `particles` (ascending order), and each component's own
    /// members are pushed in the order the BFS queue visits them — two
    /// identical worldlines therefore always yield byte-identical fragment
    /// partitions (the replay-determinism ordeal's bedrock).
    ///
    /// Only bonds with BOTH endpoints inside `particles` count as edges (a
    /// bonded body's own lattice does not fracture-merge with an unrelated
    /// body that happens to share no bond with it). A particle in `particles`
    /// but touched by no surviving bond is its own singleton fragment.
    pub fn fragment_components(&self, particles: &[usize]) -> Vec<Vec<usize>> {
        use std::collections::{BTreeSet, VecDeque};
        let members: BTreeSet<usize> = particles.iter().copied().collect();
        // Adjacency restricted to `members`, built once (index-ordered by
        // constraint list order — deterministic).
        let mut adjacency: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for &p in &members {
            adjacency.entry(p).or_default();
        }
        for c in &self.constraints {
            if members.contains(&c.a) && members.contains(&c.b) {
                adjacency.entry(c.a).or_default().push(c.b);
                adjacency.entry(c.b).or_default().push(c.a);
            }
        }
        let mut visited: BTreeSet<usize> = BTreeSet::new();
        let mut components = Vec::new();
        for &start in &members {
            if visited.contains(&start) {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);
            visited.insert(start);
            while let Some(p) = queue.pop_front() {
                component.push(p);
                if let Some(neighbors) = adjacency.get(&p) {
                    for &n in neighbors {
                        if visited.insert(n) {
                            queue.push_back(n);
                        }
                    }
                }
            }
            components.push(component);
        }
        components
    }

    /// Advance the world one fixed tick: substep integrate → solve bindings →
    /// read back velocity → tally strife → tear what love could not hold.
    ///
    /// The substep loop, the strife accumulation and the fracture pass are
    /// all order-stable (index order, always) — the determinism ordeal's
    /// bedrock.
    pub fn step(&mut self) {
        let cfg = self.config;
        let n = cfg.substeps.max(1);
        let dt_sub = cfg.dt / n as f64;
        let inv_dt2 = if dt_sub > 0.0 {
            1.0 / (dt_sub * dt_sub)
        } else {
            0.0
        };

        // A fresh tick: the strife readout is per-tick (reset here, read at
        // the tick's end).
        for c in &mut self.constraints {
            c.bond.strife = 0.0;
        }

        // P-SCALE: refresh the static collision broadphase if the collider
        // changed (cached; a no-op fingerprint check otherwise). Once per
        // tick, before the substep loop — the grid is invariant across
        // substeps (the collider is static during a step).
        self.ensure_collision_grid();

        // Which collision cluster (if any) owns each particle — cluster
        // membership never changes mid-tick (fracture only tears bonds, and
        // only at the very end of `step`, in `fracture_pass` below; a
        // rigid's particle set never changes at all), so this is built once,
        // not per substep.
        let particle_cluster = self.particle_cluster_lookup();

        for _sub in 0..n {
            // The velocity entering this substep — the incoming normal speed
            // the restitution pass reflects (captured before gravity/solve).
            let vel_pre = self.particles.vel.clone();
            self.integrate(dt_sub);
            self.reset_lambda();
            let mut contacts: Vec<Contact> = Vec::new();
            for _it in 0..cfg.iterations.max(1) {
                self.solve_distance(dt_sub);
                self.solve_shape_matching();
                self.solve_fluid();
                contacts = self.solve_collision_normal();
                contacts.extend(self.solve_body_collisions(&particle_cluster));
            }
            self.read_back_velocity(dt_sub);
            // Coulomb friction on VELOCITY, after the normal solve: rigid
            // bodies are held at the BODY granularity (one contact supports the
            // whole body's weight — a shape-matched rigid cannot build stacked
            // contact force at its base), loose particles each carry their own
            // weight. Then the restitution bounce.
            self.apply_friction(&contacts, dt_sub);
            self.apply_restitution(&contacts, &vel_pre);
            // The substep's constraint force = |lambda| / dt_sub² (XPBD:
            // f = lambda·∇C / Δt², ∇C is the unit axis). Accumulate the
            // strife the bond bore this substep.
            for c in &mut self.constraints {
                c.bond.strife += c.lambda.abs() * inv_dt2;
            }
        }

        self.fracture_pass();
        self.tick += 1;
    }

    /// P-SCALE — advance ONE tick exactly as [`Solver::step`], but wrap each
    /// phase in a wall-clock timer and return the per-phase [`PhaseProfile`].
    /// This mirrors `step`'s body line-for-line (same private phase methods,
    /// same order, same arithmetic) so the resulting world state is
    /// BIT-IDENTICAL to `step` — locked by `ordeal_step_profiled_matches_step`.
    /// The only additions are `Instant` reads between phases; no solver
    /// semantics change. Measurement-only: production uses `step` (zero timing
    /// overhead).
    pub fn step_profiled(&mut self) -> PhaseProfile {
        let mut prof = PhaseProfile::default();
        let tick_start = Instant::now();
        let cfg = self.config;
        let n = cfg.substeps.max(1);
        let dt_sub = cfg.dt / n as f64;
        let inv_dt2 = if dt_sub > 0.0 {
            1.0 / (dt_sub * dt_sub)
        } else {
            0.0
        };

        for c in &mut self.constraints {
            c.bond.strife = 0.0;
        }

        // P-SCALE: refresh the static broadphase (inside `total` — the
        // per-tick fingerprint check is a real, if tiny, tick cost).
        self.ensure_collision_grid();

        let t = Instant::now();
        let particle_cluster = self.particle_cluster_lookup();
        prof.cluster_floodfill += t.elapsed();

        let clustered = particle_cluster.iter().filter(|c| c.is_some()).count();
        prof.clustered_particles = clustered;
        let pairs_per_solve = (clustered as u64) * (clustered.saturating_sub(1) as u64) / 2;

        for _sub in 0..n {
            let vel_pre = self.particles.vel.clone();
            let t = Instant::now();
            self.integrate(dt_sub);
            prof.integrate += t.elapsed();
            self.reset_lambda();
            let mut contacts: Vec<Contact> = Vec::new();
            for _it in 0..cfg.iterations.max(1) {
                let t = Instant::now();
                self.solve_distance(dt_sub);
                prof.solve_distance += t.elapsed();
                let t = Instant::now();
                self.solve_shape_matching();
                prof.shape_matching += t.elapsed();
                let t = Instant::now();
                self.solve_fluid();
                prof.solve_fluid += t.elapsed();
                let t = Instant::now();
                contacts = self.solve_collision_normal();
                prof.collision_static += t.elapsed();
                let t = Instant::now();
                contacts.extend(self.solve_body_collisions(&particle_cluster));
                prof.collision_body += t.elapsed();
                prof.body_pair_checks += pairs_per_solve;
            }
            let t = Instant::now();
            self.read_back_velocity(dt_sub);
            self.apply_friction(&contacts, dt_sub);
            self.apply_restitution(&contacts, &vel_pre);
            for c in &mut self.constraints {
                c.bond.strife += c.lambda.abs() * inv_dt2;
            }
            prof.velocity_passes += t.elapsed();
        }

        let t = Instant::now();
        self.fracture_pass();
        prof.fracture_pass += t.elapsed();
        self.tick += 1;

        prof.particles = self.particles.pos.len();
        prof.bonds = self.constraints.len();
        prof.fluid_particles = self.fluid_particles.len();
        prof.total = tick_start.elapsed();
        prof
    }

    /// Symplectic-Euler prediction under gravity (anchors stand still).
    fn integrate(&mut self, dt_sub: f64) {
        let g = self.config.gravity;
        let p = &mut self.particles;
        for i in 0..p.pos.len() {
            if p.inv_mass[i] == 0.0 {
                p.prev[i] = p.pos[i];
                continue;
            }
            p.prev[i] = p.pos[i];
            p.vel[i] = p.vel[i] + g.scale(dt_sub);
            p.pos[i] = p.pos[i] + p.vel[i].scale(dt_sub);
        }
    }

    fn reset_lambda(&mut self) {
        for c in &mut self.constraints {
            c.lambda = 0.0;
        }
    }

    /// Solve every rigid body's shape-matching constraint, in index order
    /// (order-stable — the determinism ordeal's bedrock extends to rigids).
    fn solve_shape_matching(&mut self) {
        for body in &mut self.rigids {
            body.solve(&mut self.particles);
        }
    }

    /// The NORMAL half of world collision: project each penetrating particle
    /// out along the face normal. The effective contact radius carries the
    /// surface skin (`contact_margin`) so a resting contact stays live every
    /// substep. Returns the contacts (one per particle — its most recent face)
    /// for the friction and restitution passes. Disjoint-field borrow:
    /// `collider` read, `particles` written.
    fn solve_collision_normal(&mut self) -> Vec<Contact> {
        let mut contacts: Vec<Contact> = Vec::new();
        // Disjoint field borrows: `collider` (read), `collision_grid` (read),
        // `particles` (write). Bound separately so the borrow checker sees
        // three distinct fields, not three borrows of `self`.
        let collider = match &self.collider {
            Some(c) => c,
            None => return contacts,
        };
        let grid = if self.broadphase_enabled {
            self.collision_grid.as_ref()
        } else {
            None
        };
        let audit = self.broadphase_audit;
        let _ = &audit; // read in all builds; only asserted under debug
        let mat = collider.material;
        let p = &mut self.particles;

        // Apply one contact between particle `i` and triangle `ti`, recording
        // the touch. The SAME body the brute sweep runs — shared so the two
        // paths cannot drift.
        #[inline]
        fn resolve(
            p: &mut Particles,
            contacts: &mut Vec<Contact>,
            tri: &crate::collision::Triangle,
            i: usize,
            radius: f64,
            restitution: f64,
        ) {
            let depth = match tri.contact_depth(p.pos[i], radius) {
                Some(d) => d,
                None => return,
            };
            let n = tri.normal;
            p.pos[i] = p.pos[i] + n.scale(depth);
            match contacts.iter_mut().find(|c| c.particle == i) {
                Some(c) => c.normal = n,
                None => contacts.push(Contact {
                    particle: i,
                    normal: n,
                    restitution,
                }),
            }
        }

        match grid {
            // BROADPHASE PATH — visit only the candidate triangles the grid
            // returns for each particle's fat query AABB, in ASCENDING index
            // order (a subsequence of the brute 0..N order; pruned triangles
            // never contact, so the sequence of position updates is
            // byte-identical to the brute sweep).
            Some(grid) => {
                let mut cand: Vec<u32> = Vec::new();
                for i in 0..p.pos.len() {
                    if p.inv_mass[i] == 0.0 {
                        continue;
                    }
                    let radius = p.radius[i] + mat.contact_margin;
                    let p0 = p.pos[i]; // pre-resolution position, sweep origin
                    // FIRST-GUESS query reach: the contact radius, plus this
                    // substep's actual travel (|pos - prev|), plus one push-
                    // chain hop (a second `radius`). This is a PERF HINT only
                    // — enough to cover the common one-hop case in a single
                    // query. The fixpoint below GROWS it until it provably
                    // covers the particle's whole in-sweep displacement, so
                    // correctness never rests on this initial slack.
                    let travel = (p.pos[i] - p.prev[i]).length();
                    let mut reach = radius + travel + radius;
                    // FIXPOINT RE-QUERY. `Triangle::contact_depth` is a two-
                    // sided shell: one push is up to `2r` (particle behind the
                    // plane, `signed < 0`), and pushes CHAIN, so no fixed reach
                    // can bound the displacement in advance. Instead: query,
                    // run the forward sweep FROM p0 (byte-identical to the
                    // brute sweep restricted to the candidate set), then grow
                    // the reach until it covers the displacement the sweep
                    // actually produced, re-querying and re-sweeping from p0
                    // each time.
                    //
                    // SAFETY INVARIANT (the pruning proof): a query of L∞ half-
                    // extent `reach` centred at p0 returns EVERY triangle that
                    // can contact the particle anywhere its sweep path reaches,
                    // provided `reach >= dmax + radius`, where `dmax` is the max
                    // L∞ displacement from p0 over the sweep. Proof: a contact
                    // at path position `p` (‖p − p0‖∞ ≤ dmax) puts a triangle
                    // point within `radius` (L2 ≥ L∞) of `p`, hence within
                    // `dmax + radius ≤ reach` (L∞) of p0 — inside the query box,
                    // so the grid returns it (grid_query completeness ordeal).
                    // The loop exits ONLY when this holds, so the swept set is
                    // a complete superset and every pruned triangle is a proven
                    // no-op — the sweep equals brute bit-for-bit.
                    // TERMINATION. Each failed iteration strictly grows reach
                    // (`reach = dmax + radius` is only assigned when
                    // `dmax + radius > reach`, i.e. the `break` didn't fire),
                    // and the candidate set returned by `grid.query` is
                    // monotone non-decreasing in reach (a larger query box is
                    // a superset). So once an iteration's `cand` — and hence
                    // its resolved `dmax` — repeats the previous iteration's
                    // `dmax` unchanged, growth stops and `reach >= dmax +
                    // radius` holds, exiting the loop. The grid has a finite
                    // triangle count, so the candidate set (and thus the
                    // reachable dmax values) has finitely many distinct
                    // states; each failed pass either changes `cand` (adding
                    // at least one new triangle — bounded by the triangle
                    // count) or reproduces the same `dmax`, which is exactly
                    // the fixed point and exits. Loop passes are therefore
                    // bounded by N_triangles + 1 (this was previously commit-
                    // prose only — see c58cf54, ee2e8cd — now recorded here).
                    //
                    // PARKED HARDENING (not applied): a sub-ulp margin
                    // `reach = (dmax + radius) * (1.0 + f64::EPSILON)` was
                    // considered to guard the `>=` comparison against a
                    // float rounding edge where the true fixed point sits
                    // exactly on the reach boundary and rounds the wrong way.
                    // Not applied here — solver semantics stay in the
                    // builder's domain (this loop's contract is geometric
                    // completeness, not float-ulp defense) — parked for the
                    // next physics wave to pick up if such an edge is ever
                    // observed in practice.
                    let mut last_normal: Option<Vec3>;
                    loop {
                        let r = Vec3::new(reach, reach, reach);
                        grid.query(p0 - r, p0 + r, &mut cand);

                        // Forward sweep from p0 over the candidates (ascending
                        // index = a subsequence of the brute 0..N order). Reset
                        // to p0 so a re-query re-sweeps from the same origin.
                        p.pos[i] = p0;
                        last_normal = None;
                        let mut dmax = 0.0_f64;
                        for &ti in &cand {
                            let tri = &collider.triangles[ti as usize];
                            if let Some(depth) = tri.contact_depth(p.pos[i], radius) {
                                p.pos[i] = p.pos[i] + tri.normal.scale(depth);
                                last_normal = Some(tri.normal);
                                let d = p.pos[i] - p0;
                                let dl = d.x.abs().max(d.y.abs()).max(d.z.abs());
                                if dl > dmax {
                                    dmax = dl;
                                }
                            }
                        }

                        // Complete iff reach covers the realised displacement
                        // plus one contact radius (the invariant above). If so,
                        // stop; else grow to exactly the needed reach and redo.
                        if reach >= dmax + radius {
                            break;
                        }
                        reach = dmax + radius;
                    }

                    // TEST-ONLY AUDIT (strengthened per the adversary): every
                    // PRUNED triangle, narrow-phased at the POST-resolution
                    // position (not just the pre-resolution start — that was
                    // blind to contacts a push creates), must yield zero
                    // contact. With the fixpoint's invariant this can never
                    // trip; the audit is the belt to its braces.
                    #[cfg(debug_assertions)]
                    if audit {
                        let end = p.pos[i];
                        for (ti, tri) in collider.triangles.iter().enumerate() {
                            if cand.binary_search(&(ti as u32)).is_ok() {
                                continue;
                            }
                            debug_assert!(
                                tri.contact_depth(p0, radius).is_none()
                                    && tri.contact_depth(end, radius).is_none(),
                                "broadphase pruned triangle {ti} that DOES contact particle {i} \
                                 (reach {reach}) — the query margin is too tight"
                            );
                        }
                    }

                    // Record the contact (last resolved normal), matching the
                    // brute path's per-particle single Contact.
                    if let Some(n) = last_normal {
                        match contacts.iter_mut().find(|c| c.particle == i) {
                            Some(c) => c.normal = n,
                            None => contacts.push(Contact {
                                particle: i,
                                normal: n,
                                restitution: mat.restitution,
                            }),
                        }
                    }
                }
            }
            // BRUTE PATH — the byte-identical reference: every particle vs
            // every triangle, in index order.
            None => {
                for i in 0..p.pos.len() {
                    if p.inv_mass[i] == 0.0 {
                        continue;
                    }
                    let radius = p.radius[i] + mat.contact_margin;
                    for tri in &collider.triangles {
                        resolve(p, &mut contacts, tri, i, radius, mat.restitution);
                    }
                }
            }
        }
        contacts
    }

    /// Which collision cluster (if any) owns each particle, indexed by
    /// particle index. `None` = a free particle belonging to neither a rigid
    /// body nor a `spawn_bonded_box` lattice (an ordinary hand-built
    /// `DistanceConstraint` chain/rope/cloth particle, say) — excluded from
    /// `solve_body_collisions` exactly as before VI-2, so those scenes' cost
    /// and behavior are unchanged.
    ///
    /// Bonded clusters are numbered by flood-filling EACH registered bonded
    /// group's OWN surviving bonds independently (`fragment_components`,
    /// deterministic BFS) and offsetting the running id by the groups seen so
    /// far — collisions are checked by `ClusterId` equality, never the raw
    /// number, so the offset only needs to keep different groups' components
    /// from colliding under numerically-equal ids; it carries no other
    /// meaning.
    fn particle_cluster_lookup(&self) -> Vec<Option<ClusterId>> {
        let mut lookup = vec![None; self.particles.pos.len()];
        for (body_idx, body) in self.rigids.iter().enumerate() {
            for &i in &body.indices {
                lookup[i] = Some(ClusterId::Rigid(body_idx));
            }
        }
        let mut next_bonded_id = 0usize;
        for group in &self.bonded_groups {
            for component in self.fragment_components(group) {
                for i in component {
                    lookup[i] = Some(ClusterId::Bonded(next_bonded_id));
                }
                next_bonded_id += 1;
            }
        }
        // FLUID — every fluid particle shares the single `Fluid` cluster, so
        // fluid–fluid pairs SKIP the O(k²) contact (density constraint owns
        // them) while fluid–solid pairs collide (the coupling). Assigned last
        // so a particle is never both fluid and a rigid/bonded member.
        for &i in &self.fluid_particles {
            lookup[i] = Some(ClusterId::Fluid);
        }
        lookup
    }

    /// The BODY-vs-BODY half of collision: two distinct collision clusters'
    /// particles cannot occupy the same space — two shape-matched rigids (a
    /// stacked crate must rest on the one below it, not fall through it), a
    /// rigid and a bonded lattice/fragment, or (VI-2) two fragments of the
    /// SAME broken bonded body once it has split. Brute-force over every
    /// particle pair not sharing a cluster — adequate at this atom's
    /// particle/body counts (a handful of low-resolution lattices); a spatial
    /// broad-phase is future work once a scene needs many more bodies than a
    /// small stack.
    ///
    /// The resting gap is DERIVED from the static (particle-vs-triangle) pass'
    /// convention, not asserted independently. The static pass rests a
    /// particle at `radius + contact_margin` from a face (`solve_collision_
    /// normal`, above): a SINGLE radius, because a triangle has none of its
    /// own. Body-vs-body has TWO radii (`r_i`, `r_j`), possibly unequal, and
    /// the generalization must reduce EXACTLY to the static gap when
    /// `r_i == r_j == r` (a same-radius body resting on a same-radius body
    /// should feel the same skin as that body resting on the static world).
    /// The SUM `r_i + r_j` fails that reduction (`r + r = 2r ≠ r`); the MEAN
    /// `(r_i + r_j) / 2` reduces correctly (`(r + r) / 2 = r`). So:
    /// `gap = mean(r_i, r_j) + contact_margin`. `contact_margin` is read from
    /// the SAME `ContactMaterial` field the static pass reads (falling back
    /// to the crate's documented default when no static collider is
    /// installed at all, e.g. two bodies colliding in free space) — never a
    /// second hardcoded margin.
    fn solve_body_collisions(&mut self, particle_cluster: &[Option<ClusterId>]) -> Vec<Contact> {
        let mut contacts = Vec::new();
        // Only particles OWNED BY A CLUSTER (a rigid, or a spawn_bonded_box
        // lattice/fragment) are candidates — restricts the brute-force scan
        // to (Σ per-cluster particle counts), never the whole free-particle
        // population. An ordinary hand-built DistanceConstraint chain/rope/
        // cloth particle (never a rigid, never spawned by spawn_bonded_box)
        // has no cluster and is excluded here exactly as before VI-2 — that
        // scene shape's O(n²) cost stays at exactly zero.
        let clustered_particles: Vec<usize> = (0..particle_cluster.len())
            .filter(|&i| particle_cluster[i].is_some())
            .collect();
        // Fewer than two clustered particles from DIFFERENT clusters can't
        // happen without at least two clusters existing; cheapest short
        // circuit is just "any candidates at all" — the inner loop's own
        // same-cluster skip handles the rest with no separate count needed.
        if clustered_particles.len() < 2 {
            return contacts;
        }
        let restitution = match &self.collider {
            Some(c) => c.material.restitution,
            None => 0.0,
        };
        // F2/F3: the SAME contact_margin the static pass reads (see the
        // doc-comment above) — falls back to the crate's documented default
        // ContactMaterial when no static collider exists at all (two bodies
        // colliding in free space still get the same surface skin a
        // resting contact needs, not zero).
        let contact_margin = match &self.collider {
            Some(c) => c.material.contact_margin,
            None => ContactMaterial::default().contact_margin,
        };
        let p = &mut self.particles;
        for (a, &i) in clustered_particles.iter().enumerate() {
            let wi = p.inv_mass[i];
            for &j in &clustered_particles[(a + 1)..] {
                if particle_cluster[i] == particle_cluster[j] {
                    continue; // same cluster — held by its own constraint, not collision
                }
                let wj = p.inv_mass[j];
                let w = wi + wj;
                if w == 0.0 {
                    continue; // two anchors — nothing to push apart
                }
                let delta = p.pos[i] - p.pos[j];
                let dist = delta.length();
                // mean(r_i, r_j) + contact_margin — see the doc-comment above
                // for the derivation from the static single-radius convention.
                let min_dist = (p.radius[i] + p.radius[j]) * 0.5 + contact_margin;
                if dist >= min_dist || dist <= 0.0 {
                    continue;
                }
                let normal = delta.scale(1.0 / dist);
                let depth = min_dist - dist;
                p.pos[i] = p.pos[i] + normal.scale(depth * (wi / w));
                p.pos[j] = p.pos[j] - normal.scale(depth * (wj / w));
                contacts.push(Contact {
                    particle: i,
                    normal,
                    restitution,
                });
                contacts.push(Contact {
                    particle: j,
                    normal: normal.scale(-1.0),
                    restitution,
                });
            }
        }
        contacts
    }

    /// Coulomb friction as a change to a contact velocity. `v` is the velocity
    /// under contact, `n` the outward surface normal, `g` the world gravity,
    /// `dt` the substep. The surface supports the normal gravity `g_n = −g·n`;
    /// friction opposes the tangential velocity `v_t`. STICK (return `−v_t`,
    /// killing the slip) when the body is essentially at rest AND the
    /// tangential DRIVE cannot overcome stiction (`|g_t| ≤ μ_s·g_n` — the
    /// geometric Coulomb law `tanθ ≤ μ_s`); otherwise KINETIC, bleeding at most
    /// `μ_d·g_n·dt` of speed opposing motion. `g_n ≤ 0` (gravity not pressing
    /// into this face) yields no friction.
    fn coulomb_dv(v: Vec3, n: Vec3, g: Vec3, dt: f64, mat: &ContactMaterial) -> Vec3 {
        let g_n = -(g.dot(n));
        if g_n <= 0.0 {
            return Vec3::ZERO;
        }
        let v_t = v - n.scale(v.dot(n));
        let vt_len = v_t.length();
        if vt_len <= 0.0 {
            return Vec3::ZERO;
        }
        let g_t = (g - n.scale(g.dot(n))).length();
        let stick_capacity = mat.friction_static * g_n * dt;
        if vt_len <= stick_capacity && g_t <= mat.friction_static * g_n {
            v_t.scale(-1.0) // stiction: hold
        } else {
            let bleed = (mat.friction_dynamic * g_n * dt).min(vt_len);
            v_t.scale(-bleed / vt_len) // kinetic: bounded drag
        }
    }

    /// The friction pass (velocity level, after read-back). RIGID bodies are
    /// held at the BODY granularity: the whole body's mass-weighted velocity
    /// gets one Coulomb correction (its base contact supports the entire body
    /// weight — per-particle base penetration cannot, a shape-matched rigid
    /// builds no stacked contact pressure). Loose (non-rigid) contact
    /// particles each carry their own weight and are corrected individually.
    fn apply_friction(&mut self, contacts: &[Contact], dt_sub: f64) {
        let mat = match &self.collider {
            Some(c) => c.material,
            None => return,
        };
        let g = self.config.gravity;
        // Which particles belong to a rigid body (handled at body level).
        let mut in_rigid = vec![false; self.particles.pos.len()];
        for body in &self.rigids {
            for &i in &body.indices {
                in_rigid[i] = true;
            }
        }
        // Rigid bodies — one body-level correction each.
        for body in &self.rigids {
            // Average contact normal + mass-weighted velocity over the body's
            // contact particles (index order — determinism).
            let mut n_sum = Vec3::ZERO;
            let mut v_sum = Vec3::ZERO;
            let mut m_sum = 0.0;
            let mut touched = 0u32;
            for c in contacts {
                if let Some(k) = body.indices.iter().position(|&i| i == c.particle) {
                    n_sum = n_sum + c.normal;
                    let i = c.particle;
                    let m = body.masses[k];
                    v_sum = v_sum + self.particles.vel[i].scale(m);
                    m_sum += m;
                    touched += 1;
                }
            }
            if touched == 0 || m_sum <= 0.0 {
                continue;
            }
            let n = match n_sum.normalized() {
                Some(n) => n,
                None => continue,
            };
            let v_body = v_sum.scale(1.0 / m_sum);
            let dv = Self::coulomb_dv(v_body, n, g, dt_sub, &mat);
            if dv.x == 0.0 && dv.y == 0.0 && dv.z == 0.0 {
                continue;
            }
            for &i in &body.indices {
                if self.particles.inv_mass[i] != 0.0 {
                    self.particles.vel[i] = self.particles.vel[i] + dv;
                }
            }
        }
        // Loose particles — each carries its own weight.
        for c in contacts {
            let i = c.particle;
            if in_rigid[i] || self.particles.inv_mass[i] == 0.0 {
                continue;
            }
            let dv = Self::coulomb_dv(self.particles.vel[i], c.normal, g, dt_sub, &mat);
            self.particles.vel[i] = self.particles.vel[i] + dv;
        }
    }

    /// The restitution velocity pass. The position solve has (inelastically)
    /// killed each contact's normal velocity; here we ADD the bounce back:
    /// outgoing normal speed = `e × |incoming|`, using the velocity captured
    /// before this substep's integrate. One application per particle (a corner
    /// meeting two faces bounces once).
    fn apply_restitution(&mut self, contacts: &[Contact], vel_pre: &[Vec3]) {
        let p = &mut self.particles;
        let mut bounced = vec![false; p.pos.len()];
        for c in contacts {
            let i = c.particle;
            if bounced[i] {
                continue;
            }
            let vn_pre = vel_pre[i].dot(c.normal);
            if vn_pre < 0.0 {
                // moving into the surface — reflect scaled by restitution
                let vn = p.vel[i].dot(c.normal);
                let target = -c.restitution * vn_pre;
                p.vel[i] = p.vel[i] + c.normal.scale(target - vn);
                bounced[i] = true;
            }
        }
    }

    /// One Gauss-Seidel sweep of the compliant distance bindings. The XPBD
    /// update: `Δλ = (−C − α̃·λ) / (w + α̃)`, positions nudged by
    /// `±w·Δλ·n`. `α̃ = compliance / dt_sub²`.
    fn solve_distance(&mut self, dt_sub: f64) {
        let inv_dt2 = if dt_sub > 0.0 {
            1.0 / (dt_sub * dt_sub)
        } else {
            0.0
        };
        let p = &mut self.particles;
        for c in &mut self.constraints {
            let wa = p.inv_mass[c.a];
            let wb = p.inv_mass[c.b];
            let w = wa + wb;
            if w == 0.0 {
                continue; // two anchors — nothing to move
            }
            let delta = p.pos[c.a] - p.pos[c.b];
            let n = match delta.normalized() {
                Some(n) => n,
                None => continue, // coincident: no axis to pull along
            };
            let cval = delta.length() - c.rest;
            let a_tilde = c.compliance * inv_dt2;
            let d_lambda = (-cval - a_tilde * c.lambda) / (w + a_tilde);
            c.lambda += d_lambda;
            let corr = n.scale(d_lambda);
            p.pos[c.a] = p.pos[c.a] + corr.scale(wa);
            p.pos[c.b] = p.pos[c.b] - corr.scale(wb);
        }
    }

    /// Read velocity back from the position change — the PBD covenant
    /// `v = (x − x_prev) / dt`. This is where XPBD's characteristic
    /// numerical damping enters (documented, not denied).
    fn read_back_velocity(&mut self, dt_sub: f64) {
        if dt_sub <= 0.0 {
            return;
        }
        let inv = 1.0 / dt_sub;
        let p = &mut self.particles;
        for i in 0..p.pos.len() {
            if p.inv_mass[i] == 0.0 {
                continue;
            }
            p.vel[i] = (p.pos[i] - p.prev[i]).scale(inv);
        }
    }

    /// Tear the bindings whose strife overcame their love. Order-stable:
    /// broken bonds are collected in index order, recorded to the journal,
    /// then removed by `retain` (which preserves order).
    fn fracture_pass(&mut self) {
        let threshold = self.config.fracture_threshold;
        let tick = self.tick;
        let mut any = false;
        for c in &self.constraints {
            if c.bond.fractured(threshold) {
                self.fractures.push(FractureEvent {
                    tick,
                    a: c.a,
                    b: c.b,
                    strife: c.bond.strife,
                    love: c.bond.love,
                });
                any = true;
            }
        }
        if any {
            let t = threshold;
            self.constraints.retain(|c| !c.bond.fractured(t));
        }
    }

    /// The observable state's fingerprint at the current tick.
    pub fn state_hash(&self) -> u64 {
        self.particles.state_hash()
    }

    /// Apply an instantaneous velocity change to every particle of the rigid
    /// body at `rigid_index` — "the op is the hand": the caller (incantation
    /// layer) chooses `delta_velocity`, the solver never invents a magnitude.
    /// Anchors (`inv_mass == 0`) are left untouched, matching every other
    /// velocity pass in this file. A no-op if `rigid_index` is out of range
    /// (the caller's binding may have been fractured away).
    pub fn apply_impulse(&mut self, rigid_index: usize, delta_velocity: Vec3) {
        let Some(body) = self.rigids.get(rigid_index) else {
            return;
        };
        self.apply_impulse_to_particles(&body.indices.clone(), delta_velocity);
    }

    /// Apply an instantaneous velocity change to an explicit particle set —
    /// the generalization `apply_impulse` (rigid-only) delegates to. VI-2's
    /// bonded lattices have no `RigidBody` (see `spawn_bonded_box`'s doc), so
    /// a bonded body's impulse (e.g. an authored drop's lateral/angular
    /// component — the "op is the hand" seam, never solver-invented) is
    /// applied here directly by particle index instead. Same law as the
    /// rigid path: anchors (`inv_mass == 0`) are left untouched.
    pub fn apply_impulse_to_particles(&mut self, particles: &[usize], delta_velocity: Vec3) {
        for &i in particles {
            if self.particles.inv_mass[i] != 0.0 {
                self.particles.vel[i] = self.particles.vel[i] + delta_velocity;
            }
        }
    }

    /// Apply a RIGID-BODY-style spin (uniform angular velocity `ω` about
    /// `center`) as an instantaneous per-particle velocity change:
    /// `v += ω × (pos - center)`, the standard rigid-rotation velocity
    /// field. VI-2 uses this to give an authored bonded lattice a tumble
    /// before it falls — unlike a uniform `apply_impulse_to_particles` delta
    /// (which moves every particle the same way and so cannot, by itself,
    /// stress any bond), a spin puts particles on OPPOSITE sides of `center`
    /// moving in OPPOSITE directions, which is exactly what makes the
    /// lattice's own bonds carry real internal strife even before any
    /// external contact — an honest way to make a drop's impact (and its
    /// resulting fracture) asymmetric, driven by physics the solver already
    /// has, not a second staged effect.
    pub fn apply_spin_to_particles(
        &mut self,
        particles: &[usize],
        center: Vec3,
        angular_velocity: Vec3,
    ) {
        for &i in particles {
            if self.particles.inv_mass[i] != 0.0 {
                let r = self.particles.pos[i] - center;
                self.particles.vel[i] = self.particles.vel[i] + angular_velocity.cross(r);
            }
        }
    }
}
