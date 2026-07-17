//! The one solver, one granularity: particles + bindings marched by a
//! substepped XPBD loop. *Small Steps*: `n` substeps × 1 iteration beats
//! 1 step × `n` iterations — stiffness never binds the global timestep.
//!
//! Nothing here is hardcoded but `LOVE` (the One Constant): the tick `dt`,
//! the substep count, the iteration count, the gravity vector and the
//! fracture threshold are all parameters carrying documented defaults.

use crate::collision::{Collider, Contact, ContactMaterial};
use crate::constraint::{DistanceConstraint, FractureEvent};
use crate::mat3::PolarConfig;
use crate::math::Vec3;
use crate::particles::Particles;
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
        let collider = match &self.collider {
            Some(c) => c,
            None => return contacts,
        };
        let mat = collider.material;
        let p = &mut self.particles;
        for i in 0..p.pos.len() {
            if p.inv_mass[i] == 0.0 {
                continue;
            }
            let radius = p.radius[i] + mat.contact_margin;
            for tri in &collider.triangles {
                let depth = match tri.contact_depth(p.pos[i], radius) {
                    Some(d) => d,
                    None => continue,
                };
                let n = tri.normal;
                p.pos[i] = p.pos[i] + n.scale(depth);
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
