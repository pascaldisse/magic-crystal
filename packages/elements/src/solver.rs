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

        // Which rigid (if any) owns each particle — rigid membership never
        // changes mid-tick (fracture tears distance bonds, never a rigid's
        // particle set), so this is built once, not per substep.
        let particle_rigid = self.particle_rigid_lookup();

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
                contacts.extend(self.solve_body_collisions(&particle_rigid));
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

    /// Which rigid body (if any) owns each particle, indexed by particle
    /// index. `None` = a free (non-rigid) particle.
    fn particle_rigid_lookup(&self) -> Vec<Option<usize>> {
        let mut lookup = vec![None; self.particles.pos.len()];
        for (body_idx, body) in self.rigids.iter().enumerate() {
            for &i in &body.indices {
                lookup[i] = Some(body_idx);
            }
        }
        lookup
    }

    /// The BODY-vs-BODY half of collision: two RIGID bodies' particles cannot
    /// occupy the same space (a stacked crate must rest on the one below it,
    /// not fall through it). Brute-force over every particle pair not owned
    /// by the same rigid — adequate at this atom's particle/body counts (a
    /// handful of low-resolution lattices); a spatial broad-phase is future
    /// work once a scene needs many more bodies than a small stack.
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
    fn solve_body_collisions(&mut self, particle_rigid: &[Option<usize>]) -> Vec<Contact> {
        let mut contacts = Vec::new();
        // Fewer than two rigids ⇒ no body-vs-body pair can exist; free
        // (non-rigid) particles never collide with each other here (that was
        // never this pass's job, and skipping it keeps a chain/cloth scene's
        // O(n²) cost at exactly zero instead of scanning every free pair).
        if self.rigids.len() < 2 {
            return contacts;
        }
        // Only particles OWNED BY A RIGID are candidates — restricts the
        // brute-force scan to (Σ per-body particle counts), never the whole
        // free-particle population.
        let rigid_particles: Vec<usize> = (0..particle_rigid.len())
            .filter(|&i| particle_rigid[i].is_some())
            .collect();
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
        for (a, &i) in rigid_particles.iter().enumerate() {
            let wi = p.inv_mass[i];
            for &j in &rigid_particles[(a + 1)..] {
                if particle_rigid[i] == particle_rigid[j] {
                    continue; // same body — held by shape matching, not collision
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
        for &i in &body.indices {
            if self.particles.inv_mass[i] != 0.0 {
                self.particles.vel[i] = self.particles.vel[i] + delta_velocity;
            }
        }
    }
}
