//! PHYSICS INTO THE WORLD (P3) — the seam that binds the Elements' rigid solver
//! to a living realm. Realm data declares a `body` sigil on a vessel; the world
//! tick advances the [`elements`] solver for every declared body and collides it
//! against the STATIC realm mesh (the same triangle soup the traced light sees).
//! The moved body writes its pose back to the ECS `transform`, so its triangles
//! ride the dynamic BVH splice ([`crate::scene::Dynamics`]) and the Pleroma
//! sees it move.
//!
//! GENERIC by construction: nothing here names naruko or any realm. The `body`
//! fields are plain English, every solver dial is a parameter with a documented
//! default (never-hardcode), and the driver is inert when no body is declared
//! (zero-physics realms render byte-unchanged).
//!
//! ULTRADETERMINISM: the solver is `f64`, order-stable, seeded — two identical
//! runs fold to byte-identical [`Physics::state_hash`]. Time is the tick index
//! (entropy), never wall time.

use elements::{Collider, ContactMaterial, Mat3, Solver, SolverConfig, Triangle, Vec3};
use serde::Deserialize;

/// VI-2 — `Physics::poll_bonded`'s "still whole" half: one `(gaia_id,
/// live_centroid)` per bonded body that has not yet fractured.
type StillWholePoses = Vec<(String, [f64; 3])>;
/// VI-2 — `Physics::poll_bonded`'s "newly broken" half: one
/// `(parent_gaia_id, fragments, cube_size)` per bonded body that fractured
/// THIS tick.
type NewlyBrokenFragments = Vec<(String, Vec<fracture::Fragment>, f64)>;

/// The `body` sigil — realm data declaring a vessel as physical matter the
/// world tick simulates. Every field is plain English with a documented
/// default; only `shape` selects the discretization, the rest are solver dials.
///
/// F2 — `deny_unknown_fields`: a rigid-body sigil (`shape` + solver dials) only.
/// A `preset` (skinned vessel) is NOT a Body field — it never reaches this parse
/// (the `from_ecs` weld routes preset bodies to the RITE V skinned path and
/// refuses `{preset, shape}` outright), and an unknown key (typo'd dial) is a
/// LOUD error, never a silently-defaulted body.
#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Body {
    /// The matter's shape. `"box"` (the only P3 shape) fills a lattice box.
    pub shape: String,
    /// Full extents of the body in metres (a box's width/height/depth).
    pub size: [f64; 3],
    /// Material density in kg/m³ — mass is DERIVED (`density × volume`), never
    /// authored. Default `500` (seasoned softwood — a wooden crate).
    pub density: f64,
    /// The particle lattice the body is discretized into (nx, ny, nz). More
    /// particles = finer contact, higher cost. Default `[2, 2, 2]` (the eight
    /// corners — enough for a box resting flat on a plane).
    pub resolution: [usize; 3],
    /// Each particle's contact thickness against the world, in metres. Default
    /// `0.05` (5 cm). The body rests with its base particles this far above a
    /// supporting face.
    pub contact_radius: f64,
    /// Shape-match stiffness on `[0, 1]`: `1.0` = perfectly rigid, `< 1.0` =
    /// deformable. Default `1.0` (a rigid crate). Ignored when `bonded` is
    /// true (a bonded body is not shape-matched — see `bonded`'s doc).
    pub rigidity: f64,
    /// VI-2 — SOMETHING BREAKS: `true` makes this body a BONDED lattice
    /// (nearest-neighbor [`elements::DistanceConstraint`] bonds, each
    /// carrying a real love/strife [`elements::Bond`]) instead of a
    /// shape-matched [`elements::RigidBody`]. Only a bonded body can
    /// fracture — a shape-matched rigid keeps no per-bond bookkeeping to
    /// tear (see `elements::Solver::spawn_bonded_box`'s doc). Default
    /// `false` (every EXISTING scene's bodies stay rigid, byte-unchanged).
    pub bonded: bool,
    /// A bonded body's per-bond love in `[0, 1]`, or `None` to DERIVE it
    /// from `density` via [`elements::default_bond_love`] (the essence
    /// rule: `density` stands in for the material's essence — stone >
    /// wood > glass, GRIMOIRE). Ignored when `bonded` is false.
    pub love: Option<f64>,
    /// A bonded body's bond compliance (XPBD inverse stiffness, `m/N`;
    /// `0.0` = rigid). Default `1e-7` — near-rigid (matches the "nearly-
    /// rigid" compliance the elements ordeals already use for a stiff
    /// chain link, see `packages/elements/tests/ordeals.rs`'s comment on
    /// its own `1.0e-6` — one order tighter here since a crate's own bonds
    /// should read stiffer than a hanging chain's links). Ignored when
    /// `bonded` is false.
    pub compliance: f64,
    /// VI-2 — a bonded body's AUTHORED initial angular velocity (rad/s about
    /// its own spawn centroid), applied once at spawn via
    /// [`elements::Solver::apply_spin_to_particles`] — never a solver-
    /// invented magnitude (the "op is the hand" law: the scene author
    /// chooses it, same footing as an `Op::Impulse`'s `delta_velocity`, just
    /// applied at t=0 instead of mid-run since a spin needs a per-particle
    /// velocity FIELD, not a single delta an `Op::Impulse` can carry). A
    /// tumbling drop hits its target corner-first, which is what actually
    /// stresses a lattice's bonds ASYMMETRICALLY — see
    /// `apply_spin_to_particles`'s doc for why a uniform impulse alone
    /// cannot do this. Default `[0, 0, 0]` (no spin — every EXISTING bonded
    /// declaration keeps falling exactly as before). Ignored when `bonded`
    /// is false.
    pub spin: [f64; 3],
    /// VI-2 — a bonded body's AUTHORED initial linear velocity (m/s),
    /// applied once at spawn via [`elements::Solver::apply_impulse_to_
    /// particles`] — the uniform counterpart to `spin` (same "op is the
    /// hand" law: an authored magnitude, never solver-invented). A drop
    /// with sideways motion strikes its target OFF-AXIS: the leading
    /// particles decelerate against the surface while trailing ones still
    /// carry momentum, a shear a purely vertical drop can never produce
    /// (gravity alone is symmetric about the vertical through the crate's
    /// centroid, so a vertical-only drop's fragments retain only
    /// `spin`-induced velocity, which is itself antisymmetric about that
    /// same centroid and can largely cancel back into a flat, merely-
    /// crushed-looking settle instead of a genuine scatter). Default
    /// `[0, 0, 0]` (no drift — every EXISTING bonded declaration keeps
    /// falling exactly as before). Ignored when `bonded` is false.
    pub initial_velocity: [f64; 3],
    /// PLAYGROUND — settle-window ticks a BONDED body is pre-relaxed for at
    /// install with fracture DISARMED, then re-armed (the canonical
    /// `elements::building::settle` gesture). A near-rigid lattice spikes a
    /// huge transient constraint force in its first substeps — a numerical
    /// startup shock, not real load — that would spuriously tear a resting
    /// crate before a hand ever touched it. A crate the player walks up to
    /// and PUNCHES must sit whole at rest, so it declares e.g. `settle: 90`;
    /// the default `0` leaves every EXISTING body (VI-2's impact-break drop
    /// included) byte-unchanged. Ignored when `bonded` is false.
    pub settle: u64,
    /// PLAYABLE DESTRUCTION — anchor the bonded lattice's LOWEST layer to the
    /// world (`inv_mass = 0`), the SAME technique `elements::building::erect`
    /// uses for a load-bearing tower: an anchored base holds the WHOLE static
    /// load in its base-layer bonds (support-shear) instead of bare
    /// particle-vs-collider contact, so a resting structure stands rock-solid
    /// under its own weight (contact alone lets a tall thin lattice creep or
    /// lean under a stacked load) — yet a hard enough shove on the body
    /// still tears those same base bonds, same door as any other bonded
    /// break. Default `false` (every EXISTING bonded declaration keeps
    /// resting on contact alone, byte-unchanged). Ignored when `bonded` is
    /// false.
    #[serde(default)]
    pub anchor_base: bool,
}

impl Default for Body {
    fn default() -> Self {
        Self {
            shape: "box".to_string(),
            size: [1.0, 1.0, 1.0],
            density: 500.0,
            resolution: [2, 2, 2],
            contact_radius: 0.05,
            rigidity: 1.0,
            bonded: false,
            love: None,
            compliance: 1.0e-7,
            spin: [0.0; 3],
            initial_velocity: [0.0; 3],
            settle: 0,
            anchor_base: false,
        }
    }
}

/// A declared body wired into the solver: which vessel it animates and its
/// index into the solver's rigid bodies.
#[derive(Clone, Debug)]
pub struct BodyBinding {
    /// The owning vessel's gaia id — the write-back handle.
    pub gaia_id: String,
    /// Index into [`elements::Solver::rigids`].
    pub rigid: usize,
    /// The body's half-height (size.y / 2), kept for rest derivation.
    pub half_height: f64,
    /// The particle contact radius, kept for rest derivation.
    pub contact_radius: f64,
}

/// A body's current world pose, read from the solver after a step.
#[derive(Clone, Copy, Debug)]
pub struct BodyPose {
    /// The mass-weighted centroid — the vessel's world-space position.
    pub position: [f64; 3],
    /// The fitted rigid rotation, as three world-space column vectors
    /// (rest-frame axes mapped into the world). The caller turns this into the
    /// transform's euler triple.
    pub rotation_columns: [[f64; 3]; 3],
}

/// VI-2 — a BONDED body wired into the solver: which vessel it (used to)
/// animate, its whole particle set, and the lattice cube size its fragments'
/// render mesh is built from. `broken` flips to `true` the tick a break is
/// first observed (see `Physics::poll_bonded`) — once broken, this binding
/// stops contributing a whole-body pose (its vessel entity is gone, replaced
/// by fragment vessels the caller births).
#[derive(Clone, Debug)]
struct BondedBinding {
    gaia_id: String,
    whole: Vec<usize>,
    cube_size: f64,
    broken: bool,
}

/// The physics seam: the Elements' solver holding every declared body, plus the
/// bindings back to their vessels. Owned by the living layer; stepped once per
/// world tick.
#[derive(Clone, Debug)]
pub struct Physics {
    solver: Solver,
    bindings: Vec<BodyBinding>,
    /// VI-2 — bonded (fracturable) bodies, tracked separately from rigid
    /// `bindings` (a bonded body carries no `elements::RigidBody`, so it has
    /// no shape-matched pose to read the way `Physics::pose` does).
    bonded: Vec<BondedBinding>,
    /// PLAYGROUND — remaining runtime ticks a bonded lattice's fracture stays
    /// DISARMED after load (`max` of the declared `settle`). A near-rigid
    /// lattice spikes a huge transient constraint force in its first substeps
    /// (a numerical startup shock, not real load) that would spuriously tear a
    /// resting crate before a hand ever touched it — so for these first ticks
    /// [`Physics::step`] runs with the threshold at infinity, then it arms.
    /// Done in the RUNTIME step (never a pre-settle at install) so every rigid
    /// body still spawns exactly where the realm authored it — rigid bodies
    /// carry no bonds, so the disarm is invisible to them. `0` (every existing
    /// world) leaves the solver byte-unchanged.
    settle_remaining: u64,
    /// The armed fracture threshold captured at install, restored after each
    /// disarmed settle step.
    armed_fracture_threshold: f64,
}

impl Physics {
    /// Wire the solver from a realm's declared bodies and its static triangle
    /// soup. Returns `None` when no body is declared — the caller then does no
    /// physics at all (a zero-physics realm is byte-unchanged). `dt` is the
    /// world tick length; `seed` roots the (currently unused) deterministic
    /// jitter. Each declaration is `(gaia_id, body, world_center)` — the
    /// authored transform position is the body's spawn centroid.
    pub fn install(
        declarations: Vec<(String, Body, [f64; 3])>,
        collider_triangles: Vec<Triangle>,
        dt: f64,
        seed: u64,
    ) -> Option<Physics> {
        if declarations.is_empty() {
            return None;
        }
        let config = SolverConfig {
            dt,
            seed,
            ..SolverConfig::default()
        };
        let mut solver = Solver::new(config);
        solver.collider = Some(Collider {
            triangles: collider_triangles,
            material: ContactMaterial::default(),
        });
        let mut bindings = Vec::with_capacity(declarations.len());
        let mut bonded = Vec::new();
        let mut settle_ticks = 0u64;
        for (gaia_id, body, center) in declarations {
            let dims = Vec3::new(body.size[0], body.size[1], body.size[2]);
            let counts = (body.resolution[0], body.resolution[1], body.resolution[2]);
            if body.bonded {
                // VI-2 — SOMETHING BREAKS: a nearest-neighbor bonded lattice,
                // never shape-matched. Love defaults from essence (density)
                // when not explicitly authored.
                let love = body
                    .love
                    .unwrap_or_else(|| elements::default_bond_love(body.density));
                let whole = solver.spawn_bonded_box(
                    Vec3::new(center[0], center[1], center[2]),
                    dims,
                    counts,
                    body.density,
                    love,
                    body.compliance,
                    body.contact_radius,
                );
                let cube_size = fracture::lattice_cube_size(dims, counts);
                if body.anchor_base {
                    // PLAYABLE DESTRUCTION — pin the lowest lattice layer
                    // (`iy == 0`, the world-y-minimum row since
                    // `spawn_bonded_box` centres the lattice about `center`
                    // with `origin = center - dims/2`) to the world:
                    // `inv_mass = 0` makes it immovable, so the WHOLE static
                    // load above rides the base-layer bonds instead of bare
                    // contact. Same recovery `elements::building::erect`
                    // uses: spawn order is (ix, iy, iz) nested, so a
                    // particle's flat index is `ix*(ny*nz) + iy*nz + iz`.
                    let (nx, ny, nz) = counts;
                    for ix in 0..nx {
                        for iz in 0..nz {
                            let flat = ix * (ny * nz) + iz;
                            solver.particles.inv_mass[whole[flat]] = 0.0;
                        }
                    }
                }
                if body.spin != [0.0, 0.0, 0.0] {
                    // Applied ONCE, at spawn, about the same authored
                    // centroid spawn_bonded_box just built the lattice
                    // around — see the `spin` field's doc for why this
                    // needs the per-particle rotational field
                    // (`apply_spin_to_particles`), not a uniform impulse.
                    let spin = Vec3::new(body.spin[0], body.spin[1], body.spin[2]);
                    solver.apply_spin_to_particles(
                        &whole,
                        Vec3::new(center[0], center[1], center[2]),
                        spin,
                    );
                }
                if body.initial_velocity != [0.0, 0.0, 0.0] {
                    // Applied ONCE, at spawn — see `initial_velocity`'s doc
                    // for why a uniform drift plus `spin` together (not
                    // either alone) is what breaks a symmetric drop's
                    // tendency to fragment into a merely-flattened pile.
                    let dv = Vec3::new(
                        body.initial_velocity[0],
                        body.initial_velocity[1],
                        body.initial_velocity[2],
                    );
                    solver.apply_impulse_to_particles(&whole, dv);
                }
                settle_ticks = settle_ticks.max(body.settle);
                bonded.push(BondedBinding {
                    gaia_id,
                    whole,
                    cube_size,
                    broken: false,
                });
            } else {
                // Only the box shape is discretized in P3; any other shape
                // falls back to a box of its extents (generic, never a hard
                // error).
                let rigid = solver.spawn_rigid_box(
                    Vec3::new(center[0], center[1], center[2]),
                    dims,
                    counts,
                    body.density,
                    body.rigidity,
                    body.contact_radius,
                );
                bindings.push(BodyBinding {
                    gaia_id,
                    rigid,
                    half_height: body.size[1] * 0.5,
                    contact_radius: body.contact_radius,
                });
            }
        }
        let armed_fracture_threshold = solver.config.fracture_threshold;
        Some(Physics {
            solver,
            bindings,
            bonded,
            settle_remaining: settle_ticks,
            armed_fracture_threshold,
        })
    }

    /// Advance every declared body one fixed tick (the entropy coordinate).
    /// PLAYGROUND — for the first `settle` ticks after load, a settle-
    /// requesting bonded lattice steps with fracture DISARMED so its startup
    /// shock relaxes without spuriously tearing (see `settle_remaining`); then
    /// the authored threshold is armed for the rest of the run. Rigid-only
    /// realms (`settle == 0`) take the plain step, byte-unchanged.
    pub fn step(&mut self) {
        if self.settle_remaining > 0 {
            self.solver.config.fracture_threshold = f64::INFINITY;
            self.solver.step();
            self.solver.config.fracture_threshold = self.armed_fracture_threshold;
            self.settle_remaining -= 1;
        } else {
            self.solver.step();
        }
    }

    /// VI-2 — poll every NOT-YET-BROKEN bonded body: still whole, or broke
    /// THIS tick? Call once per tick, AFTER `step()`. Returns `(still_whole,
    /// newly_broken)`:
    /// - `still_whole`: `(gaia_id, live_centroid)` for every bonded body
    ///   that has not yet fractured — its vessel keeps riding this pose
    ///   (translation only; rotation held identity — VI-2 design note: a
    ///   bonded lattice under uniform gravity alone free-falls without
    ///   torque, so this is exact pre-impact; post-impact it is a
    ///   documented simplification, see `RITE-VI-STRIFE.md`'s VI-2 section
    ///   in this crate's example doc).
    /// - `newly_broken`: `(parent_gaia_id, fragments, cube_size)` for every
    ///   bonded body whose flood-fill just split into more than one
    ///   component — the caller (Dynamics) births fragment vessels from
    ///   this exactly once (`broken` flips true here so it is never
    ///   reported again).
    pub fn poll_bonded(&mut self) -> (StillWholePoses, NewlyBrokenFragments) {
        let mut still_whole = Vec::new();
        let mut newly_broken = Vec::new();
        for binding in &mut self.bonded {
            if binding.broken {
                continue;
            }
            let fragments = fracture::compute_fragments(&self.solver, &binding.whole);
            if fragments.len() <= 1 {
                let c = fragments
                    .first()
                    .map(|f| f.centroid)
                    .unwrap_or(elements::Vec3::ZERO);
                still_whole.push((binding.gaia_id.clone(), [c.x, c.y, c.z]));
            } else {
                binding.broken = true;
                newly_broken.push((binding.gaia_id.clone(), fragments, binding.cube_size));
            }
        }
        (still_whole, newly_broken)
    }

    /// VI-2 — the live mass-weighted centroid of an arbitrary particle set
    /// (a fragment's fixed particle indices, tracked by the caller since the
    /// tick it was born). Used to keep settling fragments moving every tick
    /// after birth (translation only, same design note as `poll_bonded`).
    pub fn group_centroid(&self, particles: &[usize]) -> [f64; 3] {
        let mut sum = elements::Vec3::ZERO;
        let mut mass = 0.0;
        for &i in particles {
            let inv_m = self.solver.particles.inv_mass[i];
            let m = if inv_m > 0.0 { 1.0 / inv_m } else { 0.0 };
            sum = sum + self.solver.particles.pos[i].scale(m);
            mass += m;
        }
        let c = if mass > 0.0 {
            sum.scale(1.0 / mass)
        } else {
            elements::Vec3::ZERO
        };
        [c.x, c.y, c.z]
    }

    /// VI-2 — read-only access to the solver (fragment mesh building needs
    /// live particle positions at the birth tick).
    pub fn solver(&self) -> &Solver {
        &self.solver
    }

    /// The bindings — each body's vessel id and rigid index.
    pub fn bindings(&self) -> &[BodyBinding] {
        &self.bindings
    }

    /// The body's current world pose (centroid + rotation columns), read from
    /// the solver's rigid readout (refreshed each step).
    pub fn pose(&self, binding: &BodyBinding) -> BodyPose {
        let body = &self.solver.rigids[binding.rigid];
        let c = body.centroid;
        let r: Mat3 = body.rotation;
        BodyPose {
            position: [c.x, c.y, c.z],
            rotation_columns: [
                [r.col0.x, r.col0.y, r.col0.z],
                [r.col1.x, r.col1.y, r.col1.z],
                [r.col2.x, r.col2.y, r.col2.z],
            ],
        }
    }

    /// The observable solver state's fingerprint — the determinism ordeal's
    /// witness (two identical runs fold identically here).
    pub fn state_hash(&self) -> u64 {
        self.solver.state_hash()
    }

    /// The current tick index (entropy coordinate).
    pub fn tick(&self) -> u64 {
        self.solver.tick
    }

    /// Apply an instantaneous velocity change to the body bound to
    /// `gaia_id` — resolves the vessel's id to its solver rigid index (the
    /// "the op is the hand" seam: crystal's `Op::Impulse` names an entity,
    /// scrying-glass resolves it, the solver just adds the delta). A no-op
    /// if no binding matches (unknown or non-physical vessel — silent, like
    /// every other op applied to a body that isn't there).
    ///
    /// PLAYGROUND (the Architect's own hands): a BONDED body carries no
    /// `RigidBody` (see `spawn_bonded_box`) so it is not in `bindings` — the
    /// SAME `Op::Impulse` id resolves here to the bonded lattice's whole
    /// particle set and rides `apply_impulse_to_particles` instead (the one
    /// solver door, no parallel physics path). A uniform shove on a lattice
    /// resting in floor contact shears its constrained base against its free
    /// body — a hard enough push tears the weak bonds and it shatters on the
    /// same route that merely topples a rigid stack.
    pub fn apply_impulse(&mut self, gaia_id: &str, delta_velocity: [f64; 3]) {
        let dv = Vec3::new(delta_velocity[0], delta_velocity[1], delta_velocity[2]);
        if let Some(binding) = self.bindings.iter().find(|b| b.gaia_id == gaia_id) {
            self.solver.apply_impulse(binding.rigid, dv);
            return;
        }
        if let Some(binding) = self
            .bonded
            .iter()
            .find(|b| b.gaia_id == gaia_id && !b.broken)
        {
            let whole = binding.whole.clone();
            self.solver.apply_impulse_to_particles(&whole, dv);
        }
    }

    /// PLAYGROUND — every pushable body's `(gaia_id, world_centroid)` this
    /// tick: rigid bodies (fitted centroid) and still-whole bonded bodies
    /// (live mass-weighted centroid). The window's push ray picks the nearest
    /// of these it is aimed at, then names it in an `Op::Impulse` — so the
    /// key and an agent op select a target through the exact same body set.
    pub fn push_targets(&self) -> Vec<(String, [f64; 3])> {
        let mut targets = Vec::with_capacity(self.bindings.len() + self.bonded.len());
        for binding in &self.bindings {
            targets.push((binding.gaia_id.clone(), self.pose(binding).position));
        }
        for binding in &self.bonded {
            if binding.broken {
                continue;
            }
            targets.push((binding.gaia_id.clone(), self.group_centroid(&binding.whole)));
        }
        targets
    }
}
