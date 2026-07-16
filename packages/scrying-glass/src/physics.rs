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

/// The `body` sigil — realm data declaring a vessel as physical matter the
/// world tick simulates. Every field is plain English with a documented
/// default; only `shape` selects the discretization, the rest are solver dials.
#[derive(Clone, Debug, Deserialize)]
pub struct Body {
    /// The matter's shape. `"box"` (the only P3 shape) fills a lattice box.
    #[serde(default = "default_shape")]
    pub shape: String,
    /// Full extents of the body in metres (a box's width/height/depth).
    #[serde(default = "default_size")]
    pub size: [f64; 3],
    /// Material density in kg/m³ — mass is DERIVED (`density × volume`), never
    /// authored. Default `500` (seasoned softwood — a wooden crate).
    #[serde(default = "default_density")]
    pub density: f64,
    /// The particle lattice the body is discretized into (nx, ny, nz). More
    /// particles = finer contact, higher cost. Default `[2, 2, 2]` (the eight
    /// corners — enough for a box resting flat on a plane).
    #[serde(default = "default_resolution")]
    pub resolution: [usize; 3],
    /// Each particle's contact thickness against the world, in metres. Default
    /// `0.05` (5 cm). The body rests with its base particles this far above a
    /// supporting face.
    #[serde(default = "default_contact_radius")]
    pub contact_radius: f64,
    /// Shape-match stiffness on `[0, 1]`: `1.0` = perfectly rigid, `< 1.0` =
    /// deformable. Default `1.0` (a rigid crate).
    #[serde(default = "default_rigidity")]
    pub rigidity: f64,
}

fn default_shape() -> String {
    "box".to_string()
}
fn default_size() -> [f64; 3] {
    [1.0, 1.0, 1.0]
}
fn default_density() -> f64 {
    500.0
}
fn default_resolution() -> [usize; 3] {
    [2, 2, 2]
}
fn default_contact_radius() -> f64 {
    0.05
}
fn default_rigidity() -> f64 {
    1.0
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

/// The physics seam: the Elements' solver holding every declared body, plus the
/// bindings back to their vessels. Owned by the living layer; stepped once per
/// world tick.
#[derive(Clone, Debug)]
pub struct Physics {
    solver: Solver,
    bindings: Vec<BodyBinding>,
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
        for (gaia_id, body, center) in declarations {
            // Only the box shape is discretized in P3; any other shape falls
            // back to a box of its extents (generic, never a hard error).
            let rigid = solver.spawn_rigid_box(
                Vec3::new(center[0], center[1], center[2]),
                Vec3::new(body.size[0], body.size[1], body.size[2]),
                (
                    body.resolution[0],
                    body.resolution[1],
                    body.resolution[2],
                ),
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
        Some(Physics { solver, bindings })
    }

    /// Advance every declared body one fixed tick (the entropy coordinate).
    pub fn step(&mut self) {
        self.solver.step();
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
}
