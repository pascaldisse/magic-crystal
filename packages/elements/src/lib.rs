//! # The Elements — DreamForge's XPBD substrate
//!
//! One solver for all matter. Matter is [`Particles`] (Structure-of-Arrays);
//! matter is held together by bindings ([`DistanceConstraint`]) whose single
//! force is LOVE — the constraint's pull toward rest. STRIFE is the readout
//! of that force (stress), and FRACTURE is the moment strife exceeds a bond's
//! love. Time is the entropy coordinate: a fixed tick, no randomness, every
//! replay byte-identical (see the ordeals in `tests/`).
//!
//! P1 is the pure substrate — particles, compliant distance bindings, the
//! bond/strife/fracture model, and determinism. P2 raises RIGID BODIES on it:
//! particle clusters held by a shape-matching constraint ([`RigidBody`]) and
//! WORLD COLLISION against a static triangle soup ([`Collider`]) with Coulomb
//! friction and restitution. Still no render, no ECS wiring (a later rite).
//! CPU only, no main thread owed.

pub mod collision;
pub mod constraint;
pub mod hash;
pub mod mat3;
pub mod math;
pub mod particles;
pub mod rigid;
pub mod solver;

pub use collision::{Collider, Contact, ContactMaterial, Triangle};
pub use constraint::{Bond, DistanceConstraint, FractureEvent};
pub use hash::{hash3, jitter, StateHasher};
pub use mat3::{polar_rotation, Mat3, PolarConfig};
pub use math::Vec3;
pub use particles::Particles;
pub use rigid::RigidBody;
pub use solver::{Solver, SolverConfig};

/// **LOVE = 1.0** — the One Constant. The never-hardcode law has exactly one
/// sanctioned exception: love, the unit of binding on `[0, 1]`, `1.0` at its
/// immutable center (the 4th Temple's `vector[32] = 1.0`, compiled). Every
/// other magnitude in the Elements is a parameter; this is not negotiable.
pub const LOVE: f64 = 1.0;
