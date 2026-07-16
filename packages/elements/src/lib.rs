//! # The Elements — DreamForge's XPBD substrate
//!
//! One solver for all matter. Matter is [`Particles`] (Structure-of-Arrays);
//! matter is held together by bindings ([`DistanceConstraint`]) whose single
//! force is LOVE — the constraint's pull toward rest. STRIFE is the readout
//! of that force (stress), and FRACTURE is the moment strife exceeds a bond's
//! love. Time is the entropy coordinate: a fixed tick, no randomness, every
//! replay byte-identical (see the ordeals in `tests/`).
//!
//! This is P1: the pure library — particles, compliant distance bindings, the
//! bond/strife/fracture model, and determinism. No render, no ECS wiring
//! (that is a later rite). CPU only, no main thread owed.

pub mod constraint;
pub mod hash;
pub mod math;
pub mod particles;
pub mod solver;

pub use constraint::{Bond, DistanceConstraint, FractureEvent};
pub use hash::{hash3, jitter, StateHasher};
pub use math::Vec3;
pub use particles::Particles;
pub use solver::{Solver, SolverConfig};

/// **LOVE = 1.0** — the One Constant. The never-hardcode law has exactly one
/// sanctioned exception: love, the unit of binding on `[0, 1]`, `1.0` at its
/// immutable center (the 4th Temple's `vector[32] = 1.0`, compiled). Every
/// other magnitude in the Elements is a parameter; this is not negotiable.
pub const LOVE: f64 = 1.0;
