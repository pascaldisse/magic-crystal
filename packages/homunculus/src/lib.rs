//! # Homunculus — DreamForge H0 body substrate
//!
//! The character-editor's skeleton, CPU + data-first. No manual rigging:
//! every joint weight and every animation frame is derived procedurally from
//! parameters. One parametric generator spans the whole morphological range
//! from a standing human to a walking cat.
//!
//! Four layers, each its own module:
//! - [`skeleton`] — hierarchical [`Bone`]s and the parametric [`BodyParams`]
//!   generator ([`Skeleton::from_params`], [`Skeleton::humanoid`],
//!   [`Skeleton::quadruped`]).
//! - [`skin`] — a capsule per bone and automatic vertex weights from capsule
//!   distance falloff ([`skin::compute_weights`]); weights always normalize.
//! - [`socket`] — named attachment points ([`Socket`], [`SocketSet`]) parented
//!   to bones with a local offset; their world pose is FK of the bone plus the
//!   offset (what coats/brooches/held props mount to).
//! - [`pose`] — joint-local rotations resolved to world transforms by forward
//!   kinematics ([`Pose`], [`Pose::forward_kinematics`], [`Pose::blend`]).
//! - [`walk`] — a deterministic, phase-driven procedural walk-cycle generator
//!   ([`walk::WalkParams`], [`walk::walk_pose`]) — no keyframe assets.
//!
//! Determinism is a law here (see `ENTROPY.md`): the walk generator is a pure
//! function of `(params, tick)` and produces a byte-identical pose stream on
//! every run.

pub mod pose;
pub mod skeleton;
pub mod skin;
pub mod socket;
pub mod walk;

pub use pose::{Pose, Transform};
pub use skeleton::{BodyParams, Bone, Skeleton};
pub use skin::{compute_weights, BoneCapsule, SkinWeights, VertexWeights};
pub use socket::{Socket, SocketSet};
pub use walk::{walk_pose, walk_pose_stream, WalkParams};
