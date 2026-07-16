//! # Sama — DreamForge S0 motion substrate
//!
//! The animation engine, CPU + data-first, over the [`homunculus`] pose
//! substrate. Three layers, each its own module:
//!
//! - [`gait`] — a deterministic, phase-driven procedural gait generator
//!   ([`GaitParams`], [`gait_pose`]). The canonical forward path for cyclic
//!   locomotion, generalizing the homunculus walk-cycle (per-limb phase
//!   offsets, tail wag, digitigrade paw flex). No keyframe assets.
//! - [`locomotion`] — a tick-indexed state machine ([`Locomotion`]) over the
//!   [`Gait`] states idle / walk / run, cross-fading between gaits by nlerp
//!   over `blend_ticks`, driven by a commanded speed.
//! - [`gesture`] — additive per-bone pose overlays ([`Gesture`]) composed over
//!   locomotion, including a clamped head [`LookAt`].
//!
//! Determinism is law (`ENTROPY.md`): every output is a pure function of
//! `(params, tick, command stream)` — no RNG, no wall-clock — so the same
//! inputs produce a byte-identical pose stream on every run.

pub mod gait;
pub mod gesture;
pub mod locomotion;

pub use gait::{gait_pose, gait_pose_stream, GaitParams};
pub use gesture::{look_at, Gesture, LookAt, LookAtParams};
pub use locomotion::{Gait, Locomotion, LocomotionParams};
