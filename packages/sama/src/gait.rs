//! Procedural gait generator — the canonical forward path for cyclic
//! locomotion poses.
//!
//! Reimplements and generalizes the homunculus walk-cycle: the pose is a pure,
//! phase-driven function of [`GaitParams`] and the integer `tick`. No keyframe
//! assets. Given the same params it emits a byte-identical pose stream on every
//! run — the determinism law (`ENTROPY.md`) made testable.
//!
//! Generalization over the seed walk-cycle:
//! - per-limb phase offsets (`right_offset`, `arm_offset`) instead of the
//!   hardcoded L/R and arm split,
//! - `tail_wag` amplitude multiplier,
//! - `digitigrade` paw flex on the hind (`.foot`) and front (`.hand`) tips —
//!   default `0.0` reproduces the plantigrade homunculus walk exactly.

use glam::Quat;
use homunculus::{Pose, Skeleton};
use std::f32::consts::TAU;

/// Parameters of a cyclic gait. Every field has a default; the [`Default`] /
/// [`GaitParams::walk`] preset reproduces the homunculus walk-cycle bit for bit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaitParams {
    /// Gait cycles per second.
    pub cadence: f32,
    /// Seconds per tick (fixed timestep).
    pub dt: f32,
    /// Leg swing amplitude, radians.
    pub stride: f32,
    /// Arm swing amplitude as a fraction of `stride`.
    pub arm_swing: f32,
    /// Knee / elbow flex amplitude, radians.
    pub flex: f32,
    /// Torso lateral sway amplitude, radians.
    pub sway: f32,
    /// Tail wag amplitude as a multiple of `sway`.
    pub tail_wag: f32,
    /// Extra paw flex on the hind/front tips (digitigrade posture), radians.
    /// `0.0` = plantigrade (feet/hands stay at bind), reproducing homunculus.
    pub digitigrade: f32,
    /// Phase offset of the right-side limbs relative to the left, in cycles.
    pub right_offset: f32,
    /// Phase offset of an arm relative to its same-side leg, in cycles.
    pub arm_offset: f32,
    /// Deterministic phase seed (chooses the starting foot / phase).
    pub seed: u64,
}

impl Default for GaitParams {
    fn default() -> Self {
        Self::walk()
    }
}

impl GaitParams {
    /// Walk preset — matches `homunculus::WalkParams::default()` exactly.
    pub fn walk() -> Self {
        Self {
            cadence: 1.0,
            dt: 1.0 / 60.0,
            stride: 0.6,
            arm_swing: 0.5,
            flex: 0.5,
            sway: 0.08,
            tail_wag: 1.5,
            digitigrade: 0.0,
            right_offset: 0.5,
            arm_offset: 0.5,
            seed: 0,
        }
    }

    /// Run preset — faster cadence, longer stride, deeper flex than [`walk`].
    ///
    /// [`walk`]: GaitParams::walk
    pub fn run() -> Self {
        Self {
            cadence: 2.0,
            dt: 1.0 / 60.0,
            stride: 0.9,
            arm_swing: 0.6,
            flex: 0.7,
            sway: 0.10,
            tail_wag: 1.5,
            digitigrade: 0.0,
            right_offset: 0.5,
            arm_offset: 0.5,
            seed: 0,
        }
    }

    /// Cycle phase in `[0, 1)` at a given tick.
    pub fn phase(&self, tick: u64) -> f32 {
        let base = seed_phase(self.seed);
        let raw = tick as f32 * self.cadence * self.dt + base;
        raw - raw.floor()
    }
}

/// The gait pose at `tick`.
///
/// Legs swing in antiphase (`right_offset`); each arm counter-swings its
/// same-side leg (`arm_offset`); knees and elbows flex on the forward half of
/// the swing; the spine sways, the head counter-sways, the tail wags; the paw
/// tips flex by `digitigrade`. Everything else stays at bind.
pub fn gait_pose(skeleton: &Skeleton, params: &GaitParams, tick: u64) -> Pose {
    let phase = params.phase(tick);
    let mut pose = Pose::bind(skeleton);

    for (i, bone) in skeleton.bones.iter().enumerate() {
        let name = bone.name.as_str();
        let side = if name.starts_with("L.") {
            Some(0.0)
        } else if name.starts_with("R.") {
            Some(params.right_offset)
        } else {
            None
        };

        pose.local_rotations[i] = if name.ends_with(".thigh") {
            // Legs: left leads, right trails by `right_offset`.
            let p = phase + side.unwrap();
            Quat::from_rotation_x(params.stride * sine(p))
        } else if name.ends_with(".upperarm") {
            // Arms counter-swing the same-side leg.
            let p = phase + side.unwrap() + params.arm_offset;
            Quat::from_rotation_x(params.stride * params.arm_swing * sine(p))
        } else if name.ends_with(".shank") {
            // Knee flexes on the forward (lifting) half of its leg's swing.
            let p = phase + side.unwrap();
            Quat::from_rotation_x(-params.flex * lift(p))
        } else if name.ends_with(".forearm") {
            let p = phase + side.unwrap() + params.arm_offset;
            Quat::from_rotation_x(-params.flex * params.arm_swing * lift(p))
        } else if name.ends_with(".foot") || name.ends_with(".hand") {
            // Digitigrade paw flex: hind (`.foot`) tracks its leg, front
            // (`.hand`) tracks its arm. `digitigrade == 0` => identity.
            let arm = name.ends_with(".hand");
            let p = phase + side.unwrap() + if arm { params.arm_offset } else { 0.0 };
            Quat::from_rotation_x(-params.digitigrade * lift(p))
        } else if name.starts_with("spine.") {
            Quat::from_rotation_y(params.sway * sine(phase))
        } else if name == "head" {
            Quat::from_rotation_y(-params.sway * sine(phase))
        } else if name.starts_with("tail.") {
            Quat::from_rotation_y(params.sway * params.tail_wag * sine(phase))
        } else {
            Quat::IDENTITY
        };
    }

    pose
}

/// A contiguous stream of gait poses for ticks `0..count`.
pub fn gait_pose_stream(skeleton: &Skeleton, params: &GaitParams, count: u64) -> Vec<Pose> {
    (0..count).map(|t| gait_pose(skeleton, params, t)).collect()
}

/// Sine of a `[0,1)` phase.
fn sine(p: f32) -> f32 {
    (p * TAU).sin()
}

/// Rectified lift term in `[0,1]`: nonzero only on the forward half of a swing.
fn lift(p: f32) -> f32 {
    sine(p).max(0.0)
}

/// Deterministic phase offset in `[0,1)` from a seed (splitmix64 finalizer).
fn seed_phase(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Top 24 bits -> [0,1); exact, no platform float variance.
    ((z >> 40) as f32) / ((1u64 << 24) as f32)
}
