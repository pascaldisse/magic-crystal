//! Deterministic procedural walk-cycle generator.
//!
//! No keyframe assets: the gait is a pure, phase-driven function of
//! [`WalkParams`] and the integer `tick`. Given the same params (seed included)
//! it emits a byte-identical pose stream on every run — the determinism law
//! (`ENTROPY.md`) made testable.

use crate::pose::Pose;
use crate::skeleton::Skeleton;
use glam::Quat;
use std::f32::consts::TAU;

/// Parameters of a walk cycle. Every field has a default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WalkParams {
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
    /// Deterministic phase seed (chooses the starting foot / phase).
    pub seed: u64,
}

impl Default for WalkParams {
    fn default() -> Self {
        Self {
            cadence: 1.0,
            dt: 1.0 / 60.0,
            stride: 0.6,
            arm_swing: 0.5,
            flex: 0.5,
            sway: 0.08,
            seed: 0,
        }
    }
}

impl WalkParams {
    /// Cycle phase in `[0, 1)` at a given tick.
    pub fn phase(&self, tick: u64) -> f32 {
        let base = seed_phase(self.seed);
        let raw = tick as f32 * self.cadence * self.dt + base;
        raw - raw.floor()
    }
}

/// The walk pose at `tick`.
///
/// Legs swing in antiphase; each arm counter-swings its same-side leg; knees
/// and elbows flex on the forward half of the swing; the spine sways and the
/// tail wags. Everything else stays at bind.
pub fn walk_pose(skeleton: &Skeleton, params: &WalkParams, tick: u64) -> Pose {
    let phase = params.phase(tick);
    let mut pose = Pose::bind(skeleton);

    for (i, bone) in skeleton.bones.iter().enumerate() {
        let name = bone.name.as_str();
        let side = if name.starts_with("L.") {
            Some(0.0)
        } else if name.starts_with("R.") {
            Some(0.5)
        } else {
            None
        };

        pose.local_rotations[i] = if name.ends_with(".thigh") {
            // Hind/legs: L leads, R is half a cycle behind.
            let p = phase + side.unwrap();
            Quat::from_rotation_x(params.stride * sine(p))
        } else if name.ends_with(".upperarm") {
            // Arms counter-swing the same-side leg (+0.5).
            let p = phase + side.unwrap() + 0.5;
            Quat::from_rotation_x(params.stride * params.arm_swing * sine(p))
        } else if name.ends_with(".shank") {
            // Knee flexes on the forward (lifting) half of its leg's swing.
            let p = phase + side.unwrap();
            Quat::from_rotation_x(-params.flex * lift(p))
        } else if name.ends_with(".forearm") {
            let p = phase + side.unwrap() + 0.5;
            Quat::from_rotation_x(-params.flex * params.arm_swing * lift(p))
        } else if name.starts_with("spine.") {
            Quat::from_rotation_y(params.sway * sine(phase))
        } else if name == "head" {
            // Counter-sway to keep the head roughly level.
            Quat::from_rotation_y(-params.sway * sine(phase))
        } else if name.starts_with("tail.") {
            Quat::from_rotation_y(params.sway * 1.5 * sine(phase))
        } else {
            Quat::IDENTITY
        };
    }

    pose
}

/// A contiguous stream of walk poses for ticks `0..count`.
pub fn walk_pose_stream(skeleton: &Skeleton, params: &WalkParams, count: u64) -> Vec<Pose> {
    (0..count).map(|t| walk_pose(skeleton, params, t)).collect()
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
    // Top 24 bits → [0,1); exact, no platform float variance.
    ((z >> 40) as f32) / ((1u64 << 24) as f32)
}
