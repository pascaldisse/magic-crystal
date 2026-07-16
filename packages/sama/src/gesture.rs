//! Gesture layer: additive pose overlays composed over locomotion.
//!
//! A [`Gesture`] is a set of per-bone additive local rotations. Applied on top
//! of a base pose (from the locomotion state machine), each overlay
//! post-multiplies the base bone rotation, so a look-at yaws the head *further*
//! from wherever locomotion already placed it. Overlays are pure data and
//! deterministic.

use glam::Quat;
use homunculus::Pose;
use std::f32::consts::FRAC_PI_4;

/// A set of additive per-bone local rotations.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gesture {
    /// `(bone index, additive local rotation)` pairs.
    pub overlays: Vec<(usize, Quat)>,
}

impl Gesture {
    /// An empty gesture (no overlays).
    pub fn none() -> Self {
        Self::default()
    }

    /// A single-bone overlay.
    pub fn single(bone: usize, rotation: Quat) -> Self {
        Self {
            overlays: vec![(bone, rotation)],
        }
    }

    /// Apply the overlays additively over `base`, returning a new pose. Bone
    /// indices outside the pose are ignored. The base is not mutated.
    pub fn apply(&self, base: &Pose) -> Pose {
        let mut out = base.clone();
        for &(bone, rot) in &self.overlays {
            if let Some(slot) = out.local_rotations.get_mut(bone) {
                *slot = (*slot * rot).normalize();
            }
        }
        out
    }
}

/// Clamps for a look-at gesture, in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookAtParams {
    /// Maximum absolute yaw the head may turn.
    pub max_yaw: f32,
    /// Maximum absolute pitch the head may turn.
    pub max_pitch: f32,
}

impl Default for LookAtParams {
    fn default() -> Self {
        // ~70 deg yaw, 45 deg pitch — a comfortable human head cone.
        Self {
            max_yaw: 70.0_f32.to_radians(),
            max_pitch: FRAC_PI_4,
        }
    }
}

/// The clamped yaw/pitch a look-at request resolves to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookAt {
    /// Clamped yaw, radians.
    pub yaw: f32,
    /// Clamped pitch, radians.
    pub pitch: f32,
}

impl LookAt {
    /// Resolve a requested yaw/pitch against the clamps.
    pub fn resolve(target_yaw: f32, target_pitch: f32, params: &LookAtParams) -> Self {
        Self {
            yaw: target_yaw.clamp(-params.max_yaw, params.max_yaw),
            pitch: target_pitch.clamp(-params.max_pitch, params.max_pitch),
        }
    }

    /// The additive head rotation (yaw about local Y, then pitch about local X).
    pub fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }

    /// Build a head look-at [`Gesture`] for the given head bone.
    pub fn gesture(&self, head_bone: usize) -> Gesture {
        Gesture::single(head_bone, self.rotation())
    }
}

/// Convenience: a clamped head look-at gesture in one call.
pub fn look_at(
    head_bone: usize,
    target_yaw: f32,
    target_pitch: f32,
    params: &LookAtParams,
) -> Gesture {
    LookAt::resolve(target_yaw, target_pitch, params).gesture(head_bone)
}
