//! Pose substrate: joint-local transforms, forward kinematics, and blending.

use crate::skeleton::Skeleton;
use glam::{Affine3A, Quat, Vec3};

/// A rigid local transform of a bone relative to its parent.
///
/// Uniform (no non-uniform scale): translation + rotation only. The bind
/// transform of a bone lives here, and so does the per-bone pose delta.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    /// Translation relative to the parent bone's frame.
    pub translation: Vec3,
    /// Rotation relative to the parent bone's frame.
    pub rotation: Quat,
}

impl Transform {
    /// The identity transform (no translation, no rotation).
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// Build a transform from a translation and rotation.
    pub fn new(translation: Vec3, rotation: Quat) -> Self {
        Self {
            translation,
            rotation,
        }
    }

    /// A pure translation.
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            rotation: Quat::IDENTITY,
        }
    }

    /// This transform as a `glam` affine matrix.
    pub fn to_affine(self) -> Affine3A {
        Affine3A::from_rotation_translation(self.rotation, self.translation)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A pose: one joint-local rotation delta per bone, applied on top of the
/// skeleton's bind transform.
///
/// The identity pose (all rotations [`Quat::IDENTITY`]) reproduces the bind
/// pose exactly under [`Pose::forward_kinematics`] — this roundtrip is an
/// ordeal of the substrate.
#[derive(Clone, Debug, PartialEq)]
pub struct Pose {
    /// Per-bone local rotation delta, indexed by bone.
    pub local_rotations: Vec<Quat>,
}

impl Pose {
    /// The identity pose for a skeleton of `bone_count` bones.
    pub fn identity(bone_count: usize) -> Self {
        Self {
            local_rotations: vec![Quat::IDENTITY; bone_count],
        }
    }

    /// The identity pose sized to a given skeleton.
    pub fn bind(skeleton: &Skeleton) -> Self {
        Self::identity(skeleton.bones.len())
    }

    /// Number of bones this pose addresses.
    pub fn len(&self) -> usize {
        self.local_rotations.len()
    }

    /// Whether the pose addresses no bones.
    pub fn is_empty(&self) -> bool {
        self.local_rotations.is_empty()
    }

    /// The effective local transform of bone `i`: its bind transform with the
    /// pose's local rotation delta applied.
    fn local_transform(&self, skeleton: &Skeleton, i: usize) -> Transform {
        let bind = skeleton.bones[i].local_bind;
        Transform {
            translation: bind.translation,
            rotation: bind.rotation * self.local_rotations[i],
        }
    }

    /// Resolve this pose to a world transform per bone via forward kinematics.
    ///
    /// Bones are stored parents-before-children (guaranteed by the generators),
    /// so a single forward pass suffices.
    pub fn forward_kinematics(&self, skeleton: &Skeleton) -> Vec<Affine3A> {
        assert_eq!(
            self.local_rotations.len(),
            skeleton.bones.len(),
            "pose bone count must match skeleton"
        );
        let mut world: Vec<Affine3A> = Vec::with_capacity(skeleton.bones.len());
        for i in 0..skeleton.bones.len() {
            let local = self.local_transform(skeleton, i).to_affine();
            let w = match skeleton.bones[i].parent {
                Some(p) => {
                    debug_assert!(p < i, "parent must precede child");
                    world[p] * local
                }
                None => local,
            };
            world.push(w);
        }
        world
    }

    /// Blend from `self` toward `other` by `t` in `[0, 1]`, per-bone, using
    /// normalized-lerp on the rotations (shortest arc, unit output).
    pub fn blend(&self, other: &Pose, t: f32) -> Pose {
        assert_eq!(
            self.local_rotations.len(),
            other.local_rotations.len(),
            "blended poses must share bone count"
        );
        let local_rotations = self
            .local_rotations
            .iter()
            .zip(other.local_rotations.iter())
            .map(|(a, b)| nlerp(*a, *b, t))
            .collect();
        Pose { local_rotations }
    }

    /// Serialize the pose's rotations to raw little-endian bytes — the canonical
    /// form for byte-identity checks in the determinism ordeal.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.local_rotations.len() * 16);
        for q in &self.local_rotations {
            out.extend_from_slice(&q.x.to_le_bytes());
            out.extend_from_slice(&q.y.to_le_bytes());
            out.extend_from_slice(&q.z.to_le_bytes());
            out.extend_from_slice(&q.w.to_le_bytes());
        }
        out
    }
}

/// Normalized linear interpolation of two unit quaternions along the shortest
/// arc. Cheaper than slerp and, unlike slerp, has no branch on the angle — good
/// for determinism.
pub fn nlerp(a: Quat, b: Quat, t: f32) -> Quat {
    // Pick the closer hemisphere so the arc is the short one.
    let b = if a.dot(b) < 0.0 { -b } else { b };
    let x = a.x + (b.x - a.x) * t;
    let y = a.y + (b.y - a.y) * t;
    let z = a.z + (b.z - a.z) * t;
    let w = a.w + (b.w - a.w) * t;
    Quat::from_xyzw(x, y, z, w).normalize()
}
