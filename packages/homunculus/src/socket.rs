//! Attachment sockets: named mount points rigidly parented to a bone.
//!
//! A [`Socket`] is a bone index plus a local offset [`Transform`] in that
//! bone's frame — nothing more. Its world pose under any [`Pose`] is pure
//! forward kinematics of the bone followed by the local offset
//! ([`Socket::world`]); no meshes live here (vessel/render consume the
//! transforms later — a coat rides `back`, a brooch rides `chest`, a lantern
//! rides `right_hand_grip`).
//!
//! The generator ships standard sets per morphology
//! ([`SocketSet::humanoid`], [`SocketSet::quadruped`], and [`SocketSet::standard`]
//! which picks by [`BodyParams::stance`]); user-defined sockets are added with
//! [`SocketSet::add`]. Every offset is DERIVED from the skeleton's own bone
//! lengths and radius, so a socket stays glued to its bone across any
//! reparametrization.

use crate::pose::{Pose, Transform};
use crate::skeleton::{BodyParams, Skeleton};
use glam::{Affine3A, Vec3};

/// A named attachment point rigidly parented to bone `bone`, offset by `local`
/// in that bone's frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Socket {
    /// Human-readable, stable name (e.g. `"right_hand_grip"`, `"head_top"`).
    pub name: String,
    /// Index of the bone this socket is parented to.
    pub bone: usize,
    /// Offset transform relative to the parent bone's frame.
    pub local: Transform,
}

impl Socket {
    /// Build a socket from a name, parent bone index and local offset.
    pub fn new(name: impl Into<String>, bone: usize, local: Transform) -> Self {
        Self {
            name: name.into(),
            bone,
            local,
        }
    }

    /// World transform of the socket given per-bone world transforms (as
    /// produced by [`Pose::forward_kinematics`]): the parent bone's world
    /// transform composed with the socket's local offset.
    pub fn world(&self, bone_world: &[Affine3A]) -> Affine3A {
        bone_world[self.bone] * self.local.to_affine()
    }
}

/// A named collection of [`Socket`]s for one skeleton.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SocketSet {
    /// The sockets, in insertion order.
    pub sockets: Vec<Socket>,
}

impl SocketSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of sockets.
    pub fn len(&self) -> usize {
        self.sockets.len()
    }
    /// Whether the set has no sockets.
    pub fn is_empty(&self) -> bool {
        self.sockets.is_empty()
    }

    /// Look up a socket by name.
    pub fn get(&self, name: &str) -> Option<&Socket> {
        self.sockets.iter().find(|s| s.name == name)
    }

    /// Add (or replace, by name) a user-defined socket. Returns `&mut self` for
    /// chaining.
    pub fn add(&mut self, socket: Socket) -> &mut Self {
        if let Some(slot) = self.sockets.iter_mut().find(|s| s.name == socket.name) {
            *slot = socket;
        } else {
            self.sockets.push(socket);
        }
        self
    }

    /// World transform of one named socket under a pose, or `None` if unknown.
    pub fn world_of(&self, name: &str, skeleton: &Skeleton, pose: &Pose) -> Option<Affine3A> {
        let world = pose.forward_kinematics(skeleton);
        self.get(name).map(|s| s.world(&world))
    }

    /// World transforms of every socket under a pose, in socket order. FK is
    /// computed once and shared.
    pub fn world_transforms(&self, skeleton: &Skeleton, pose: &Pose) -> Vec<Affine3A> {
        let world = pose.forward_kinematics(skeleton);
        self.sockets.iter().map(|s| s.world(&world)).collect()
    }

    /// The standard socket set for a skeleton, chosen by morphology: an upright
    /// biped ([`BodyParams::stance`] `< 0.5`) gets the humanoid set, anything
    /// more horizontal gets the quadruped set.
    pub fn standard(skeleton: &Skeleton, params: &BodyParams) -> SocketSet {
        if params.stance < 0.5 {
            Self::humanoid(skeleton)
        } else {
            Self::quadruped(skeleton)
        }
    }

    /// The standard humanoid socket set:
    /// `right_hand_grip`, `left_hand_grip`, `head_top`, `chest`, `back`,
    /// `hip_left`, `hip_right`. Missing bones are skipped (never panics).
    pub fn humanoid(skeleton: &Skeleton) -> SocketSet {
        let mut set = SocketSet::new();
        let r = base_radius(skeleton);

        // Grips at the mid-point of each hand bone.
        for (name, bone) in [("right_hand_grip", "R.hand"), ("left_hand_grip", "L.hand")] {
            if let Some((i, len)) = bone_len(skeleton, bone) {
                set.push(
                    name,
                    i,
                    Transform::from_translation(Vec3::new(0.0, len * 0.5, 0.0)),
                );
            }
        }

        // Top of the head, at the head bone's tip.
        if let Some((i, len)) = bone_len(skeleton, "head") {
            set.push(
                "head_top",
                i,
                Transform::from_translation(Vec3::new(0.0, len, 0.0)),
            );
        }

        // Chest (front) and back, on the topmost spine bone.
        if let Some((i, len)) = chest_bone(skeleton) {
            set.push(
                "chest",
                i,
                Transform::from_translation(Vec3::new(0.0, len * 0.5, r)),
            );
            set.push(
                "back",
                i,
                Transform::from_translation(Vec3::new(0.0, len * 0.5, -r)),
            );
        }

        // Hips, on the pelvis, offset sideways by half the derived hip width.
        if let Some((i, _)) = bone_len(skeleton, "pelvis") {
            let half_hip = hip_half_width(skeleton);
            set.push(
                "hip_left",
                i,
                Transform::from_translation(Vec3::new(half_hip, 0.0, 0.0)),
            );
            set.push(
                "hip_right",
                i,
                Transform::from_translation(Vec3::new(-half_hip, 0.0, 0.0)),
            );
        }

        set
    }

    /// The standard quadruped socket set: `head_top`, `saddle` (mid-back),
    /// `tail_tip`. Missing bones are skipped.
    pub fn quadruped(skeleton: &Skeleton) -> SocketSet {
        let mut set = SocketSet::new();
        let r = base_radius(skeleton);

        if let Some((i, len)) = bone_len(skeleton, "head") {
            set.push(
                "head_top",
                i,
                Transform::from_translation(Vec3::new(0.0, len, 0.0)),
            );
        }

        // Saddle: the top of the mid-back spine segment.
        if let Some((i, len)) = mid_spine_bone(skeleton) {
            set.push(
                "saddle",
                i,
                Transform::from_translation(Vec3::new(0.0, len * 0.5, r)),
            );
        }

        // Tail tip: the end of the last tail segment.
        if let Some((i, len)) = last_tail_bone(skeleton) {
            set.push(
                "tail_tip",
                i,
                Transform::from_translation(Vec3::new(0.0, len, 0.0)),
            );
        }

        set
    }

    fn push(&mut self, name: &str, bone: usize, local: Transform) {
        self.sockets.push(Socket::new(name, bone, local));
    }
}

/// `(index, length)` of a named bone, if present.
fn bone_len(skeleton: &Skeleton, name: &str) -> Option<(usize, f32)> {
    skeleton
        .index_of(name)
        .map(|i| (i, skeleton.bones[i].length))
}

/// The base capsule radius (any bone carries it; the pelvis always exists).
fn base_radius(skeleton: &Skeleton) -> f32 {
    skeleton.bones.first().map(|b| b.radius).unwrap_or(0.0)
}

/// The topmost spine bone (chest) — the highest `spine.N`.
fn chest_bone(skeleton: &Skeleton) -> Option<(usize, f32)> {
    highest_indexed(skeleton, "spine.")
}

/// The middle spine bone, for the saddle.
fn mid_spine_bone(skeleton: &Skeleton) -> Option<(usize, f32)> {
    let spines: Vec<usize> = (0..skeleton.bones.len())
        .filter(|&i| skeleton.bones[i].name.starts_with("spine."))
        .collect();
    spines
        .get(spines.len() / 2)
        .map(|&i| (i, skeleton.bones[i].length))
}

/// The last tail bone (highest `tail.N`).
fn last_tail_bone(skeleton: &Skeleton) -> Option<(usize, f32)> {
    highest_indexed(skeleton, "tail.")
}

/// Among bones named `<prefix><n>`, the one with the largest `n`.
fn highest_indexed(skeleton: &Skeleton, prefix: &str) -> Option<(usize, f32)> {
    let mut best: Option<(u32, usize)> = None;
    for (i, b) in skeleton.bones.iter().enumerate() {
        if let Some(rest) = b.name.strip_prefix(prefix) {
            if let Ok(n) = rest.parse::<u32>() {
                if best.map(|(bn, _)| n > bn).unwrap_or(true) {
                    best = Some((n, i));
                }
            }
        }
    }
    best.map(|(_, i)| (i, skeleton.bones[i].length))
}

/// Half the hip width, derived from the two thigh bones' sideways bind offset.
fn hip_half_width(skeleton: &Skeleton) -> f32 {
    skeleton
        .index_of("L.thigh")
        .map(|i| skeleton.bones[i].local_bind.translation.x.abs())
        .unwrap_or(0.0)
}
