//! Procedural skinning: an implicit capsule per bone and automatic vertex
//! weights from capsule distance falloff.
//!
//! No manual weights exist here (a DreamForge law). Every vertex is bound by
//! how near it sits to each bone's capsule; the falloff is strictly positive,
//! so the per-vertex weights always normalize to sum 1.

use crate::pose::Pose;
use crate::skeleton::Skeleton;
use glam::{Affine3A, Vec3, Vec3A};

/// Small floor added to squared distance so the falloff never divides by zero
/// and every vertex keeps a positive weight to every bone.
const FALLOFF_EPS: f32 = 1.0e-6;

/// An implicit capsule for one bone: a segment `a`..`b` swept by `radius`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoneCapsule {
    /// Segment start (bone world origin).
    pub a: Vec3,
    /// Segment end (bone world tip).
    pub b: Vec3,
    /// Sweep radius.
    pub radius: f32,
}

impl BoneCapsule {
    /// Unsigned distance from `p` to the capsule's surface (0 inside).
    pub fn distance(&self, p: Vec3) -> f32 {
        (distance_point_segment(p, self.a, self.b) - self.radius).max(0.0)
    }
}

/// Weights binding a single vertex to bones, as `(bone_index, weight)` pairs,
/// sorted by descending weight and summing to 1.
pub type VertexWeights = Vec<(usize, f32)>;

/// Skin weights for a whole mesh, one [`VertexWeights`] per input vertex.
#[derive(Clone, Debug, PartialEq)]
pub struct SkinWeights {
    /// Per-vertex influence lists.
    pub per_vertex: Vec<VertexWeights>,
}

impl SkinWeights {
    /// The weight sum of each vertex — every entry should be 1 within float
    /// tolerance. Used by the normalization ordeal.
    pub fn sums(&self) -> Vec<f32> {
        self.per_vertex
            .iter()
            .map(|v| v.iter().map(|(_, w)| *w).sum())
            .collect()
    }

    /// Largest deviation of any vertex weight-sum from 1.0.
    pub fn max_sum_error(&self) -> f32 {
        self.sums()
            .into_iter()
            .map(|s| (s - 1.0).abs())
            .fold(0.0f32, f32::max)
    }
}

/// Build the bone capsules for a skeleton in a given world pose.
pub fn capsules(skeleton: &Skeleton, world: &[Affine3A]) -> Vec<BoneCapsule> {
    assert_eq!(
        world.len(),
        skeleton.bones.len(),
        "world transform count must match skeleton"
    );
    skeleton
        .bones
        .iter()
        .zip(world.iter())
        .map(|(bone, w)| {
            let a = w.transform_point3(Vec3::ZERO);
            let b = w.transform_point3(Vec3::new(0.0, bone.length, 0.0));
            BoneCapsule {
                a,
                b,
                radius: bone.radius,
            }
        })
        .collect()
}

/// Compute skin weights for `vertices` against `capsules`.
///
/// The falloff is `1 / (d^2 + eps)` where `d` is the distance to the capsule
/// surface — strictly positive, so every vertex gets a valid, normalized set of
/// influences. If `max_influences` is `Some(n)`, only the `n` strongest bones
/// per vertex are kept and the remainder renormalized.
pub fn compute_weights(
    capsules: &[BoneCapsule],
    vertices: &[Vec3],
    max_influences: Option<usize>,
) -> SkinWeights {
    let per_vertex = vertices
        .iter()
        .map(|v| weights_for_vertex(capsules, *v, max_influences))
        .collect();
    SkinWeights { per_vertex }
}

/// Convenience: skin `vertices` to a skeleton in its bind (identity) pose.
pub fn bind_weights(
    skeleton: &Skeleton,
    vertices: &[Vec3],
    max_influences: Option<usize>,
) -> SkinWeights {
    let world = Pose::bind(skeleton).forward_kinematics(skeleton);
    let caps = capsules(skeleton, &world);
    compute_weights(&caps, vertices, max_influences)
}

fn weights_for_vertex(
    capsules: &[BoneCapsule],
    v: Vec3,
    max_influences: Option<usize>,
) -> VertexWeights {
    let mut raw: Vec<(usize, f32)> = capsules
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let d = c.distance(v);
            (i, 1.0 / (d * d + FALLOFF_EPS))
        })
        .collect();

    // Strongest first.
    raw.sort_by(|a, b| b.1.total_cmp(&a.1));
    if let Some(n) = max_influences {
        raw.truncate(n.max(1));
    }

    let total: f32 = raw.iter().map(|(_, w)| *w).sum();
    debug_assert!(total > 0.0, "falloff guarantees a positive weight sum");
    for (_, w) in raw.iter_mut() {
        *w /= total;
    }
    raw
}

/// Squared-free distance from a point to a segment.
fn distance_point_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let p = Vec3A::from(p);
    let a = Vec3A::from(a);
    let b = Vec3A::from(b);
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 <= f32::EPSILON {
        0.0
    } else {
        (p - a).dot(ab) / len2
    };
    let t = t.clamp(0.0, 1.0);
    let closest = a + ab * t;
    (p - closest).length()
}
