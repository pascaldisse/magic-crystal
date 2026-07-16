//! The body signed-distance field: a smooth union of the skeleton's bone
//! capsules in bind pose.
//!
//! Each bone is an implicit capsule (segment `a`..`b` swept by `radius`, the
//! very [`BoneCapsule`] homunculus derives for skinning). Its signed distance
//! is negative inside, zero on the surface, positive outside. The body field
//! is the polynomial smooth-minimum of every capsule's signed distance — the
//! blend radius `k` fuses limbs into one continuous surface instead of a
//! collection of separate tubes. Meshing (`crate::mesh`) contours the `iso = 0`
//! level set of this field; there is no explicit geometry until then.

use glam::Vec3;
use homunculus::BoneCapsule;

/// Signed distance from `p` to a capsule: negative inside, zero on the swept
/// surface, positive outside. (homunculus's [`BoneCapsule::distance`] clamps the
/// inside to zero for skinning falloff; meshing needs the true signed value.)
pub fn capsule_sdf(p: Vec3, c: &BoneCapsule) -> f32 {
    distance_point_segment(p, c.a, c.b) - c.radius
}

/// Polynomial smooth-minimum of two distances with blend radius `k`.
///
/// For `k -> 0` this converges to `a.min(b)` (a hard union); for larger `k`
/// the two surfaces fuse over a `~k`-wide seam. `smin(INFINITY, d, k) == d`, so
/// it folds cleanly across a capsule list starting from `+INFINITY`.
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

/// The body field over a fixed set of bind-pose capsules.
#[derive(Clone, Debug)]
pub struct BodySdf {
    /// One implicit capsule per bone, in bind-pose world space.
    pub capsules: Vec<BoneCapsule>,
    /// Smooth-union blend radius (metres).
    pub k: f32,
}

impl BodySdf {
    /// Wrap a capsule set with a blend radius.
    pub fn new(capsules: Vec<BoneCapsule>, k: f32) -> Self {
        Self { capsules, k }
    }

    /// Evaluate the field at `p` (negative inside the body, positive outside).
    pub fn eval(&self, p: Vec3) -> f32 {
        let mut d = f32::INFINITY;
        for c in &self.capsules {
            d = smin(d, capsule_sdf(p, c), self.k);
        }
        d
    }

    /// Surface normal at `p`: the normalized field gradient (central
    /// differences with step `eps`), pointing outward (toward increasing
    /// distance). Falls back to `+Y` if the gradient is degenerate.
    pub fn normal(&self, p: Vec3, eps: f32) -> Vec3 {
        let dx = self.eval(p + Vec3::new(eps, 0.0, 0.0)) - self.eval(p - Vec3::new(eps, 0.0, 0.0));
        let dy = self.eval(p + Vec3::new(0.0, eps, 0.0)) - self.eval(p - Vec3::new(0.0, eps, 0.0));
        let dz = self.eval(p + Vec3::new(0.0, 0.0, eps)) - self.eval(p - Vec3::new(0.0, 0.0, eps));
        let g = Vec3::new(dx, dy, dz);
        let len = g.length();
        if len > 1.0e-20 {
            g / len
        } else {
            Vec3::Y
        }
    }

    /// Axis-aligned bounds fully enclosing the `iso = 0` surface, expanded by
    /// `margin` on every side so a contouring grid built from these bounds
    /// keeps the surface strictly interior (a watertight-mesh precondition).
    pub fn bounds(&self, margin: f32) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for c in &self.capsules {
            let r = c.radius + margin;
            lo = lo.min(c.a - r).min(c.b - r);
            hi = hi.max(c.a + r).max(c.b + r);
        }
        (lo, hi)
    }
}

/// Unsigned distance from a point to a segment `a`..`b`.
fn distance_point_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 <= f32::EPSILON {
        0.0
    } else {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    };
    (p - (a + ab * t)).length()
}
