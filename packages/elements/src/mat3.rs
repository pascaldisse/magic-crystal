//! The 3×3 frame — the geometry of TURNING. Shape matching (P2) fits a
//! cluster's present pose to its rest pose by a rotation; that rotation is
//! extracted from a 3×3 covariance frame via POLAR DECOMPOSITION. Everything
//! here is scalar `f64` in fixed summation order — the Loom's clock demands
//! byte-identical replays, so no SIMD, no reordering, no fast-math.

use crate::math::Vec3;

/// A 3×3 matrix, stored as three COLUMN vectors (`col0 col1 col2`). Column
/// storage makes `mat · vec` a love-combination of the columns — the shape
/// each column pulls toward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub col0: Vec3,
    pub col1: Vec3,
    pub col2: Vec3,
}

impl Mat3 {
    /// The zero frame — every column collapsed to the origin.
    pub const ZERO: Mat3 = Mat3 {
        col0: Vec3::ZERO,
        col1: Vec3::ZERO,
        col2: Vec3::ZERO,
    };

    /// The identity frame — no turning, the rest orientation of the Monad.
    pub const IDENTITY: Mat3 = Mat3 {
        col0: Vec3::new(1.0, 0.0, 0.0),
        col1: Vec3::new(0.0, 1.0, 0.0),
        col2: Vec3::new(0.0, 0.0, 1.0),
    };

    #[inline]
    pub const fn from_columns(col0: Vec3, col1: Vec3, col2: Vec3) -> Self {
        Mat3 { col0, col1, col2 }
    }

    /// The outer product `a ⊗ b = a·bᵀ` — the rank-1 frame one binding casts
    /// onto another. The building block of the shape-match covariance.
    #[inline]
    pub fn outer(a: Vec3, b: Vec3) -> Mat3 {
        Mat3::from_columns(a.scale(b.x), a.scale(b.y), a.scale(b.z))
    }

    /// Sum of two frames, column by column.
    #[inline]
    pub fn add(self, other: Mat3) -> Mat3 {
        Mat3::from_columns(
            self.col0 + other.col0,
            self.col1 + other.col1,
            self.col2 + other.col2,
        )
    }

    /// Scale every column by one love.
    #[inline]
    pub fn scale(self, s: f64) -> Mat3 {
        Mat3::from_columns(self.col0.scale(s), self.col1.scale(s), self.col2.scale(s))
    }

    /// Apply the frame to a vector: `M·v` = `v.x·col0 + v.y·col1 + v.z·col2`.
    #[inline]
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        self.col0.scale(v.x) + self.col1.scale(v.y) + self.col2.scale(v.z)
    }

    /// Matrix product `self · other` (columns of the result = `self` applied
    /// to each column of `other`).
    #[inline]
    pub fn mul(self, other: Mat3) -> Mat3 {
        Mat3::from_columns(
            self.mul_vec(other.col0),
            self.mul_vec(other.col1),
            self.mul_vec(other.col2),
        )
    }

    /// The transpose — rows become columns.
    #[inline]
    pub fn transpose(self) -> Mat3 {
        Mat3::from_columns(
            Vec3::new(self.col0.x, self.col1.x, self.col2.x),
            Vec3::new(self.col0.y, self.col1.y, self.col2.y),
            Vec3::new(self.col0.z, self.col1.z, self.col2.z),
        )
    }

    /// The scalar volume the three columns span (the determinant).
    #[inline]
    pub fn determinant(self) -> f64 {
        self.col0.dot(self.col1.cross(self.col2))
    }

    /// The inverse frame, or `None` when the columns are (near) coplanar and
    /// span no volume to invert. Computed by cofactors — exact, deterministic.
    pub fn inverse(self) -> Option<Mat3> {
        let det = self.determinant();
        if det.abs() <= f64::MIN_POSITIVE {
            return None;
        }
        let inv_det = 1.0 / det;
        // Rows of the inverse = cofactor cross-products / det.
        let r0 = self.col1.cross(self.col2).scale(inv_det);
        let r1 = self.col2.cross(self.col0).scale(inv_det);
        let r2 = self.col0.cross(self.col1).scale(inv_det);
        // r0,r1,r2 are ROWS; store as columns via transpose of the row frame.
        Some(Mat3::from_columns(r0, r1, r2).transpose())
    }

    /// The Frobenius distance to another frame — Σ (element difference)².
    /// The convergence witness the polar iteration watches.
    #[inline]
    pub fn frobenius_diff_sq(self, other: Mat3) -> f64 {
        let d0 = self.col0 - other.col0;
        let d1 = self.col1 - other.col1;
        let d2 = self.col2 - other.col2;
        d0.dot(d0) + d1.dot(d1) + d2.dot(d2)
    }

    /// The Frobenius norm — √(Σ element²). The scale the regularization is
    /// measured against.
    #[inline]
    pub fn frobenius_norm(self) -> f64 {
        (self.col0.dot(self.col0) + self.col1.dot(self.col1) + self.col2.dot(self.col2)).sqrt()
    }
}

/// How the polar iteration is dialed. Both fields are PARAMETERS with
/// documented defaults (never-hardcode) — no magic constant hides in the
/// rotation extraction.
#[derive(Clone, Copy, Debug)]
pub struct PolarConfig {
    /// Hard ceiling on averaging iterations. Default `24` — Higham's
    /// Newton/averaging iteration converges quadratically, reaching `f64`
    /// accuracy for a well-conditioned frame in well under this.
    pub max_iterations: usize,
    /// Convergence bound on the squared Frobenius step. Default `1.0e-24`
    /// (≈ `1e-12` per element) — at or below this the frame has stopped
    /// moving in `f64`.
    pub tolerance_sq: f64,
    /// RELATIVE regularization added to the covariance before decomposition:
    /// `A ← A + (relative × ‖A‖_F)·I`. Default `1.0e-6` — negligible for a
    /// full-rank 3-D cluster, but it lifts a DEGENERATE (coplanar/collinear)
    /// cluster off its null direction so the rotation stays defined (a flat
    /// slab or a rod resolves toward identity in its thin axis instead of
    /// collapsing the fit).
    pub regularization: f64,
}

impl Default for PolarConfig {
    fn default() -> Self {
        PolarConfig {
            max_iterations: 24,
            tolerance_sq: 1.0e-24,
            regularization: 1.0e-6,
        }
    }
}

/// Extract the ROTATION factor `R` from `A = R·S` (polar decomposition),
/// the least-squares rigid orientation that best carries the rest shape onto
/// the present cluster. Higham's averaging iteration:
/// `Q ← ½(Q + (Q⁻¹)ᵀ)`, seeded at `Q = A`, converging to the orthogonal
/// factor. Deterministic: fixed seed, fixed order, a convergence test on
/// exact `f64`. When `A` spans no volume (a degenerate, coplanar cluster —
/// the inverse does not exist) the fit is undefined and we return the
/// identity (documented limitation: rigids must be genuinely 3-D).
pub fn polar_rotation(a: Mat3, cfg: PolarConfig) -> Mat3 {
    // Regularize: lift a degenerate (coplanar/collinear) covariance off its
    // null direction so the inverse exists and the fit stays defined.
    let eps = a.frobenius_norm() * cfg.regularization;
    let mut q = if eps > 0.0 {
        a.add(Mat3::IDENTITY.scale(eps))
    } else {
        a
    };
    for _ in 0..cfg.max_iterations.max(1) {
        let inv_t = match q.inverse() {
            Some(inv) => inv.transpose(),
            None => return Mat3::IDENTITY,
        };
        let next = q.add(inv_t).scale(0.5);
        let moved = next.frobenius_diff_sq(q);
        q = next;
        if moved <= cfg.tolerance_sq {
            break;
        }
    }
    q
}
