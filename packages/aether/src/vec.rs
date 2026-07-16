//! A dependency-free `f64` 3-vector — the substrate's world coordinate.
//!
//! Kept local (no `glam`) so the CPU reference stays a pure-`std` witness the
//! GPU port (Rite VI) can be measured against. All arithmetic is exact
//! IEEE-754 `f64`; nothing here draws a random number (ENTROPY law).

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A point or direction in world space, `f64` per axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// X axis.
    pub x: f64,
    /// Y axis.
    pub y: f64,
    /// Z axis.
    pub z: f64,
}

/// Construct a [`Vec3`].
#[inline]
pub fn vec3(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Dot product.
    #[inline]
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// Squared Euclidean length (no `sqrt`).
    #[inline]
    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f64 {
        self.length_sq().sqrt()
    }

    /// Unit vector in the same direction. Returns [`Vec3::ZERO`] for a zero
    /// input (no NaN — deterministic degenerate case).
    #[inline]
    pub fn normalize(self) -> Vec3 {
        let len = self.length();
        if len == 0.0 {
            Vec3::ZERO
        } else {
            self / len
        }
    }

    /// Component-wise minimum.
    #[inline]
    pub fn min(self, o: Vec3) -> Vec3 {
        vec3(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }

    /// Component-wise maximum.
    #[inline]
    pub fn max(self, o: Vec3) -> Vec3 {
        vec3(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, o: Vec3) -> Vec3 {
        vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, o: Vec3) -> Vec3 {
        vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    #[inline]
    fn neg(self) -> Vec3 {
        vec3(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, s: f64) -> Vec3 {
        vec3(self.x * s, self.y * s, self.z * s)
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn div(self, s: f64) -> Vec3 {
        vec3(self.x / s, self.y / s, self.z / s)
    }
}
