//! Vec3 — the only vector algebra L0 needs. f64 throughout (reference truth,
//! not runtime; precision over speed).

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub const fn vec3(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub const ZERO: Vec3 = vec3(0.0, 0.0, 0.0);
    pub const ONE: Vec3 = vec3(1.0, 1.0, 1.0);

    pub fn splat(v: f64) -> Vec3 {
        vec3(v, v, v)
    }
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: Vec3) -> Vec3 {
        vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }
    pub fn normalize(self) -> Vec3 {
        self / self.length()
    }
    pub fn hadamard(self, o: Vec3) -> Vec3 {
        vec3(self.x * o.x, self.y * o.y, self.z * o.z)
    }
    pub fn max_component(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }
    /// Build an orthonormal basis with `self` as +z (Duff et al. 2017,
    /// branchless — stable, deterministic). Returns (tangent, bitangent).
    pub fn onb(self) -> (Vec3, Vec3) {
        let n = self;
        let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
        let a = -1.0 / (sign + n.z);
        let b = n.x * n.y * a;
        let t = vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        let bt = vec3(b, sign + n.y * n.y * a, -n.y);
        (t, bt)
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        vec3(self.x * s, self.y * s, self.z * s)
    }
}
impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}
impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f64) -> Vec3 {
        vec3(self.x / s, self.y / s, self.z / s)
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        vec3(-self.x, -self.y, -self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_cross_hand_values() {
        let a = vec3(1.0, 0.0, 0.0);
        let b = vec3(0.0, 1.0, 0.0);
        assert_eq!(a.dot(b), 0.0);
        assert_eq!(a.cross(b), vec3(0.0, 0.0, 1.0));
        assert_eq!(vec3(2.0, 3.0, 4.0).dot(vec3(5.0, 6.0, 7.0)), 56.0);
    }

    #[test]
    fn onb_is_orthonormal() {
        for n in [
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, -1.0),
            vec3(1.0, 2.0, 3.0).normalize(),
        ] {
            let (t, bt) = n.onb();
            assert!((t.length() - 1.0).abs() < 1e-12);
            assert!((bt.length() - 1.0).abs() < 1e-12);
            assert!(t.dot(bt).abs() < 1e-12);
            assert!(t.dot(n).abs() < 1e-12);
            assert!(bt.dot(n).abs() < 1e-12);
        }
    }
}
