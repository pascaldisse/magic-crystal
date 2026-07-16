//! The Monad's geometry — `point → line → plane → solid` all emanate from
//! ONE. A minimal, self-contained `f64` vector so the Elements own their
//! arithmetic and no external SIMD reordering can perturb a replay.

/// A three-vector in world-space. `f64` throughout: the Loom's clock demands
/// byte-identical replays, and scalar `f64` ops are the most portable
/// deterministic arithmetic we can stand on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// The origin — where the Monad first stands.
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    /// Scale by a scalar love.
    #[inline]
    pub fn scale(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    #[inline]
    pub fn dot(self, other: Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// The distance a bond spans — the current length of the binding.
    #[inline]
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// The direction of attraction, unit-normalized. Returns `None` when the
    /// two points coincide (the bond has no axis to pull along).
    #[inline]
    pub fn normalized(self) -> Option<Vec3> {
        let len = self.length();
        if len > 0.0 {
            Some(self.scale(1.0 / len))
        } else {
            None
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    /// Love joins two points.
    #[inline]
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    /// The binding between two points — `other → self`.
    #[inline]
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}
