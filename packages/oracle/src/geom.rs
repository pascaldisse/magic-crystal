//! Minimal vector / bounds math for pure-geometry senses. No GPU, no linear
//! algebra crate — a Glance is a function you call, so keep the math local.

use serde_json::Value;

pub type Vec3 = [f32; 3];

pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn length(a: Vec3) -> f32 {
    dot(a, a).sqrt()
}
pub fn normalize(a: Vec3) -> Vec3 {
    let len = length(a);
    if len <= f32::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [a[0] / len, a[1] / len, a[2] / len]
    }
}

/// Engine yaw/pitch convention, lifted verbatim from `client/kernel/player.js`
/// (`forward = (-sinθcosφ, sinφ, -cosθcosφ)`). yaw 0 looks toward -Z, the
/// three.js camera default; the sense MUST agree with the renderer/player or
/// its convictions (backwards, facing) would be wrong.
pub fn forward(yaw: f32, pitch: f32) -> Vec3 {
    [
        -yaw.sin() * pitch.cos(),
        pitch.sin(),
        -yaw.cos() * pitch.cos(),
    ]
}

/// Right-handed camera basis (forward, right, up) for a yaw/pitch eye. World up
/// is +Y. Matches player.js `right = (cosθ, 0, -sinθ)` at pitch 0.
pub fn camera_basis(yaw: f32, pitch: f32) -> (Vec3, Vec3, Vec3) {
    let fwd = forward(yaw, pitch);
    let world_up = [0.0, 1.0, 0.0];
    let right = normalize(cross(fwd, world_up));
    let up = normalize(cross(right, fwd));
    (fwd, right, up)
}

/// World-space axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}
impl Aabb {
    pub fn point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }
    pub fn from_center_half(center: Vec3, half: Vec3) -> Self {
        Self {
            min: [
                center[0] - half[0],
                center[1] - half[1],
                center[2] - half[2],
            ],
            max: [
                center[0] + half[0],
                center[1] + half[1],
                center[2] + half[2],
            ],
        }
    }
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }
    pub fn center(&self) -> Vec3 {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }
    pub fn size(&self) -> Vec3 {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }
    pub fn max_extent(&self) -> f32 {
        let s = self.size();
        s[0].max(s[1]).max(s[2])
    }
    /// The 8 corners, for frustum projection.
    pub fn corners(&self) -> [Vec3; 8] {
        let (a, b) = (self.min, self.max);
        [
            [a[0], a[1], a[2]],
            [b[0], a[1], a[2]],
            [a[0], b[1], a[2]],
            [b[0], b[1], a[2]],
            [a[0], a[1], b[2]],
            [b[0], a[1], b[2]],
            [a[0], b[1], b[2]],
            [b[0], b[1], b[2]],
        ]
    }
}

/// An affine transform `world = M·local + t`, where `M` (3 columns) folds the
/// GAIA rotation (Euler XYZ) and scale exactly as the renderer's
/// `Mat4::from_scale_rotation_translation` does (`T·R·S`). Kept local — the
/// RAIN law: a Glance is a function you call, so the math stays in-crate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    /// Columns of the 3×3 linear part (rotation ∘ scale).
    pub cols: [Vec3; 3],
    pub t: Vec3,
}
impl Affine {
    pub const IDENTITY: Affine = Affine {
        cols: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t: [0.0, 0.0, 0.0],
    };
    /// Build from GAIA transform fields: `translation·R(euler XYZ)·S(scale)`.
    pub fn from_trs(translation: Vec3, euler_xyz: Vec3, scale: Vec3) -> Self {
        let r = euler_xyz_cols(euler_xyz);
        // Scale multiplies each rotation column (post-multiply by diag(scale)).
        let cols = [
            [r[0][0] * scale[0], r[0][1] * scale[0], r[0][2] * scale[0]],
            [r[1][0] * scale[1], r[1][1] * scale[1], r[1][2] * scale[1]],
            [r[2][0] * scale[2], r[2][1] * scale[2], r[2][2] * scale[2]],
        ];
        Affine {
            cols,
            t: translation,
        }
    }
    /// Apply to a local point.
    pub fn apply(&self, p: Vec3) -> Vec3 {
        [
            self.cols[0][0] * p[0] + self.cols[1][0] * p[1] + self.cols[2][0] * p[2] + self.t[0],
            self.cols[0][1] * p[0] + self.cols[1][1] * p[1] + self.cols[2][1] * p[2] + self.t[1],
            self.cols[0][2] * p[0] + self.cols[1][2] * p[1] + self.cols[2][2] * p[2] + self.t[2],
        ]
    }
    /// Compose so that `self.then(inner).apply(p) == self.apply(inner.apply(p))`
    /// — i.e. `self` is the OUTER (entity) transform, `inner` the part.
    pub fn then(&self, inner: &Affine) -> Affine {
        let c0 = self.linear(inner.cols[0]);
        let c1 = self.linear(inner.cols[1]);
        let c2 = self.linear(inner.cols[2]);
        Affine {
            cols: [c0, c1, c2],
            t: self.apply(inner.t),
        }
    }
    fn linear(&self, v: Vec3) -> Vec3 {
        [
            self.cols[0][0] * v[0] + self.cols[1][0] * v[1] + self.cols[2][0] * v[2],
            self.cols[0][1] * v[0] + self.cols[1][1] * v[1] + self.cols[2][1] * v[2],
            self.cols[0][2] * v[0] + self.cols[1][2] * v[1] + self.cols[2][2] * v[2],
        ]
    }
    /// World AABB of a local AABB carried through this transform (8 corners).
    pub fn transform_aabb(&self, local: &Aabb) -> Aabb {
        let mut out: Option<Aabb> = None;
        for c in local.corners() {
            let p = self.apply(c);
            out = Some(match out {
                Some(a) => a.union(&Aabb::point(p)),
                None => Aabb::point(p),
            });
        }
        out.unwrap_or(Aabb::point(self.t))
    }
}

/// Columns of the rotation matrix for GAIA's Euler XYZ convention
/// (`Rx(x)·Ry(y)·Rz(z)`, matching glam `Quat::from_euler(EulerRot::XYZ, …)`).
pub fn euler_xyz_cols(r: Vec3) -> [Vec3; 3] {
    let (sx, cx) = (r[0].sin(), r[0].cos());
    let (sy, cy) = (r[1].sin(), r[1].cos());
    let (sz, cz) = (r[2].sin(), r[2].cos());
    // Rx*Ry*Rz, returned as columns (col i = image of basis vector e_i).
    let m00 = cy * cz;
    let m01 = -cy * sz;
    let m02 = sy;
    let m10 = sx * sy * cz + cx * sz;
    let m11 = -sx * sy * sz + cx * cz;
    let m12 = -sx * cy;
    let m20 = -cx * sy * cz + sx * sz;
    let m21 = cx * sy * sz + sx * cz;
    let m22 = cx * cy;
    [[m00, m10, m20], [m01, m11, m21], [m02, m12, m22]]
}

/// Slab ray/AABB test. `origin`+`dir` (dir need NOT be unit); returns the entry
/// parameter `t` (in `dir` units) of the nearest intersection within
/// `[tmin, tmax]`, or `None`. With a UNIT `dir`, `t` is the Euclidean distance
/// to the nearest point of the box along that ray.
pub fn ray_aabb(origin: Vec3, dir: Vec3, aabb: &Aabb, tmin: f32, tmax: f32) -> Option<f32> {
    let mut t0 = tmin;
    let mut t1 = tmax;
    for axis in 0..3 {
        let d = dir[axis];
        let (lo, hi) = (aabb.min[axis], aabb.max[axis]);
        if d.abs() < 1e-9 {
            // Ray parallel to this slab: miss if origin outside the slab.
            if origin[axis] < lo || origin[axis] > hi {
                return None;
            }
        } else {
            let inv = 1.0 / d;
            let mut near = (lo - origin[axis]) * inv;
            let mut far = (hi - origin[axis]) * inv;
            if near > far {
                std::mem::swap(&mut near, &mut far);
            }
            t0 = t0.max(near);
            t1 = t1.min(far);
            if t0 > t1 {
                return None;
            }
        }
    }
    Some(t0)
}

/// An inward-facing half-space `dot(n, p) >= offset`.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub n: Vec3,
    pub offset: f32,
}
impl Plane {
    /// True if the AABB lies entirely on the outside (negative) side.
    pub fn aabb_outside(&self, aabb: &Aabb) -> bool {
        // Positive vertex: the AABB corner farthest along the inward normal.
        let px = if self.n[0] >= 0.0 {
            aabb.max[0]
        } else {
            aabb.min[0]
        };
        let py = if self.n[1] >= 0.0 {
            aabb.max[1]
        } else {
            aabb.min[1]
        };
        let pz = if self.n[2] >= 0.0 {
            aabb.max[2]
        } else {
            aabb.min[2]
        };
        dot(self.n, [px, py, pz]) < self.offset
    }
}

/// The six inward-facing planes of a symmetric perspective frustum (aspect 1).
/// The four side planes pass through the eye (apex); near/far are offset along
/// forward. Used for conservative AABB culling in the caption layer.
pub fn frustum_planes(
    eye: Vec3,
    fwd: Vec3,
    right: Vec3,
    up: Vec3,
    tan_half: f32,
    near: f32,
    far: f32,
) -> [Plane; 6] {
    let fix = |n: Vec3| -> Vec3 {
        // Orient inward: the frustum interior lies along +forward.
        if dot(n, fwd) < 0.0 {
            [-n[0], -n[1], -n[2]]
        } else {
            n
        }
    };
    let side = |edge: Vec3, axis: Vec3| -> Plane {
        let n = fix(normalize(cross(axis, normalize(edge))));
        Plane {
            n,
            offset: dot(n, eye),
        }
    };
    let left = side(sub(fwd, scale3(right, tan_half)), up);
    let rightp = side(add(fwd, scale3(right, tan_half)), up);
    let bottom = side(sub(fwd, scale3(up, tan_half)), right);
    let top = side(add(fwd, scale3(up, tan_half)), right);
    let near_p = Plane {
        n: fwd,
        offset: dot(fwd, eye) + near,
    };
    let far_p = Plane {
        n: [-fwd[0], -fwd[1], -fwd[2]],
        offset: -(dot(fwd, eye) + far),
    };
    [left, rightp, bottom, top, near_p, far_p]
}

/// True if `aabb` intersects (or is inside) the frustum described by `planes`.
pub fn frustum_intersects_aabb(planes: &[Plane], aabb: &Aabb) -> bool {
    !planes.iter().any(|p| p.aabb_outside(aabb))
}

pub fn scale3(a: Vec3, s: f32) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Read a JSON array as a Vec3, tolerating ints/floats and short arrays.
pub fn read_vec3(value: Option<&Value>, default: Vec3) -> Vec3 {
    match value.and_then(Value::as_array) {
        Some(a) => {
            let get = |i: usize, d: f32| {
                a.get(i)
                    .and_then(Value::as_f64)
                    .map(|v| v as f32)
                    .unwrap_or(d)
            };
            [get(0, default[0]), get(1, default[1]), get(2, default[2])]
        }
        None => default,
    }
}

pub fn read_f32(value: Option<&Value>, key: &str, default: f32) -> f32 {
    value
        .and_then(|v| v.get(key))
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(default)
}
