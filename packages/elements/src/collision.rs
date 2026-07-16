//! WORLD COLLISION (P2) — particles struck against a STATIC triangle soup.
//! Engine-agnostic: the crate takes a plain list of triangles (positions +
//! normals); a host (the Glass, later) feeds leaf triangles from whatever
//! world it holds. Contact is point-vs-triangle-face: a particle that sinks
//! within its radius of a face is projected back out along the face normal,
//! held by Coulomb friction, and bounced by a restitution velocity pass.
//!
//! Scope (named, deliberate): FACE contact only — a particle whose foot
//! projects inside the triangle. Edge/vertex contact and particle-vs-particle
//! collision are later rites; for grounds, inclines and large faces (what the
//! ordeals need) face contact is exact.

use crate::math::Vec3;

/// One static triangle of the world, with an outward `normal` (the free side
/// the particle is kept on). Winding is the host's business; the normal is
/// authoritative.
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    /// Unit outward normal — the direction a resting particle is pushed.
    pub normal: Vec3,
}

impl Triangle {
    /// A triangle whose normal is derived from the winding `(v1-v0)×(v2-v0)`,
    /// normalized. Degenerate (zero-area) triangles keep a zero normal and
    /// never register contact.
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        let normal = (v1 - v0).cross(v2 - v0).normalized().unwrap_or(Vec3::ZERO);
        Triangle { v0, v1, v2, normal }
    }

    /// A triangle with an explicitly supplied outward `normal` (the host may
    /// know the free side better than the winding does).
    pub fn with_normal(v0: Vec3, v1: Vec3, v2: Vec3, normal: Vec3) -> Self {
        Triangle {
            v0,
            v1,
            v2,
            normal: normal.normalized().unwrap_or(Vec3::ZERO),
        }
    }

    /// Does the point's projection onto this triangle's plane fall inside the
    /// triangle? Barycentric sign test — all three edge cross-products must
    /// agree with the normal.
    fn contains_projection(&self, point: Vec3) -> bool {
        let n = self.normal;
        let c0 = (self.v1 - self.v0).cross(point - self.v0).dot(n);
        let c1 = (self.v2 - self.v1).cross(point - self.v1).dot(n);
        let c2 = (self.v0 - self.v2).cross(point - self.v2).dot(n);
        (c0 >= 0.0 && c1 >= 0.0 && c2 >= 0.0) || (c0 <= 0.0 && c1 <= 0.0 && c2 <= 0.0)
    }

    /// If `pos` (a particle of the given `radius`) is in contact with this
    /// face, return the penetration `depth` (how far to push along `normal`).
    /// `None` when the particle is clear of the face or off its footprint.
    pub fn contact_depth(&self, pos: Vec3, radius: f64) -> Option<f64> {
        let signed = (pos - self.v0).dot(self.normal);
        if signed >= radius {
            return None; // above the face by more than the contact radius
        }
        let foot = pos - self.normal.scale(signed); // projection onto plane
        if !self.contains_projection(foot) {
            return None; // off the triangle's footprint
        }
        Some(radius - signed)
    }
}

/// The friction + restitution law of a contact surface. Every field is a
/// PARAMETER with a documented default (never-hardcode). Coulomb: a resting
/// particle holds while `tan(slope) < friction_static`, and slides at
/// `friction_dynamic` once broken free.
#[derive(Clone, Copy, Debug)]
pub struct ContactMaterial {
    /// Static (stiction) coefficient μ_s. Default `0.6` — the critical repose
    /// angle is `atan(0.6) ≈ 30.96°`; a box on a gentler slope holds.
    pub friction_static: f64,
    /// Kinetic coefficient μ_d, `≤ μ_s`. Default `0.4`.
    pub friction_dynamic: f64,
    /// Restitution `e ∈ [0,1]` — outgoing normal speed / incoming. Default
    /// `0.2` (a dull thud; `0` = fully plastic, `1` = lossless bounce).
    pub restitution: f64,
    /// Contact skin, in metres: a particle is in contact while it is within
    /// `radius + contact_margin` of a face. Default `1.0e-3` (1 mm). This
    /// keeps a RESTING contact live every substep — without it a particle
    /// riding exactly at its radius registers contact only intermittently,
    /// and gravity's tangential pull leaks past friction on the empty
    /// substeps (a resting body would creep). The particle hovers within the
    /// margin, well under authoring tolerances.
    pub contact_margin: f64,
}

impl Default for ContactMaterial {
    fn default() -> Self {
        ContactMaterial {
            friction_static: 0.6,
            friction_dynamic: 0.4,
            restitution: 0.2,
            contact_margin: 1.0e-3,
        }
    }
}

/// The static world a particle body strikes: a soup of [`Triangle`]s and the
/// [`ContactMaterial`] that governs every contact against it. Immutable during
/// a step (the Glass owns any rebuild between steps).
#[derive(Clone, Debug, Default)]
pub struct Collider {
    pub triangles: Vec<Triangle>,
    pub material: ContactMaterial,
}

impl Collider {
    /// An empty world — no faces, no contact.
    pub fn new(material: ContactMaterial) -> Self {
        Collider {
            triangles: Vec::new(),
            material,
        }
    }

    /// A finite horizontal ground plane at height `y`, a square of half-extent
    /// `half` centred on the origin in xz, normal `+y`. Two triangles — the
    /// canonical floor the drop ordeals fall onto.
    pub fn ground_plane(y: f64, half: f64, material: ContactMaterial) -> Self {
        let up = Vec3::new(0.0, 1.0, 0.0);
        let a = Vec3::new(-half, y, -half);
        let b = Vec3::new(half, y, -half);
        let c = Vec3::new(half, y, half);
        let d = Vec3::new(-half, y, half);
        Collider {
            triangles: vec![
                Triangle::with_normal(a, b, c, up),
                Triangle::with_normal(a, c, d, up),
            ],
            material,
        }
    }

    /// A finite planar RAMP tilted by `angle` radians about the world x-axis,
    /// passing through the origin, of half-extent `half`. The outward normal
    /// tilts with it (`sin θ` in +z, `cos θ` in +y) — a slope to test repose
    /// against. Down-slope points toward `-z` as `angle` grows.
    pub fn incline(angle: f64, half: f64, material: ContactMaterial) -> Self {
        let (s, c) = (angle.sin(), angle.cos());
        let normal = Vec3::new(0.0, c, s);
        // Two in-plane axes: world x (unchanged), and the up-slope direction.
        let along_x = Vec3::new(1.0, 0.0, 0.0);
        let up_slope = Vec3::new(0.0, s, -c); // perpendicular to normal & x
        let corner = |sx: f64, su: f64| along_x.scale(sx * half) + up_slope.scale(su * half);
        let a = corner(-1.0, -1.0);
        let b = corner(1.0, -1.0);
        let cc = corner(1.0, 1.0);
        let d = corner(-1.0, 1.0);
        Collider {
            triangles: vec![
                Triangle::with_normal(a, b, cc, normal),
                Triangle::with_normal(a, cc, d, normal),
            ],
            material,
        }
    }
}

/// A recorded touch this substep: the particle, the surface normal it met,
/// and the restitution owed. Read by the velocity pass to bounce the body.
#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub particle: usize,
    pub normal: Vec3,
    pub restitution: f64,
}
