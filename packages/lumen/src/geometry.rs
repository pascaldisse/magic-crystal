//! Geometry — analytic intersectors. Sphere, axis-aligned box, infinite
//! plane. Each returns the nearest hit in [t_min, t_max] with a unit
//! geometric normal. Hand-checked in tests; these are the ground the whole
//! reference stands on, so they are exact, not approximate.

use crate::vec::{vec3, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, dir: Vec3) -> Ray {
        Ray { origin, dir }
    }
    pub fn at(&self, t: f64) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// A geometric hit: distance, world point, and the OUTWARD unit normal.
/// `front` = ray met the surface from outside.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f64,
    pub point: Vec3,
    pub normal: Vec3,
    pub front: bool,
}

fn oriented(ray: &Ray, t: f64, outward: Vec3) -> Hit {
    let front = ray.dir.dot(outward) < 0.0;
    Hit {
        t,
        point: ray.at(t),
        normal: if front { outward } else { -outward },
        front,
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Shape {
    Sphere {
        center: Vec3,
        radius: f64,
    },
    /// Axis-aligned box given by opposite corners.
    Box {
        min: Vec3,
        max: Vec3,
    },
    /// Infinite plane: point on plane + unit normal.
    Plane {
        point: Vec3,
        normal: Vec3,
    },
}

impl Shape {
    pub fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        match *self {
            Shape::Sphere { center, radius } => sphere(ray, center, radius, t_min, t_max),
            Shape::Box { min, max } => aabb(ray, min, max, t_min, t_max),
            Shape::Plane { point, normal } => plane(ray, point, normal, t_min, t_max),
        }
    }
}

fn sphere(ray: &Ray, center: Vec3, radius: f64, t_min: f64, t_max: f64) -> Option<Hit> {
    // |o + t d - c|^2 = r^2  →  (d·d)t^2 + 2 d·(o-c) t + (|o-c|^2 - r^2) = 0
    let oc = ray.origin - center;
    let a = ray.dir.dot(ray.dir);
    let half_b = oc.dot(ray.dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = half_b * half_b - a * c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    // nearer root first, then farther (so we can hit from inside)
    let mut t = (-half_b - sq) / a;
    if t < t_min || t > t_max {
        t = (-half_b + sq) / a;
        if t < t_min || t > t_max {
            return None;
        }
    }
    let outward = (ray.at(t) - center) / radius;
    Some(oriented(ray, t, outward))
}

fn plane(ray: &Ray, point: Vec3, normal: Vec3, t_min: f64, t_max: f64) -> Option<Hit> {
    let denom = ray.dir.dot(normal);
    if denom.abs() < 1e-12 {
        return None; // parallel
    }
    let t = (point - ray.origin).dot(normal) / denom;
    if t < t_min || t > t_max {
        return None;
    }
    Some(oriented(ray, t, normal))
}

fn aabb(ray: &Ray, min: Vec3, max: Vec3, t_min: f64, t_max: f64) -> Option<Hit> {
    // Slab method, tracking which axis/side the entry (or exit, if inside)
    // came from so we can emit the correct face normal.
    let inv = vec3(1.0 / ray.dir.x, 1.0 / ray.dir.y, 1.0 / ray.dir.z);
    // tenter/texit are the true slab overlap, independent of the query
    // window (so rays starting INSIDE the box get the exit face, not t_min).
    let mut tenter = f64::NEG_INFINITY;
    let mut texit = f64::INFINITY;
    let mut lo_axis = 0usize;
    let mut hi_axis = 0usize;
    let mut lo_neg = false;
    let mut hi_neg = false;
    let o = [ray.origin.x, ray.origin.y, ray.origin.z];
    let mn = [min.x, min.y, min.z];
    let mx = [max.x, max.y, max.z];
    let iv = [inv.x, inv.y, inv.z];
    for axis in 0..3 {
        let mut t0 = (mn[axis] - o[axis]) * iv[axis];
        let mut t1 = (mx[axis] - o[axis]) * iv[axis];
        if iv[axis] < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        // Near face outward normal points toward -axis when the ray travels
        // in +axis (it enters through the min plane), toward +axis otherwise.
        let n0_neg = iv[axis] >= 0.0;
        if t0 > tenter {
            tenter = t0;
            lo_axis = axis;
            lo_neg = n0_neg;
        }
        if t1 < texit {
            texit = t1;
            hi_axis = axis;
            hi_neg = !n0_neg;
        }
        if texit < tenter {
            return None;
        }
    }
    // Entry if it lies in the query window, else the exit face (origin
    // inside the box). Miss if neither is in (t_min, t_max].
    let (t, axis, neg) = if tenter >= t_min && tenter <= t_max {
        (tenter, lo_axis, lo_neg)
    } else if texit >= t_min && texit <= t_max {
        (texit, hi_axis, hi_neg)
    } else {
        return None;
    };
    let mut n = [0.0, 0.0, 0.0];
    n[axis] = if neg { -1.0 } else { 1.0 };
    Some(oriented(ray, t, vec3(n[0], n[1], n[2])))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_front_hit() {
        // unit sphere at origin, ray from -3z toward +z: enter at z=-1, t=2.
        let s = Shape::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let r = Ray::new(vec3(0.0, 0.0, -3.0), vec3(0.0, 0.0, 1.0));
        let h = s.hit(&r, 1e-4, f64::INFINITY).unwrap();
        assert!((h.t - 2.0).abs() < 1e-12);
        assert_eq!(h.point, vec3(0.0, 0.0, -1.0));
        assert_eq!(h.normal, vec3(0.0, 0.0, -1.0));
        assert!(h.front);
    }

    #[test]
    fn sphere_from_inside_flips_normal() {
        let s = Shape::Sphere {
            center: Vec3::ZERO,
            radius: 2.0,
        };
        let r = Ray::new(Vec3::ZERO, vec3(1.0, 0.0, 0.0));
        let h = s.hit(&r, 1e-4, f64::INFINITY).unwrap();
        assert!((h.t - 2.0).abs() < 1e-12);
        // outward normal is +x, but we hit from inside → shading normal -x
        assert_eq!(h.normal, vec3(-1.0, 0.0, 0.0));
        assert!(!h.front);
    }

    #[test]
    fn sphere_miss() {
        let s = Shape::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let r = Ray::new(vec3(0.0, 5.0, -3.0), vec3(0.0, 0.0, 1.0));
        assert!(s.hit(&r, 1e-4, f64::INFINITY).is_none());
    }

    #[test]
    fn plane_hit_and_parallel() {
        let p = Shape::Plane {
            point: Vec3::ZERO,
            normal: vec3(0.0, 1.0, 0.0),
        };
        let r = Ray::new(vec3(0.0, 4.0, 0.0), vec3(0.0, -1.0, 0.0));
        let h = p.hit(&r, 1e-4, f64::INFINITY).unwrap();
        assert!((h.t - 4.0).abs() < 1e-12);
        assert_eq!(h.point, Vec3::ZERO);
        assert_eq!(h.normal, vec3(0.0, 1.0, 0.0));
        // parallel ray misses
        let par = Ray::new(vec3(0.0, 4.0, 0.0), vec3(1.0, 0.0, 0.0));
        assert!(p.hit(&par, 1e-4, f64::INFINITY).is_none());
    }

    #[test]
    fn box_face_normal() {
        // unit cube [0,1]^3, ray from x=-2 toward +x through center → hits
        // the -x face at (0,0.5,0.5), t=2.
        let b = Shape::Box {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        };
        let r = Ray::new(vec3(-2.0, 0.5, 0.5), vec3(1.0, 0.0, 0.0));
        let h = b.hit(&r, 1e-4, f64::INFINITY).unwrap();
        assert!((h.t - 2.0).abs() < 1e-12);
        assert_eq!(h.point, vec3(0.0, 0.5, 0.5));
        assert_eq!(h.normal, vec3(-1.0, 0.0, 0.0));
    }

    #[test]
    fn box_from_inside() {
        let b = Shape::Box {
            min: vec3(-1.0, -1.0, -1.0),
            max: vec3(1.0, 1.0, 1.0),
        };
        let r = Ray::new(Vec3::ZERO, vec3(0.0, 1.0, 0.0));
        let h = b.hit(&r, 1e-4, f64::INFINITY).unwrap();
        assert!((h.t - 1.0).abs() < 1e-12);
        assert_eq!(h.point, vec3(0.0, 1.0, 0.0));
        // exit through +y face; outward is +y but hit from inside → -y
        assert_eq!(h.normal, vec3(0.0, -1.0, 0.0));
        assert!(!h.front);
    }

    #[test]
    fn box_miss() {
        let b = Shape::Box {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        };
        let r = Ray::new(vec3(-2.0, 5.0, 0.5), vec3(1.0, 0.0, 0.0));
        assert!(b.hit(&r, 1e-4, f64::INFINITY).is_none());
    }
}
