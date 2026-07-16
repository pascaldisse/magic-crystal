//! Camera — a simple pinhole. Builds primary rays through a virtual sensor.
//! Right-handed, look-from/look-at, vertical FOV. Deterministic pixel
//! jitter comes from the sampler (dims reserved so it never collides with
//! path dims).

use crate::geometry::Ray;
use crate::sampler::uniform;
use crate::vec::{vec3, Vec3};

pub const DIM_PIXEL_X: u64 = 1_000_000; // far from path dims
pub const DIM_PIXEL_Y: u64 = 1_000_001;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    origin: Vec3,
    lower_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    /// `vfov_deg` = vertical field of view in degrees; `aspect` = w/h.
    pub fn new(look_from: Vec3, look_at: Vec3, up: Vec3, vfov_deg: f64, aspect: f64) -> Camera {
        let theta = vfov_deg.to_radians();
        let half_h = (theta * 0.5).tan();
        let half_w = aspect * half_h;
        let w = (look_from - look_at).normalize(); // points back toward camera
        let u = up.cross(w).normalize();
        let v = w.cross(u);
        Camera {
            origin: look_from,
            lower_left: look_from - u * half_w - v * half_h - w,
            horizontal: u * (2.0 * half_w),
            vertical: v * (2.0 * half_h),
        }
    }

    /// Primary ray for normalized screen coords (s,t) ∈ `[0,1]`, s→+x, t→+y up.
    pub fn ray(&self, s: f64, t: f64) -> Ray {
        let dir = self.lower_left + self.horizontal * s + self.vertical * t - self.origin;
        Ray::new(self.origin, dir.normalize())
    }

    /// Jittered primary ray for pixel (px,py) on a (w,h) sensor, sample-keyed.
    pub fn pixel_ray(&self, px: u32, py: u32, w: u32, h: u32, seed: u64, sample: u64) -> Ray {
        let pixel = py as u64 * w as u64 + px as u64;
        let jx = uniform(seed, pixel, sample, 0, DIM_PIXEL_X);
        let jy = uniform(seed, pixel, sample, 0, DIM_PIXEL_Y);
        let s = (px as f64 + jx) / w as f64;
        // flip t so row 0 is the top of the image
        let t = 1.0 - (py as f64 + jy) / h as f64;
        self.ray(s, t)
    }

    #[allow(dead_code)]
    fn _touch() -> Vec3 {
        vec3(0.0, 0.0, 0.0)
    }
}
