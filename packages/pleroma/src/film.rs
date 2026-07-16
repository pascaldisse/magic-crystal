//! Film — the f32 radiance buffer and its PNG relic writer. The buffer is
//! LINEAR HDR radiance (the honest output); the PNG applies sRGB encoding
//! for viewing only. Buffer bytes are the determinism ordeal's subject:
//! same seed ⇒ byte-identical f32 buffer.

use crate::camera::Camera;
use crate::integrator::{radiance, Params};
use crate::scene::Scene;
use crate::vec::{vec3, Vec3};
use std::path::Path;

/// Linear f32 radiance image, row-major, RGB triplets.
#[derive(Clone, Debug)]
pub struct Film {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>, // width*height*3, linear
}

impl Film {
    pub fn new(width: u32, height: u32) -> Film {
        Film {
            width,
            height,
            data: vec![0.0; (width * height * 3) as usize],
        }
    }

    pub fn get(&self, x: u32, y: u32) -> Vec3 {
        let i = ((y * self.width + x) * 3) as usize;
        vec3(
            self.data[i] as f64,
            self.data[i + 1] as f64,
            self.data[i + 2] as f64,
        )
    }

    fn put(&mut self, x: u32, y: u32, c: Vec3) {
        let i = ((y * self.width + x) * 3) as usize;
        self.data[i] = c.x as f32;
        self.data[i + 1] = c.y as f32;
        self.data[i + 2] = c.z as f32;
    }

    /// Render a whole scene through a camera. Single-threaded, pixel-major —
    /// but because sampling is keyed by (pixel,sample,…) the RESULT is
    /// order-independent (a future threaded backend produces the same bytes).
    pub fn render(scene: &Scene, cam: &Camera, w: u32, h: u32, p: &Params) -> Film {
        let mut film = Film::new(w, h);
        for py in 0..h {
            for px in 0..w {
                let pixel = py as u64 * w as u64 + px as u64;
                let mut sum = Vec3::ZERO;
                for s in 0..p.spp {
                    let ray = cam.pixel_ray(px, py, w, h, p.seed, s as u64);
                    sum = sum + radiance(scene, ray, pixel, s as u64, p);
                }
                film.put(px, py, sum / p.spp as f64);
            }
        }
        film
    }
}

/// sRGB opto-electronic transfer (linear → display), inverse of color::parse.
fn linear_to_srgb(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Write the film as an 8-bit sRGB PNG relic. `exposure` scales linear
/// radiance before tonemapping (a dial — no magic constant baked in).
pub fn write_png(film: &Film, path: &Path, exposure: f64) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut bytes = Vec::with_capacity((film.width * film.height * 3) as usize);
    for &v in &film.data {
        let s = linear_to_srgb(v as f64 * exposure);
        bytes.push((s * 255.0 + 0.5) as u8);
    }
    let file = std::fs::File::create(path)?;
    let w = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(w, film.width, film.height);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(std::io::Error::other)?;
    writer
        .write_image_data(&bytes)
        .map_err(std::io::Error::other)?;
    Ok(())
}
