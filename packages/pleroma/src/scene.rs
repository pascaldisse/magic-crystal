//! Scene — a bag of (shape, material) primitives and the nearest-hit query.
//! L0 is a linear scan: correctness first, no BVH (that is a later rite's
//! acceleration, and it must converge to THIS truth).

use crate::geometry::{Hit, Ray, Shape};
use crate::material::Material;
use crate::medium::Medium;

#[derive(Clone, Copy, Debug)]
pub struct Primitive {
    pub shape: Shape,
    pub material: Material,
}

#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub prims: Vec<Primitive>,
    /// An optional participating medium marched inside the light pass (Rite VI
    /// A1). `None` = a vacuum scene (the L0/L1 surface-only path, unchanged).
    pub medium: Option<Medium>,
}

impl Scene {
    pub fn new() -> Scene {
        Scene {
            prims: Vec::new(),
            medium: None,
        }
    }

    pub fn add(&mut self, shape: Shape, material: Material) -> &mut Self {
        self.prims.push(Primitive { shape, material });
        self
    }

    /// Nearest hit in (t_min, t_max]; returns the hit and the material it
    /// belongs to.
    pub fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<(Hit, Material)> {
        let mut best: Option<(Hit, Material)> = None;
        let mut closest = t_max;
        for p in &self.prims {
            if let Some(h) = p.shape.hit(ray, t_min, closest) {
                closest = h.t;
                best = Some((h, p.material));
            }
        }
        best
    }
}
