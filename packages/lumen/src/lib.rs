//! # Lumen Naturae — the CPU reference path integrator (L0)
//!
//! Paracelsus' light of nature: the one true light, computed offline, tiny,
//! and correct. This crate is the GROUND TRUTH other DreamForge renderers
//! (the GPU integrator, ReSTIR, the SDF/voxel intersector) are tested
//! against. It is unidirectional Monte-Carlo path transport with a
//! deterministic keyed sampler — no fallbacks, no ambient, no randomness.
//!
//! Laws it obeys:
//! - RENDER.md: ONE integrator, real path tracing, no raster lighting path.
//! - ENTROPY.md: no randomness — every draw = hash(seed, entropy, entity).
//! - GRIMOIRE.md: Darkness is truly unlit; there is NO fake ambient term.
//!
//! The four ORDEALS (tests/ordeals.rs) are the point of L0:
//! furnace · analytic direct light · determinism · energy.
//!
//! ## Quick use
//! ```
//! use lumen::{Scene, Shape, Material, Camera, Film, Params, vec3, color};
//! let mut scene = Scene::new();
//! scene.add(
//!     Shape::Sphere { center: vec3(0.0, 0.0, 0.0), radius: 1.0 },
//!     Material::emissive(color::parse("white").unwrap()),
//! );
//! let cam = Camera::new(vec3(0.0, 0.0, 5.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), 40.0, 1.0);
//! let p = Params { spp: 4, ..Params::default() };
//! let film = Film::render(&scene, &cam, 16, 16, &p);
//! assert_eq!(film.width, 16);
//! ```

pub mod camera;
pub mod color;
pub mod film;
pub mod geometry;
pub mod integrator;
pub mod material;
pub mod sampler;
pub mod scene;
pub mod vec;

pub use camera::Camera;
pub use film::{write_png, Film};
pub use geometry::{Hit, Ray, Shape};
pub use integrator::{estimate, radiance, Params};
pub use material::Material;
pub use scene::{Primitive, Scene};
pub use vec::{vec3, Vec3};
