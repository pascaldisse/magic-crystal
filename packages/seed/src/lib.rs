//! # seed — DreamForge's procedural substrate (S0)
//!
//! Deterministic generation primitives, CPU-side. The crate embodies the
//! entropy law (§ ENTROPY.md): there is no randomness — every value is
//! `hash(seed, coords)`. Three layers:
//!
//! - [`hash`] — splittable deterministic hash streams and hierarchical
//!   sub-seeds (world → region → entity → part); any node regenerates from its
//!   coordinates alone (the zero-loading law).
//! - [`fields`] — value/gradient noise, fBm stacks, and domain warp, built
//!   only on the hash (no `rand` crate).
//! - [`scatter`] — grid-jitter, Poisson-disk, and density-map scatter (the
//!   foliage-as-density substrate; § GEOMETRY.md).

//! - [`terrain`] — seed → Mesh tile sampler (RITE VII "THE PLANET-WALKER",
//!   VII-0 "THE FIRST GROUND"): a height field pure in `(seed, world x/z)`
//!   only, so grid tiles meet at seam-free shared edges by construction.

pub mod fields;
pub mod hash;
pub mod scatter;
pub mod terrain;
pub mod terrain_sigil;

pub use fields::{Fbm, Noise};
pub use hash::{
    coord_key, coord_key_i64, domain, hash_seq, mix64, signed_f32, unit_f32, Seed, GOLDEN,
};
pub use scatter::{density_scatter, grid_jitter, poisson_disk, Region};
pub use terrain::{height, tile_mesh, tile_origin_m, tile_seed, TerrainParams, TerrainTile};
pub use terrain_sigil::TerrainSigil;
