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

pub mod fields;
pub mod hash;
pub mod scatter;

pub use fields::{Fbm, Noise};
pub use hash::{coord_key, domain, hash_seq, mix64, signed_f32, unit_f32, Seed, GOLDEN};
pub use scatter::{density_scatter, grid_jitter, poisson_disk, Region};
