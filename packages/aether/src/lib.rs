//! # Aether — the participating-media substrate (DreamForge Rite VI, A0)
//!
//! The CPU reference for volumetric transport: clouds, fire, smoke and steam
//! are **participating media inside the one traced light** (DREAMFORGE
//! VOLUMETRIC LAW — 2D billboards are forbidden vocabulary). This crate is the
//! ground truth the GPU port (Rite VI) will be measured against; here
//! everything is pure-`std`, deterministic `f64`.
//!
//! ## The pieces
//! - [`HomogeneousMedium`] — optical coefficients + Henyey-Greenstein phase.
//! - [`DensityGrid`] — dense, `f16`-convertible density volume (trilinear).
//! - [`Density`] sources — [`SphereFalloff`], [`NoiseStack`], and the
//!   [`SteamColumn`] / [`CloudPuff`] presets (param sets, documented defaults).
//! - [`transport`] — ray-march [`transmittance`] (Beer-Lambert) and the
//!   [`single_scatter`] estimator against a [`Light`].
//!
//! ## Determinism (ENTROPY law)
//! Nothing draws a random number: every "noise" value is `hash(seed, cell,
//! octave)` via [`hash`]. Same seed → byte-identical grid and march results.
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![forbid(unsafe_code)]

pub mod grid;
pub mod half;
pub mod hash;
pub mod medium;
pub mod sources;
pub mod transport;
pub mod vec;

pub use grid::DensityGrid;
pub use medium::HomogeneousMedium;
pub use sources::{CloudPuff, Constant, Density, NoiseStack, SphereFalloff, SteamColumn};
pub use transport::{in_scattered_energy, optical_depth, single_scatter, transmittance, Light};
pub use vec::{vec3, Vec3};
