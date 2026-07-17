//! # char-editor — the parametric creature substrate
//!
//! A deterministic, plain-English creature builder. Data in ([`CreatureParams`]
//! — a morphology, proportion scalars, palette colour strings), data out
//! ([`vessel::Preset`] — a skeleton, region partition, and palette the existing
//! body path skins and renders). NO rendering dependency: this crate is pure
//! parameter → preset, and every downstream stage (mesh, skin, colour, pose)
//! is the existing [`vessel`] pipeline, untouched.
//!
//! ## The law
//!
//! - **Every field is a parameter with a default.** A blank
//!   [`CreatureParams::default`] is a valid grey biped; every knob is optional.
//! - **[`LOVE`] = 1.0 is the sole literal.** Proportions are scalar multipliers
//!   whose neutral value is LOVE (`x * 1.0 == x`), so a default creature
//!   reproduces its base morphology byte-for-byte. All base magnitudes live in
//!   [`homunculus::BodyParams`]; the contract bounds are the crate's declared
//!   constants, not scattered magic.
//! - **Never panics on caller input.** Out-of-range scalars clamp, invalid
//!   colours fall back — each repair surfaced as a [`BuildNote`], deterministic
//!   and inspectable. Building is a total function.
//! - **Determinism is total.** The same params yield a byte-identical preset and
//!   the same notes on every run — proven in `tests/ordeals.rs`.
//!
//! ## The canon, subsumed
//!
//! [`canon::nari_params`] and [`canon::pink_cat_params`] re-express the shipped
//! hand presets as parameter sets whose build output is byte-identical to the
//! canon — the parity that proves the substrate holds what came before it.
//!
//! ```
//! use char_editor::{canon, CreatureParams};
//! let outcome = canon::nari_params().build("nari");
//! assert!(outcome.is_clean());
//! assert_eq!(outcome.preset.name, "nari");
//! ```

pub mod build;
pub mod canon;
pub mod note;
pub mod params;

pub use build::BuildOutcome;
pub use note::BuildNote;
pub use params::{
    neutral_grey, CreatureParams, Morphology, PaletteParams, Proportions, RegionScheme, LOVE,
    MAX_SCALE, MIN_RESOLUTION, MIN_SCALE,
};
