//! Build NOTES — the surfaced record of every clamp or substitution.
//!
//! The builder NEVER panics on bad input (LAW). When a parameter falls outside
//! its contract range, or a colour string is not a valid schema colour, the
//! builder repairs the value (clamp to the range, substitute the palette
//! default) and records ONE note describing exactly what it changed. A caller
//! that wants strict input treats a non-empty note list as a rejection; a
//! caller that wants best-effort output ignores it. The notes are deterministic
//! — same input, same notes in the same order.

use std::fmt;

/// One repair the builder made to keep a bad parameter from producing an
/// invalid body. Each note names the field, the offending value, and the value
/// that was used instead.
#[derive(Clone, Debug, PartialEq)]
pub enum BuildNote {
    /// A proportion scalar was non-finite (NaN or infinity) and was reset to the
    /// neutral value (LOVE = 1.0).
    ScalarNotFinite {
        /// The scalar field's plain-English name.
        field: &'static str,
        /// The neutral value substituted (LOVE).
        replaced_with: f32,
    },
    /// A proportion scalar was outside `[minimum, maximum]` and was clamped.
    ScalarClamped {
        /// The scalar field's plain-English name.
        field: &'static str,
        /// The value the caller supplied.
        supplied: f32,
        /// The value after clamping into range.
        clamped_to: f32,
    },
    /// A mesh integer parameter was below its floor and was raised.
    MeshFloored {
        /// The mesh field's plain-English name.
        field: &'static str,
        /// The value the caller supplied.
        supplied: usize,
        /// The floor value used instead.
        floored_to: usize,
    },
    /// A palette colour string was not a valid schema colour and was replaced
    /// with the palette default.
    ColorInvalid {
        /// The palette slot's name.
        slot: String,
        /// The invalid colour string the caller supplied.
        supplied: String,
        /// The default colour substituted.
        replaced_with: String,
    },
    /// The palette DEFAULT colour string was itself invalid and was replaced
    /// with the last-resort neutral grey.
    DefaultColorInvalid {
        /// The invalid default the caller supplied.
        supplied: String,
        /// The neutral grey substituted.
        replaced_with: String,
    },
}

impl fmt::Display for BuildNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildNote::ScalarNotFinite {
                field,
                replaced_with,
            } => write!(f, "scalar `{field}` not finite -> {replaced_with}"),
            BuildNote::ScalarClamped {
                field,
                supplied,
                clamped_to,
            } => write!(f, "scalar `{field}` {supplied} clamped -> {clamped_to}"),
            BuildNote::MeshFloored {
                field,
                supplied,
                floored_to,
            } => write!(f, "mesh `{field}` {supplied} floored -> {floored_to}"),
            BuildNote::ColorInvalid {
                slot,
                supplied,
                replaced_with,
            } => write!(
                f,
                "colour slot `{slot}` {supplied:?} invalid -> {replaced_with:?}"
            ),
            BuildNote::DefaultColorInvalid {
                supplied,
                replaced_with,
            } => write!(
                f,
                "palette default {supplied:?} invalid -> {replaced_with:?}"
            ),
        }
    }
}
