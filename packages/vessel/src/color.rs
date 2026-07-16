//! Schema color strings ↔ linear RGB.
//!
//! DreamForge law: colors are authored and surfaced as STRINGS (the
//! EMISSIVE/color = string law — a part carries `"color": "#f3e0d0"`, never a
//! float triple). This module is the geometry-side bridge: it parses the schema
//! vocabulary (`#rgb` / `#rrggbb` hex + the CSS names the realms author with)
//! into LINEAR RGB for blending, and re-encodes a blended linear color back
//! into a canonical lowercase `#rrggbb` string. Every per-vertex color the
//! vessel emits therefore stays a valid schema color string.
//!
//! The vocabulary mirrors the client (`geometry.js` → `THREE.Color`), which
//! treats the literals as sRGB; we do the exact IEC 61966-2-1 sRGB↔linear
//! decode so blends happen in linear light, not in gamma space.

use glam::Vec3;

/// sRGB electro-optical transfer (decode), per channel. Exact — no gamma-2.2
/// shortcut.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse sRGB transfer (encode), per channel.
fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Parse a schema color string → linear RGB. `None` if unrecognized (callers
/// decide the default; nothing is silently invented here).
pub fn parse(s: &str) -> Option<Vec3> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix('#') {
        return parse_hex(hex);
    }
    named(&s.to_ascii_lowercase()).and_then(parse_hex)
}

/// Whether `s` is a color string this vocabulary accepts.
pub fn is_valid(s: &str) -> bool {
    parse(s).is_some()
}

/// Encode a LINEAR RGB color into a canonical lowercase `#rrggbb` schema
/// string (sRGB-encoded, clamped to `[0,1]`, rounded to 8-bit). Deterministic:
/// identical input → identical string.
pub fn to_hex(linear: Vec3) -> String {
    let ch = |c: f32| (linear_to_srgb(c) * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8;
    format!(
        "#{:02x}{:02x}{:02x}",
        ch(linear.x),
        ch(linear.y),
        ch(linear.z)
    )
}

fn parse_hex(hex: &str) -> Option<Vec3> {
    let full = match hex.len() {
        3 => hex.chars().flat_map(|c| [c, c]).collect::<String>(),
        6 => hex.to_string(),
        _ => return None,
    };
    let r = u8::from_str_radix(&full[0..2], 16).ok()?;
    let g = u8::from_str_radix(&full[2..4], 16).ok()?;
    let b = u8::from_str_radix(&full[4..6], 16).ok()?;
    Some(Vec3::new(
        srgb_to_linear(r as f32 / 255.0),
        srgb_to_linear(g as f32 / 255.0),
        srgb_to_linear(b as f32 / 255.0),
    ))
}

/// The named-color slice the schema leans on (CSS names `THREE.Color` knows).
/// Not exhaustive of CSS — the vocab the realms actually author with, plus the
/// avatar-palette names nari and the naruko cat are dressed in.
fn named(name: &str) -> Option<&'static str> {
    Some(match name {
        "black" => "000000",
        "white" => "ffffff",
        "red" => "ff0000",
        "green" => "008000",
        "lime" => "00ff00",
        "blue" => "0000ff",
        "yellow" => "ffff00",
        "cyan" | "aqua" => "00ffff",
        "magenta" | "fuchsia" => "ff00ff",
        "gray" | "grey" => "808080",
        "silver" => "c0c0c0",
        "orange" => "ffa500",
        "purple" => "800080",
        "violet" => "7c3aed",
        "crimson" => "dc143c",
        "gold" => "ffd700",
        "teal" => "008080",
        "navy" => "000080",
        "brown" => "a52a2a",
        "pink" => "ffc0cb",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_black_roundtrip() {
        assert_eq!(to_hex(parse("white").unwrap()), "#ffffff");
        assert_eq!(to_hex(parse("black").unwrap()), "#000000");
        assert_eq!(parse("white"), parse("#ffffff"));
    }

    #[test]
    fn short_hex_expands() {
        assert_eq!(parse("#f00"), parse("#ff0000"));
    }

    #[test]
    fn hex_roundtrips_canonical() {
        for h in ["#f3e0d0", "#7c3aed", "#ffc0cb", "#1a1a1a"] {
            assert_eq!(to_hex(parse(h).unwrap()), h, "roundtrip {h}");
        }
    }

    #[test]
    fn unknown_is_none() {
        assert!(parse("chartreuse-of-mars").is_none());
        assert!(parse("#12").is_none());
        assert!(!is_valid("nonsense"));
    }
}
