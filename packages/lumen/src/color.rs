//! Color — parse schema color strings into LINEAR RGB.
//!
//! Schema vocab (client/kernel/geometry.js → THREE.Color): CSS-style
//! `#rgb` / `#rrggbb` hex plus a set of CSS named colors. THREE treats those
//! literals as sRGB and converts to linear for lighting; the reference
//! integrator works in LINEAR space, so we do the same sRGB→linear decode
//! here. No ambient, no tint — a string in, a linear radiance/albedo out.

use crate::vec::{vec3, Vec3};

/// sRGB electro-optical transfer, per channel (IEC 61966-2-1). Exact — no
/// gamma-2.2 shortcut.
fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Parse a schema color string → linear RGB. Returns None if unrecognized
/// (callers decide the default; nothing is silently invented here).
pub fn parse(s: &str) -> Option<Vec3> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix('#') {
        return parse_hex(hex);
    }
    named(&s.to_ascii_lowercase()).and_then(parse_hex)
}

/// Parse then fall back to a caller-supplied default (already linear).
pub fn parse_or(s: &str, default: Vec3) -> Vec3 {
    parse(s).unwrap_or(default)
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
    Some(vec3(
        srgb_to_linear(r as f64 / 255.0),
        srgb_to_linear(g as f64 / 255.0),
        srgb_to_linear(b as f64 / 255.0),
    ))
}

/// The named-color slice the schema leans on (CSS names THREE.Color knows).
/// Not exhaustive of CSS — the vocab the realms actually author with.
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
        "crimson" => "dc143c",
        "gold" => "ffd700",
        "teal" => "008080",
        "navy" => "000080",
        "brown" => "a52a2a",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_is_linear_one() {
        let c = parse("#ffffff").unwrap();
        assert!((c.x - 1.0).abs() < 1e-9 && (c.y - 1.0).abs() < 1e-9 && (c.z - 1.0).abs() < 1e-9);
        assert_eq!(parse("white"), parse("#ffffff"));
    }

    #[test]
    fn black_is_zero() {
        assert_eq!(parse("#000000").unwrap(), Vec3::ZERO);
        assert_eq!(parse("black").unwrap(), Vec3::ZERO);
    }

    #[test]
    fn short_hex_expands() {
        assert_eq!(parse("#f00"), parse("#ff0000"));
        assert_eq!(parse("red"), parse("#ff0000"));
    }

    #[test]
    fn mid_gray_decodes_below_half() {
        // 0x80 = 128/255 ≈ 0.502 sRGB → ~0.2159 linear (the whole point of
        // the decode; a naive linear read would keep 0.502).
        let c = parse("#808080").unwrap();
        assert!((c.x - 0.2158605).abs() < 1e-5, "got {}", c.x);
    }

    #[test]
    fn unknown_is_none() {
        assert!(parse("chartreuse-of-mars").is_none());
        assert!(parse("#12").is_none());
    }
}
