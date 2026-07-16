//! IEEE-754 binary16 (`f16`) pack/unpack, dependency-free.
//!
//! The density grid is stored as `f32` for the CPU reference but is
//! *`f16`-convertible*: the GPU port (Rite VI) uploads half-precision volume
//! textures, so the substrate must be able to round-trip its densities
//! through 16-bit floats and know the error that introduces. Round-to-nearest,
//! ties-to-even; subnormals, infinities and NaN handled.

/// Convert an `f32` to the bit pattern of the nearest `f16`
/// (round-to-nearest, ties-to-even).
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf or NaN. Preserve NaN-ness with a non-zero mantissa.
        let m = if mant != 0 { 0x0200 } else { 0 };
        return sign | 0x7c00 | m;
    }

    // Unbias f32 (127) and rebias f16 (15).
    let new_exp = exp - 127 + 15;

    if new_exp >= 0x1f {
        // Overflow to infinity.
        return sign | 0x7c00;
    }

    if new_exp <= 0 {
        // Subnormal f16 or underflow to zero.
        if new_exp < -10 {
            return sign;
        }
        // Restore the implicit leading 1, then shift into the subnormal range.
        let m = mant | 0x0080_0000;
        let shift = 14 - new_exp; // in [14, 24]
        let half_mant = m >> shift;
        // Round to nearest, ties to even.
        let remainder = m & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let round_up = remainder > halfway || (remainder == halfway && (half_mant & 1) == 1);
        return sign | ((half_mant as u16) + u16::from(round_up));
    }

    // Normal f16.
    let half_mant = mant >> 13;
    let remainder = mant & 0x1fff;
    let round_up = remainder > 0x1000 || (remainder == 0x1000 && (half_mant & 1) == 1);
    let mut out = sign | ((new_exp as u16) << 10) | (half_mant as u16);
    if round_up {
        // Carrying into the exponent is handled naturally by the +1 on the
        // combined exponent|mantissa field.
        out += 1;
    }
    out
}

/// Convert an `f16` bit pattern to `f32` (exact — every `f16` is representable
/// in `f32`).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign); // signed zero
        }
        // Subnormal f16 → normalized f32.
        let mut e = -1i32;
        let mut m = mant;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03ff;
        let new_exp = (e + 127 + 1) as u32;
        return f32::from_bits(sign | (new_exp << 23) | (m << 13));
    }

    if exp == 0x1f {
        // Inf or NaN.
        let m = if mant != 0 { mant << 13 } else { 0 };
        return f32::from_bits(sign | 0x7f80_0000 | m);
    }

    let new_exp = exp + 127 - 15;
    f32::from_bits(sign | (new_exp << 23) | (mant << 13))
}

/// Round-trip an `f32` through `f16` (the loss the GPU texture would incur).
#[inline]
pub fn quantize_f16(value: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(value))
}
