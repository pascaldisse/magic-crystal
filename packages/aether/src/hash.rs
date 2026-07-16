//! The Loom's dice that are not dice (ENTROPY law: *there is no randomness*).
//!
//! Every value a lesser engine would draw from an RNG is here a pure function
//! of an integer coordinate: `hash(seed, …)`. Rasterizing a density field two
//! times with the same seed reads the same book, byte for byte — the
//! determinism ordeal is a property of the substrate, not of luck.

/// SplitMix64 finalizer — avalanches every input bit. The single mixing
/// primitive all samplers in this crate are built from.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Hash a `(seed, x, y, z, stream)` lattice key into a `u64`. Order-free of
/// any global state — a pure function of its arguments, portable across
/// machines. `stream` separates independent draws at the same lattice cell
/// (e.g. octave index) so re-seeding can never collide.
#[inline]
pub fn hash5(seed: u64, x: u64, y: u64, z: u64, stream: u64) -> u64 {
    // Golden-ratio odd constants keep the per-field folds well separated.
    let mut h = mix64(seed ^ 0x9e37_79b9_7f4a_7c15);
    h = mix64(h ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd));
    h = mix64(h ^ y.wrapping_mul(0xc4ce_b9fe_1a85_ec53));
    h = mix64(h ^ z.wrapping_mul(0xd6e8_feb8_6659_fd93));
    mix64(h ^ stream.wrapping_mul(0xa076_1d64_78bd_642f))
}

/// A uniform `f64` in `[0, 1)` — 53 significant bits from [`hash5`]. Never
/// reaches `1.0` (matches the `f64` mantissa width).
#[inline]
pub fn uniform(seed: u64, x: u64, y: u64, z: u64, stream: u64) -> f64 {
    let h = hash5(seed, x, y, z, stream);
    ((h >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// Fold a stream of `f64`/`u64` into one fingerprint by hashing raw bits.
/// The determinism ordeal's witness: identical worldlines fold identical.
#[derive(Clone)]
pub struct StateHasher {
    h: u64,
}

impl Default for StateHasher {
    fn default() -> Self {
        StateHasher::new()
    }
}

impl StateHasher {
    /// A fresh hasher seeded with the FNV offset basis.
    #[inline]
    pub fn new() -> Self {
        StateHasher {
            h: 0xcbf2_9ce4_8422_2325,
        }
    }

    /// Absorb one scalar by its exact bit pattern (no rounding, no epsilon).
    #[inline]
    pub fn absorb_f64(&mut self, v: f64) {
        self.absorb_u64(v.to_bits());
    }

    /// Absorb one `u64` word.
    #[inline]
    pub fn absorb_u64(&mut self, v: u64) {
        self.h ^= v;
        self.h = mix64(self.h);
    }

    /// The current fingerprint.
    #[inline]
    pub fn finish(&self) -> u64 {
        self.h
    }
}
