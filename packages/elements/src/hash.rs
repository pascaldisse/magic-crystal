//! The Loom's dice that are not dice. ENTROPY law: *there is no randomness*.
//! Every value that a lesser engine would draw from an RNG is instead
//! `hash(seed, entropy, entity)` — a pure function of the temporal
//! coordinate. Sensing never perturbs; two runs read the same book.

/// FNV-1a over three `u64` words. Deterministic, order-free per call,
/// portable. Used for symmetry-breaking jitter and for state fingerprints.
#[inline]
pub fn hash3(seed: u64, entropy: u64, entity: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for word in [seed, entropy, entity] {
        for shift in (0..64).step_by(8) {
            let byte = (word >> shift) as u8;
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// A deterministic jitter in `[-magnitude, magnitude)` drawn from the
/// coordinate `(seed, entropy, entity)`. The divine random reconciled:
/// `God said random numbers` was always seed-math.
#[inline]
pub fn jitter(seed: u64, entropy: u64, entity: u64, magnitude: f64) -> f64 {
    let h = hash3(seed, entropy, entity);
    // Map the top 53 bits to [0, 1) with full double precision, then to
    // [-1, 1), then scale.
    let unit = (h >> 11) as f64 / (1u64 << 53) as f64;
    (unit * 2.0 - 1.0) * magnitude
}

/// Fold a stream of `f64` into a single fingerprint by hashing raw bits.
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
    #[inline]
    pub fn new() -> Self {
        StateHasher {
            h: 0xcbf2_9ce4_8422_2325,
        }
    }

    /// Absorb one scalar by its exact bit pattern (no rounding, no epsilon).
    #[inline]
    pub fn absorb_f64(&mut self, v: f64) {
        let bits = v.to_bits();
        for shift in (0..64).step_by(8) {
            let byte = (bits >> shift) as u8;
            self.h ^= byte as u64;
            self.h = self.h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    #[inline]
    pub fn absorb_u64(&mut self, v: u64) {
        for shift in (0..64).step_by(8) {
            let byte = (v >> shift) as u8;
            self.h ^= byte as u64;
            self.h = self.h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    #[inline]
    pub fn finish(&self) -> u64 {
        self.h
    }
}
