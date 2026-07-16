//! Deterministic entropy hash. No RNG, no wall clock: every "random" value is
//! `hash(seed, entropy, entity)` (`gaia-dreamforge/ENTROPY.md`). SplitMix64
//! finalizer over the three mixed inputs — stateless, reproducible forever.

/// Mix `(seed, entropy, entity)` into a uniformly distributed `u64`.
pub fn hash(seed: u64, entropy: u64, entity: u64) -> u64 {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(entropy.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(entity.wrapping_mul(0x94D0_49BB_1331_11EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
