//! Splittable deterministic hash streams — the entropy law's arithmetic.
//!
//! Every value in DreamForge is `hash(seed, entropy, entity)`, never a rolled
//! die (§ ENTROPY.md "THERE IS NO RANDOMNESS"). This module is that hash: a
//! stateless integer mixer that folds a seed plus a sequence of `u64` keys
//! into a `u64`, plus a [`Seed`] handle that derives hierarchical sub-seeds
//! (world → region → entity → part). Because a sub-seed depends only on its
//! ancestors' keys, any node regenerates in isolation from its coordinates
//! alone — the zero-loading law (§ DREAMFORGE.md "UNIVERSE SCALE, ZERO
//! LOADING").

use glam::{Vec2, Vec3};

/// The golden-ratio odd constant used to decorrelate successive keys.
pub const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Named key-space domains, so distinct generators never collide by accident.
///
/// Any `u64` works as a domain; these are the conventional hierarchy tags.
pub mod domain {
    /// A region of the world (the coarse streaming/generation tile).
    pub const REGION: u64 = 0x01;
    /// An entity within a region.
    pub const ENTITY: u64 = 0x02;
    /// A part within an entity (mesh part, limb, sub-shape).
    pub const PART: u64 = 0x03;
    /// A scatter instance stream.
    pub const SCATTER: u64 = 0x04;
    /// A noise field's lattice.
    pub const FIELD: u64 = 0x05;
}

/// SplitMix64 finalizer — a bijective avalanche mix of a single `u64`.
#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Fold a `seed` and an ordered slice of `keys` into a `u64`.
///
/// Order-sensitive by construction: each key is weighted by its position, so
/// `hash_seq(s, &[1, 2]) != hash_seq(s, &[2, 1])`. Deterministic and
/// allocation-free.
#[inline]
pub fn hash_seq(seed: u64, keys: &[u64]) -> u64 {
    let mut h = mix64(seed ^ GOLDEN);
    for (i, &k) in keys.iter().enumerate() {
        let weighted = k.wrapping_add(GOLDEN.wrapping_mul(i as u64 + 1));
        h = mix64(h ^ mix64(weighted));
    }
    h
}

/// Map raw bits to a `f32` in `[0, 1)`, exactly (top 24 bits / 2^24).
///
/// The result uses only integers representable in the `f32` mantissa and a
/// power-of-two divisor, so it is byte-identical on every platform.
#[inline]
pub fn unit_f32(bits: u64) -> f32 {
    // Top 24 bits — a value in [0, 2^24), exact as f32; divide by 2^24.
    ((bits >> 40) as u32 as f32) / ((1u32 << 24) as f32)
}

/// Map raw bits to a `f32` in `[-1, 1)`.
#[inline]
pub fn signed_f32(bits: u64) -> f32 {
    unit_f32(bits) * 2.0 - 1.0
}

/// A splittable seed: a `u64` plus the derivation operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Seed(pub u64);

impl Seed {
    /// Wrap a raw root seed.
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Seed(seed)
    }

    /// Derive a child seed for a whole `domain` (a named sub-tree).
    #[inline]
    pub fn child(self, domain: u64) -> Seed {
        Seed(hash_seq(self.0, &[domain]))
    }

    /// Derive a child seed for one indexed node within a `domain`.
    ///
    /// This is the hierarchy operator: `world.sub_at(REGION, r)` then
    /// `.sub_at(ENTITY, e)` reaches an entity's stream using only the keys on
    /// the path to it — the isolation property.
    #[inline]
    pub fn sub_at(self, domain: u64, index: u64) -> Seed {
        Seed(hash_seq(self.0, &[domain, index]))
    }

    /// A `u64` draw from this stream at `index`.
    #[inline]
    pub fn u64(self, index: u64) -> u64 {
        hash_seq(self.0, &[index])
    }

    /// A `f32` in `[0, 1)` from this stream at `index`.
    #[inline]
    pub fn f32(self, index: u64) -> f32 {
        unit_f32(self.u64(index))
    }

    /// A `f32` in `[-1, 1)` from this stream at `index`.
    #[inline]
    pub fn signed(self, index: u64) -> f32 {
        signed_f32(self.u64(index))
    }

    /// A `Vec2` with each lane in `[0, 1)`, drawn at `index`.
    #[inline]
    pub fn vec2(self, index: u64) -> Vec2 {
        Vec2::new(
            unit_f32(hash_seq(self.0, &[index, 0])),
            unit_f32(hash_seq(self.0, &[index, 1])),
        )
    }

    /// A `Vec3` with each lane in `[0, 1)`, drawn at `index`.
    #[inline]
    pub fn vec3(self, index: u64) -> Vec3 {
        Vec3::new(
            unit_f32(hash_seq(self.0, &[index, 0])),
            unit_f32(hash_seq(self.0, &[index, 1])),
            unit_f32(hash_seq(self.0, &[index, 2])),
        )
    }
}

/// Encode a signed integer coordinate as a hash key (wrap into `u64`).
///
/// Keeps negative lattice coordinates distinct and bounded.
#[inline]
pub fn coord_key(v: i32) -> u64 {
    v as u32 as u64
}
