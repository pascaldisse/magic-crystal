//! Sampler — deterministic, keyed, stateless (ENTROPY law: there is no
//! randomness anywhere; every "random" value = hash(seed, entropy, entity)).
//!
//! Here the entropy coordinate is the tuple (pixel, sample, bounce, dim).
//! Every draw is a pure function of (seed, pixel, sample, bounce, dim) — a
//! cranked counter, PCG-mixed. No hidden state advances, so any traversal
//! order (pixel-major, tiled, threaded later) yields the SAME buffer. Two
//! renders with the same seed are byte-identical by construction, not by
//! luck.
//!
//! `dim` allocation within one path (documented so re-seeding never collides):
//!   dim = bounce * DIMS_PER_BOUNCE + slot
//!   slot 0,1 → BSDF direction sample (u1,u2 — cosine hemisphere OR GGX half)
//!   slot 2   → russian roulette
//!   slot 3   → lobe selection (diffuse vs specular)
pub const DIMS_PER_BOUNCE: u64 = 4;
pub const DIM_HEMI_U1: u64 = 0;
pub const DIM_HEMI_U2: u64 = 1;
pub const DIM_RR: u64 = 2;
pub const DIM_LOBE: u64 = 3;

use crate::vec::{vec3, Vec3};
use std::f64::consts::PI;

/// SplitMix64 finalizer — the mixing primitive (avalanches every input bit).
#[inline]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Hash the full key into a u64. Order-independent of any global state — a
/// pure function of its arguments.
#[inline]
pub fn hash(seed: u64, pixel: u64, sample: u64, bounce: u64, dim: u64) -> u64 {
    // Golden-ratio odd constants keep the per-field folds well separated.
    let mut h = mix64(seed ^ 0x9e3779b97f4a7c15);
    h = mix64(h ^ pixel.wrapping_mul(0xff51afd7ed558ccd));
    h = mix64(h ^ sample.wrapping_mul(0xc4ceb9fe1a85ec53));
    h = mix64(h ^ bounce.wrapping_mul(0xd6e8feb86659fd93));
    mix64(h ^ dim.wrapping_mul(0xa0761d6478bd642f))
}

/// A uniform f64 in `[0,1)` — 53 significant bits from the hash.
#[inline]
pub fn uniform(seed: u64, pixel: u64, sample: u64, bounce: u64, dim: u64) -> f64 {
    let h = hash(seed, pixel, sample, bounce, dim);
    // top 53 bits → [0,1); matches the mantissa of f64, never reaches 1.0.
    ((h >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// Cosine-weighted hemisphere direction about unit `normal` (Malley's
/// method: disk-sample then lift). Returns (direction, pdf) with
/// pdf = cosθ / π. For the lambertian BRDF this pdf cancels the cosine and
/// the 1/π, so the path throughput multiply is exactly the albedo.
pub fn cosine_hemisphere(normal: Vec3, u1: f64, u2: f64) -> (Vec3, f64) {
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - u1).max(0.0).sqrt(); // cosθ
    let (t, bt) = normal.onb();
    let dir = (t * x + bt * y + normal * z).normalize();
    let pdf = z / PI;
    (dir, pdf)
}

/// Reflect incident direction `d` about unit normal `n` (both any handedness).
/// `reflect(d, n) = d - 2(d·n)n`.
pub fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    d - n * (2.0 * d.dot(n))
}

/// Sample a GGX (Trowbridge-Reitz) microfacet HALF-VECTOR about unit `normal`
/// with roughness α (α = roughness², Disney remap — do the remap at the call
/// site). Standard NDF importance sampling (Walter et al. 2007):
///   cosθ_h = sqrt((1-u1) / (1 + (α²-1) u1)),   φ = 2π u2
/// Returns the world-space unit half-vector m. The reflected direction is then
/// `reflect(ω_o_incident, m)`; the caller weights by the Smith G ratio.
pub fn ggx_half(normal: Vec3, alpha: f64, u1: f64, u2: f64) -> Vec3 {
    let a2 = alpha * alpha;
    let cos_t2 = ((1.0 - u1) / (1.0 + (a2 - 1.0) * u1)).clamp(0.0, 1.0);
    let cos_t = cos_t2.sqrt();
    let sin_t = (1.0 - cos_t2).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    let x = sin_t * phi.cos();
    let y = sin_t * phi.sin();
    let (t, bt) = normal.onb();
    (t * x + bt * y + normal * cos_t).normalize()
}

/// Smith masking-shadowing G2 for GGX (height-correlated, Heitz 2014), given
/// the cosines of the incident/outgoing directions with the macro-normal.
/// Used to form the BSDF-sampling weight `G2 · |ω_o·m| / (|ω_o·n|·|m·n|)`.
pub fn smith_g2(cos_o: f64, cos_i: f64, alpha: f64) -> f64 {
    let a2 = alpha * alpha;
    let lambda = |c: f64| -> f64 {
        let c = c.abs().max(1e-6);
        let tan2 = (1.0 - c * c).max(0.0) / (c * c);
        0.5 * (-1.0 + (1.0 + a2 * tan2).sqrt())
    };
    1.0 / (1.0 + lambda(cos_o) + lambda(cos_i))
}

/// The Vec3 unused-import guard for callers that only pull cosine sampling.
#[allow(dead_code)]
fn _touch() -> Vec3 {
    vec3(0.0, 0.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_stable_and_in_range() {
        let a = uniform(42, 1000, 7, 2, 1);
        let b = uniform(42, 1000, 7, 2, 1);
        assert_eq!(a, b); // pure function
        assert!((0.0..1.0).contains(&a));
    }

    #[test]
    fn distinct_keys_differ() {
        let base = uniform(1, 0, 0, 0, 0);
        assert_ne!(base, uniform(2, 0, 0, 0, 0));
        assert_ne!(base, uniform(1, 1, 0, 0, 0));
        assert_ne!(base, uniform(1, 0, 1, 0, 0));
        assert_ne!(base, uniform(1, 0, 0, 1, 0));
        assert_ne!(base, uniform(1, 0, 0, 0, 1));
    }

    #[test]
    fn uniform_mean_near_half() {
        // 100k draws over dim → mean ≈ 0.5 (sanity on the hash's flatness).
        let n = 100_000u64;
        let mut sum = 0.0;
        for i in 0..n {
            sum += uniform(7, 0, i, 0, 0);
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    }

    #[test]
    fn cosine_pdf_matches_and_hemisphere_respected() {
        let n = vec3(0.0, 1.0, 0.0);
        for i in 0..1000u64 {
            let u1 = uniform(3, 0, i, 0, DIM_HEMI_U1);
            let u2 = uniform(3, 0, i, 0, DIM_HEMI_U2);
            let (d, pdf) = cosine_hemisphere(n, u1, u2);
            assert!((d.length() - 1.0).abs() < 1e-9);
            let cos = d.dot(n);
            assert!(cos >= -1e-9, "below hemisphere: {cos}");
            assert!((pdf - cos.max(0.0) / PI).abs() < 1e-9);
        }
    }

    #[test]
    fn cosine_average_is_two_thirds() {
        // E[cosθ] under cosine weighting = ∫ cosθ (cosθ/π) dω = 2/3.
        let n = vec3(0.0, 0.0, 1.0);
        let m = 200_000u64;
        let mut sum = 0.0;
        for i in 0..m {
            let u1 = uniform(9, 0, i, 0, 0);
            let u2 = uniform(9, 0, i, 0, 1);
            let (d, _) = cosine_hemisphere(n, u1, u2);
            sum += d.dot(n);
        }
        let mean = sum / m as f64;
        assert!((mean - 2.0 / 3.0).abs() < 0.005, "mean {mean}");
    }
}
