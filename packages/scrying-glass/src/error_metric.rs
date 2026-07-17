//! RITE VIII-0 — THE NOISE AND THE TRUTH: the error metric between two
//! resolved images. No net lands in this wave; this module is the ruler every
//! later denoiser claim (VIII-1+) measures against, so it must itself be
//! trustworthy: f64 accumulation (a resolved image can be large; f32 summation
//! drifts), a FIXED index order (plain `for i in 0..n`, never a HashMap or any
//! other order-unstable structure — byte/bit-deterministic across runs), and a
//! self-test baked into the ordeals (`error(ref, ref) == 0e0` EXACTLY, not
//! "close to zero").
//!
//! Scope note for the BAN grep-gate (see `tests/viii0_ordeals.rs`): this file
//! is a NEW module added whole-cloth for VIII-0, so the gate scans it in
//! full — via the `// BAN-SCOPED` marker below, the same forward-proof
//! mechanism the VIII-1 net module is expected to carry.

// BAN-SCOPED

use glam::Vec3;

/// Root-mean-square error between two resolved images (per-channel), summed
/// in f64 to avoid f32 accumulation drift over large images, then reduced to
/// a single f32 for reporting (the accumulation precision matters; the report
/// precision does not).
///
/// Panics if `a.len() != b.len()` — comparing images of different pixel
/// counts is a caller bug, not a metric result.
pub fn rmse(a: &[Vec3], b: &[Vec3]) -> f64 {
    assert_eq!(a.len(), b.len(), "rmse: image pixel counts differ");
    if a.is_empty() {
        return 0.0;
    }
    let mut sum_sq = 0.0f64;
    for i in 0..a.len() {
        let dx = (a[i].x - b[i].x) as f64;
        let dy = (a[i].y - b[i].y) as f64;
        let dz = (a[i].z - b[i].z) as f64;
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    let n = (a.len() * 3) as f64;
    (sum_sq / n).sqrt()
}

/// Mean absolute error between two resolved images (per-channel), f64
/// accumulation, fixed index order — see [`rmse`] for the rationale.
pub fn mae(a: &[Vec3], b: &[Vec3]) -> f64 {
    assert_eq!(a.len(), b.len(), "mae: image pixel counts differ");
    if a.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        sum += (a[i].x - b[i].x).abs() as f64;
        sum += (a[i].y - b[i].y).abs() as f64;
        sum += (a[i].z - b[i].z).abs() as f64;
    }
    let n = (a.len() * 3) as f64;
    sum / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmse_self_is_exactly_zero() {
        let img = vec![Vec3::new(0.1, 0.2, 0.3), Vec3::new(1.0, 0.0, 5.0)];
        assert_eq!(rmse(&img, &img), 0.0f64);
    }

    #[test]
    fn mae_self_is_exactly_zero() {
        let img = vec![Vec3::new(0.1, 0.2, 0.3), Vec3::new(1.0, 0.0, 5.0)];
        assert_eq!(mae(&img, &img), 0.0f64);
    }

    #[test]
    fn rmse_and_mae_discriminate_a_real_difference() {
        let a = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.5, 0.5, 0.5)];
        // b differs from a at every channel of every pixel — not a degenerate
        // all-zero-diff case.
        let b = vec![Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 0.0)];
        assert!(rmse(&a, &b) > 0.0);
        assert!(mae(&a, &b) > 0.0);
    }

    #[test]
    fn empty_images_are_zero_not_nan() {
        let empty: Vec<Vec3> = Vec::new();
        assert_eq!(rmse(&empty, &empty), 0.0f64);
        assert_eq!(mae(&empty, &empty), 0.0f64);
    }
}
