//! Deterministic point processes — the foliage substrate: density in, instances
//! out (§ GEOMETRY.md foliage-as-density).
//!
//! Grid-jitter, Poisson-disk (hash-driven dart throwing), and density-map
//! scatter. Every point is a [`hash_seq`]/[`unit_f32`] draw keyed by the cell
//! or dart index, so a region's instances depend only on the region's seed —
//! regenerating one region in isolation reproduces it byte-for-byte (the
//! zero-loading property).

use crate::hash::{hash_seq, unit_f32};
use glam::Vec2;

/// An axis-aligned 2D region over which points are scattered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    /// Minimum corner (inclusive).
    pub min: Vec2,
    /// Maximum corner (exclusive).
    pub max: Vec2,
}

impl Region {
    /// A region from corner coordinates.
    #[inline]
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Region { min, max }
    }

    /// Width along x.
    #[inline]
    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    /// Height along y.
    #[inline]
    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }

    /// Is `p` inside `[min, max)`?
    #[inline]
    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.min.x && p.x < self.max.x && p.y >= self.min.y && p.y < self.max.y
    }
}

/// One jittered point per grid cell of edge `cell`, over `region`.
///
/// Each cell `(cx, cy)` places a single point offset within the cell by a hash
/// draw. Points depend only on their cell index, so the result is independent
/// of iteration order and identical when a sub-region's cells are regenerated
/// alone.
pub fn grid_jitter(seed: u64, region: Region, cell: f32) -> Vec<Vec2> {
    assert!(cell > 0.0, "cell size must be positive");
    let nx = (region.width() / cell).ceil() as i32;
    let ny = (region.height() / cell).ceil() as i32;
    let mut out = Vec::with_capacity((nx.max(0) * ny.max(0)) as usize);
    for cy in 0..ny {
        for cx in 0..nx {
            let jx = unit_f32(hash_seq(seed, &[cx as u64, cy as u64, 0]));
            let jy = unit_f32(hash_seq(seed, &[cx as u64, cy as u64, 1]));
            let p = region.min + Vec2::new((cx as f32 + jx) * cell, (cy as f32 + jy) * cell);
            if region.contains(p) {
                out.push(p);
            }
        }
    }
    out
}

/// Density-driven scatter: one candidate per grid cell, accepted with
/// probability `density(point)` (clamped to `[0, 1]`).
///
/// This is the foliage substrate — a density field goes in, instances come
/// out, counts proportional to density. Deterministic and order-independent.
pub fn density_scatter<F: Fn(Vec2) -> f32>(
    seed: u64,
    region: Region,
    cell: f32,
    density: F,
) -> Vec<Vec2> {
    assert!(cell > 0.0, "cell size must be positive");
    let nx = (region.width() / cell).ceil() as i32;
    let ny = (region.height() / cell).ceil() as i32;
    let mut out = Vec::new();
    for cy in 0..ny {
        for cx in 0..nx {
            let jx = unit_f32(hash_seq(seed, &[cx as u64, cy as u64, 0]));
            let jy = unit_f32(hash_seq(seed, &[cx as u64, cy as u64, 1]));
            let p = region.min + Vec2::new((cx as f32 + jx) * cell, (cy as f32 + jy) * cell);
            if !region.contains(p) {
                continue;
            }
            let accept = unit_f32(hash_seq(seed, &[cx as u64, cy as u64, 2]));
            if accept < density(p).clamp(0.0, 1.0) {
                out.push(p);
            }
        }
    }
    out
}

/// Poisson-disk sampling by hash-driven dart throwing.
///
/// Throws `darts` candidate points (each a deterministic hash draw over the
/// region) and keeps a candidate only when it is at least `radius` from every
/// already-kept point. The min-distance property holds for EVERY kept pair by
/// construction. Deterministic in the dart order.
pub fn poisson_disk(seed: u64, region: Region, radius: f32, darts: u32) -> Vec<Vec2> {
    assert!(radius > 0.0, "radius must be positive");
    let r2 = radius * radius;
    let mut kept: Vec<Vec2> = Vec::new();
    for i in 0..darts as u64 {
        let px = region.min.x + unit_f32(hash_seq(seed, &[i, 0])) * region.width();
        let py = region.min.y + unit_f32(hash_seq(seed, &[i, 1])) * region.height();
        let cand = Vec2::new(px, py);
        if kept.iter().all(|&p| p.distance_squared(cand) >= r2) {
            kept.push(cand);
        }
    }
    kept
}
