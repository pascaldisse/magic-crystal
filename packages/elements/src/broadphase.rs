//! CONSERVATIVE COLLISION BROADPHASE — prune particle-vs-triangle pairs that
//! provably cannot touch, EXACT BY CONSTRUCTION.
//!
//! A uniform spatial grid over the STATIC collider's triangle soup. Each
//! triangle is binned into EVERY cell its axis-aligned bounding box overlaps;
//! a per-particle query gathers the triangles held by the cells the particle's
//! FAT query-AABB overlaps. The exactness rests on two facts:
//!
//!   1. Two AABBs that overlap in space share at least one grid cell (the cell
//!      containing any point of their intersection). So binning triangles by
//!      their AABB and querying by an AABB returns EVERY triangle whose AABB
//!      meets the query box — no false negatives.
//!   2. A particle-vs-triangle CONTACT (`Triangle::contact_depth` returns
//!      `Some`) requires the in-face foot point to lie within the contact
//!      radius of the particle centre — i.e. the triangle has a point within
//!      `radius` of the particle, so its AABB meets the ball of that radius,
//!      so it meets any query AABB that contains that ball.
//!
//! Therefore, as long as the query half-extent (`reach`) is at least the
//! contact radius PLUS the distance the particle can travel while its contacts
//! are resolved this substep, the surviving candidate set is a SUPERSET of
//! every real contact: nothing real is pruned. The narrow phase
//! (`contact_depth`) and the ITERATION ORDER (ascending triangle index) are
//! left byte-identical to the brute-force sweep — the grid only decides WHICH
//! triangles are visited, never how or in what order the surviving ones are
//! resolved. See `tests/pscale_broadphase.rs` for the byte-identical replay
//! and the pruned-pair zero-contact audit that lock this in.
//!
//! The grid is static per scene: it is rebuilt only when the collider's
//! triangle soup changes, detected by a deterministic FNV fingerprint over
//! every vertex (`fingerprint`, built on `hash::StateHasher`).

use crate::collision::Triangle;
use crate::hash::StateHasher;
use crate::math::Vec3;

/// A uniform grid binning a static triangle soup for conservative
/// particle-vs-triangle broadphase. Rebuilt only when [`TriangleGrid::
/// fingerprint`] of the collider changes.
#[derive(Clone, Debug)]
pub struct TriangleGrid {
    /// Edge length of one cubic cell (metres).
    cell: f64,
    inv_cell: f64,
    /// Min corner of the triangle-cloud AABB — grid cell `(0,0,0)` origin.
    origin: Vec3,
    /// Grid resolution per axis (≥ 1 each).
    nx: usize,
    ny: usize,
    nz: usize,
    /// Flattened cells (`(iz*ny + iy)*nx + ix`); each holds the ASCENDING
    /// triangle indices whose AABB overlaps that cell.
    cells: Vec<Vec<u32>>,
    /// The fingerprint of the triangle soup this grid was built from.
    pub fingerprint: u64,
    /// The triangle count this grid was built from.
    pub triangle_count: usize,
}

/// The min/max corners of a triangle's AABB.
#[inline]
fn tri_aabb(t: &Triangle) -> (Vec3, Vec3) {
    let min = Vec3::new(
        t.v0.x.min(t.v1.x).min(t.v2.x),
        t.v0.y.min(t.v1.y).min(t.v2.y),
        t.v0.z.min(t.v1.z).min(t.v2.z),
    );
    let max = Vec3::new(
        t.v0.x.max(t.v1.x).max(t.v2.x),
        t.v0.y.max(t.v1.y).max(t.v2.y),
        t.v0.z.max(t.v1.z).max(t.v2.z),
    );
    (min, max)
}

impl TriangleGrid {
    /// A deterministic fingerprint over every triangle vertex (and the count),
    /// bit-exact via [`StateHasher`]. Two identical soups fingerprint equal;
    /// any moved/added/removed vertex changes it — the grid-rebuild trigger.
    pub fn fingerprint(tris: &[Triangle]) -> u64 {
        let mut h = StateHasher::new();
        h.absorb_u64(tris.len() as u64);
        for t in tris {
            for v in [t.v0, t.v1, t.v2, t.normal] {
                h.absorb_f64(v.x);
                h.absorb_f64(v.y);
                h.absorb_f64(v.z);
            }
        }
        h.finish()
    }

    /// DERIVE the cell edge length from the geometry and the max contact
    /// `reach` — never a plucked constant. Two forces set it:
    ///
    ///   * RESOLUTION: a cell should be at least the mean triangle AABB extent
    ///     (so a triangle spans O(1) cells, not thousands) and at least the
    ///     query reach (so a query touches O(1) cells, not one-per-triangle).
    ///   * MEMORY: the cell count is capped at `budget × triangle_count`
    ///     (a derived density target of ≤ `1/budget` triangles per cell on
    ///     average); if the resolution choice would exceed it over the world
    ///     AABB, the cell grows by the cube-root of the overshoot to fit.
    ///
    /// `reach` is the STATIC contact reach (max particle radius +
    /// contact_margin); the per-substep travel slack is added at QUERY time,
    /// not baked into the static cell size.
    pub fn derive_cell_size(tris: &[Triangle], reach: f64) -> f64 {
        if tris.is_empty() {
            return reach.max(f64::EPSILON);
        }
        // Mean triangle AABB max-extent.
        let mut sum_extent = 0.0;
        let mut world_min = tri_aabb(&tris[0]).0;
        let mut world_max = tri_aabb(&tris[0]).1;
        for t in tris {
            let (lo, hi) = tri_aabb(t);
            let ext = (hi.x - lo.x).max(hi.y - lo.y).max(hi.z - lo.z);
            sum_extent += ext;
            world_min = Vec3::new(world_min.x.min(lo.x), world_min.y.min(lo.y), world_min.z.min(lo.z));
            world_max = Vec3::new(world_max.x.max(hi.x), world_max.y.max(hi.y), world_max.z.max(hi.z));
        }
        let mean_extent = sum_extent / tris.len() as f64;
        let mut cell = mean_extent.max(reach).max(f64::EPSILON);

        // Memory cap: keep total cells ≤ CELL_BUDGET × triangle_count.
        const CELL_BUDGET: f64 = 8.0;
        let span = Vec3::new(
            (world_max.x - world_min.x).max(0.0),
            (world_max.y - world_min.y).max(0.0),
            (world_max.z - world_min.z).max(0.0),
        );
        let cap = CELL_BUDGET * tris.len() as f64;
        loop {
            let dx = (span.x / cell).floor() + 1.0;
            let dy = (span.y / cell).floor() + 1.0;
            let dz = (span.z / cell).floor() + 1.0;
            let total = dx * dy * dz;
            if total <= cap || !total.is_finite() {
                break;
            }
            cell *= (total / cap).cbrt();
        }
        cell
    }

    /// Build the grid over `tris` with the given `cell` size and precomputed
    /// `fingerprint`. Triangles are binned in index order, so each cell's
    /// index list is ASCENDING by construction.
    pub fn build(tris: &[Triangle], cell: f64, fingerprint: u64) -> Self {
        let cell = cell.max(f64::EPSILON);
        let inv_cell = 1.0 / cell;
        if tris.is_empty() {
            return TriangleGrid {
                cell,
                inv_cell,
                origin: Vec3::ZERO,
                nx: 1,
                ny: 1,
                nz: 1,
                cells: vec![Vec::new()],
                fingerprint,
                triangle_count: 0,
            };
        }
        let (mut world_min, mut world_max) = tri_aabb(&tris[0]);
        for t in tris {
            let (lo, hi) = tri_aabb(t);
            world_min = Vec3::new(world_min.x.min(lo.x), world_min.y.min(lo.y), world_min.z.min(lo.z));
            world_max = Vec3::new(world_max.x.max(hi.x), world_max.y.max(hi.y), world_max.z.max(hi.z));
        }
        let nx = (((world_max.x - world_min.x) * inv_cell).floor() as i64 + 1).max(1) as usize;
        let ny = (((world_max.y - world_min.y) * inv_cell).floor() as i64 + 1).max(1) as usize;
        let nz = (((world_max.z - world_min.z) * inv_cell).floor() as i64 + 1).max(1) as usize;
        let mut cells: Vec<Vec<u32>> = vec![Vec::new(); nx * ny * nz];

        let idx = |ix: usize, iy: usize, iz: usize| (iz * ny + iy) * nx + ix;
        let clamp = |v: f64, span: usize| -> usize {
            if v < 0.0 {
                0
            } else {
                (v as usize).min(span - 1)
            }
        };
        for (ti, t) in tris.iter().enumerate() {
            let (lo, hi) = tri_aabb(t);
            let ix0 = clamp(((lo.x - world_min.x) * inv_cell).floor(), nx);
            let ix1 = clamp(((hi.x - world_min.x) * inv_cell).floor(), nx);
            let iy0 = clamp(((lo.y - world_min.y) * inv_cell).floor(), ny);
            let iy1 = clamp(((hi.y - world_min.y) * inv_cell).floor(), ny);
            let iz0 = clamp(((lo.z - world_min.z) * inv_cell).floor(), nz);
            let iz1 = clamp(((hi.z - world_min.z) * inv_cell).floor(), nz);
            for iz in iz0..=iz1 {
                for iy in iy0..=iy1 {
                    for ix in ix0..=ix1 {
                        cells[idx(ix, iy, iz)].push(ti as u32);
                    }
                }
            }
        }
        TriangleGrid {
            cell,
            inv_cell,
            origin: world_min,
            nx,
            ny,
            nz,
            cells,
            fingerprint,
            triangle_count: tris.len(),
        }
    }

    /// Gather the ASCENDING, DEDUPED triangle indices whose cells overlap the
    /// AABB `[min, max]` into `out` (cleared first). A superset of every
    /// triangle whose own AABB meets `[min, max]`.
    pub fn query(&self, min: Vec3, max: Vec3, out: &mut Vec<u32>) {
        out.clear();
        // Clamp the query index range to the grid; skip if it misses entirely.
        let axis = |mn: f64, mx: f64, origin: f64, span: usize| -> Option<(usize, usize)> {
            let a = ((mn - origin) * self.inv_cell).floor();
            let b = ((mx - origin) * self.inv_cell).floor();
            if b < 0.0 || a >= span as f64 {
                return None; // range misses the grid on this axis
            }
            let lo = if a < 0.0 { 0 } else { a as usize };
            let hi = (b as usize).min(span - 1);
            Some((lo, hi))
        };
        let (ix0, ix1) = match axis(min.x, max.x, self.origin.x, self.nx) {
            Some(r) => r,
            None => return,
        };
        let (iy0, iy1) = match axis(min.y, max.y, self.origin.y, self.ny) {
            Some(r) => r,
            None => return,
        };
        let (iz0, iz1) = match axis(min.z, max.z, self.origin.z, self.nz) {
            Some(r) => r,
            None => return,
        };
        for iz in iz0..=iz1 {
            for iy in iy0..=iy1 {
                let row = (iz * self.ny + iy) * self.nx;
                for ix in ix0..=ix1 {
                    for &ti in &self.cells[row + ix] {
                        out.push(ti);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// The cell edge length (metres) — exposed for the cell-size derivation
    /// evidence.
    pub fn cell_size(&self) -> f64 {
        self.cell
    }

    /// Grid resolution `(nx, ny, nz)`.
    pub fn resolution(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}
