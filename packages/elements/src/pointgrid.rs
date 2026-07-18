//! CONSERVATIVE PARTICLE NEIGHBOUR BROADPHASE — the point-cloud sibling of
//! [`crate::broadphase::TriangleGrid`], for the Position-Based-Fluids density
//! constraint (FLUID lane). Where `TriangleGrid` bins a STATIC triangle soup
//! and answers particle-vs-triangle, `PointGrid` bins a LIVE particle cloud
//! and answers particle-vs-particle: "which particles lie within the
//! smoothing radius `h` of this one?".
//!
//! Same exactness discipline as its sibling. Two facts make the pruning exact:
//!
//!   1. Two AABBs that overlap share at least one grid cell (the cell holding
//!      a point of their intersection). Binning each particle by the single
//!      cell of its POINT and querying by the fat AABB `[c − r, c + r]`
//!      returns EVERY particle whose point lies inside that box — no false
//!      negatives.
//!   2. A neighbour within Euclidean distance `r` of the query centre lies
//!      inside the L∞ box `[c − r, c + r]` (L∞ ≤ L2), so it is never pruned.
//!
//! Therefore a query of half-extent `reach = h` returns a SUPERSET of every
//! particle within `h` — the narrow phase (the actual `‖p_i − p_j‖ ≤ h`
//! distance test) is left to the caller and is byte-identical to the brute
//! all-pairs scan restricted to the returned set. The grid only decides WHICH
//! candidate pairs are distance-tested, never the result of the test. Locked
//! by the neighbour-superset ordeal in `tests/fluid_ordeals.rs`.
//!
//! DERIVED cell size: the cell edge is the query reach `h` itself — the
//! physical kernel support radius, never a plucked constant. A query then
//! touches the 3×3×3 block of cells about the centre (`ceil(h/cell)=1` per
//! side), O(1) cells, not one-per-particle. Rebuilt every substep (the cloud
//! moves); cheap — one bin per particle, one FNV-free integer hash per query.

use crate::math::Vec3;

/// A uniform grid binning a LIVE particle cloud for conservative
/// particle-vs-particle neighbour queries. Rebuilt each substep from the
/// current positions of the indexed subset (the fluid particles).
#[derive(Clone, Debug)]
pub struct PointGrid {
    cell: f64,
    inv_cell: f64,
    origin: Vec3,
    nx: usize,
    ny: usize,
    nz: usize,
    /// Flattened cells (`(iz*ny + iy)*nx + ix`); each holds the ASCENDING
    /// particle indices whose point falls in that cell.
    cells: Vec<Vec<u32>>,
}

impl PointGrid {
    /// DERIVE the cell edge from the query reach `h` (the smoothing radius) —
    /// the physical kernel support, never a tuning literal. A neighbour search
    /// within `h` then visits exactly the 3×3×3 cell block about the query
    /// point. Floored at `f64::EPSILON` so a degenerate `h` still yields a
    /// valid one-cell grid.
    pub fn cell_size(reach: f64) -> f64 {
        reach.max(f64::EPSILON)
    }

    /// Build the grid over the particles named by `indices`, reading their
    /// points from `pos`, with the given `cell` size (use [`PointGrid::
    /// cell_size`]). Particles are binned in the order `indices` lists them,
    /// so — when `indices` is ascending — each cell's list is ASCENDING by
    /// construction (the determinism bedrock).
    pub fn build(pos: &[Vec3], indices: &[usize], cell: f64) -> Self {
        let cell = cell.max(f64::EPSILON);
        let inv_cell = 1.0 / cell;
        if indices.is_empty() {
            return PointGrid {
                cell,
                inv_cell,
                origin: Vec3::ZERO,
                nx: 1,
                ny: 1,
                nz: 1,
                cells: vec![Vec::new()],
            };
        }
        let mut world_min = pos[indices[0]];
        let mut world_max = pos[indices[0]];
        for &i in indices {
            let p = pos[i];
            world_min = Vec3::new(world_min.x.min(p.x), world_min.y.min(p.y), world_min.z.min(p.z));
            world_max = Vec3::new(world_max.x.max(p.x), world_max.y.max(p.y), world_max.z.max(p.z));
        }
        let nx = (((world_max.x - world_min.x) * inv_cell).floor() as i64 + 1).max(1) as usize;
        let ny = (((world_max.y - world_min.y) * inv_cell).floor() as i64 + 1).max(1) as usize;
        let nz = (((world_max.z - world_min.z) * inv_cell).floor() as i64 + 1).max(1) as usize;
        let mut cells: Vec<Vec<u32>> = vec![Vec::new(); nx * ny * nz];
        let clamp = |v: f64, span: usize| -> usize {
            if v < 0.0 {
                0
            } else {
                (v as usize).min(span - 1)
            }
        };
        for &i in indices {
            let p = pos[i];
            let ix = clamp(((p.x - world_min.x) * inv_cell).floor(), nx);
            let iy = clamp(((p.y - world_min.y) * inv_cell).floor(), ny);
            let iz = clamp(((p.z - world_min.z) * inv_cell).floor(), nz);
            cells[(iz * ny + iy) * nx + ix].push(i as u32);
        }
        PointGrid {
            cell,
            inv_cell,
            origin: world_min,
            nx,
            ny,
            nz,
            cells,
        }
    }

    /// Gather the ASCENDING particle indices whose cell overlaps the ball of
    /// radius `reach` about `center` into `out` (cleared first). A SUPERSET of
    /// every particle within Euclidean `reach` of `center` (see the module
    /// proof). Indices arrive ASCENDING because cells are traversed in row-
    /// major order and each cell's own list is ascending — no sort needed when
    /// only ONE cell block is walked, but the block spans several cells whose
    /// index ranges interleave, so the caller must treat the result as a SET.
    /// Kept ascending-within-cell; the caller's narrow phase iterates in the
    /// order returned. For byte-identical determinism the caller should sort
    /// the gathered set if cross-cell global order matters — here the density
    /// sum is order-independent (addition) except for float non-associativity,
    /// so we SORT to pin the summation order exactly.
    pub fn query_ball(&self, center: Vec3, reach: f64, out: &mut Vec<u32>) {
        out.clear();
        let axis = |c: f64, origin: f64, span: usize| -> Option<(usize, usize)> {
            let a = ((c - reach - origin) * self.inv_cell).floor();
            let b = ((c + reach - origin) * self.inv_cell).floor();
            if b < 0.0 || a >= span as f64 {
                return None;
            }
            let lo = if a < 0.0 { 0 } else { a as usize };
            let hi = (b as usize).min(span - 1);
            Some((lo, hi))
        };
        let (ix0, ix1) = match axis(center.x, self.origin.x, self.nx) {
            Some(r) => r,
            None => return,
        };
        let (iy0, iy1) = match axis(center.y, self.origin.y, self.ny) {
            Some(r) => r,
            None => return,
        };
        let (iz0, iz1) = match axis(center.z, self.origin.z, self.nz) {
            Some(r) => r,
            None => return,
        };
        for iz in iz0..=iz1 {
            for iy in iy0..=iy1 {
                let row = (iz * self.ny + iy) * self.nx;
                for ix in ix0..=ix1 {
                    for &pi in &self.cells[row + ix] {
                        out.push(pi);
                    }
                }
            }
        }
        out.sort_unstable();
    }

    /// The cell edge length (metres) — for evidence/derivation.
    pub fn cell(&self) -> f64 {
        self.cell
    }

    /// Grid resolution `(nx, ny, nz)`.
    pub fn resolution(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}
