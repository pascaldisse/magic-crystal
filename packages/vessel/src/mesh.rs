//! Marching-cubes contouring of the body SDF into a triangle mesh.
//!
//! The grid uses cubic cells (edge = longest-bounds axis / `resolution`) so
//! aspect never skews the sampling. Iso-crossing vertices are welded per grid
//! EDGE — both cells sharing an edge reference the same vertex index — which is
//! what makes the output watertight (every interior edge shared by exactly two
//! triangles). Positions interpolate the field linearly along the crossed edge;
//! normals are the SDF gradient at each welded vertex (smooth, shared).
//!
//! No UVs are emitted — texturing is virtual (a DreamForge law); this is pure
//! position + normal + index geometry.

use crate::sdf::BodySdf;
use crate::tables::{CORNER_OFFSETS, EDGE_CORNERS, EDGE_TABLE, TRI_TABLE};
use glam::Vec3;
use std::collections::HashMap;

/// A triangle mesh: parallel position/normal arrays indexed by `indices`
/// (three per triangle). No UVs (virtual texturing — DreamForge law).
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    /// Vertex positions.
    pub positions: Vec<Vec3>,
    /// Per-vertex normals (unit length).
    pub normals: Vec<Vec3>,
    /// Triangle vertex indices (`3 * triangle_count` entries).
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Canonical little-endian byte serialization (positions, then normals,
    /// then indices) — the form the determinism ordeal compares.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.positions.len() * 24 + self.indices.len() * 4);
        for p in &self.positions {
            out.extend_from_slice(&p.x.to_le_bytes());
            out.extend_from_slice(&p.y.to_le_bytes());
            out.extend_from_slice(&p.z.to_le_bytes());
        }
        for n in &self.normals {
            out.extend_from_slice(&n.x.to_le_bytes());
            out.extend_from_slice(&n.y.to_le_bytes());
            out.extend_from_slice(&n.z.to_le_bytes());
        }
        for i in &self.indices {
            out.extend_from_slice(&i.to_le_bytes());
        }
        out
    }
}

/// Contour the `iso` level set of `sdf` over the axis-aligned box `[lo, hi]`.
///
/// `resolution` sets the number of cubic cells along the longest box axis; the
/// other axes get as many cells of the same size as fit. The box must fully
/// enclose the surface with a margin (see [`BodySdf::bounds`]) so no crossing
/// touches the boundary — that is what guarantees a closed mesh.
pub fn marching_cubes(sdf: &BodySdf, lo: Vec3, hi: Vec3, resolution: usize, iso: f32) -> Mesh {
    let resolution = resolution.max(1);
    let extent = hi - lo;
    let longest = extent.max_element().max(1.0e-6);
    let cell = longest / resolution as f32;

    // Cell counts per axis (>= 1); the grid may slightly overshoot `hi` so the
    // cells stay cubic — that keeps the surface strictly interior, never clips.
    let nx = ((extent.x / cell).ceil() as usize).max(1);
    let ny = ((extent.y / cell).ceil() as usize).max(1);
    let nz = ((extent.z / cell).ceil() as usize).max(1);
    let (cx, cy, cz) = (nx + 1, ny + 1, nz + 1); // corner counts per axis

    // Gradient step for normals: a small fraction of the cell.
    let grad_eps = cell * 0.25;

    // Snap window: field values within this of `iso` are nudged off it so no
    // edge-crossing lands exactly on a corner (which would spawn coincident,
    // degenerate triangles). Derived from cell size, tiny relative to it.
    let snap = cell * 1.0e-3;

    let corner_pos = |ix: usize, iy: usize, iz: usize| {
        Vec3::new(
            lo.x + ix as f32 * cell,
            lo.y + iy as f32 * cell,
            lo.z + iz as f32 * cell,
        )
    };
    let corner_lin = |ix: usize, iy: usize, iz: usize| ix + iy * cx + iz * cx * cy;

    // Precompute (snapped) field value at every grid corner — deterministic,
    // and each corner sampled once.
    let mut field = vec![0.0f32; cx * cy * cz];
    for iz in 0..cz {
        for iy in 0..cy {
            for ix in 0..cx {
                let mut v = sdf.eval(corner_pos(ix, iy, iz)) - iso;
                if v.abs() < snap {
                    v = if v >= 0.0 { snap } else { -snap };
                }
                field[corner_lin(ix, iy, iz)] = v;
            }
        }
    }

    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    // Welds an iso-crossing to a single vertex per grid edge (keyed by the two
    // global corner linear indices), so shared edges share vertices.
    let mut edge_verts: HashMap<u64, u32> = HashMap::new();

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                // Sample the 8 corners in table order.
                let mut vals = [0.0f32; 8];
                let mut lins = [0usize; 8];
                let mut cube = 0u8;
                for (corner, off) in CORNER_OFFSETS.iter().enumerate() {
                    let (jx, jy, jz) = (ix + off[0], iy + off[1], iz + off[2]);
                    let lin = corner_lin(jx, jy, jz);
                    lins[corner] = lin;
                    vals[corner] = field[lin];
                    // Bourke convention: bit set when the corner is inside
                    // (field < 0, i.e. within the body).
                    if vals[corner] < 0.0 {
                        cube |= 1 << corner;
                    }
                }

                let edges = EDGE_TABLE[cube as usize];
                if edges == 0 {
                    continue;
                }

                // Resolve the (up to) 12 crossed-edge vertices, welding to the
                // shared per-grid-edge vertex.
                let mut vert_idx = [u32::MAX; 12];
                for (e, corners) in EDGE_CORNERS.iter().enumerate() {
                    if edges & (1 << e) == 0 {
                        continue;
                    }
                    let (a, b) = (corners[0], corners[1]);
                    let (la, lb) = (lins[a], lins[b]);
                    let key = if la < lb {
                        (la as u64) << 32 | lb as u64
                    } else {
                        (lb as u64) << 32 | la as u64
                    };
                    let idx = *edge_verts.entry(key).or_insert_with(|| {
                        // Corner positions from linear indices.
                        let pa = lin_to_pos(la, cx, cy, lo, cell);
                        let pb = lin_to_pos(lb, cx, cy, lo, cell);
                        let (va, vb) = (field[la], field[lb]);
                        // Both non-zero and opposite sign (snap guaranteed):
                        // interior linear crossing.
                        let t = va / (va - vb);
                        let p = pa + (pb - pa) * t;
                        let n = sdf.normal(p, grad_eps);
                        let id = positions.len() as u32;
                        positions.push(p);
                        normals.push(n);
                        id
                    });
                    vert_idx[e] = idx;
                }

                // Emit triangles from the tri-table.
                let tris = &TRI_TABLE[cube as usize];
                let mut t = 0;
                while t < 16 && tris[t] >= 0 {
                    let e0 = tris[t] as usize;
                    let e1 = tris[t + 1] as usize;
                    let e2 = tris[t + 2] as usize;
                    let (i0, i1, i2) = (vert_idx[e0], vert_idx[e1], vert_idx[e2]);
                    debug_assert!(
                        i0 != u32::MAX && i1 != u32::MAX && i2 != u32::MAX,
                        "tri-table referenced an uncrossed edge"
                    );
                    // Orient so the geometric normal agrees with the SDF
                    // gradient (outward); watertightness is winding-agnostic.
                    let (p0, p1, p2) = (
                        positions[i0 as usize],
                        positions[i1 as usize],
                        positions[i2 as usize],
                    );
                    let geo = (p1 - p0).cross(p2 - p0);
                    let outward =
                        normals[i0 as usize] + normals[i1 as usize] + normals[i2 as usize];
                    if geo.dot(outward) >= 0.0 {
                        indices.extend_from_slice(&[i0, i1, i2]);
                    } else {
                        indices.extend_from_slice(&[i0, i2, i1]);
                    }
                    t += 3;
                }
            }
        }
    }

    Mesh {
        positions,
        normals,
        indices,
    }
}

fn lin_to_pos(lin: usize, cx: usize, cy: usize, lo: Vec3, cell: f32) -> Vec3 {
    let ix = lin % cx;
    let iy = (lin / cx) % cy;
    let iz = lin / (cx * cy);
    Vec3::new(
        lo.x + ix as f32 * cell,
        lo.y + iy as f32 * cell,
        lo.z + iz as f32 * cell,
    )
}
