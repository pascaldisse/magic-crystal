//! The acceleration — a CPU-built bounding-volume hierarchy over the Great
//! Chain's leaf triangles (Rite IV, L1). The traced integrator (`integrator.rs`)
//! walks THIS to find the nearest surface a ray meets and to answer shadow-ray
//! occlusion. Built once at load, uploaded as two storage buffers (nodes +
//! triangles) the GPU compute integrator reads.
//!
//! Structure: a binary BVH, median-split on the widest centroid axis (Rite IV
//! keeps it simple and correct — SAH/refit is a later performance rite; RENDER.md
//! rules "never optimize" until the truth is right). Every threshold is a param
//! (IRON LAW: never hardcode).
//!
//! Node layout is GPU-ready and matches the WGSL `Node` struct byte-for-byte:
//! an internal node stores its left child index in `left_first` (right child is
//! `left_first + 1`); a leaf stores its first triangle index in `left_first` and
//! a nonzero `count`.

use bytemuck::{Pod, Zeroable};

use crate::scene::LeafTriangle;

/// A triangle as the GPU integrator reads it: three corners (w padding) plus
/// lambertian albedo and emissive radiance. 80 bytes; matches WGSL `Tri`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuTri {
    pub v0: [f32; 4],
    pub v1: [f32; 4],
    pub v2: [f32; 4],
    pub albedo: [f32; 4],
    pub emission: [f32; 4],
}

/// A BVH node. 32 bytes; matches WGSL `Node` (vec3 + u32, vec3 + u32).
/// `count == 0` → internal (children at `left_first`, `left_first + 1`).
/// `count > 0`  → leaf (triangles `[left_first, left_first + count)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuNode {
    pub min: [f32; 3],
    pub left_first: u32,
    pub max: [f32; 3],
    pub count: u32,
}

/// BVH build parameters. Defaults per Rite IV; every one a dial.
#[derive(Clone, Copy, Debug)]
pub struct BvhParams {
    /// A node with this many triangles or fewer becomes a leaf.
    pub leaf_max: usize,
    /// Hard recursion cap (safety against degenerate splits).
    pub max_depth: usize,
}

impl Default for BvhParams {
    fn default() -> Self {
        Self {
            leaf_max: 4,
            max_depth: 64,
        }
    }
}

/// The built hierarchy: a flat node array (root at index 0) + the triangles in
/// leaf-visitation order (both uploaded verbatim as storage buffers).
#[derive(Clone, Debug, Default)]
pub struct Bvh {
    pub nodes: Vec<GpuNode>,
    pub tris: Vec<GpuTri>,
}

#[derive(Clone, Copy)]
struct BuildTri {
    tri: GpuTri,
    centroid: [f32; 3],
    min: [f32; 3],
    max: [f32; 3],
}

fn tri_bounds(t: &LeafTriangle) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in &t.positions {
        for i in 0..3 {
            mn[i] = mn[i].min(p[i]);
            mx[i] = mx[i].max(p[i]);
        }
    }
    let centroid = [
        (mn[0] + mx[0]) * 0.5,
        (mn[1] + mx[1]) * 0.5,
        (mn[2] + mx[2]) * 0.5,
    ];
    (mn, mx, centroid)
}

impl Bvh {
    /// Build a BVH over world-space leaf triangles. An empty input yields a
    /// single empty leaf so the GPU always has a valid root to read.
    pub fn build(triangles: &[LeafTriangle], params: &BvhParams) -> Bvh {
        let mut build: Vec<BuildTri> = triangles
            .iter()
            .map(|t| {
                let (min, max, centroid) = tri_bounds(t);
                BuildTri {
                    tri: GpuTri {
                        v0: [t.positions[0][0], t.positions[0][1], t.positions[0][2], 0.0],
                        v1: [t.positions[1][0], t.positions[1][1], t.positions[1][2], 0.0],
                        v2: [t.positions[2][0], t.positions[2][1], t.positions[2][2], 0.0],
                        albedo: [t.albedo[0], t.albedo[1], t.albedo[2], 0.0],
                        emission: [t.emission[0], t.emission[1], t.emission[2], 0.0],
                    },
                    centroid,
                    min,
                    max,
                }
            })
            .collect();

        let mut nodes: Vec<GpuNode> = Vec::new();
        if build.is_empty() {
            nodes.push(GpuNode {
                min: [0.0; 3],
                left_first: 0,
                max: [0.0; 3],
                count: 0,
            });
            return Bvh {
                nodes,
                tris: Vec::new(),
            };
        }
        // Reserve the root; subdivide fills it and appends children.
        nodes.push(GpuNode::zeroed());
        let count = build.len();
        subdivide(&mut nodes, &mut build, 0, 0, count, 0, params);

        let tris = build.into_iter().map(|b| b.tri).collect();
        Bvh { nodes, tris }
    }

    /// Nearest ray-triangle hit in (t_min, t_max]. Returns `(t, tri_index)`.
    /// CPU mirror of the WGSL traversal — the ordeals' ground for occlusion.
    pub fn hit(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        t_min: f32,
        t_max: f32,
    ) -> Option<(f32, u32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = [1.0 / dir[0], 1.0 / dir[1], 1.0 / dir[2]];
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;
        let mut best_t = t_max;
        let mut best_i: Option<u32> = None;
        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if !aabb_hit(node.min, node.max, origin, inv, t_min, best_t) {
                continue;
            }
            if node.count > 0 {
                for k in 0..node.count {
                    let ti = node.left_first + k;
                    let t = &self.tris[ti as usize];
                    if let Some(t_hit) = tri_hit(origin, dir, t, t_min, best_t) {
                        best_t = t_hit;
                        best_i = Some(ti);
                    }
                }
            } else if sp + 2 <= stack.len() {
                stack[sp] = node.left_first;
                sp += 1;
                stack[sp] = node.left_first + 1;
                sp += 1;
            }
        }
        best_i.map(|i| (best_t, i))
    }

    /// Is any triangle within (t_min, t_max] along the ray? (Shadow test.)
    pub fn occluded(&self, origin: [f32; 3], dir: [f32; 3], t_min: f32, t_max: f32) -> bool {
        self.hit(origin, dir, t_min, t_max).is_some()
    }
}

#[allow(clippy::too_many_arguments)]
fn subdivide(
    nodes: &mut Vec<GpuNode>,
    build: &mut [BuildTri],
    node_index: usize,
    start: usize,
    count: usize,
    depth: usize,
    params: &BvhParams,
) {
    // Bounds over this node's triangles.
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    let mut cmn = [f32::INFINITY; 3];
    let mut cmx = [f32::NEG_INFINITY; 3];
    for b in &build[start..start + count] {
        for i in 0..3 {
            mn[i] = mn[i].min(b.min[i]);
            mx[i] = mx[i].max(b.max[i]);
            cmn[i] = cmn[i].min(b.centroid[i]);
            cmx[i] = cmx[i].max(b.centroid[i]);
        }
    }
    nodes[node_index].min = mn;
    nodes[node_index].max = mx;

    // Leaf when small enough or too deep.
    if count <= params.leaf_max || depth >= params.max_depth {
        nodes[node_index].left_first = start as u32;
        nodes[node_index].count = count as u32;
        return;
    }

    // Split on the widest centroid axis at its median (stable, deterministic).
    let extent = [cmx[0] - cmn[0], cmx[1] - cmn[1], cmx[2] - cmn[2]];
    let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    };
    if extent[axis] <= 0.0 {
        // Degenerate (all centroids coincide) → leaf.
        nodes[node_index].left_first = start as u32;
        nodes[node_index].count = count as u32;
        return;
    }
    let slice = &mut build[start..start + count];
    slice.sort_by(|a, b| {
        a.centroid[axis]
            .partial_cmp(&b.centroid[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = count / 2;

    let left_index = nodes.len();
    nodes.push(GpuNode::zeroed());
    nodes.push(GpuNode::zeroed());
    nodes[node_index].left_first = left_index as u32;
    nodes[node_index].count = 0;

    subdivide(nodes, build, left_index, start, mid, depth + 1, params);
    subdivide(
        nodes,
        build,
        left_index + 1,
        start + mid,
        count - mid,
        depth + 1,
        params,
    );
}

fn aabb_hit(
    min: [f32; 3],
    max: [f32; 3],
    origin: [f32; 3],
    inv: [f32; 3],
    t_min: f32,
    t_max: f32,
) -> bool {
    let mut tmin = t_min;
    let mut tmax = t_max;
    for i in 0..3 {
        let t0 = (min[i] - origin[i]) * inv[i];
        let t1 = (max[i] - origin[i]) * inv[i];
        let (lo, hi) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        tmin = tmin.max(lo);
        tmax = tmax.min(hi);
        if tmax < tmin {
            return false;
        }
    }
    true
}

/// Möller–Trumbore, matching the WGSL. Returns hit distance in (t_min, t_max].
fn tri_hit(origin: [f32; 3], dir: [f32; 3], t: &GpuTri, t_min: f32, t_max: f32) -> Option<f32> {
    let v0 = [t.v0[0], t.v0[1], t.v0[2]];
    let e1 = [t.v1[0] - v0[0], t.v1[1] - v0[1], t.v1[2] - v0[2]];
    let e2 = [t.v2[0] - v0[0], t.v2[1] - v0[1], t.v2[2] - v0[2]];
    let p = cross(dir, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = [origin[0] - v0[0], origin[1] - v0[1], origin[2] - v0[2]];
    let u = dot(tvec, p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(tvec, e1);
    let v = dot(dir, q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t_hit = dot(e2, q) * inv_det;
    if t_hit > t_min && t_hit <= t_max {
        Some(t_hit)
    } else {
        None
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(y: f32, half: f32, albedo: [f32; 3], emission: [f32; 3]) -> [LeafTriangle; 2] {
        // Two triangles forming an axis-aligned square in the y-plane.
        let a = [-half, y, -half];
        let b = [half, y, -half];
        let c = [half, y, half];
        let d = [-half, y, half];
        [
            LeafTriangle {
                positions: [a, b, c],
                albedo,
                emission,
            },
            LeafTriangle {
                positions: [a, c, d],
                albedo,
                emission,
            },
        ]
    }

    #[test]
    fn empty_build_has_one_leaf() {
        let bvh = Bvh::build(&[], &BvhParams::default());
        assert_eq!(bvh.nodes.len(), 1);
        assert_eq!(bvh.nodes[0].count, 0);
        assert!(bvh.tris.is_empty());
    }

    #[test]
    fn leaf_indices_cover_every_triangle_once() {
        // A grid of quads → many triangles; every one must land in exactly one leaf.
        let mut tris = Vec::new();
        for i in 0..20 {
            let y = i as f32;
            tris.extend(quad(y, 1.0, [0.5, 0.5, 0.5], [0.0; 3]));
        }
        let bvh = Bvh::build(&tris, &BvhParams::default());
        let mut covered = vec![0u32; bvh.tris.len()];
        for node in &bvh.nodes {
            if node.count > 0 {
                for k in 0..node.count {
                    covered[(node.left_first + k) as usize] += 1;
                }
            }
        }
        assert_eq!(bvh.tris.len(), tris.len());
        assert!(covered.iter().all(|&c| c == 1), "each triangle in one leaf");
    }

    #[test]
    fn ray_hits_and_shadow_occludes() {
        // Floor quad at y=0, occluder quad at y=5. A ray up from the floor is
        // occluded by the ceiling; a ray up from above the ceiling escapes.
        let mut tris = Vec::new();
        tris.extend(quad(0.0, 10.0, [0.6, 0.6, 0.6], [0.0; 3]));
        tris.extend(quad(5.0, 2.0, [0.0; 3], [1.0, 1.0, 1.0]));
        let bvh = Bvh::build(&tris, &BvhParams::default());

        // Straight down onto the floor from above → hits.
        let hit = bvh.hit([0.0, 3.0, 0.0], [0.0, -1.0, 0.0], 1e-3, 1e9);
        assert!(hit.is_some());
        let (t, _) = hit.unwrap();
        assert!((t - 3.0).abs() < 1e-3, "floor at t=3, got {t}");

        // From the floor toward +y → the ceiling occludes the sky.
        assert!(bvh.occluded([0.0, 0.01, 0.0], [0.0, 1.0, 0.0], 1e-3, 1e9));
        // From above the ceiling toward +y → nothing above, escapes.
        assert!(!bvh.occluded([0.0, 6.0, 0.0], [0.0, 1.0, 0.0], 1e-3, 1e9));
    }
}
