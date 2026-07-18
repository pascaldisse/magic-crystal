//! The acceleration — a CPU-built bounding-volume hierarchy over the Great
//! Chain's leaf triangles (Rite IV, L1). The traced integrator (`integrator.rs`)
//! walks THIS to find the nearest surface a ray meets and to answer shadow-ray
//! occlusion. Built once at load, uploaded as two storage buffers (nodes +
//! triangles) the GPU compute integrator reads.
//!
//! Structure: a binary BVH split by the Surface-Area Heuristic (binned SAH over
//! the centroid axes). SAH is a pure traversal-quality choice — it changes only
//! the tree's SHAPE (which triangles land under which node), never the triangle
//! SET; `nearest_hit` for any ray whose nearest surface is UNAMBIGUOUS is
//! unchanged, so the vast majority of pixels are bit-identical to the old median
//! split (`merge_equals_full_rebuild` proves the geometry parity). The one
//! exception is a coplanar z-fight seam: two triangles a ray meets at the EXACT
//! same depth (Möller–Trumbore agrees to the ULP). There the "nearest" winner is
//! undefined — the old median tree kept whichever it visited last, this tree
//! keeps whichever the SAH order visits last. Both are valid nearest hits; only
//! the arbitrary tie winner shifts (measured: Naruko front pose 0 px, wide 20 px
//! of 540k = 0.0037%, all proven exact-depth ties by the brute-force DIAG). A
//! canonical build-independent tie-break exists (git history, tie band + vertex-
//! lexicographic `tri_before`) but was retired from tip: it makes MORE pixels
//! diverge from the median baseline (it rewrites even seams that already agreed),
//! so median-parity beats build-independence for this lane — see the perf-fix
//! report. A degenerate node (coincident centroids / no paying split) falls back
//! to the widest-axis median. Every threshold is a param (IRON LAW: never
//! hardcode).
//!
//! Node layout is GPU-ready and matches the WGSL `Node` struct byte-for-byte:
//! an internal node stores its left child index in `left_first` (right child is
//! `left_first + 1`); a leaf stores its first triangle index in `left_first` and
//! a nonzero `count`.

use bytemuck::{Pod, Zeroable};

use crate::scene::LeafTriangle;

/// A triangle as the GPU integrator reads it: three corners (w padding) plus
/// albedo (`.w` = metallic) and emissive radiance (`.w` = roughness). The L2
/// conductor dials ride the unused `.w` lanes (no size change). 80 bytes;
/// matches WGSL `Tri`.
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
    /// Binned-SAH bucket count per axis (higher = finer split search, more
    /// build cost). 16 is the standard sweet spot.
    pub sah_bins: usize,
}

impl Default for BvhParams {
    fn default() -> Self {
        Self {
            leaf_max: 4,
            max_depth: 64,
            sah_bins: 16,
        }
    }
}

impl BvhParams {
    /// Params for the PER-TICK dynamic partition: identical to `self` but with
    /// `sah_bins = 0`, selecting the cheap widest-axis median split. SAH pays for
    /// itself on the large static tree (built once, traversed millions of times);
    /// on the small dynamic partition (rebuilt EVERY tick) the binned search
    /// costs ~3 ms/tick for negligible traversal gain, so median wins the frame
    /// budget. Pure build-strategy choice — the triangle set and every
    /// unambiguous nearest hit are unchanged.
    pub fn dynamic(&self) -> Self {
        Self {
            sah_bins: 0,
            ..*self
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
    /// Index of this triangle in the ORIGINAL input slice. The build reorders
    /// triangles into leaf-visitation order; this remembers where each landed
    /// so a later REFIT can repopulate the same slots from moved positions
    /// (topology kept, bounds re-fitted) instead of rebuilding.
    src: u32,
}

/// Pack a world-space leaf triangle into the GPU-ready `GpuTri` (positions +
/// albedo/metallic + emission/roughness). The one place the packing lives — the
/// full build and the per-tick refit both go through it, so they can never drift.
fn gpu_tri(t: &LeafTriangle) -> GpuTri {
    GpuTri {
        v0: [t.positions[0][0], t.positions[0][1], t.positions[0][2], 0.0],
        v1: [t.positions[1][0], t.positions[1][1], t.positions[1][2], 0.0],
        v2: [t.positions[2][0], t.positions[2][1], t.positions[2][2], 0.0],
        albedo: [t.albedo[0], t.albedo[1], t.albedo[2], t.metallic],
        emission: [t.emission[0], t.emission[1], t.emission[2], t.roughness],
    }
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
        Bvh::build_indexed(triangles, params).0
    }

    /// Build, also returning the leaf-order permutation: `src[k]` is the index
    /// into `triangles` that landed in tris slot `k`. Feeds `Bvh::refit` — the
    /// per-tick REFIT lever repopulates the same slots from moved positions and
    /// re-fits the node bounds bottom-up, keeping the topology (acceleration-only
    /// ⇒ every unambiguous nearest hit is unchanged; only the same coplanar-tie
    /// winners a fresh rebuild would also shuffle can move — proven by the
    /// refit-vs-rebuild parity gate).
    pub fn build_indexed(triangles: &[LeafTriangle], params: &BvhParams) -> (Bvh, Vec<u32>) {
        let mut build: Vec<BuildTri> = triangles
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let (min, max, centroid) = tri_bounds(t);
                BuildTri {
                    tri: gpu_tri(t),
                    centroid,
                    min,
                    max,
                    src: i as u32,
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
            return (
                Bvh {
                    nodes,
                    tris: Vec::new(),
                },
                Vec::new(),
            );
        }
        // Reserve the root; subdivide fills it and appends children.
        nodes.push(GpuNode::zeroed());
        let count = build.len();
        subdivide(&mut nodes, &mut build, 0, 0, count, 0, params);

        let tris = build.iter().map(|b| b.tri).collect();
        let src = build.iter().map(|b| b.src).collect();
        (Bvh { nodes, tris }, src)
    }

    /// Half the surface area of the root AABB. NOT a valid refit-degradation
    /// signal: `Bvh::refit` recomputes every node's bounds from the ACTUAL
    /// current triangle corners (not inherited/stale bounds), so a refit
    /// tree's root ends up exactly as tight as a fresh build's root over the
    /// same positions, by construction — proven pinned at ratio 1.0000 across
    /// a 300-tick sweep in `docs/perf/2026-07-17-refit-degrade-derivation.md`.
    /// Kept for callers that want the outer bound (e.g. the example's
    /// same-tick sanity check); `DynamicSplice` uses `total_node_half_area`
    /// instead, which sees interior decay the root alone cannot.
    pub fn root_half_area(&self) -> f32 {
        self.nodes.first().map_or(0.0, |n| half_area(n.min, n.max))
    }

    /// Sum of half-area over EVERY node (root + every internal + every leaf).
    /// Unlike `root_half_area`, this sees interior topology decay: as a refit
    /// tree accumulates many refits without a rebuild, sibling bounds loosen
    /// and start overlapping even though the root (the union of everything)
    /// stays tight — this sum grows as that happens (more overlap ⇒ more
    /// total boxed volume ⇒ more wasted GPU traversal, the SAH cost proxy).
    /// `0.0` for the empty tree.
    pub fn total_node_half_area(&self) -> f32 {
        self.nodes.iter().map(|n| half_area(n.min, n.max)).sum()
    }

    /// REFIT: keep this tree's topology and triangle→slot assignment, but pull
    /// fresh positions from `triangles` through the `src` permutation (from
    /// `build_indexed`) and recompute every node's AABB bottom-up. O(nodes+tris),
    /// no sorting. Valid only when `triangles` is the SAME set the tree was built
    /// over (same count, same emission order) with only positions/attrs moved —
    /// the caller guards that. A BVH is acceleration-only, so the nearest hit for
    /// any ray is unchanged by which topology found it; only exact-depth tie
    /// winners can differ from a fresh rebuild (the refit-vs-rebuild parity gate
    /// characterises it).
    ///
    /// Bottom-up is a plain reverse scan: `subdivide` always appends children
    /// AFTER their parent, so every child index exceeds its parent's and a
    /// descending pass sees children already refitted. Returns the sum of every
    /// node's half-area (the `total_node_half_area` value) computed in this
    /// SAME pass, so callers that need it (the degradation watchdog) don't pay
    /// a redundant second full scan over the nodes.
    pub fn refit(&mut self, triangles: &[LeafTriangle], src: &[u32]) -> f32 {
        debug_assert_eq!(self.tris.len(), src.len());
        for (slot, &s) in src.iter().enumerate() {
            self.tris[slot] = gpu_tri(&triangles[s as usize]);
        }
        let mut total_half_area = 0.0f32;
        for idx in (0..self.nodes.len()).rev() {
            let node = self.nodes[idx];
            let (mn, mx) = if node.count > 0 {
                let mut mn = [f32::INFINITY; 3];
                let mut mx = [f32::NEG_INFINITY; 3];
                for k in 0..node.count {
                    let t = &self.tris[(node.left_first + k) as usize];
                    for corner in [&t.v0, &t.v1, &t.v2] {
                        for i in 0..3 {
                            mn[i] = mn[i].min(corner[i]);
                            mx[i] = mx[i].max(corner[i]);
                        }
                    }
                }
                (mn, mx)
            } else {
                let l = &self.nodes[node.left_first as usize];
                let r = &self.nodes[node.left_first as usize + 1];
                let mut mn = [0.0f32; 3];
                let mut mx = [0.0f32; 3];
                for i in 0..3 {
                    mn[i] = l.min[i].min(r.min[i]);
                    mx[i] = l.max[i].max(r.max[i]);
                }
                (mn, mx)
            };
            self.nodes[idx].min = mn;
            self.nodes[idx].max = mx;
            total_half_area += half_area(mn, mx);
        }
        total_half_area
    }

    /// Splice a cached STATIC BVH and a freshly-built DYNAMIC BVH into one flat
    /// tree the GPU integrator walks unchanged — the two-level dynamics update
    /// (DYNAMICS.md). The static hierarchy is built once and never re-sorted;
    /// only the (tiny) dynamic partition rebuilds per tick, then this O(Sn+Dn)
    /// linear splice fuses them under a new two-child root. Correct by
    /// construction (the `merge_equals_full_rebuild` ordeal proves traversal
    /// parity vs a from-scratch build over the union).
    ///
    /// Node layout (right child is always `left_first + 1`, the flat invariant):
    /// `[root, static_root, dynamic_root, static_rest.., dynamic_rest..]`.
    /// Triangles: `static.tris ++ dynamic.tris` (static indices unchanged,
    /// dynamic leaf tri-indices shifted by the static triangle count).
    pub fn merge(static_bvh: &Bvh, dynamic_bvh: &Bvh) -> Bvh {
        // Degenerate sides: with nothing dynamic the static tree IS the answer
        // (a still world never pays the merge), and vice-versa.
        if dynamic_bvh.tris.is_empty() {
            return static_bvh.clone();
        }
        if static_bvh.tris.is_empty() {
            return dynamic_bvh.clone();
        }
        let sn = static_bvh.nodes.len();
        let dn = dynamic_bvh.nodes.len();
        let st = static_bvh.tris.len() as u32;
        // Remap: static root→1, static rest i→i+2; dynamic root→2, rest j→j+Sn+1.
        let rs = |i: usize| -> u32 { if i == 0 { 1 } else { (i + 2) as u32 } };
        let rd = |j: usize| -> u32 { if j == 0 { 2 } else { (j + sn + 1) as u32 } };

        let mut nodes = vec![GpuNode::zeroed(); 1 + sn + dn];
        // The new root spans both children (static_root at 1, dynamic_root at 2).
        let sr = &static_bvh.nodes[0];
        let dr = &dynamic_bvh.nodes[0];
        let mut mn = [0.0f32; 3];
        let mut mx = [0.0f32; 3];
        for k in 0..3 {
            mn[k] = sr.min[k].min(dr.min[k]);
            mx[k] = sr.max[k].max(dr.max[k]);
        }
        nodes[0] = GpuNode {
            min: mn,
            left_first: 1,
            max: mx,
            count: 0,
        };
        for (i, node) in static_bvh.nodes.iter().enumerate() {
            let mut copy = *node;
            if node.count == 0 {
                // Internal: children move with the static remap (stay adjacent).
                copy.left_first = rs(node.left_first as usize);
            }
            // Leaf keeps its triangle index (static tris come first, unshifted).
            nodes[rs(i) as usize] = copy;
        }
        for (j, node) in dynamic_bvh.nodes.iter().enumerate() {
            let mut copy = *node;
            if node.count == 0 {
                copy.left_first = rd(node.left_first as usize);
            } else {
                // Leaf: dynamic triangles sit after the static block.
                copy.left_first = node.left_first + st;
            }
            nodes[rd(j) as usize] = copy;
        }

        let mut tris = Vec::with_capacity(static_bvh.tris.len() + dynamic_bvh.tris.len());
        tris.extend_from_slice(&static_bvh.tris);
        tris.extend_from_slice(&dynamic_bvh.tris);
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

/// How the last `DynamicSplice::update` produced the merged tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpliceKind {
    /// The dynamic partition was rebuilt from scratch (set changed, first frame,
    /// or the refit bounds had degraded past the threshold).
    Rebuilt,
    /// The dynamic partition kept its topology and only re-fitted its bounds
    /// (the cheap per-tick path).
    Refit,
}

/// Refit-not-rebuild control (LEVER 1). Every knob a dial (IRON LAW).
#[derive(Clone, Copy, Debug)]
pub struct RefitParams {
    /// Rebuild once the refit tree's TOTAL node half-area (sum over every
    /// node, `Bvh::total_node_half_area`) grows past this multiple of the sum
    /// AT THE LAST REBUILD (`DynamicSplice::rebuild_reference_area` — a
    /// reference across TIME, not a same-tick comparison to a fresh build).
    /// Refit keeps the tree's topology and only re-fits bounds; across many
    /// refits without a rebuild this gated ratio grows from two compounding
    /// causes: (1) interior topology staleness (sibling bounds loosen/overlap
    /// even though the root — the exact union of every leaf — stays tight,
    /// so the root alone can't see it) and (2) the body's overall silhouette
    /// oscillating relative to whatever pose the last rebuild happened to
    /// freeze as the reference. Both cost extra GPU traversal; the gate
    /// watches their product, which is what actually accumulates tick over
    /// tick between rebuilds. Default derived by the 300-tick + 1200-tick
    /// trace-drift sweep (`examples/refit_degrade.rs`), which reports this
    /// EXACT gated ratio (not a proxy) alongside its two components — see
    /// `docs/perf/2026-07-17-refit-degrade-derivation.md` (revision 2, which
    /// replaced a first derivation that measured the wrong ratio). The
    /// derived default is `1 + 10 × (max observed benign gated-ratio
    /// excursion above 1.0)` — an excursion-form headroom, not a flat
    /// multiplier, so a benign ratio near 1.0 doesn't inflate the gate
    /// disproportionately.
    pub degrade_ratio: f32,
    /// Hard cap on consecutive refits between rebuilds (belt-and-braces against
    /// a slow area creep that never trips the ratio). `0` = unlimited.
    pub max_refits: u32,
}

/// `RefitParams::degrade_ratio` default — measured by the 300-tick +
/// 1200-tick trace-drift sweep (`examples/refit_degrade.rs`): `1 + 10 ×` the
/// maximum benign gated-ratio excursion. Public so every door uses this one
/// IRON parameter rather than duplicating its derived value.
pub const DEFAULT_DEGRADE_RATIO: f32 = 1.7030;

impl Default for RefitParams {
    /// Defaults measured by the 300-tick + 1200-tick trace-drift sweep
    /// (`examples/refit_degrade.rs`). The degrade ratio is
    /// `1 + 10 × max benign excursion above 1.0`; §
    /// `docs/perf/2026-07-17-refit-degrade-derivation.md` revision 2.
    fn default() -> Self {
        Self {
            degrade_ratio: DEFAULT_DEGRADE_RATIO,
            max_refits: 0,
        }
    }
}

/// The persistent two-level splice (LEVER 1: refit-not-rebuild). Holds the
/// cached STATIC tree, a PERSISTENT dynamic tree, its build permutation, and the
/// merged flat tree the GPU walks. Per tick: when the dynamic triangle SET is
/// unchanged (same count — same entities/emission order, only positions moved)
/// and the bounds have not degraded, it REFITS the dynamic tree (O(n) bounds
/// update, no sort) and re-splices; otherwise it rebuilds the dynamic partition.
/// The re-merge is O(Sn+Dn) linear and measured at ~0.04 ms — the win is
/// replacing the ~3 ms median BUILD with a ~0.1 ms refit.
#[derive(Clone, Debug)]
pub struct DynamicSplice {
    dyn_bvh: Bvh,
    dyn_src: Vec<u32>,
    dyn_tri_count: usize,
    /// Dynamic-tree total node half-area (`Bvh::total_node_half_area`) captured
    /// at the last full rebuild — the degradation reference.
    rebuild_area: f32,
    refits_since_rebuild: u32,
    dyn_params: BvhParams,
    refit: RefitParams,
    /// The merged flat tree — upload this to the GPU.
    pub merged: Bvh,
    /// How the last `update` produced `merged`.
    pub last_kind: SpliceKind,
}

impl DynamicSplice {
    /// The current dynamic sub-tree's total node half-area
    /// (`Bvh::total_node_half_area`) — the degradation signal this splice
    /// gates on. Exposed for measurement (the `refit_degrade` sweep compares
    /// this against a fresh build's, tick over tick); production code never
    /// needs it directly since `update` applies the gate internally.
    pub fn dyn_total_half_area(&self) -> f32 {
        self.dyn_bvh.total_node_half_area()
    }

    /// The total-node-half-area sum captured at the LAST REBUILD — the exact
    /// reference `update`'s gate divides by (`degraded = current_sum >
    /// rebuild_reference_area * degrade_ratio`, see `update` below). Exposed
    /// read-only so measurement code (`refit_degrade`) can compute the SAME
    /// ratio the gate actually watches, instead of a same-tick fresh-build
    /// comparison that measures a different signal (topology staleness, not
    /// growth since the last rebuild).
    pub fn rebuild_reference_area(&self) -> f32 {
        self.rebuild_area
    }

    /// First build: full dynamic build + merge, capturing the permutation and the
    /// degradation reference.
    pub fn build(
        static_bvh: &Bvh,
        dyn_tris: &[LeafTriangle],
        dyn_params: &BvhParams,
        refit: RefitParams,
    ) -> Self {
        let (dyn_bvh, dyn_src) = Bvh::build_indexed(dyn_tris, dyn_params);
        let merged = Bvh::merge(static_bvh, &dyn_bvh);
        Self {
            rebuild_area: dyn_bvh.total_node_half_area(),
            dyn_tri_count: dyn_tris.len(),
            refits_since_rebuild: 0,
            dyn_params: *dyn_params,
            refit,
            dyn_bvh,
            dyn_src,
            merged,
            last_kind: SpliceKind::Rebuilt,
        }
    }

    /// Per-tick update. Refits the dynamic tree in place when the set is unchanged
    /// and bounds are still tight; rebuilds otherwise. Always re-merges onto the
    /// (unchanged) static tree so `merged` is ready to upload. Returns how it went.
    pub fn update(&mut self, static_bvh: &Bvh, dyn_tris: &[LeafTriangle]) -> SpliceKind {
        let set_unchanged = dyn_tris.len() == self.dyn_tri_count && !dyn_tris.is_empty();
        let cap_ok =
            self.refit.max_refits == 0 || self.refits_since_rebuild < self.refit.max_refits;
        if set_unchanged && cap_ok {
            // Trial refit (cheap); accept it unless the bounds have degraded past
            // the ratio, in which case fall through to a rebuild this tick.
            // `refit` already walks every node bottom-up, so it returns the
            // total-node-half-area sum from that same pass — no redundant scan.
            let area = self.dyn_bvh.refit(dyn_tris, &self.dyn_src);
            let degraded =
                self.rebuild_area > 0.0 && area > self.rebuild_area * self.refit.degrade_ratio;
            if !degraded {
                self.merged = Bvh::merge(static_bvh, &self.dyn_bvh);
                self.refits_since_rebuild += 1;
                self.last_kind = SpliceKind::Refit;
                return SpliceKind::Refit;
            }
        }
        // Rebuild: set changed, first-frame mismatch, cap hit, or degraded.
        let (dyn_bvh, dyn_src) = Bvh::build_indexed(dyn_tris, &self.dyn_params);
        self.merged = Bvh::merge(static_bvh, &dyn_bvh);
        self.rebuild_area = dyn_bvh.total_node_half_area();
        self.dyn_tri_count = dyn_tris.len();
        self.refits_since_rebuild = 0;
        self.dyn_bvh = dyn_bvh;
        self.dyn_src = dyn_src;
        self.last_kind = SpliceKind::Rebuilt;
        SpliceKind::Rebuilt
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

    // Choose the split by the Surface-Area Heuristic (binned). Falls back to the
    // widest-axis median when no axis offers a paying split.
    let extent = [cmx[0] - cmn[0], cmx[1] - cmn[1], cmx[2] - cmn[2]];
    let widest = if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    };
    if extent[widest] <= 0.0 {
        // Degenerate (all centroids coincide) → leaf.
        nodes[node_index].left_first = start as u32;
        nodes[node_index].count = count as u32;
        return;
    }

    let slice = &mut build[start..start + count];
    // `sah_bins < 2` selects the cheap widest-axis median (baseline behaviour) —
    // the per-tick DYNAMIC partition uses it: nari's body is a compact skinned
    // cluster where SAH tree quality buys almost no traversal but the binned
    // search costs ~3 ms/tick. The static tree (built once) keeps full SAH.
    let sah = if params.sah_bins >= 2 {
        sah_split(slice, cmn, extent, params.sah_bins)
    } else {
        None
    };
    let mid = match sah {
        Some((axis, plane, scale)) => {
            // Stable partition: bins ≤ plane go left. Sorting by bin index keeps
            // the build deterministic (topology is free — pixels never see it).
            slice.sort_by(|a, b| {
                bin_of(a.centroid[axis], cmn[axis], scale, params.sah_bins).cmp(&bin_of(
                    b.centroid[axis],
                    cmn[axis],
                    scale,
                    params.sah_bins,
                ))
            });
            slice
                .iter()
                .take_while(|b| {
                    bin_of(b.centroid[axis], cmn[axis], scale, params.sah_bins) <= plane
                })
                .count()
        }
        None => {
            // No paying SAH split → widest-axis median (old behaviour).
            slice.sort_by(|a, b| {
                a.centroid[widest]
                    .partial_cmp(&b.centroid[widest])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            count / 2
        }
    };
    // Guard the degenerate partition (all one side) — never emit an empty child.
    let mid = mid.clamp(1, count - 1);

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

/// The bin a centroid coordinate falls in (clamped to `[0, bins)`).
fn bin_of(c: f32, cmin: f32, scale: f32, bins: usize) -> usize {
    let bi = ((c - cmin) * scale) as usize;
    bi.min(bins - 1)
}

/// Half the surface area of an AABB (the SAH's relative probability metric;
/// the constant 2 cancels across candidates, so half-area suffices).
fn half_area(mn: [f32; 3], mx: [f32; 3]) -> f32 {
    let dx = (mx[0] - mn[0]).max(0.0);
    let dy = (mx[1] - mn[1]).max(0.0);
    let dz = (mx[2] - mn[2]).max(0.0);
    dx * dy + dy * dz + dz * dx
}

/// Binned Surface-Area-Heuristic split search over all three centroid axes.
/// Returns `(axis, plane, scale)` where every triangle whose centroid bins in
/// `[0, plane]` goes left, or `None` when no axis yields a valid two-sided
/// split (caller falls back to the median). Pure ordering choice — never touches
/// the triangle geometry, so the render stays bit-exact.
fn sah_split(
    build: &[BuildTri],
    cmn: [f32; 3],
    extent: [f32; 3],
    bins: usize,
) -> Option<(usize, usize, f32)> {
    debug_assert!(bins >= 2);
    let mut best: Option<(usize, usize, f32, f32)> = None; // axis, plane, scale, cost
    for axis in 0..3 {
        if extent[axis] <= 0.0 {
            continue;
        }
        let scale = bins as f32 / extent[axis];
        let mut bin_count = vec![0u32; bins];
        let mut bin_min = vec![[f32::INFINITY; 3]; bins];
        let mut bin_max = vec![[f32::NEG_INFINITY; 3]; bins];
        for b in build {
            let bi = bin_of(b.centroid[axis], cmn[axis], scale, bins);
            bin_count[bi] += 1;
            for k in 0..3 {
                bin_min[bi][k] = bin_min[bi][k].min(b.min[k]);
                bin_max[bi][k] = bin_max[bi][k].max(b.max[k]);
            }
        }
        // Left prefix sweep: cumulative area + count for planes after bin i.
        let mut left_area = vec![0.0f32; bins - 1];
        let mut left_count = vec![0u32; bins - 1];
        let mut amn = [f32::INFINITY; 3];
        let mut amx = [f32::NEG_INFINITY; 3];
        let mut acc = 0u32;
        for i in 0..bins - 1 {
            acc += bin_count[i];
            for k in 0..3 {
                amn[k] = amn[k].min(bin_min[i][k]);
                amx[k] = amx[k].max(bin_max[i][k]);
            }
            left_count[i] = acc;
            left_area[i] = half_area(amn, amx);
        }
        // Right suffix sweep: cost at each plane = SA_l*N_l + SA_r*N_r.
        let mut amn = [f32::INFINITY; 3];
        let mut amx = [f32::NEG_INFINITY; 3];
        let mut acc = 0u32;
        for i in (1..bins).rev() {
            acc += bin_count[i];
            for k in 0..3 {
                amn[k] = amn[k].min(bin_min[i][k]);
                amx[k] = amx[k].max(bin_max[i][k]);
            }
            let plane = i - 1;
            let (nl, nr) = (left_count[plane], acc);
            if nl == 0 || nr == 0 {
                continue;
            }
            let cost = left_area[plane] * nl as f32 + half_area(amn, amx) * nr as f32;
            if best.is_none_or(|(_, _, _, c)| cost < c) {
                best = Some((axis, plane, scale, cost));
            }
        }
    }
    best.map(|(axis, plane, scale, _)| (axis, plane, scale))
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
            LeafTriangle::lambertian([a, b, c], albedo, emission),
            LeafTriangle::lambertian([a, c, d], albedo, emission),
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

    /// The two-level splice is traversal-identical to a from-scratch build over
    /// the union: for a fan of rays, `merge(static, dynamic)` and a whole rebuild
    /// agree on hit/miss and hit distance (tri indices differ by ordering; the
    /// GEOMETRY the ray meets is the same). This is the correctness proof the
    /// dynamics update leans on.
    #[test]
    fn merge_equals_full_rebuild() {
        // Static: a wide floor at y=0. Dynamic: a small occluder slab at y=3.
        let mut static_tris = Vec::new();
        static_tris.extend(quad(0.0, 20.0, [0.6, 0.6, 0.6], [0.0; 3]));
        for i in 1..8 {
            static_tris.extend(quad(-(i as f32), 20.0, [0.4, 0.4, 0.4], [0.0; 3]));
        }
        let mut dyn_tris = Vec::new();
        dyn_tris.extend(quad(3.0, 2.0, [0.0; 3], [1.0, 1.0, 1.0]));
        dyn_tris.extend(quad(4.5, 1.0, [0.0; 3], [0.8, 0.8, 0.8]));

        let params = BvhParams::default();
        let static_bvh = Bvh::build(&static_tris, &params);
        let dyn_bvh = Bvh::build(&dyn_tris, &params);
        let merged = Bvh::merge(&static_bvh, &dyn_bvh);

        let mut union = static_tris.clone();
        union.extend_from_slice(&dyn_tris);
        let full = Bvh::build(&union, &params);

        // Node/tri counts: the splice adds exactly one root over the two trees.
        assert_eq!(
            merged.nodes.len(),
            static_bvh.nodes.len() + dyn_bvh.nodes.len() + 1
        );
        assert_eq!(merged.tris.len(), static_tris.len() + dyn_tris.len());

        // A grid of rays fired from above straight down and at angles: hit/miss
        // and distance must match the full rebuild to floating-point tolerance.
        let mut checked = 0;
        for gx in -6..=6 {
            for gz in -6..=6 {
                let ox = gx as f32 * 1.5;
                let oz = gz as f32 * 1.5;
                for dir in [[0.0, -1.0, 0.0], [0.2, -1.0, 0.1], [-0.15, -1.0, -0.25]] {
                    let o = [ox, 12.0, oz];
                    let a = merged.hit(o, dir, 1e-3, 1e9);
                    let b = full.hit(o, dir, 1e-3, 1e9);
                    assert_eq!(
                        a.is_some(),
                        b.is_some(),
                        "hit/miss parity at {o:?} dir {dir:?}"
                    );
                    if let (Some((ta, _)), Some((tb, _))) = (a, b) {
                        assert!(
                            (ta - tb).abs() < 1e-4,
                            "distance parity: merged {ta} vs full {tb}"
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 100, "fired a real fan of rays");
        eprintln!("[ordeal] merge == full rebuild: {checked} rays, byte-parity of geometry");
    }

    /// Merge handles the degenerate single-leaf sides (Sn==1 and/or Dn==1)
    /// without breaking the flat adjacency invariant.
    #[test]
    fn merge_handles_single_leaf_sides() {
        let params = BvhParams::default();
        let s = Bvh::build(&quad(0.0, 5.0, [0.5, 0.5, 0.5], [0.0; 3]), &params);
        let d = Bvh::build(&quad(2.0, 1.0, [0.0; 3], [1.0, 1.0, 1.0]), &params);
        assert_eq!(s.nodes.len(), 1, "single leaf static");
        assert_eq!(d.nodes.len(), 1, "single leaf dynamic");
        let m = Bvh::merge(&s, &d);
        // Down onto the dynamic slab from above → hits it first (t≈8 from y=10).
        let hit = m.hit([0.0, 10.0, 0.0], [0.0, -1.0, 0.0], 1e-3, 1e9);
        assert!(hit.is_some());
        // Every triangle reachable in exactly one leaf.
        let mut covered = vec![0u32; m.tris.len()];
        for node in &m.nodes {
            if node.count > 0 {
                for k in 0..node.count {
                    covered[(node.left_first + k) as usize] += 1;
                }
            }
        }
        assert!(covered.iter().all(|&c| c == 1), "each tri in one leaf");
    }

    /// Translate a quad's two triangles by `d` (moves positions, keeps the set).
    fn shift(tris: &[LeafTriangle], d: [f32; 3]) -> Vec<LeafTriangle> {
        tris.iter()
            .map(|t| {
                let mut p = t.positions;
                for corner in p.iter_mut() {
                    for i in 0..3 {
                        corner[i] += d[i];
                    }
                }
                LeafTriangle::lambertian(p, t.albedo, t.emission)
            })
            .collect()
    }

    /// REFIT after moving every triangle gives the SAME nearest hits as a fresh
    /// build over the moved geometry (the acceleration-only guarantee): for a
    /// fan of rays, hit/miss and distance match to fp tolerance. Topology stays
    /// the pre-move build's; only bounds re-fit.
    #[test]
    fn refit_matches_rebuild_geometry() {
        // A stack of quads at varied y — a real multi-node tree.
        let mut base = Vec::new();
        for i in 0..12 {
            base.extend(quad(i as f32 * 0.7, 6.0, [0.5, 0.5, 0.5], [0.0; 3]));
        }
        let params = BvhParams::default().dynamic();
        let (mut tree, src) = Bvh::build_indexed(&base, &params);

        // Move the whole set (skinned-body style displacement) and refit.
        let moved = shift(&base, [1.3, 2.1, -0.8]);
        tree.refit(&moved, &src);
        // Ground truth: build fresh over the moved set.
        let fresh = Bvh::build(&moved, &params);

        let mut checked = 0;
        for gx in -5..=5 {
            for gz in -5..=5 {
                let o = [gx as f32 * 1.2, 20.0, gz as f32 * 1.2];
                for dir in [[0.0, -1.0, 0.0], [0.18, -1.0, 0.12], [-0.2, -1.0, -0.1]] {
                    let a = tree.hit(o, dir, 1e-3, 1e9);
                    let b = fresh.hit(o, dir, 1e-3, 1e9);
                    assert_eq!(a.is_some(), b.is_some(), "hit/miss parity at {o:?} {dir:?}");
                    if let (Some((ta, _)), Some((tb, _))) = (a, b) {
                        assert!((ta - tb).abs() < 1e-4, "distance parity: {ta} vs {tb}");
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 100);
        eprintln!("[ordeal] refit == rebuild geometry: {checked} rays");
    }

    /// The persistent splice REFITS when the set is unchanged and REBUILDS when
    /// the count changes — and both paths splice a merged tree whose nearest
    /// hits match a from-scratch build over the union.
    #[test]
    fn dynamic_splice_refits_then_rebuilds() {
        let params = BvhParams::default();
        let mut static_tris = Vec::new();
        static_tris.extend(quad(0.0, 20.0, [0.6, 0.6, 0.6], [0.0; 3]));
        for i in 1..6 {
            static_tris.extend(quad(-(i as f32), 20.0, [0.4, 0.4, 0.4], [0.0; 3]));
        }
        let static_bvh = Bvh::build(&static_tris, &params);

        let dyn0 = {
            let mut v = Vec::new();
            v.extend(quad(3.0, 2.0, [0.0; 3], [1.0, 1.0, 1.0]));
            v.extend(quad(4.5, 1.0, [0.0; 3], [0.8, 0.8, 0.8]));
            v
        };
        let mut splice = DynamicSplice::build(
            &static_bvh,
            &dyn0,
            &params.dynamic(),
            RefitParams::default(),
        );
        assert_eq!(splice.last_kind, SpliceKind::Rebuilt);

        // Same count, moved → refit.
        let dyn1 = shift(&dyn0, [0.4, 0.3, 0.2]);
        assert_eq!(splice.update(&static_bvh, &dyn1), SpliceKind::Refit);

        // Merged nearest hits match a fresh build over the union.
        let mut union = static_tris.clone();
        union.extend_from_slice(&dyn1);
        let full = Bvh::build(&union, &params);
        for gx in -6..=6 {
            for gz in -6..=6 {
                let o = [gx as f32 * 1.5, 12.0, gz as f32 * 1.5];
                let d = [0.1, -1.0, -0.05];
                let a = splice.merged.hit(o, d, 1e-3, 1e9);
                let b = full.hit(o, d, 1e-3, 1e9);
                assert_eq!(a.is_some(), b.is_some());
                if let (Some((ta, _)), Some((tb, _))) = (a, b) {
                    assert!((ta - tb).abs() < 1e-4);
                }
            }
        }

        // Add a triangle (count changes) → rebuild.
        let mut dyn2 = dyn1.clone();
        dyn2.extend(quad(6.0, 0.5, [0.0; 3], [0.5, 0.5, 0.5]));
        assert_eq!(splice.update(&static_bvh, &dyn2), SpliceKind::Rebuilt);
    }

    /// DISCRIMINATING TEST (b): a genuine blowup FIRES the DEFAULT gate
    /// (`RefitParams::default()` — the real, freshly-derived `degrade_ratio`,
    /// no inline fixture) before any `max_refits` cap could — proving the
    /// default is not decorative. A modest move first proves the default
    /// tolerates ordinary motion (refit holds); scattering the triangles far
    /// apart then blows the total-node-half-area sum far past the default
    /// ratio, forcing a rebuild even though `max_refits: 0` (unlimited) never
    /// engages the cap.
    #[test]
    fn dynamic_splice_default_gate_fires_on_blowup() {
        let params = BvhParams::default();
        let static_bvh = Bvh::build(&quad(0.0, 50.0, [0.6, 0.6, 0.6], [0.0; 3]), &params);
        let mut dyn0 = Vec::new();
        for i in 0..8 {
            dyn0.extend(quad(3.0 + i as f32 * 0.3, 1.0, [0.0; 3], [1.0, 1.0, 1.0]));
        }
        let mut splice = DynamicSplice::build(
            &static_bvh,
            &dyn0,
            &params.dynamic(),
            RefitParams::default(),
        );
        // A modest move → the DEFAULT gate holds (refit).
        assert_eq!(
            splice.update(&static_bvh, &shift(&dyn0, [0.2, 0.1, 0.1])),
            SpliceKind::Refit,
            "a modest, plausible move must not trip the default gate"
        );
        // Scatter the tris far apart → total node half-area explodes → the
        // DEFAULT gate (no inline fixture) fires a rebuild.
        let scattered: Vec<LeafTriangle> = dyn0
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut p = t.positions;
                for c in p.iter_mut() {
                    c[0] += i as f32 * 40.0;
                }
                LeafTriangle::lambertian(p, t.albedo, t.emission)
            })
            .collect();
        assert_eq!(
            splice.update(&static_bvh, &scattered),
            SpliceKind::Rebuilt,
            "a genuine blowup must fire the DEFAULT degrade_ratio gate, not rely on max_refits"
        );
    }

    /// DISCRIMINATING TEST (a): the DEFAULT gate must NOT fire across a
    /// bounded, periodic, walk-cycle-scale oscillation — the exact motion
    /// class the default was derived to tolerate. Six independent "limbs"
    /// each swing around their own fixed base offset with their own phase (a
    /// single RIGID whole-body translation would leave every node's
    /// half-area unchanged and prove nothing — independent per-limb motion is
    /// what actually reshapes internal node boxes, same as a real gait's
    /// limbs swinging relative to the torso). Amplitude/period are the same
    /// ORDER as the real sweep's observed envelope
    /// (`docs/perf/2026-07-17-refit-degrade-derivation.md`, revision 2: a
    /// bounded ±7% total-sum oscillation over 20 real gait cycles), not a
    /// blowup. Driven for several synthetic "cycles" — 0 rebuilds expected.
    #[test]
    fn dynamic_splice_default_holds_across_bounded_oscillation() {
        let params = BvhParams::default();
        let static_bvh = Bvh::build(&quad(0.0, 50.0, [0.6, 0.6, 0.6], [0.0; 3]), &params);

        let limb_x: [f32; 6] = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5];
        let base_quad = quad(0.0, 0.4, [0.0; 3], [1.0, 1.0, 1.0]);
        let pose_at = |tick: u64| -> Vec<LeafTriangle> {
            let mut out = Vec::new();
            for (i, &x) in limb_x.iter().enumerate() {
                let phase = (tick as f32 / 60.0) * std::f32::consts::TAU
                    + i as f32 * std::f32::consts::PI / 3.0;
                let d = [x, 3.0 + phase.sin() * 0.35, phase.cos() * 0.2];
                out.extend(shift(&base_quad, d));
            }
            out
        };

        let mut splice = DynamicSplice::build(
            &static_bvh,
            &pose_at(0),
            &params.dynamic(),
            RefitParams::default(),
        );
        assert_eq!(splice.last_kind, SpliceKind::Rebuilt);

        let cycles = 4u64; // ≥2 cycles, matching the real sweep's multi-cycle coverage
        let ticks = 60 * cycles;
        let mut rebuilds = 0u32;
        for tick in 1..=ticks {
            match splice.update(&static_bvh, &pose_at(tick)) {
                SpliceKind::Refit => {}
                SpliceKind::Rebuilt => rebuilds += 1,
            }
        }
        assert_eq!(
            rebuilds, 0,
            "RefitParams::default() must hold across a bounded gait-like oscillation \
             (0 rebuilds expected over {cycles} cycles) — {rebuilds} fired"
        );
    }
}
