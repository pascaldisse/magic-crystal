//! Transmutation: mesh → shards (leaves) → group → simplify → re-shardize →
//! parents-reference-children → repeat to root. The result is the Great Chain
//! (RENDER.md §1, the sole geometry pipeline, offline/import half).
//!
//! Follows research/nanite-recon.md: ≤max_triangles-tri shards, adjacency
//! groups, ratio-simplify per group, DAG with monotone error so a runtime cut
//! `parentError > τ ≥ clusterError` is crack-free. CPU-only (this lane).
//!
//! CRACK-FREE INVARIANTS (the reason the group machinery exists):
//!  - BOUNDARY LOCKING (finding 1): positions shared across two groups are
//!    LOCKED during each group's simplify, so neighboring groups cut their
//!    shared border identically → the seam matches at every LOD.
//!  - SHARED GROUP METRIC (finding 2): every cluster a group produces (and
//!    every child it consumes) transitions on the group's ONE shared LOD
//!    bounds sphere + error, never a per-cluster self-sphere.
//!  - UV-SEAM-SAFE WELD (finding 5): canonical POSITION identity is tracked
//!    separately from attribute WEDGES; UV/normal discontinuities are never
//!    collapsed, so seams survive simplification.

use crate::mesh::{Mesh, Vertex, POSITION_OFFSET, VERTEX_STRIDE};
use crate::partition::{AdjacencyGraph, Partitioner};
use meshopt::{SimplifyOptions, VertexDataAdapter};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

// Engine ceilings (RENDER.md §1). Params may be smaller but never exceed these.
/// Hard ceiling on vertices per shard (M1 shard budget).
pub const MAX_VERTICES_CEIL: usize = 64;
/// Hard ceiling on triangles per shard (M1 shard budget).
pub const MAX_TRIANGLES_CEIL: usize = 124;
/// meshopt contract: `max_triangles` must be a multiple of this.
pub const TRI_MULTIPLE: usize = 4;

/// Typed transmutation errors (finding 6): params are validated BEFORE any
/// unsafe meshopt FFI, so illegal budgets are rejected loudly, never handed to
/// C where they would corrupt or trap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransmuteError {
    /// A meshlet budget field is zero.
    ZeroBudget(&'static str),
    /// `max_vertices` exceeds the engine ceiling (`MAX_VERTICES_CEIL`).
    VerticesTooLarge(usize),
    /// `max_triangles` exceeds the engine ceiling (`MAX_TRIANGLES_CEIL`).
    TrianglesTooLarge(usize),
    /// `max_triangles` is not a multiple of `TRI_MULTIPLE` (meshopt contract).
    TrianglesNotMultiple(usize),
    /// A ratio/weight/tolerance is not finite or out of its documented range.
    OutOfRange(&'static str),
}

impl std::fmt::Display for TransmuteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransmuteError::ZeroBudget(w) => write!(f, "budget `{w}` must be nonzero"),
            TransmuteError::VerticesTooLarge(v) => {
                write!(f, "max_vertices {v} exceeds engine ceiling {MAX_VERTICES_CEIL}")
            }
            TransmuteError::TrianglesTooLarge(t) => {
                write!(f, "max_triangles {t} exceeds engine ceiling {MAX_TRIANGLES_CEIL}")
            }
            TransmuteError::TrianglesNotMultiple(t) => write!(
                f,
                "max_triangles {t} must be a multiple of {TRI_MULTIPLE} (meshopt contract)"
            ),
            TransmuteError::OutOfRange(w) => write!(f, "param `{w}` is out of range or non-finite"),
        }
    }
}
impl std::error::Error for TransmuteError {}

/// Shardization budgets. Defaults per RENDER.md (≤124 tris / ≤64 verts).
/// meshopt requires `max_triangles % 4 == 0` and `<= 512`, `max_vertices <= 256`;
/// the engine ceiling is stricter (`MAX_VERTICES_CEIL` / `MAX_TRIANGLES_CEIL`).
#[derive(Clone, Copy, Debug)]
pub struct MeshletParams {
    pub max_vertices: usize,
    pub max_triangles: usize,
    pub cone_weight: f32,
}

impl Default for MeshletParams {
    fn default() -> Self {
        Self {
            max_vertices: 64,
            max_triangles: 124,
            cone_weight: 0.0,
        }
    }
}

impl MeshletParams {
    /// Validate against the meshopt contract + engine ceilings (finding 6).
    pub fn validate(&self) -> Result<(), TransmuteError> {
        if self.max_vertices == 0 {
            return Err(TransmuteError::ZeroBudget("max_vertices"));
        }
        if self.max_triangles == 0 {
            return Err(TransmuteError::ZeroBudget("max_triangles"));
        }
        if self.max_vertices > MAX_VERTICES_CEIL {
            return Err(TransmuteError::VerticesTooLarge(self.max_vertices));
        }
        if self.max_triangles > MAX_TRIANGLES_CEIL {
            return Err(TransmuteError::TrianglesTooLarge(self.max_triangles));
        }
        if !self.max_triangles.is_multiple_of(TRI_MULTIPLE) {
            return Err(TransmuteError::TrianglesNotMultiple(self.max_triangles));
        }
        if !self.cone_weight.is_finite() || !(0.0..=1.0).contains(&self.cone_weight) {
            return Err(TransmuteError::OutOfRange("cone_weight"));
        }
        Ok(())
    }
}

/// Weld tolerances + attribute weights. Split out so every tolerance is a
/// documented param (IRON LAW: never hardcode) — finding 5's "hidden weld
/// constants" fix.
#[derive(Clone, Copy, Debug)]
pub struct WeldParams {
    /// Position quantum as a FRACTION of the mesh bounding radius. Two positions
    /// within `pos_quant_frac * radius` fuse into one canonical position.
    pub pos_quant_frac: f32,
    /// Absolute floor for the position quantum (tiny-scale meshes — finding 5).
    pub pos_quant_min: f32,
    /// Normal quantum (radians-ish, cosine space). Wedges whose normals differ
    /// by more than this stay separate (hard shading edges preserved).
    pub normal_quant: f32,
    /// UV quantum. Wedges whose UVs differ by more than this stay separate
    /// (texture seams preserved).
    pub uv_quant: f32,
    /// meshopt attribute weight applied to the normal channel during simplify.
    pub normal_weight: f32,
    /// meshopt attribute weight applied to the UV channel during simplify.
    pub uv_weight: f32,
}

impl Default for WeldParams {
    fn default() -> Self {
        Self {
            pos_quant_frac: 1e-5,
            pos_quant_min: 1e-6,
            normal_quant: 1e-3,
            uv_quant: 1e-4,
            normal_weight: 0.5,
            uv_weight: 1.0,
        }
    }
}

impl WeldParams {
    pub fn validate(&self) -> Result<(), TransmuteError> {
        let finite_pos = |x: f32| x.is_finite() && x > 0.0;
        if !finite_pos(self.pos_quant_frac) {
            return Err(TransmuteError::OutOfRange("pos_quant_frac"));
        }
        if !finite_pos(self.pos_quant_min) {
            return Err(TransmuteError::OutOfRange("pos_quant_min"));
        }
        if !finite_pos(self.normal_quant) {
            return Err(TransmuteError::OutOfRange("normal_quant"));
        }
        if !finite_pos(self.uv_quant) {
            return Err(TransmuteError::OutOfRange("uv_quant"));
        }
        if !self.normal_weight.is_finite() || self.normal_weight < 0.0 {
            return Err(TransmuteError::OutOfRange("normal_weight"));
        }
        if !self.uv_weight.is_finite() || self.uv_weight < 0.0 {
            return Err(TransmuteError::OutOfRange("uv_weight"));
        }
        Ok(())
    }
}

/// Full transmutation parameters. Every threshold is a param (IRON LAW: never
/// hardcode).
#[derive(Clone, Copy, Debug)]
pub struct TransmuteParams {
    pub meshlet: MeshletParams,
    pub weld: WeldParams,
    /// Target shards per group (the adjacency-group size).
    pub group_size: usize,
    /// Per-group simplification target as a fraction of the group's tri count.
    pub simplify_ratio: f32,
    /// Relative error ceiling handed to meshopt's simplifier.
    pub target_error: f32,
    /// Safety cap on DAG levels.
    pub max_levels: usize,
    /// Stop once a level has this many clusters or fewer (root).
    pub min_clusters: usize,
}

impl Default for TransmuteParams {
    fn default() -> Self {
        Self {
            meshlet: MeshletParams::default(),
            weld: WeldParams::default(),
            group_size: 4,
            simplify_ratio: 0.5,
            target_error: 1.0,
            max_levels: 32,
            min_clusters: 1,
        }
    }
}

impl TransmuteParams {
    /// Validate the whole param set BEFORE any FFI (finding 6).
    pub fn validate(&self) -> Result<(), TransmuteError> {
        self.meshlet.validate()?;
        self.weld.validate()?;
        if self.group_size == 0 {
            return Err(TransmuteError::ZeroBudget("group_size"));
        }
        if self.max_levels == 0 {
            return Err(TransmuteError::ZeroBudget("max_levels"));
        }
        if !self.simplify_ratio.is_finite() || !(0.0..=1.0).contains(&self.simplify_ratio) {
            return Err(TransmuteError::OutOfRange("simplify_ratio"));
        }
        if !self.target_error.is_finite() || self.target_error < 0.0 {
            return Err(TransmuteError::OutOfRange("target_error"));
        }
        Ok(())
    }
}

/// Bounding sphere as center + radius (cheap runtime cull metric AND the shared
/// LOD metric when it lives on a `Group`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub center: [f32; 3],
    pub radius: f32,
}

/// One node of the Great Chain: self-contained geometry + LOD error + links.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cluster {
    pub id: u32,
    pub level: u32,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Geometric error of the simplification that produced this geometry
    /// (0 for leaves). Absolute (world units). Mirrors `group(self.group).error`.
    pub error: f32,
    /// Error threshold to switch UP to the parent group (>= `error`, monotone).
    /// `∞` for a terminal cluster (no parent). Mirrors the consuming group error.
    pub parent_error: f32,
    /// Child cluster ids (in the level below). Empty for leaves.
    pub children: Vec<u32>,
    /// Self bounding sphere (frustum culling).
    pub bounds: Bounds,
    /// Group that PRODUCED this cluster (its shared LOD metric lives there).
    /// `None` for leaves (produced by shardization, error 0).
    pub group: Option<u32>,
    /// Group that CONSUMES this cluster as a child (its shared metric drives the
    /// switch-UP decision). `None` for a terminal/root cluster.
    pub parent_group: Option<u32>,
}

impl Cluster {
    pub fn tri_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// An explicit group record (finding 2). Every member cluster — the children it
/// consumes AND the parents it produces — shares this ONE bounds sphere + error,
/// so they all cross the same screen-space threshold: crack-free by construction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Group {
    pub id: u32,
    /// Level of the CHILDREN this group consumes.
    pub level: u32,
    /// Child cluster ids grouped together (the shared cut).
    pub children: Vec<u32>,
    /// Parent cluster ids produced by simplifying this group.
    pub parents: Vec<u32>,
    /// SHARED LOD bounds sphere (the merged child geometry's sphere).
    pub bounds: Bounds,
    /// SHARED monotone LOD error (>= every child group's error).
    pub error: f32,
}

/// The transmuted Great Chain: flat cluster store + explicit groups + levels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dag {
    /// Flat store; `clusters[id as usize].id == id`.
    pub clusters: Vec<Cluster>,
    /// Explicit group records; `groups[id as usize].id == id`.
    pub groups: Vec<Group>,
    /// Cluster ids per level; level 0 = leaves (finest), last = root(s).
    pub levels: Vec<Vec<u32>>,
    /// Triangle count of the input mesh (leaf sum invariant).
    pub input_tri_count: u32,
    /// Partitioner backend(s) actually used ("metis" | "greedy" | "metis,greedy").
    pub partitioner: String,
}

impl Dag {
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
    pub fn cluster(&self, id: u32) -> &Cluster {
        &self.clusters[id as usize]
    }
    pub fn group(&self, id: u32) -> &Group {
        &self.groups[id as usize]
    }
    /// Sum of triangle counts of every leaf cluster.
    pub fn leaf_tri_sum(&self) -> usize {
        self.levels
            .first()
            .map(|ids| ids.iter().map(|&id| self.cluster(id).tri_count()).sum())
            .unwrap_or(0)
    }
}

fn compute_bounds(vertices: &[Vertex]) -> Bounds {
    if vertices.is_empty() {
        return Bounds::default();
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for v in vertices {
        for i in 0..3 {
            mn[i] = mn[i].min(v.position[i]);
            mx[i] = mx[i].max(v.position[i]);
        }
    }
    let center = [
        (mn[0] + mx[0]) * 0.5,
        (mn[1] + mx[1]) * 0.5,
        (mn[2] + mx[2]) * 0.5,
    ];
    let mut r2 = 0.0f32;
    for v in vertices {
        let d = [
            v.position[0] - center[0],
            v.position[1] - center[1],
            v.position[2] - center[2],
        ];
        r2 = r2.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    }
    Bounds {
        center,
        radius: r2.sqrt(),
    }
}

/// Shardize a mesh into self-contained clusters (each with its own compact
/// vertex list + local u32 index buffer). Loss-free triangle partition.
fn shardize(mesh: &Mesh, p: &MeshletParams) -> Vec<(Vec<Vertex>, Vec<u32>)> {
    if mesh.indices.is_empty() {
        return Vec::new();
    }
    let adapter = VertexDataAdapter::new(mesh.vertex_bytes(), VERTEX_STRIDE, POSITION_OFFSET)
        .expect("vertex adapter");
    let meshlets = meshopt::build_meshlets(
        &mesh.indices,
        &adapter,
        p.max_vertices,
        p.max_triangles,
        p.cone_weight,
    );
    let mut out = Vec::with_capacity(meshlets.len());
    for m in meshlets.iter() {
        let verts: Vec<Vertex> = m.vertices.iter().map(|&gi| mesh.vertices[gi as usize]).collect();
        let indices: Vec<u32> = m.triangles.iter().map(|&t| t as u32).collect();
        out.push((verts, indices));
    }
    out
}

/// Quantize a position to a canonical lattice key (position identity only).
#[inline]
fn pos_key(p: [f32; 3], quant: f32) -> [i64; 3] {
    let q = |x: f32| (x / quant).round() as i64;
    [q(p[0]), q(p[1]), q(p[2])]
}

/// Lexicographic order on raw float coordinates (total order, NaN-safe) — the
/// deterministic representative rule for canonical coordinates (finding 1).
#[inline]
fn lex_less(a: [f32; 3], b: [f32; 3]) -> bool {
    for i in 0..3 {
        match a[i].total_cmp(&b[i]) {
            std::cmp::Ordering::Less => return true,
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal => {}
        }
    }
    false
}

/// ONE canonical physical coordinate per position key across a WHOLE level
/// (finding 1). Without this each group kept its own first-seen coordinate for a
/// shared key, so two grids offset < quantum wrote divergent border coordinates
/// (69 shared keys → 68 mismatched boundaries). The deterministic representative
/// is the lexicographically-smallest source coordinate carrying the key, applied
/// identically in every group that touches it.
fn canonical_coords(clusters: &[&Cluster], pos_quant: f32) -> BTreeMap<[i64; 3], [f32; 3]> {
    let mut canon: BTreeMap<[i64; 3], [f32; 3]> = BTreeMap::new();
    for c in clusters {
        for v in &c.vertices {
            let k = pos_key(v.position, pos_quant);
            canon
                .entry(k)
                .and_modify(|cur| {
                    if lex_less(v.position, *cur) {
                        *cur = v.position;
                    }
                })
                .or_insert(v.position);
        }
    }
    canon
}

/// Full attribute WEDGE key: position + normal + uv, each with its own quantum.
/// UV/normal discontinuities produce DISTINCT keys → seams survive (finding 5).
#[inline]
fn wedge_key(v: &Vertex, w: &WeldParams, pos_quant: f32) -> [i64; 8] {
    let qp = |x: f32| (x / pos_quant).round() as i64;
    let qn = |x: f32| (x / w.normal_quant).round() as i64;
    let qu = |x: f32| (x / w.uv_quant).round() as i64;
    [
        qp(v.position[0]),
        qp(v.position[1]),
        qp(v.position[2]),
        qn(v.normal[0]),
        qn(v.normal[1]),
        qn(v.normal[2]),
        qu(v.uv[0]),
        qu(v.uv[1]),
    ]
}

/// A welded group ready to simplify: welded mesh + per-vertex canonical-position
/// id (so boundary locks are addressed by POSITION, not by wedge).
struct Welded {
    mesh: Mesh,
    /// Parallel to `mesh.vertices`: canonical position key per welded vertex.
    canonical: Vec<[i64; 3]>,
}

/// Weld a group of clusters, deduplicating only EXACT attribute wedges (finding
/// 5). Canonical position identity is tracked separately so UV/material/normal
/// discontinuities stay as distinct vertices (the simplifier then treats seam
/// edges as borders and cannot smear across them).
fn weld_group(
    group: &[&Cluster],
    w: &WeldParams,
    pos_quant: f32,
    canon: &BTreeMap<[i64; 3], [f32; 3]>,
) -> Welded {
    let mut wedge_map: BTreeMap<[i64; 8], u32> = BTreeMap::new();
    let mut mesh = Mesh::default();
    let mut canonical: Vec<[i64; 3]> = Vec::new();
    for c in group {
        let mut local: Vec<u32> = Vec::with_capacity(c.vertices.len());
        for v in &c.vertices {
            let k = wedge_key(v, w, pos_quant);
            let id = *wedge_map.entry(k).or_insert_with(|| {
                let ck = pos_key(v.position, pos_quant);
                // Snap to the level's ONE canonical coordinate for this key so
                // every group writes byte-identical shared-border positions
                // (finding 1). Normal/UV are untouched → seams still survive.
                let mut vv = *v;
                vv.position = *canon.get(&ck).unwrap_or(&v.position);
                mesh.vertices.push(vv);
                canonical.push(ck);
                (mesh.vertices.len() - 1) as u32
            });
            local.push(id);
        }
        for &i in &c.indices {
            mesh.indices.push(local[i as usize]);
        }
    }
    Welded { mesh, canonical }
}

/// Build the shard adjacency graph for one level: nodes = clusters, edges
/// weighted by shared canonical-position count. Deterministic (BTree ordering).
fn build_adjacency(clusters: &[&Cluster], pos_quant: f32) -> AdjacencyGraph {
    let n = clusters.len();
    // canonical position -> sorted set of cluster ids touching it
    let mut vert_clusters: BTreeMap<[i64; 3], BTreeSet<usize>> = BTreeMap::new();
    for (ci, c) in clusters.iter().enumerate() {
        for v in &c.vertices {
            vert_clusters
                .entry(pos_key(v.position, pos_quant))
                .or_default()
                .insert(ci);
        }
    }
    // accumulate pair weights (commutative → order-independent, but BTree keeps
    // the whole pipeline deterministic).
    let mut weights: BTreeMap<(usize, usize), i32> = BTreeMap::new();
    for cls in vert_clusters.values() {
        let ids: Vec<usize> = cls.iter().copied().collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                *weights.entry((ids[i], ids[j])).or_insert(0) += 1;
            }
        }
    }
    let mut adj: Vec<Vec<(usize, i32)>> = vec![Vec::new(); n];
    for (&(a, b), &wt) in &weights {
        adj[a].push((b, wt));
        adj[b].push((a, wt));
    }
    let mut xadj = Vec::with_capacity(n + 1);
    let mut adjncy = Vec::new();
    let mut adjwgt = Vec::new();
    xadj.push(0i32);
    for nbrs in &mut adj {
        nbrs.sort_by_key(|&(nb, _)| nb);
        for &(nb, wt) in nbrs.iter() {
            adjncy.push(nb as i32);
            adjwgt.push(wt);
        }
        xadj.push(adjncy.len() as i32);
    }
    AdjacencyGraph {
        node_count: n,
        xadj,
        adjncy,
        adjwgt,
    }
}

/// Everything a single accepted level contributes, held OUT of the DAG until the
/// progress check passes (finding 3 — staged commit, never a half-mutated DAG).
struct StagedLevel {
    new_clusters: Vec<Cluster>,
    new_groups: Vec<Group>,
    /// (child_id, parent_error, parent_group_id) to stamp on the children.
    child_links: Vec<(u32, f32, u32)>,
    next_ids: Vec<u32>,
    backend: Option<&'static str>,
    progressed: bool,
}

/// Transmute a mesh into a cluster Great Chain. Params are validated up front
/// (finding 6); illegal budgets never reach meshopt.
pub fn transmute(
    mesh: &Mesh,
    params: &TransmuteParams,
    partitioner: &dyn Partitioner,
) -> Result<Dag, TransmuteError> {
    params.validate()?;

    // welding quantum derived from mesh extent so it scales with content, with
    // an absolute floor for tiny meshes (finding 5).
    let b = compute_bounds(&mesh.vertices);
    let pos_quant = (b.radius.max(params.weld.pos_quant_min) * params.weld.pos_quant_frac)
        .max(params.weld.pos_quant_min);

    let mut clusters: Vec<Cluster> = Vec::new();
    let mut groups: Vec<Group> = Vec::new();
    let mut levels: Vec<Vec<u32>> = Vec::new();
    let mut backends: BTreeSet<&'static str> = BTreeSet::new();

    // ---- Level 0: leaves ----
    let mut level_ids: Vec<u32> = Vec::new();
    for (verts, idx) in shardize(mesh, &params.meshlet) {
        let bounds = compute_bounds(&verts);
        let id = clusters.len() as u32;
        clusters.push(Cluster {
            id,
            level: 0,
            vertices: verts,
            indices: idx,
            error: 0.0,
            parent_error: f32::INFINITY, // stamped when a parent group claims it
            children: Vec::new(),
            bounds,
            group: None, // leaves are shardized, not simplified
            parent_group: None,
        });
        level_ids.push(id);
    }
    levels.push(level_ids);

    // ---- Build up (staged, one level at a time) ----
    let mut level = 0u32;
    while levels.last().unwrap().len() > params.min_clusters
        && (level as usize) < params.max_levels
    {
        let prev_ids = levels.last().unwrap().clone();
        // owned snapshot so we can read old clusters while staging new ones.
        let prev: Vec<Cluster> =
            prev_ids.iter().map(|&id| clusters[id as usize].clone()).collect();
        let prev_ref: Vec<&Cluster> = prev.iter().collect();

        let staged = stage_level(
            &prev,
            &prev_ref,
            level,
            pos_quant,
            params,
            partitioner,
            clusters.len() as u32,
            groups.len() as u32,
        );

        // Progress gate: only accept a level that actually shrinks the front.
        if !staged.progressed
            || staged.next_ids.is_empty()
            || staged.next_ids.len() >= prev_ids.len()
        {
            // ROLLBACK: nothing committed; children keep parent_error = ∞ and
            // parent_group = None → they stay terminal, no orphans (finding 3).
            break;
        }

        // COMMIT: append clusters + groups, stamp children, push the level.
        if let Some(bk) = staged.backend {
            backends.insert(bk);
        }
        clusters.extend(staged.new_clusters);
        groups.extend(staged.new_groups);
        for (child, perr, pg) in staged.child_links {
            let c = &mut clusters[child as usize];
            c.parent_error = perr;
            c.parent_group = Some(pg);
        }
        levels.push(staged.next_ids);
        level += 1;
    }

    let partitioner = if backends.is_empty() {
        partitioner.name().to_string()
    } else {
        // Canonical order per FORMAT.md ("metis,greedy"), not BTreeSet's alpha
        // order ("greedy,metis") — advisory.
        ["metis", "greedy"]
            .into_iter()
            .filter(|b| backends.contains(b))
            .collect::<Vec<_>>()
            .join(",")
    };

    Ok(Dag {
        clusters,
        groups,
        levels,
        input_tri_count: mesh.tri_count() as u32,
        partitioner,
    })
}

/// Build ONE level entirely in staging (no DAG mutation). Returns the parents,
/// groups, and child link stamps to apply iff the caller accepts.
#[allow(clippy::too_many_arguments)]
fn stage_level(
    prev: &[Cluster],
    prev_ref: &[&Cluster],
    level: u32,
    pos_quant: f32,
    params: &TransmuteParams,
    partitioner: &dyn Partitioner,
    mut next_cluster_id: u32,
    mut next_group_id: u32,
) -> StagedLevel {
    let nparts = prev.len().div_ceil(params.group_size.max(1)).max(1);
    let graph = build_adjacency(prev_ref, pos_quant);
    let partition = partitioner.partition(&graph, nparts);

    // ONE canonical coordinate per position key for the whole level (finding 1),
    // computed before any group welds so every touching group snaps identically.
    let canon = canonical_coords(prev_ref, pos_quant);

    // bucket clusters by part id — BTreeMap keeps group processing order stable
    // (finding 8: HashMap iteration fed METIS nondeterministically).
    let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &pid) in partition.parts.iter().enumerate() {
        buckets.entry(pid).or_default().push(i);
    }

    // BOUNDARY LOCKING setup (finding 1): a canonical position touched by >1
    // part is a shared border and must be LOCKED in every group that owns it,
    // so neighboring groups cut it identically.
    let mut pos_parts: BTreeMap<[i64; 3], BTreeSet<usize>> = BTreeMap::new();
    for (i, &pid) in partition.parts.iter().enumerate() {
        for v in &prev[i].vertices {
            pos_parts
                .entry(pos_key(v.position, pos_quant))
                .or_default()
                .insert(pid);
        }
    }
    let locked_positions: BTreeSet<[i64; 3]> = pos_parts
        .into_iter()
        .filter(|(_, parts)| parts.len() > 1)
        .map(|(k, _)| k)
        .collect();

    let mut staged = StagedLevel {
        new_clusters: Vec::new(),
        new_groups: Vec::new(),
        child_links: Vec::new(),
        next_ids: Vec::new(),
        backend: Some(partition.backend),
        progressed: false,
    };

    for members in buckets.values() {
        let group: Vec<&Cluster> = members.iter().map(|&i| &prev[i]).collect();
        let mut child_ids: Vec<u32> = group.iter().map(|c| c.id).collect();
        child_ids.sort_unstable();
        let child_err_max = group.iter().fold(0.0f32, |a, c| a.max(c.error));

        let welded = weld_group(&group, &params.weld, pos_quant, &canon);
        let merged_tris = welded.mesh.tri_count();
        if merged_tris == 0 {
            continue;
        }
        // SHARED LOD bounds sphere: the merged child geometry, computed once
        // for the whole group (finding 2).
        let group_bounds = compute_bounds(&welded.mesh.vertices);

        // per-vertex lock flags from the canonical-position lock set (finding 1).
        let locks: Vec<bool> = welded
            .canonical
            .iter()
            .map(|k| locked_positions.contains(k))
            .collect();

        let target_tris = ((merged_tris as f32) * params.simplify_ratio).ceil() as usize;
        let target_index_count = target_tris.max(1) * 3;
        let (simplified, abs_error) = simplify_group(
            &welded.mesh,
            &locks,
            target_index_count,
            params,
        );

        // group LOD error is monotone: at least the children's, plus this step.
        let group_error = child_err_max.max(abs_error);

        let simplified_mesh = Mesh::new(welded.mesh.vertices.clone(), simplified.clone());
        let parents = shardize(&simplified_mesh, &params.meshlet);
        if parents.is_empty() {
            continue;
        }

        let produced_tris: usize = parents.iter().map(|(_, i)| i.len() / 3).sum();
        if produced_tris < merged_tris {
            staged.progressed = true;
        }

        let group_id = next_group_id;
        next_group_id += 1;
        let mut parent_ids: Vec<u32> = Vec::with_capacity(parents.len());

        for (verts, idx) in parents {
            let bounds = compute_bounds(&verts);
            let id = next_cluster_id;
            next_cluster_id += 1;
            staged.new_clusters.push(Cluster {
                id,
                level: level + 1,
                vertices: verts,
                indices: idx,
                error: group_error,
                parent_error: f32::INFINITY,
                children: child_ids.clone(),
                bounds,
                group: Some(group_id),
                parent_group: None,
            });
            parent_ids.push(id);
            staged.next_ids.push(id);
        }

        // stamp children (applied only on commit) with the SHARED metric.
        for &cid in &child_ids {
            staged.child_links.push((cid, group_error, group_id));
        }

        staged.new_groups.push(Group {
            id: group_id,
            level,
            children: child_ids,
            parents: parent_ids,
            bounds: group_bounds,
            error: group_error,
        });
    }

    staged
}

/// Simplify a welded group toward `target_index_count`, LOCKING shared-boundary
/// vertices and weighing normal/uv attributes so seams don't smear (findings
/// 1 + 5). Returns new indices + ABSOLUTE geometric error (world units).
fn simplify_group(
    mesh: &Mesh,
    locks: &[bool],
    target_index_count: usize,
    params: &TransmuteParams,
) -> (Vec<u32>, f32) {
    let adapter = VertexDataAdapter::new(mesh.vertex_bytes(), VERTEX_STRIDE, POSITION_OFFSET)
        .expect("vertex adapter");
    // attribute stream: normal.xyz + uv.xy per vertex (finding 5 — attribute-
    // aware error keeps UV/normal from being smeared by the collapse).
    let mut attrs: Vec<f32> = Vec::with_capacity(mesh.vertices.len() * 5);
    for v in &mesh.vertices {
        attrs.extend_from_slice(&v.normal);
        attrs.extend_from_slice(&v.uv);
    }
    let weights = [
        params.weld.normal_weight,
        params.weld.normal_weight,
        params.weld.normal_weight,
        params.weld.uv_weight,
        params.weld.uv_weight,
    ];
    let mut rel_error = 0.0f32;
    // LockBorder is the minimum floor even beyond our explicit locks, so an
    // open border a partition never shared is still not cut loose.
    let options = SimplifyOptions::LockBorder;
    let result = meshopt::simplify_with_attributes_and_locks(
        &mesh.indices,
        &adapter,
        &attrs,
        &weights,
        5 * std::mem::size_of::<f32>(),
        locks,
        target_index_count,
        params.target_error,
        options,
        Some(&mut rel_error),
    );
    let scale = meshopt::simplify_scale(&adapter);
    let abs_error = rel_error * scale;
    if result.len() >= mesh.indices.len() || result.is_empty() {
        (mesh.indices.clone(), abs_error)
    } else {
        (result, abs_error)
    }
}
