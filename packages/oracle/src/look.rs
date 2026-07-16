//! `look()` — the Matrix-vision Glance. Pure-geometry projection of entity
//! bounds into a view frustum: CAPTIONS (default) + an on-demand GLANCE GRID
//! whose cells carry dominant entityId + per-cell geometry depth. No pixels, no
//! streaming, no timers.
//!
//! FRUSTUM TRUTH: caption membership is a real 6-plane frustum/AABB test — an
//! entity off to the side (outside the lateral/vertical FOV) is REJECTED, never
//! clamped onto a border cell. Grid cells are filled by casting a ray through
//! each cell and intersecting the entity AABBs, so both the id and the depth a
//! cell reports are the geometry actually projecting there.
//!
//! DOMINANCE RULE (public, deterministic): a cell is owned by the entity whose
//! AABB has the NEAREST intersection distance along that cell's ray. Ties
//! (within [`DEPTH_TIE_EPS`]) break toward the lexicographically smaller entity
//! id, so a Glance is a pure, stable function of world DATA.
//!
//! LAYERS (RAIN context diet): the grid stack is CHANNEL-SEPARATED (SoA) and
//! computed ONLY for the layers requested in [`LookParams::layers`]. With no
//! grid layer requested, no buffers are computed at all (captions only). The id
//! channel is TRULY lazy: neither its per-cell buffer NOR the interned id table
//! is allocated, and its per-cell work never runs, unless ids are asked (proven
//! by instrumented op AND alloc counters in the tests). Depth is the dominance
//! key — the nearest-entity rule cannot be resolved without it — so it is always
//! computed as scratch, but its buffer is only MATERIALIZED as output (moved,
//! never cloned) when the depth layer is requested; otherwise it is dropped.
//!
//! IDS ARE INTERNED: the id channel is a table of unique id strings
//! ([`Glance::id_table`]) plus a per-cell `u32` index buffer ([`Glance::ids`])
//! whose [`EMPTY_CELL`] sentinel marks an unstamped cell. Each id string is
//! cloned exactly once (into the table), never per cell — so a fully stamped
//! grid holds at most `entities` id strings, not `grid²`.
//!
//! COST: each gaze scans every entity once (O(entities)); a spatial index is the
//! tracked future optimization (not built yet).

use crate::geom::{
    camera_basis, dot, forward, frustum_intersects_aabb, frustum_planes, length, normalize,
    ray_aabb, scale3, sub, Aabb, Vec3,
};
use crate::model::{EntityGeom, World};
use serde::Serialize;
use std::fmt;

/// Default depth equality tolerance for the dominance tiebreak (meters). It is
/// the default of [`LookParams::tie_eps`], never a hardcoded constant in the
/// hot loop.
pub const DEPTH_TIE_EPS: f32 = 1e-3;

/// Hard grid-cell ceiling (`4096²`) that bounds the allocation path at ANY
/// `max_grid`. `checked_mul` catches `usize` overflow — but the REAL non-OOM
/// guarantee is [`LookParams::max_grid_bytes`], a byte budget enforced BEFORE
/// any reserve (an overcommitting OS lazily "succeeds" a multi-terabyte reserve
/// and only OOM-kills when pages are touched, so a cell count alone is not
/// enough). 2^24 cells is far beyond any real glance (study = 128²), and a grid
/// past it is unusable data anyway. This is a safety ceiling, not a tuning knob
/// (unlike `max_grid`), so it is a fixed documented constant.
pub const MAX_GRID_CELLS: usize = 1 << 24;

/// Worst-case per-cell allocation across the SoA grid buffers (bytes), used to
/// derive [`LookParams::max_grid_bytes`]'s default. At the ceiling with BOTH
/// layers we reserve: the `f32` depth scratch (4 B, moved into output when depth
/// is asked — no extra), the `Option<usize>` dominance-index scratch
/// (`size_of` = 16 B on 64-bit), and the `u32` per-cell interned-id output
/// (4 B). 4 + 16 + 4 = 24 B/cell. Depth-only reserves just the 4 B scratch. The
/// O(entities) id side (mapping + interned table) is budgeted SEPARATELY (see
/// [`DEFAULT_ID_SIDE_HEADROOM`]) since it scales with the world, not the grid.
pub const GRID_BYTES_PER_CELL: usize = 4 + std::mem::size_of::<Option<usize>>() + 4;

/// Headroom (bytes) the default [`LookParams::max_grid_bytes`] adds on top of
/// the O(cells) grid worst case for the O(entities) ID SIDE — the visible→table
/// mapping (`Option<u32>` per visible entity) plus the interned `id_table`
/// (unique id strings). Unlike the grid buffers this scales with the world's
/// entity count, not the grid resolution, and is far smaller in any real glance
/// (entities ≪ cells). The default reserves a generous fixed allowance so the
/// id side is covered without conflating it with the grid ceiling; an explicit
/// lower `max_grid_bytes` still rejects a pathological table BEFORE allocation
/// (the reserve is fallible either way). A param-derived default, not magic.
pub const DEFAULT_ID_SIDE_HEADROOM: usize = 8 << 20; // 8 MiB

/// Sentinel per-cell id index meaning "no entity stamped this cell". `u32::MAX`
/// can never be a real table index (a glance can hold at most `entities` ids,
/// far below `2^32`), so it is an unambiguous empty marker in [`Glance::ids`].
pub const EMPTY_CELL: u32 = u32::MAX;

/// Serialized sentinel for a depth cell with NO cover. In memory an unhit cell
/// holds `+inf` ([`Glance::cell_depth`]); JSON has no `+inf` and serde would
/// emit `null` for it — a placeholder the RAIN protocol forbids (an unrequested
/// or empty channel must never look like a real value). A HIT depth is a
/// ray-entry distance, always `> 0` (≥ `near`), so `-1` is an unambiguous
/// "no cover" marker that keeps the serialized depth layer null-free. This is
/// the DOCUMENTED sparse-cell encoding: in-memory `+inf`, on the wire `-1`.
pub const SPARSE_DEPTH_SENTINEL: f32 = -1.0;

/// Which grid layers a gaze should compute. Unrequested layers are not stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Layers {
    /// Dominant-entity-id channel (the glance grid proper).
    pub ids: bool,
    /// Per-cell geometry depth channel (meters along the cell ray).
    pub depth: bool,
}
impl Layers {
    pub const NONE: Layers = Layers {
        ids: false,
        depth: false,
    };
    pub const IDS: Layers = Layers {
        ids: true,
        depth: false,
    };
    pub const DEPTH: Layers = Layers {
        ids: false,
        depth: true,
    };
    pub const BOTH: Layers = Layers {
        ids: true,
        depth: true,
    };
    pub fn any(&self) -> bool {
        self.ids || self.depth
    }
}
impl Default for Layers {
    /// RAIN default mode = captions + the glance grid (ids). Depth is a pulled
    /// layer, not computed unless asked.
    fn default() -> Self {
        Layers::IDS
    }
}

/// Where the eye is and where it points. Position is the EYE, like a GAIA spawn.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct EyePose {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

/// Every resolution / fov / range / count is a param with a default — the RAIN
/// iron law: never hardcode. Deeper levels (32, 128) are just a bigger `grid`.
#[derive(Clone, Copy, Debug)]
pub struct LookParams {
    /// Vertical field of view in degrees (grid is square, aspect 1).
    pub fov_deg: f32,
    /// Glance grid resolution N (N×N). 8 = glance, 32 = regard, 128 = study.
    pub grid: usize,
    /// Nearest-N entities listed in the captions.
    pub nearest_n: usize,
    /// Near plane; geometry closer than this is ignored.
    pub near: f32,
    /// Far cull for the captions/grid.
    pub far: f32,
    /// Which grid layers to compute (RAIN context diet).
    pub layers: Layers,
    /// Hard cap on `grid` — rejects absurd resolutions before any allocation.
    /// Raising it never makes allocation unsafe: `grid²` is checked and the
    /// grid buffers are fallibly reserved (typed errors, never OOM/abort).
    pub max_grid: usize,
    /// Byte budget for the SoA grid buffers — the REAL non-OOM guarantee. The
    /// estimated reserve (`grid² ×` per-cell bytes for the requested layers) is
    /// checked against this BEFORE any allocation, so an overcommitting OS can
    /// never "succeed" a giant reserve and OOM-kill on first write. Covers BOTH
    /// the O(cells) grid buffers AND the O(entities) id side (mapping + interned
    /// table). Default is [`GRID_BYTES_PER_CELL`]`×`[`MAX_GRID_CELLS`] (the 4096²
    /// both-layers grid worst case) `+` [`DEFAULT_ID_SIDE_HEADROOM`] — a param,
    /// never hidden magic.
    pub max_grid_bytes: usize,
    /// An entity counts as world-support (excluded from captions) when its
    /// largest extent exceeds `support_ratio × range`.
    pub support_ratio: f32,
    /// Include world-support surfaces (ground/sea) in captions anyway.
    pub include_support: bool,
    /// Depth equality tolerance (meters) for the dominance tiebreak. Default
    /// [`DEPTH_TIE_EPS`]; a param, never hardcoded in the loop.
    pub tie_eps: f32,
}
impl Default for LookParams {
    fn default() -> Self {
        Self {
            fov_deg: 60.0,
            grid: 8,
            nearest_n: 5,
            near: 0.05,
            far: 5000.0,
            layers: Layers::default(),
            max_grid: 512,
            max_grid_bytes: GRID_BYTES_PER_CELL * MAX_GRID_CELLS + DEFAULT_ID_SIDE_HEADROOM,
            support_ratio: 8.0,
            include_support: false,
            tie_eps: DEPTH_TIE_EPS,
        }
    }
}

/// Typed rejection of nonsensical look parameters — no panic, no alloc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LookError {
    /// `grid` was 0 or exceeded `max_grid`.
    InvalidGrid { grid: usize, max: usize },
    /// `fov_deg` was ≤0, ≥180, or NaN.
    InvalidFov { fov_deg: f32 },
    /// `near`/`far` were not a finite range with `0 < near < far`.
    InvalidRange { near: f32, far: f32 },
    /// `grid²` overflowed `usize` (only reachable with an enormous `max_grid`).
    GridOverflow { grid: usize },
    /// The grid buffers could not be allocated (`grid²` cells too large for
    /// memory) — reported, never a panic or abort.
    AllocFailed { cells: usize },
    /// The estimated SoA grid reserve exceeded [`LookParams::max_grid_bytes`].
    /// Rejected BEFORE any allocation — the real non-OOM guard.
    ByteBudget { bytes: usize, budget: usize },
    /// `tie_eps` was negative or not finite (NaN/±inf).
    InvalidTieEps { tie_eps: f32 },
}
impl fmt::Display for LookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LookError::InvalidGrid { grid, max } => {
                write!(f, "grid {grid} out of range (1..={max})")
            }
            LookError::InvalidFov { fov_deg } => {
                write!(f, "fov {fov_deg} out of range (0 < fov < 180)")
            }
            LookError::InvalidRange { near, far } => {
                write!(
                    f,
                    "near/far invalid: need 0 < near < far (got {near}/{far})"
                )
            }
            LookError::GridOverflow { grid } => {
                write!(f, "grid {grid} squared overflows usize")
            }
            LookError::AllocFailed { cells } => {
                write!(f, "could not allocate {cells}-cell glance grid")
            }
            LookError::ByteBudget { bytes, budget } => {
                write!(
                    f,
                    "glance grid needs {bytes} bytes, over the {budget}-byte budget"
                )
            }
            LookError::InvalidTieEps { tie_eps } => {
                write!(f, "tie_eps {tie_eps} invalid (need finite, >= 0)")
            }
        }
    }
}
impl std::error::Error for LookError {}

/// One nearby entity in the caption layer.
#[derive(Clone, Debug, Serialize)]
pub struct NearEntity {
    pub id: String,
    /// Horizontal bearing, degrees: negative = left, positive = right.
    pub bearing_deg: f32,
    /// Vertical angle, degrees: positive = above the eye.
    pub elevation_deg: f32,
    pub range: f32,
    /// Largest world-space dimension of the entity's bounds.
    pub size: f32,
    /// Emissive color string of the entity's first emissive part, if any.
    /// Absent from serialized output when the entity has none (never `null`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emissive: Option<String>,
    /// World-support surface (extent ≫ range) — normally hidden from captions.
    pub support: bool,
}

/// The full Glance: captions + (optional) CHANNEL-SEPARATED grid layers.
///
/// The grid is Struct-of-Arrays: [`Glance::depth`] and [`Glance::ids`] are each
/// `Some` only when their layer was requested (an unrequested channel is `None`
/// — zero allocation, zero per-cell work). Both, when present, are row-major
/// (`grid*grid`, row 0 = TOP). The id channel is INTERNED: `ids` holds a
/// per-cell `u32` index into [`Glance::id_table`] (unique id strings), with the
/// [`EMPTY_CELL`] sentinel for an unstamped cell.
#[derive(Clone, Debug, Serialize)]
pub struct Glance {
    pub eye: EyePose,
    pub fov_deg: f32,
    pub grid: usize,
    /// Number of renderable entities intersecting the frustum.
    pub entity_count: usize,
    pub nearest: Vec<NearEntity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
    /// Which grid layers were computed.
    pub layers: Layers,
    /// Per-cell nearest geometry depth (meters), row-major. `None` unless the
    /// depth layer was requested (then the whole field is ABSENT from serialized
    /// output — `skip_serializing_if`). In memory a cell with no cover holds
    /// `+inf`; SERIALIZED it is [`SPARSE_DEPTH_SENTINEL`] (`-1`), never `null`.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_depth"
    )]
    pub depth: Option<Vec<f32>>,
    /// Per-cell dominant-entity index into [`Glance::id_table`], row-major.
    /// `None` unless the ids layer was requested — then ABSENT from serialized
    /// output. [`EMPTY_CELL`] (`u32::MAX`, a number, never `null`) = no cover.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ids: Option<Vec<u32>>,
    /// Interned unique dominant-entity ids referenced by [`Glance::ids`]. Empty
    /// (and ABSENT from serialized output) when ids were not requested.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub id_table: Vec<String>,
}

/// Serialize the depth channel null-free: map every unhit (`+inf`) cell to
/// [`SPARSE_DEPTH_SENTINEL`] (`-1`) so the emitted array carries no `null`
/// token. Only invoked when the layer is `Some` (`skip_serializing_if` handles
/// `None` — an unrequested channel is absent entirely).
fn serialize_depth<S: serde::Serializer>(
    depth: &Option<Vec<f32>>,
    s: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    let v = depth
        .as_ref()
        .expect("serialize_depth is only reached for Some (skip_serializing_if None)");
    let mut seq = s.serialize_seq(Some(v.len()))?;
    for &d in v {
        let out = if d.is_finite() {
            d
        } else {
            SPARSE_DEPTH_SENTINEL
        };
        seq.serialize_element(&out)?;
    }
    seq.end()
}

impl Glance {
    /// Number of grid cells actually computed (`grid²`, or 0 when no grid layer
    /// was requested).
    pub fn cell_count(&self) -> usize {
        self.depth
            .as_ref()
            .map(Vec::len)
            .or_else(|| self.ids.as_ref().map(Vec::len))
            .unwrap_or(0)
    }
    /// Dominant entity id at row-major cell `i` (`None` if ids not requested,
    /// cell empty, or `i` out of range).
    pub fn cell_id(&self, i: usize) -> Option<&str> {
        let ids = self.ids.as_ref()?;
        let idx = *ids.get(i)?;
        if idx == EMPTY_CELL {
            None
        } else {
            self.id_table.get(idx as usize).map(String::as_str)
        }
    }
    /// Nearest cover depth at row-major cell `i` (meters). `+inf` when depth was
    /// not requested, the cell is empty, or `i` is out of range.
    pub fn cell_depth(&self, i: usize) -> f32 {
        self.depth
            .as_ref()
            .and_then(|d| d.get(i).copied())
            .unwrap_or(f32::INFINITY)
    }
    /// The horizon row index under the current pitch: the row where the world
    /// horizontal at eye height projects. Rows strictly less than this are
    /// "above the horizon". Only equals `grid/2` at pitch 0.
    pub fn horizon_row(&self) -> usize {
        let (fwd, _r, up) = camera_basis(self.eye.yaw, self.eye.pitch);
        let h = forward(self.eye.yaw, 0.0); // world-horizontal in the view's yaw plane
        let zc = dot(h, fwd);
        if zc <= 1e-4 {
            // Horizon is at/behind the view edge (near-vertical gaze).
            return if self.eye.pitch > 0.0 { self.grid } else { 0 };
        }
        let tan_half = (self.fov_deg.to_radians() * 0.5).tan().max(1e-4);
        let ndc_y = (dot(h, up) / zc) / tan_half;
        let row_f = (0.5 - ndc_y * 0.5) * self.grid as f32;
        row_f.round().clamp(0.0, self.grid as f32) as usize
    }
    /// ids-only layer (dominant entity per cell), row-major. Empty when the ids
    /// layer was not requested.
    pub fn ids_layer(&self) -> Vec<Option<String>> {
        (0..self.cell_count())
            .map(|i| self.cell_id(i).map(str::to_string))
            .collect()
    }
    /// depth-only layer (nearest cover depth per cell; +inf = empty), row-major.
    /// Empty when the depth layer was not requested.
    pub fn depth_layer(&self) -> Vec<f32> {
        self.depth.clone().unwrap_or_default()
    }
    /// True if `id` occupies any cell strictly above the horizon row.
    pub fn is_above_horizon(&self, id: &str) -> bool {
        let horizon = self.horizon_row();
        (0..self.cell_count()).any(|i| (i / self.grid) < horizon && self.cell_id(i) == Some(id))
    }
    /// Every cell (row, col) a given id covers.
    pub fn cells_of(&self, id: &str) -> Vec<(usize, usize)> {
        (0..self.cell_count())
            .filter(|&i| self.cell_id(i) == Some(id))
            .map(|i| (i / self.grid, i % self.grid))
            .collect()
    }
}

/// Project the world's entity bounds into the frustum at `eye`. The Glance is a
/// pure function of world DATA (read LIVE from the ECS) — call it when asked,
/// never on a clock. Invalid params are rejected as typed [`LookError`]s.
pub fn look(world: &World, eye: EyePose, params: LookParams) -> Result<Glance, LookError> {
    // -- input validation (no panic, no unbounded alloc) ------------------
    if params.grid == 0 || params.grid > params.max_grid {
        return Err(LookError::InvalidGrid {
            grid: params.grid,
            max: params.max_grid,
        });
    }
    if !params.fov_deg.is_finite() || params.fov_deg <= 0.0 || params.fov_deg >= 180.0 {
        return Err(LookError::InvalidFov {
            fov_deg: params.fov_deg,
        });
    }
    if !params.near.is_finite()
        || !params.far.is_finite()
        || params.near <= 0.0
        || params.near >= params.far
    {
        return Err(LookError::InvalidRange {
            near: params.near,
            far: params.far,
        });
    }
    if !params.tie_eps.is_finite() || params.tie_eps < 0.0 {
        return Err(LookError::InvalidTieEps {
            tie_eps: params.tie_eps,
        });
    }

    let grid = params.grid;
    let (fwd, right, up) = camera_basis(eye.yaw, eye.pitch);
    let tan_half = (params.fov_deg.to_radians() * 0.5).tan().max(1e-4);
    let planes = frustum_planes(
        eye.position,
        fwd,
        right,
        up,
        tan_half,
        params.near,
        params.far,
    );

    // -- caption pass: real frustum/AABB culling over the LIVE ECS --------
    // Collect the in-frustum renderable entities once; the grid reuses them.
    // N1(a): `visible` carries the ENTITY INDEX (not a cloned id string) with
    // its bounds — the collection phase does ZERO id-string work, so a
    // depth-only glance never touches an id. The id string is resolved from the
    // entity list ONLY at intern time, and ONLY when the ids layer is requested
    // (see `rasterize`). The caption id clone below is unrelated to the grid:
    // captions always list ids regardless of layers.
    let mut visible: Vec<(usize, Aabb)> = Vec::new();
    let mut nearest: Vec<NearEntity> = Vec::new();

    for (ei, ent) in world.entities.iter().enumerate() {
        let Some(geom) = world.geometry(&ent.id) else {
            continue;
        };
        let Some(bounds) = geom.bounds else { continue };
        if !frustum_intersects_aabb(&planes, &bounds) {
            continue;
        }
        let (bearing, elevation, range) = bearing_of(&geom, eye, fwd, right, up);
        let size = geom.max_extent();
        let support = range > 1e-3 && size > params.support_ratio * range;
        nearest.push(NearEntity {
            id: ent.id.clone(),
            bearing_deg: bearing,
            elevation_deg: elevation,
            range,
            size,
            emissive: geom.emissive.clone(),
            support,
        });
        visible.push((ei, bounds));
    }
    let entity_count = visible.len();

    // Captions rank by range; world-support surfaces (ground/sea) are demoted
    // out unless explicitly included, so they never eat a nearest-N slot.
    if !params.include_support {
        nearest.retain(|n| !n.support);
    }
    nearest.sort_by(|a, b| a.range.total_cmp(&b.range).then_with(|| a.id.cmp(&b.id)));
    nearest.truncate(params.nearest_n);

    // -- grid pass: computed ONLY for requested layers --------------------
    let grid_buffers = if params.layers.any() {
        let ctx = RasterCtx {
            eye,
            fwd,
            right,
            up,
            tan_half,
            grid,
            near: params.near,
            far: params.far,
            layers: params.layers,
            tie_eps: params.tie_eps,
            max_grid_bytes: params.max_grid_bytes,
        };
        rasterize(&visible, world, &ctx)?
    } else {
        GridBuffers::default()
    };

    Ok(Glance {
        eye,
        fov_deg: params.fov_deg,
        grid,
        entity_count,
        nearest,
        environment: environment_summary(world),
        layers: params.layers,
        depth: grid_buffers.depth,
        ids: grid_buffers.ids,
        id_table: grid_buffers.id_table,
    })
}

/// Bundle of the per-gaze geometry the rasterizer needs (keeps `rasterize` to a
/// single argument — no long parameter list).
struct RasterCtx {
    eye: EyePose,
    fwd: Vec3,
    right: Vec3,
    up: Vec3,
    tan_half: f32,
    grid: usize,
    near: f32,
    far: f32,
    layers: Layers,
    tie_eps: f32,
    max_grid_bytes: usize,
}

/// Channel-separated grid output (SoA). Each channel is `Some` only when its
/// layer was requested; the id table is interned (unique strings, referenced by
/// the per-cell `u32` indices with the [`EMPTY_CELL`] sentinel).
#[derive(Default)]
struct GridBuffers {
    depth: Option<Vec<f32>>,
    ids: Option<Vec<u32>>,
    id_table: Vec<String>,
}

#[cfg(test)]
thread_local! {
    /// Instrumentation for the lazy-layer proof: counts every per-cell
    /// ID-channel operation. It stays 0 when ids are not requested (the id
    /// branch is never entered), proving the channel does no work when unasked.
    pub(crate) static ID_STAMP_OPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Companion counter: increments once per ID-channel BUFFER allocation, each
    /// AFTER its fallible reserve SUCCEEDS (the per-cell index scratch, the
    /// visible→table mapping, the interned table, and the per-cell `u32`
    /// output). Stays 0 depth-only, proving no id-channel memory is even
    /// reserved when unasked.
    pub(crate) static ID_ALLOC_OPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Grid id-STRING birth counter: increments once per unique id string cloned
    /// into [`Glance::id_table`] — the ONLY place a grid id string is
    /// materialized. Measured across the WHOLE glance (collection + rasterize),
    /// it stays 0 when ids are unrequested, proving the depth-only path allocates
    /// no id string anywhere (the collection phase carries indices, not ids).
    pub(crate) static ID_STRING_OPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Cast a ray through each candidate cell and stamp the nearest-AABB entity
/// and/or depth. A cell is stamped ONLY on a TRUE ray/AABB intersection — no
/// off-ray depth is ever fabricated. The id channel is allocated and worked
/// ONLY when requested; depth is always computed as the dominance key but its
/// output is withheld unless the depth layer asked.
fn rasterize(
    visible: &[(usize, Aabb)],
    world: &World,
    ctx: &RasterCtx,
) -> Result<GridBuffers, LookError> {
    let grid = ctx.grid;
    // Grid id strings are resolved from the entity list ONLY here (the id
    // channel). `visible` carries entity INDICES, so depth-only never reaches
    // this and allocates no id string (see ID_STRING_OPS).
    let id_of = |vi: usize| world.entities[visible[vi].0].id.as_str();
    // Non-bypassable allocation safety: `grid²` is checked (typed overflow error),
    // capped by MAX_GRID_CELLS, gated by a BYTE budget (the real non-OOM guard),
    // and every buffer is FALLIBLY reserved (typed alloc error) — safe at ANY
    // `max_grid`, never OOM, never abort, never panic.
    let n = grid
        .checked_mul(grid)
        .ok_or(LookError::GridOverflow { grid })?;
    if n > MAX_GRID_CELLS {
        // Refuse BEFORE touching memory — an overcommitting allocator would
        // accept the reserve and OOM-kill only on first write.
        return Err(LookError::AllocFailed { cells: n });
    }
    // BYTE BUDGET (N5): sum the bytes we will actually reserve and reject over
    // budget BEFORE any allocation. Two terms:
    //  (1) PER-CELL grid buffers (O(cells)): the 4 B dominance-key scratch
    //      (moved into the depth output, so depth adds nothing beyond it) plus,
    //      when ids are asked, the `Option<usize>` index scratch + `u32` output.
    //  (2) ID SIDE (O(entities), N5 remainder): the visible→table `mapping`
    //      (one `Option<u32>` per visible entity) and the interned `id_table`
    //      (headers + the REAL id byte lengths, worst case every visible id
    //      covers a cell). These were previously allocated INFALLIBLY and left
    //      out of the budget; they are counted here and reserved fallibly below.
    let per_cell_bytes = 4 // best_depth scratch (f32), moved into depth output
        + if ctx.layers.ids {
            std::mem::size_of::<Option<usize>>() + 4 // index scratch + u32 output
        } else {
            0
        };
    let grid_bytes = n.saturating_mul(per_cell_bytes);
    let id_side_bytes = if ctx.layers.ids {
        let mapping = visible
            .len()
            .saturating_mul(std::mem::size_of::<Option<u32>>());
        let table_headers = visible.len().saturating_mul(std::mem::size_of::<String>());
        let table_content: usize = visible
            .iter()
            .map(|(ei, _)| world.entities[*ei].id.len())
            .sum();
        mapping
            .saturating_add(table_headers)
            .saturating_add(table_content)
    } else {
        0
    };
    let est_bytes = grid_bytes.saturating_add(id_side_bytes);
    if est_bytes > ctx.max_grid_bytes {
        return Err(LookError::ByteBudget {
            bytes: est_bytes,
            budget: ctx.max_grid_bytes,
        });
    }

    let mut best_depth: Vec<f32> = Vec::new();
    best_depth
        .try_reserve_exact(n)
        .map_err(|_| LookError::AllocFailed { cells: n })?;
    best_depth.resize(n, f32::INFINITY);

    // TRUE LAZY LAYER: the id index buffer is not even CONSTRUCTED unless ids are
    // requested (type-level proof: it is `None`, no allocation, no per-cell work).
    let mut best_id: Option<Vec<Option<usize>>> = if ctx.layers.ids {
        let mut v: Vec<Option<usize>> = Vec::new();
        v.try_reserve_exact(n)
            .map_err(|_| LookError::AllocFailed { cells: n })?;
        v.resize(n, None);
        // N1(b): count AFTER the reserve succeeds, never before.
        #[cfg(test)]
        ID_ALLOC_OPS.with(|c| c.set(c.get() + 1));
        Some(v)
    } else {
        None
    };

    // Unit ray direction per cell (row 0 = top).
    let cell_dir = |row: usize, col: usize| -> Vec3 {
        let ndc_x = ((col as f32 + 0.5) / grid as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((row as f32 + 0.5) / grid as f32) * 2.0;
        normalize(add3(
            ctx.fwd,
            add3(
                scale3(ctx.right, ndc_x * ctx.tan_half),
                scale3(ctx.up, ndc_y * ctx.tan_half),
            ),
        ))
    };

    for (ei, (_ent_idx, bounds)) in visible.iter().enumerate() {
        // Candidate cell rectangle: an OPTIMIZATION only — it bounds which cells
        // are ray-tested. The actual stamp is gated on a real ray hit, so the
        // rect is a conservative superset (widened by one cell) and never
        // fabricates coverage.
        let (cc0, cc1, rr0, rr1) = candidate_rect(bounds, ctx);
        for row in rr0..=rr1 {
            for col in cc0..=cc1 {
                let dir = cell_dir(row, col);
                // NO FABRICATED DEPTH: only a true ray/AABB intersection stamps a
                // cell, and the stamped depth IS that ray-true entry distance.
                // A cell whose center ray misses the box is left untouched.
                let Some(depth) = ray_aabb(ctx.eye.position, dir, bounds, ctx.near, ctx.far) else {
                    continue;
                };
                let i = row * grid + col;
                match best_id.as_mut() {
                    // ids requested: resolve the dominant entity with the tie rule
                    // (this whole branch is the per-cell ID-channel work).
                    Some(ids) => {
                        #[cfg(test)]
                        ID_STAMP_OPS.with(|c| c.set(c.get() + 1));
                        let closer = depth < best_depth[i] - ctx.tie_eps;
                        let tie = (depth - best_depth[i]).abs() <= ctx.tie_eps
                            && ids[i].is_none_or(|cur| id_of(ei) < id_of(cur));
                        if closer || tie {
                            best_depth[i] = depth.min(best_depth[i]);
                            ids[i] = Some(ei);
                        }
                    }
                    // depth-only: no id buffer, no id work — just the nearest depth.
                    None => {
                        if depth < best_depth[i] {
                            best_depth[i] = depth;
                        }
                    }
                }
            }
        }
    }

    // -- materialize the SoA output --------------------------------------
    // Depth output MOVES the scratch (no clone) when requested, else is dropped.
    let depth = if ctx.layers.depth {
        Some(best_depth)
    } else {
        None
    };

    // IDS: intern into a table of UNIQUE id strings (each cloned exactly ONCE,
    // never per cell) plus a per-cell `u32` index buffer with the EMPTY_CELL
    // sentinel — the SoA id channel.
    let (ids, id_table) = match best_id {
        Some(index_of) => {
            let mut per_cell: Vec<u32> = Vec::new();
            per_cell
                .try_reserve_exact(n)
                .map_err(|_| LookError::AllocFailed { cells: n })?;
            // N1(b): count AFTER the reserve succeeds.
            #[cfg(test)]
            ID_ALLOC_OPS.with(|c| c.set(c.get() + 1));
            // visible-index -> table-index, filled lazily so the table holds
            // only the ids that actually cover a cell (one clone each). N5: both
            // buffers are FALLIBLY reserved (their bytes were budgeted above) so
            // no id-side allocation escapes the byte budget or aborts on OOM.
            let mut mapping: Vec<Option<u32>> = Vec::new();
            mapping
                .try_reserve_exact(visible.len())
                .map_err(|_| LookError::AllocFailed { cells: n })?;
            mapping.resize(visible.len(), None);
            #[cfg(test)]
            ID_ALLOC_OPS.with(|c| c.set(c.get() + 1));
            let mut table: Vec<String> = Vec::new();
            // At most one entry per visible entity ever interns.
            table
                .try_reserve(visible.len())
                .map_err(|_| LookError::AllocFailed { cells: n })?;
            #[cfg(test)]
            ID_ALLOC_OPS.with(|c| c.set(c.get() + 1));
            for slot in index_of {
                match slot {
                    Some(ei) => {
                        let ti = match mapping[ei] {
                            Some(ti) => ti,
                            None => {
                                let ti = table.len() as u32;
                                // The ONLY grid id-string birth (one clone each).
                                #[cfg(test)]
                                ID_STRING_OPS.with(|c| c.set(c.get() + 1));
                                table.push(id_of(ei).to_string());
                                mapping[ei] = Some(ti);
                                ti
                            }
                        };
                        per_cell.push(ti);
                    }
                    None => per_cell.push(EMPTY_CELL),
                }
            }
            (Some(per_cell), table)
        }
        None => (None, Vec::new()),
    };

    Ok(GridBuffers {
        depth,
        ids,
        id_table,
    })
}

/// Candidate cell rectangle for a box: the projected span of its 8 corners,
/// widened by one cell (a conservative superset). A box straddling the near
/// plane (eye inside/behind it — ground/sea) has no meaningful projection, so
/// the whole grid is the candidate set; the per-cell ray test still gates every
/// stamp, so this only affects how many cells are TESTED, never which are filled.
fn candidate_rect(bounds: &Aabb, ctx: &RasterCtx) -> (usize, usize, usize, usize) {
    let grid = ctx.grid;
    let (mut c0, mut c1, mut r0, mut r1) = (grid, 0usize, grid, 0usize);
    let mut straddle = false;
    for corner in bounds.corners() {
        let d = sub(corner, ctx.eye.position);
        let z = dot(d, ctx.fwd);
        if z <= ctx.near {
            straddle = true;
            break;
        }
        let ndc_x = dot(d, ctx.right) / (z * ctx.tan_half);
        let ndc_y = dot(d, ctx.up) / (z * ctx.tan_half);
        let col =
            (((ndc_x * 0.5 + 0.5) * grid as f32).floor()).clamp(0.0, grid as f32 - 1.0) as usize;
        let row =
            (((0.5 - ndc_y * 0.5) * grid as f32).floor()).clamp(0.0, grid as f32 - 1.0) as usize;
        c0 = c0.min(col);
        c1 = c1.max(col);
        r0 = r0.min(row);
        r1 = r1.max(row);
    }
    if straddle {
        return (0, grid - 1, 0, grid - 1);
    }
    // Widen by one cell so an edge cell whose center ray hits near the border is
    // never missed by floor rounding.
    (
        c0.saturating_sub(1),
        (c1 + 1).min(grid - 1),
        r0.saturating_sub(1),
        (r1 + 1).min(grid - 1),
    )
}

fn add3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Bearing/elevation/range of an entity's bounds center relative to the eye.
fn bearing_of(
    geom: &EntityGeom,
    eye: EyePose,
    fwd: Vec3,
    right: Vec3,
    up: Vec3,
) -> (f32, f32, f32) {
    let center = geom.bounds.map(|b| b.center()).unwrap_or(geom.origin);
    let d = sub(center, eye.position);
    let range = length(d);
    let bearing = dot(d, right).atan2(dot(d, fwd)).to_degrees();
    let horiz = (dot(d, right).powi(2) + dot(d, fwd).powi(2)).sqrt();
    let elevation = dot(d, up).atan2(horiz).to_degrees();
    (bearing, elevation, range)
}

fn environment_summary(world: &World) -> Option<String> {
    let env = world.environment()?;
    let sky = env
        .get("sky")
        .and_then(|s| s.get("preset"))
        .and_then(|p| p.as_str());
    let fog = env
        .get("fog")
        .and_then(|f| f.get("density"))
        .and_then(|d| d.as_f64());
    let mut parts = Vec::new();
    if let Some(sky) = sky {
        parts.push(format!("sky {sky}"));
    }
    if let Some(fog) = fog {
        parts.push(format!("fog density {fog}"));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(", "))
    }
}
