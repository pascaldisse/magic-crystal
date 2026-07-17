//! RITE VII · VII-2 — THE HORIZON RESIDENCY RING (data-driven, no stored
//! geometry). DREAMFORGE world law: a universe regenerates from its
//! coordinates alone, so the ground around a moving walker is held in FINITE
//! memory by DATA-DRIVEN RESIDENCY — never by authored streaming volumes (a
//! forbidden concept). This ring is the terrain sibling of `jormungandr`'s
//! disk-page `Ring`: same three invariants, a different backing store. A
//! `jormungandr` page is READ from a `.cbdg` artifact; a horizon tile is
//! MATERIALIZED from `(seed, tile)` through VII-0a's `seed::tile_mesh`
//! (byte-identical on regeneration — proven in `seed`), so "residency" here
//! means "which tiles are currently generated-and-held", capped by a hard
//! byte budget.
//!
//! ## The three invariants (mirrored from `jormungandr::Ring`)
//! 1. **Budget never exceeded** — resident bytes ≤ `budget_bytes` after EVERY
//!    [`HorizonRing::update`].
//! 2. **Required always resident** — every tile of the observer's current
//!    residency square is resident after the update that names it (it is
//!    never an eviction victim).
//! 3. **Determinism** — an identical observer flight replays an identical
//!    load/evict sequence (the field is a pure function of `(seed, coords)`,
//!    the required set a pure function of the observer tile, eviction a
//!    farthest-first order with a deterministic key tie-break).
//!
//! ## Load-ahead / evict-behind
//! The observer's REQUIRED set is a Chebyshev SQUARE of `radius_tiles` around
//! the tile it currently stands in — so the walker always has `radius_tiles`
//! of materialized ground in every direction, INCLUDING ahead (the
//! load-ahead). When the walker crosses a tile boundary the leading column
//! enters the required set (materializes) and the trailing column leaves it;
//! the trailing tiles are then evicted FARTHEST-FROM-OBSERVER first to hold
//! the budget (the evict-behind). Square, not disc, so the required tile
//! count is EXACTLY `(2·radius+1)²` — which lets [`horizon_radius_tiles`]
//! derive the radius so the required set provably fits the budget for ANY
//! sub-tile observer offset (no runtime "did it fit" surprise). The visible
//! horizon is the disc inscribed in that square.
//!
//! ## The byte budget (the sole law, env-tunable)
//! `budget_bytes` = `GAIA_HORIZON_BUDGET_BYTES` (env) or, absent, a default
//! DERIVED to hold a [`DEFAULT_HORIZON_RADIUS_TILES`]-tile square of THIS
//! params' tiles (never a plucked byte count — it scales with the tile's own
//! vertex cost). Whatever the budget, [`horizon_radius_tiles`] re-derives the
//! reach to fit it, so the ONE knob is the budget and world size never
//! appears in the resident cost.
//!
//! ## Reads are synchronous this wave
//! Materialization is a synchronous `seed::tile_mesh` call inside
//! [`HorizonRing::update`] (RITE-VII §OPEN 6 — "sync first, numbers rule").
//! No async job graph yet; the seam for it is the same as `jormungandr`'s.

use std::collections::BTreeMap;

use crate::scene::{RenderScene, SceneParameters};
use crystal::{ComponentDescriptor, EcsWorld};
use seed::terrain::{tile_mesh, tile_origin_m, TerrainParams, TerrainTile};
use serde_json::json;
use std::collections::BTreeMap as FieldMap;
use transmutation::Mesh;

/// Default residency reach when the budget is not overridden: `3` tiles in
/// every direction (a 7×7 = 49-tile resident square). A documented DEFAULT
/// VIEW RADIUS in the house style of `seed`'s `DEFAULT_SLOPE_FRACTION` — the
/// real law is the byte budget, which re-derives the reach when set; this is
/// only the reach the default budget is SIZED to afford. Chosen ≥ 2 so the
/// walker always has its own tile plus a full ring of load-ahead margin
/// before it can reach an unmaterialized tile within one tile-crossing.
pub const DEFAULT_HORIZON_RADIUS_TILES: i64 = 3;

/// Env var naming the hard residency byte budget — the ONE knob (hardcode
/// law: the budget is a param, never a literal in the loop).
pub const BUDGET_ENV: &str = "GAIA_HORIZON_BUDGET_BYTES";

/// The exact byte cost of one tile's materialized mesh, computed analytically
/// from the params' grid — every tile of a given params has the SAME cost
/// (identical vertex/index counts), so the budget arithmetic needs no probe
/// generation. `(n+1)²` vertices · `size_of::<Vertex>()` + `n²·6` indices ·
/// `size_of::<u32>()`.
pub fn tile_byte_cost(params: &TerrainParams) -> u64 {
    let n = params.grid_resolution as u64;
    let verts = (n + 1) * (n + 1);
    let indices = n * n * 6;
    verts * std::mem::size_of::<transmutation::Vertex>() as u64
        + indices * std::mem::size_of::<u32>() as u64
}

/// The ACTUAL byte size of a materialized tile mesh — the resident memory the
/// budget caps. Equal to [`tile_byte_cost`] for the tile's params (uniform
/// grid), but measured from the held mesh so residency accounting is the real
/// allocation, never a promised one. (Planning — budget/radius derivation —
/// uses the analytic [`tile_byte_cost`], like `jormungandr` sizes loads from
/// its index before reading.)
fn mesh_bytes(mesh: &Mesh) -> u64 {
    mesh.vertices.len() as u64 * std::mem::size_of::<transmutation::Vertex>() as u64
        + mesh.indices.len() as u64 * std::mem::size_of::<u32>() as u64
}

/// The largest Chebyshev radius `R` whose required square `(2R+1)²` tiles fit
/// the budget: `max R with (2R+1)²·tile_bytes ≤ budget`. DERIVED, not chosen —
/// the budget is the law, the reach is whatever it affords. Returns `0` when
/// not even a `1×1` tile fits (caller rejects: a horizon needs ≥ a `3×3`).
pub fn horizon_radius_tiles(budget_bytes: u64, tile_bytes: u64) -> i64 {
    if tile_bytes == 0 {
        return 0;
    }
    let max_tiles = budget_bytes / tile_bytes; // floor
    if max_tiles == 0 {
        return 0;
    }
    // (2R+1)² ≤ max_tiles  ⇒  R ≤ (√max_tiles − 1) / 2.
    let side = (max_tiles as f64).sqrt().floor() as i64; // 2R+1 ≤ side
    (side - 1) / 2
}

/// Typed failure — the ring never panics on a mis-sized budget.
#[derive(Debug, PartialEq)]
pub enum HorizonError {
    /// The budget cannot hold even a minimal `3×3` residency square around the
    /// walker — the two invariants (fit the budget / keep the walker's ground
    /// resident) cannot both hold.
    BudgetTooSmall { tile_bytes: u64, budget_bytes: u64 },
}

impl std::fmt::Display for HorizonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HorizonError::BudgetTooSmall {
                tile_bytes,
                budget_bytes,
            } => write!(
                f,
                "horizon budget {budget_bytes} B cannot hold a 3×3 residency square \
                 of {tile_bytes} B tiles (need ≥ {} B)",
                9 * tile_bytes
            ),
        }
    }
}

impl std::error::Error for HorizonError {}

/// A resident tile record: its materialized byte cost (the memory the budget
/// caps) and its world-space centre (the eviction-distance metric, in `f64` so
/// it stays exact at planetary tile magnitude). Mirrors `jormungandr::
/// ResidentPage`, which likewise holds `{bytes, aabb}` and DISCARDS the decoded
/// geometry after measuring it — the geometry is (re)materialized on demand by
/// the consumer (here, [`HorizonRing::scene_at`]'s weld), never double-stored.
struct ResidentTile {
    bytes: u64,
    center_world: [f64; 2],
}

/// Cumulative counters (ordeals + honest telemetry).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HorizonStats {
    pub loads: u64,
    pub evictions: u64,
    /// High-water mark of resident bytes across all updates (≤ budget always).
    pub peak_resident_bytes: u64,
}

/// The outcome of one [`HorizonRing::update`].
#[derive(Clone, Debug, PartialEq)]
pub struct HorizonTick {
    /// Every tile resident after this update, in stable key order.
    pub resident: Vec<TerrainTile>,
    /// Tiles materialized this update (stable key order).
    pub loaded: Vec<TerrainTile>,
    /// Tiles evicted this update, farthest-first.
    pub evicted: Vec<TerrainTile>,
    /// Resident bytes after this update (≤ budget, always).
    pub resident_bytes: u64,
    /// The tile the observer stands in this update.
    pub observer_tile: TerrainTile,
}

/// The horizon residency ring over one terrain field.
pub struct HorizonRing {
    seed: u64,
    params: TerrainParams,
    color: Option<String>,
    budget_bytes: u64,
    tile_bytes: u64,
    radius_tiles: i64,
    resident: BTreeMap<(i64, i64), ResidentTile>,
    resident_bytes: u64,
    stats: HorizonStats,
}

impl HorizonRing {
    /// Build a ring over `(seed, params)` with the budget read from
    /// [`BUDGET_ENV`] (or the derived default). `color` is the terrain sigil's
    /// optional material colour, threaded verbatim into every tile.
    pub fn from_env(
        seed: u64,
        params: TerrainParams,
        color: Option<String>,
    ) -> Result<Self, HorizonError> {
        let tile_bytes = tile_byte_cost(&params);
        let default_budget = {
            let side = 2 * DEFAULT_HORIZON_RADIUS_TILES + 1;
            (side * side) as u64 * tile_bytes
        };
        let budget_bytes = std::env::var(BUDGET_ENV)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(default_budget);
        Self::new(seed, params, color, budget_bytes)
    }

    /// Build a ring with an explicit byte budget (tests pin the budget so the
    /// invariants are checked at a known scale — never the ambient env).
    pub fn new(
        seed: u64,
        params: TerrainParams,
        color: Option<String>,
        budget_bytes: u64,
    ) -> Result<Self, HorizonError> {
        let tile_bytes = tile_byte_cost(&params);
        let radius_tiles = horizon_radius_tiles(budget_bytes, tile_bytes);
        if radius_tiles < 1 {
            return Err(HorizonError::BudgetTooSmall {
                tile_bytes,
                budget_bytes,
            });
        }
        Ok(Self {
            seed,
            params,
            color,
            budget_bytes,
            tile_bytes,
            radius_tiles,
            resident: BTreeMap::new(),
            resident_bytes: 0,
            stats: HorizonStats::default(),
        })
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }
    pub fn tile_bytes(&self) -> u64 {
        self.tile_bytes
    }
    pub fn radius_tiles(&self) -> i64 {
        self.radius_tiles
    }
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }
    pub fn stats(&self) -> HorizonStats {
        self.stats
    }
    pub fn params(&self) -> &TerrainParams {
        &self.params
    }
    pub fn is_resident(&self, tile: TerrainTile) -> bool {
        self.resident.contains_key(&(tile.tile_x, tile.tile_y))
    }
    /// Resident tiles in stable key order (diagnostics / scene build).
    pub fn resident_tiles(&self) -> Vec<TerrainTile> {
        self.resident
            .keys()
            .map(|&(x, y)| TerrainTile::new(x, y))
            .collect()
    }

    /// The tile a world-space xz stands in (`floor(world / tile_size_m)`).
    /// `f64` throughout so it stays exact at planetary tile magnitude.
    pub fn tile_at(&self, world_x: f64, world_z: f64) -> TerrainTile {
        let ts = self.params.tile_size_m as f64;
        TerrainTile::new((world_x / ts).floor() as i64, (world_z / ts).floor() as i64)
    }

    /// One tile's world-space centre (`tile_origin + tile_size/2`), `f64`.
    fn tile_center(&self, tile: TerrainTile) -> [f64; 2] {
        let (ox, oz) = tile_origin_m(tile, &self.params);
        let half = self.params.tile_size_m as f64 / 2.0;
        [ox + half, oz + half]
    }

    /// Advance the ring one update: given the observer's WORLD-space xz, make
    /// every tile of the observer's residency square resident under the byte
    /// budget, evicting farthest-behind non-required tiles farthest-first to
    /// make room. Order (compute required → evict → materialize) guarantees
    /// the budget is never transiently exceeded, exactly as `jormungandr::
    /// Ring::update` does with its index-known page sizes.
    pub fn update(&mut self, observer_x: f64, observer_z: f64) -> HorizonTick {
        let observer_tile = self.tile_at(observer_x, observer_z);

        // Required = Chebyshev square of radius_tiles around the observer tile.
        let mut required: Vec<TerrainTile> = Vec::new();
        for dy in -self.radius_tiles..=self.radius_tiles {
            for dx in -self.radius_tiles..=self.radius_tiles {
                required.push(observer_tile.neighbor(dx, dy));
            }
        }
        // (2R+1)²·tile_bytes ≤ budget by construction of radius_tiles — the
        // required set always fits, for any sub-tile observer offset.
        debug_assert!(required.len() as u64 * self.tile_bytes <= self.budget_bytes);
        let required_keys: std::collections::BTreeSet<(i64, i64)> =
            required.iter().map(|t| (t.tile_x, t.tile_y)).collect();

        // New materializations needed.
        let mut to_load: Vec<TerrainTile> = required
            .iter()
            .copied()
            .filter(|t| !self.resident.contains_key(&(t.tile_x, t.tile_y)))
            .collect();
        to_load.sort_by_key(|t| (t.tile_x, t.tile_y));
        let needed_new = to_load.len() as u64 * self.tile_bytes;

        // Evict farthest-from-observer non-required tiles until the new loads
        // fit. Because required_bytes ≤ budget, evicting every non-required
        // tile always frees enough — the loop can never fail.
        let mut evicted: Vec<TerrainTile> = Vec::new();
        while self.resident_bytes + needed_new > self.budget_bytes {
            let victim = self
                .resident
                .iter()
                .filter(|(k, _)| !required_keys.contains(k))
                .max_by(|(ka, a), (kb, b)| {
                    let da = dist2(a.center_world, [observer_x, observer_z]);
                    let db = dist2(b.center_world, [observer_x, observer_z]);
                    da.partial_cmp(&db)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(ka.cmp(kb))
                })
                .map(|(k, _)| *k);
            let Some(victim) = victim else { break };
            let rt = self.resident.remove(&victim).expect("victim resident");
            self.resident_bytes -= rt.bytes;
            self.stats.evictions += 1;
            evicted.push(TerrainTile::new(victim.0, victim.1));
        }

        // Materialize the new required tiles (space already reserved). The tile
        // is generated to MEASURE its real byte cost, then — like `jormungandr`
        // after computing a page's aabb — the geometry is discarded; the record
        // keeps only {bytes, centre}. `scene_at` regenerates identical meshes
        // (VII-0a determinism) when the resident set must be drawn/collided.
        for &tile in &to_load {
            let mesh = tile_mesh(seed::Seed(self.seed), tile, &self.params);
            let center_world = self.tile_center(tile);
            let bytes = mesh_bytes(&mesh);
            self.resident.insert(
                (tile.tile_x, tile.tile_y),
                ResidentTile { bytes, center_world },
            );
            self.resident_bytes += bytes;
            self.stats.loads += 1;
        }

        if self.resident_bytes > self.stats.peak_resident_bytes {
            self.stats.peak_resident_bytes = self.resident_bytes;
        }

        HorizonTick {
            resident: self.resident_tiles(),
            loaded: to_load,
            evicted,
            resident_bytes: self.resident_bytes,
            observer_tile,
        }
    }

    /// The render_origin the camera-relative frame should sit at for a given
    /// observer world xz: the ORIGIN of the observer's current tile (`f64`,
    /// never rounded through `f32`). Rebasing to this every time the observer
    /// changes tile keeps every resident tile's placement offset (`tile_origin
    /// − render_origin`) bounded by the residency reach — so the `f32` cast in
    /// `scene::terrain_placement_offset` never sees a planet-scale magnitude
    /// (ruling 4 / the camera-relative-rendering guarantee).
    pub fn render_origin_for(&self, observer_x: f64, observer_z: f64) -> [f64; 3] {
        let tile = self.tile_at(observer_x, observer_z);
        let (ox, oz) = tile_origin_m(tile, &self.params);
        [ox, 0.0, oz]
    }

    /// Build a [`RenderScene`] of the CURRENTLY RESIDENT tiles, placed
    /// camera-relative to `render_origin` — the production weld, threading the
    /// MOVING render_origin through the SAME `RenderScene::from_ecs_at` seal
    /// path VII-0b/VII-1 proved (the SOLE geometry path; the terrain rides the
    /// Great Chain like all matter). Only the resident tiles are authored into
    /// the throwaway ECS, so the scene's cost tracks residency, never world
    /// size. The generated meshes are byte-identical to the ring's held ones
    /// (VII-0a determinism), so this regeneration cannot diverge from the
    /// budgeted residency.
    pub fn scene_at(
        &self,
        render_origin: [f64; 3],
        params: &SceneParameters,
    ) -> Result<RenderScene, String> {
        let mut world = EcsWorld::default();
        world.register_component(ComponentDescriptor {
            name: "terrain".to_string(),
            fields: FieldMap::new(),
            enableable: false,
            buffer: true,
            default: None,
        })?;
        let terrain_id = world
            .component_id("terrain")
            .ok_or("terrain component not registered")?;
        for tile in self.resident_tiles() {
            let sigil = self.terrain_sigil_value(tile);
            let entity = world.create_entity(vec![(terrain_id, sigil)])?;
            world.bind_gaia_id(
                format!("horizon_tile_{}_{}", tile.tile_x, tile.tile_y),
                entity,
            )?;
        }
        RenderScene::from_ecs_at(world, params, render_origin)
    }

    /// The full, explicit terrain sigil for one tile — every param dial named
    /// so `TerrainSigil::params` reconstructs THIS ring's params exactly (no
    /// silent re-derivation drift between the ring's held mesh and the weld's
    /// regenerated one).
    fn terrain_sigil_value(&self, tile: TerrainTile) -> serde_json::Value {
        let p = &self.params;
        let mut v = json!({
            "seed": self.seed,
            "tile_x": tile.tile_x,
            "tile_y": tile.tile_y,
            "tile_size_m": p.tile_size_m,
            "grid_resolution": p.grid_resolution,
            "height_amplitude": p.height_amplitude,
            "base_wavelength_m": p.base_wavelength_m,
            "octaves": p.fbm.octaves,
            "lacunarity": p.fbm.lacunarity,
            "gain": p.fbm.gain,
            "warp_strength": p.warp_strength,
            "warp_enabled": p.warp_enabled,
        });
        if let Some(color) = &self.color {
            v.as_object_mut()
                .unwrap()
                .insert("color".to_string(), json!(color));
        }
        v
    }
}

/// Squared distance between two world-space xz points, `f64` (stable,
/// deterministic eviction ordering at any magnitude).
fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dz = a[1] - b[1];
    dx * dx + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> TerrainParams {
        TerrainParams::default()
    }

    #[test]
    fn radius_derives_from_budget_and_fits() {
        let p = params();
        let tb = tile_byte_cost(&p);
        // A budget sized for exactly a 5×5 = 25-tile square.
        let budget = 25 * tb;
        let r = horizon_radius_tiles(budget, tb);
        assert_eq!(r, 2, "√25 = 5 = 2·2+1 ⇒ radius 2");
        // The required square provably fits.
        assert!(((2 * r + 1) * (2 * r + 1)) as u64 * tb <= budget);
    }

    #[test]
    fn tiny_budget_is_rejected_not_panicked() {
        let p = params();
        let tb = tile_byte_cost(&p);
        match HorizonRing::new(7, p, None, tb * 4) {
            Err(err) => assert_eq!(
                err,
                HorizonError::BudgetTooSmall {
                    tile_bytes: tb,
                    budget_bytes: tb * 4
                }
            ),
            Ok(_) => panic!("a 4-tile budget cannot hold a 3×3 square — must be rejected"),
        }
    }
}
