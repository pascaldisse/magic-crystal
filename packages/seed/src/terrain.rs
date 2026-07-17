//! Terrain tiles — seed → Mesh, the first link of RITE VII's chain (VII-0
//! "THE FIRST GROUND", docs/proposals/RITE-VII-THE-PLANET-WALKER.md).
//!
//! A tile is a `(grid_resolution+1)^2` grid sampled from a single world-space
//! height field, `height()`. The height field is a pure function of
//! `(seed, params, world_x, world_z)` — it never sees the tile key — so two
//! tiles that happen to sample the same world-space point always agree,
//! byte-for-byte. That's the whole seam story: shared edges aren't stitched
//! after the fact, they're identical by construction because both sides ran
//! the same function on the same numbers.
//!
//! Flat ground here is deliberately the flat-frame case; Ruling 3 (NARUKO.md
//! "GUARDIAN RULINGS UNDER DELEGATION") makes flat ground the infinite-radius
//! limit of a radial `up(r)` — this atom (VII-0a) only pays for the flat
//! limit, the sphere-domain field is later rite scope (RITE-VII §OPEN 4).
//!
//! Coordinates: tile keys are `i64` and vertex positions are tile-local
//! `f32` (Ruling 4, "64-bit/camera-relative coords: PAID AT VII-0") — never
//! an `f32` measured from the world origin, which is exactly the precision
//! trap 64-bit coords exist to avoid.

use glam::Vec3;
use transmutation::{Mesh, Vertex};

use crate::fields::{Fbm, Noise};
use crate::hash::{coord_key_i64, domain, Seed};

/// A terrain tile's integer address on the world's tile lattice.
///
/// `i64`, not `i32` (Ruling 4): a planet-scale world must be able to name
/// tiles arbitrarily far from the origin without wrapping or truncating.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TerrainTile {
    pub tile_x: i64,
    pub tile_y: i64,
}

impl TerrainTile {
    /// Construct a tile key.
    #[inline]
    pub const fn new(tile_x: i64, tile_y: i64) -> Self {
        TerrainTile { tile_x, tile_y }
    }

    /// The tile offset by `(dx, dy)` tiles — the edge-sharing neighbor used
    /// by the seam ordeal (`+x`/`-x`/`+z`/`-z` are `(1,0)`/`(-1,0)`/`(0,1)`/`(0,-1)`).
    #[inline]
    pub const fn neighbor(self, dx: i64, dy: i64) -> Self {
        TerrainTile {
            tile_x: self.tile_x + dx,
            tile_y: self.tile_y + dy,
        }
    }
}

/// Nyquist floor: the minimum samples-per-wavelength for a sinusoid to be
/// representable at all without aliasing (the sampling theorem's `2`, not a
/// stylistic oversampling choice). `grid_resolution`'s derived default uses
/// exactly this floor against the fBm's finest octave, so the mesh is the
/// coarsest grid that can still legally represent the terrain field it was
/// given — raising `octaves` or `lacunarity` sharpens the finest wavelength
/// and the derived default grid tightens with it.
pub const NYQUIST_SAMPLES_PER_WAVELENGTH: f32 = 2.0;

/// Every tunable of a terrain tile, each a named field with a documented,
/// derived default (IRON LAW: no silently plucked numbers).
#[derive(Clone, Copy, Debug)]
pub struct TerrainParams {
    /// Tile edge length in meters. Default `64.0`: matches the region size
    /// already established as this crate's scatter-ordeal scale
    /// (`packages/seed/tests/ordeals.rs`), so a tile and a scatter region
    /// read as the same physical footprint.
    pub tile_size_m: f32,
    /// Vertex grid cells per tile edge (mesh is `(grid_resolution+1)^2`
    /// vertices). Default DERIVED, not chosen: see [`TerrainParams::default`].
    pub grid_resolution: u32,
    /// Peak height-field amplitude in meters (the fBm's `[-1,1]` output is
    /// scaled by this). Default `tile_size_m * DEFAULT_SLOPE_FRACTION`.
    pub height_amplitude: f32,
    /// World-space wavelength (meters) of the fBm's first (coarsest) octave.
    /// Default `tile_size_m / 2.0`: two undulations of the coarsest octave
    /// span one tile, a visible-relief scale for the VII-0 "first ground"
    /// proof shot without dwarfing the tile itself.
    pub base_wavelength_m: f32,
    /// The fBm octave stack sampled by [`height`]. `frequency` is set from
    /// `base_wavelength_m` (see [`TerrainParams::default`]); `octaves`,
    /// `lacunarity`, `gain` reuse [`Fbm::default`]'s engine defaults.
    pub fbm: Fbm,
    /// Domain-warp offset strength (world meters), applied before the fBm
    /// sample when `warp_enabled`. Default `0.0` (inert until enabled).
    pub warp_strength: f32,
    /// Gate for `domain_warp2`. Default `false`: VII-0 is "the first
    /// ground" — the plainest correct field, so the seam/determinism
    /// ordeals reason about the fBm alone. Warp is wired and provably
    /// seam-safe (it's still a pure function of `(seed, world_x, world_z)`,
    /// tile-key-free like everything else in [`height`]) but left off by
    /// default until a later wave wants the visual variety.
    pub warp_enabled: bool,
}

/// Fraction of `tile_size_m` used as [`TerrainParams::height_amplitude`]'s
/// default. Chosen so the coarsest-octave max slope
/// (`height_amplitude / base_wavelength_m`, with `base_wavelength_m =
/// tile_size_m / 2`) is `2 * DEFAULT_SLOPE_FRACTION`; at `0.15` that's a
/// `0.3` rise-over-run (~17°), comfortably below a walkable-floor grade
/// (Ruling 6) while still reading as relief in a screenshot.
pub const DEFAULT_SLOPE_FRACTION: f32 = 0.15;

impl Default for TerrainParams {
    fn default() -> Self {
        let tile_size_m = 64.0_f32;
        let base_wavelength_m = tile_size_m / 2.0;
        let fbm = Fbm {
            frequency: 1.0 / base_wavelength_m,
            ..Fbm::default()
        };

        // Derived grid_resolution: the finest octave's wavelength must still
        // clear the Nyquist floor against the vertex spacing, or the mesh
        // can't represent the detail the field is generating.
        let finest_wavelength_m =
            base_wavelength_m / fbm.lacunarity.powi(fbm.octaves as i32 - 1);
        let max_cell_size_m = finest_wavelength_m / NYQUIST_SAMPLES_PER_WAVELENGTH;
        let grid_resolution = (tile_size_m / max_cell_size_m).ceil().max(1.0) as u32;

        TerrainParams {
            tile_size_m,
            grid_resolution,
            height_amplitude: tile_size_m * DEFAULT_SLOPE_FRACTION,
            base_wavelength_m,
            fbm,
            warp_strength: 0.0,
            warp_enabled: false,
        }
    }
}

impl TerrainParams {
    /// Grid cell edge length in meters (`tile_size_m / grid_resolution`).
    #[inline]
    pub fn cell_size_m(&self) -> f32 {
        self.tile_size_m / self.grid_resolution as f32
    }
}

/// Derive this world's tile sub-seed stream.
///
/// `world_seed.child(TILE).sub_at(...)` over the tile's `(tile_x, tile_y)` —
/// the hierarchical sub-seed path the whole crate uses for isolation
/// (`hash.rs` module doc: "any node regenerates in isolation from its
/// coordinates alone"). This seed is for anything keyed by the TILE's
/// *identity* (e.g. per-tile scatter/props in later waves); [`height`]
/// deliberately does NOT use it — see [`height`]'s doc for why.
pub fn tile_seed(world_seed: Seed, tile: TerrainTile) -> Seed {
    world_seed
        .child(domain::TILE)
        .sub_at(coord_key_i64(tile.tile_x), coord_key_i64(tile.tile_y))
}

/// The height field: world meters in, world meters up.
///
/// CRITICAL PROPERTY: depends on `(seed, params, world_x, world_z)` ONLY —
/// never on a tile key. Two different tiles sampling the same world-space
/// point therefore get byte-identical results, which is what makes shared
/// tile edges seam-free by construction rather than by post-hoc stitching.
pub fn height(world_seed: Seed, params: &TerrainParams, world_x: f32, world_z: f32) -> f32 {
    // Noise::new folds in domain::FIELD itself; seeded directly off the
    // world seed (not a tile sub-seed) so this stays tile-independent.
    let noise = Noise::new(world_seed.0);
    let (x, z) = if params.warp_enabled {
        noise.domain_warp2(world_x, world_z, params.warp_strength)
    } else {
        (world_x, world_z)
    };
    let field = params.fbm.sample2(|a, b| noise.value2(a, b), x, z);
    field * params.height_amplitude
}

/// A tile's world-space origin (its `(i=0, j=0)` grid corner), in meters.
///
/// Computed via the SAME `global_index * cell_size` arithmetic path as every
/// grid vertex ([`grid_vertex_world_position`]) — see that function's doc for
/// why the shared arithmetic path, not the origin value itself, is what
/// makes seams exact.
pub fn tile_origin_m(tile: TerrainTile, params: &TerrainParams) -> (f64, f64) {
    let n = params.grid_resolution as i64;
    let cell_size = params.cell_size_m() as f64;
    (
        (tile.tile_x * n) as f64 * cell_size,
        (tile.tile_y * n) as f64 * cell_size,
    )
}

/// World-space position (meters, `f32`) of grid vertex `(i, j)` within
/// `tile`, for `i, j` in `0..=grid_resolution`.
///
/// SEAM-EXACTNESS CRUX: this is computed from a single `i64` GLOBAL grid
/// index (`tile.tile_x * grid_resolution + i`), not from a per-tile local
/// offset added to a separately-rounded origin. That means tile `T`'s vertex
/// `i = grid_resolution` and tile `T+1`'s vertex `i = 0` reduce to the exact
/// same global index via ordinary `i64` integer arithmetic (exact, no
/// rounding), then run through the identical `(global_index as f64) *
/// cell_size as f64) as f32` cast — so they produce the bit-identical `f32`
/// regardless of `tile_size_m` / `grid_resolution`'s divisibility. Computing
/// each tile's origin independently and adding a local offset would instead
/// sum two different rounding errors on each side of the seam.
pub fn grid_vertex_world_position(tile: TerrainTile, params: &TerrainParams, i: u32, j: u32) -> (f32, f32) {
    let n = params.grid_resolution as i64;
    let cell_size = params.cell_size_m() as f64;
    let global_i = tile.tile_x * n + i as i64;
    let global_j = tile.tile_y * n + j as i64;
    (
        (global_i as f64 * cell_size) as f32,
        (global_j as f64 * cell_size) as f32,
    )
}

/// Analytic-ish normal at a world point via central differences.
///
/// Step size = `cell_size_m` (the grid's own sampling spacing), not an
/// independently plucked epsilon: the mesh can only ever display slope
/// resolved at grid resolution, so a smaller step would measure sub-grid
/// detail the triangles can't show, and a larger one would blur across
/// neighboring cells. `cell_size_m` is therefore the natural — and derived —
/// choice.
fn normal_at(world_seed: Seed, params: &TerrainParams, world_x: f32, world_z: f32, step: f32) -> [f32; 3] {
    let h_neg_x = height(world_seed, params, world_x - step, world_z);
    let h_pos_x = height(world_seed, params, world_x + step, world_z);
    let h_neg_z = height(world_seed, params, world_x, world_z - step);
    let h_pos_z = height(world_seed, params, world_x, world_z + step);
    let slope_x = (h_pos_x - h_neg_x) / (2.0 * step);
    let slope_z = (h_pos_z - h_neg_z) / (2.0 * step);
    // Height field is y = h(x, z); surface normal = normalize(-dh/dx, 1, -dh/dz).
    let n = Vec3::new(-slope_x, 1.0, -slope_z).normalize();
    [n.x, n.y, n.z]
}

/// Build one tile's mesh: `(grid_resolution+1)^2` vertices, two triangles per
/// cell, winding/index-emission convention mirrored from transmute's
/// `add_grid_face` (row-major, `[a, c, b, b, c, d]` per quad).
///
/// Vertex positions are TILE-LOCAL `f32` (world position minus the tile's
/// own origin, Ruling 4) so precision stays anchored near the tile rather
/// than the world origin. `Mesh`/`Vertex` come straight from `transmutation`
/// (see `packages/seed/Cargo.toml` for why `seed -> transmutation` is a
/// legal forward edge, not a cycle).
pub fn tile_mesh(world_seed: Seed, tile: TerrainTile, params: &TerrainParams) -> Mesh {
    let n = params.grid_resolution;
    let row = n + 1;
    let step = params.cell_size_m();
    let (origin_x, origin_z) = tile_origin_m(tile, params);

    let mut vertices = Vec::with_capacity((row * row) as usize);
    for j in 0..=n {
        for i in 0..=n {
            let (world_x, world_z) = grid_vertex_world_position(tile, params, i, j);
            let h = height(world_seed, params, world_x, world_z);
            let local_position = [world_x - origin_x as f32, h, world_z - origin_z as f32];
            let normal = normal_at(world_seed, params, world_x, world_z, step);
            let uv = [i as f32 / n as f32, j as f32 / n as f32];
            vertices.push(Vertex::new(local_position, normal, uv));
        }
    }

    let mut indices = Vec::with_capacity((n * n * 6) as usize);
    for j in 0..n {
        for i in 0..n {
            let a = j * row + i;
            let b = a + 1;
            let c = (j + 1) * row + i;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    Mesh::new(vertices, indices)
}
