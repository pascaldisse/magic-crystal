//! Terrain tiles — seed → Mesh, the first link of RITE VII's chain (VII-0
//! "THE FIRST GROUND", docs/proposals/RITE-VII-THE-PLANET-WALKER.md).
//!
//! A tile is a `(grid_resolution+1)^2` grid. Two properties make shared tile
//! edges seam-free BY CONSTRUCTION (never by post-hoc stitching):
//!
//! 1. Every grid vertex's height comes from [`height_at_grid_index`], keyed
//!    on an EXACT `i64` global grid index (`tile.tile_x * grid_resolution +
//!    i`) — never a tile key, and (in the default, non-warped path) never a
//!    float world coordinate either. Two tiles sampling the same global
//!    index therefore agree byte-for-byte, at ANY tile-coordinate magnitude.
//! 2. Every vertex POSITION stored in the mesh is tile-local `f32` — derived
//!    from the LOCAL index alone (`i as f64 * cell_size`, magnitude bounded
//!    by `tile_size_m`), never from a world-magnitude value cast down to
//!    `f32`. Ruling 4 ("64-bit/camera-relative coords: PAID AT VII-0") is
//!    explicit that vertex positions must be tile-local `f32`, not
//!    world-absolute `f32` — an absolute world coordinate stops being exact
//!    in `f32` past `2^24` meters (~16,777km; ordinary tile ranges blow past
//!    this in the tens of thousands of tiles), so baking it into the mesh
//!    would silently collapse vertices at exactly the distances 64-bit
//!    coordinates exist to survive. The tile's ORIGIN is carried separately,
//!    in `f64`/`i64` ([`tile_origin_m`]), for a consumer to place the tile
//!    in the world without ever rounding that placement through `f32`.
//!
//! Flat ground here is deliberately the flat-frame case; Ruling 3 (NARUKO.md
//! "GUARDIAN RULINGS UNDER DELEGATION") makes flat ground the infinite-radius
//! limit of a radial `up(r)` — this atom (VII-0a) only pays for the flat
//! limit, the sphere-domain field is later rite scope (RITE-VII §OPEN 4).
//!
//! KNOWN LIMIT (documented, not silently wrong): [`TerrainParams::warp_enabled`]
//! routes through [`height`], which takes an `f32` world coordinate and so
//! inherits ordinary `f32` precision limits at extreme tile magnitudes. Warp
//! defaults OFF; the exact-integer path ([`height_at_grid_index`]) is what
//! `tile_mesh` uses whenever warp is off, which is the planet-scale-exact
//! case this atom is required to prove.

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

/// Nyquist floor: the minimum number of GRID VERTICES that must fall across
/// one world-space WAVELENGTH for a sinusoid of that wavelength to be
/// representable at all without aliasing (the sampling theorem's `2`, not a
/// stylistic oversampling choice) — i.e. the maximum allowed vertex spacing
/// (`cell_size_m`) is `wavelength_m / NYQUIST_SAMPLES_PER_WAVELENGTH`.
/// `grid_resolution`'s derived default applies exactly this floor against
/// the fBm's finest (shortest-wavelength) octave, so the mesh is the
/// coarsest grid that can still legally represent the terrain field it was
/// given — raising `octaves` or `lacunarity` shortens the finest wavelength
/// and the derived default grid tightens (more vertices) with it.
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
    /// scaled by this). Default `tile_size_m * DEFAULT_SLOPE_FRACTION` — a
    /// RELIEF-SCALE parameter, sized relative to the tile so the terrain
    /// reads as visible relief in the VII-0 proof shot without dwarfing the
    /// tile. NOT a walkable-grade guarantee: a multi-octave fBm sum can
    /// locally exceed the single-octave slope estimate its ratio to
    /// `base_wavelength_m` suggests, so no walkable-floor bound (Ruling 6)
    /// is claimed here — that's VII-1's collider-fit ordeal to derive and
    /// enforce, not this atom's.
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
/// default (a relief-scale ratio, not a slope guarantee — see that field's
/// doc). `0.15` keeps the coarsest octave's amplitude modest relative to its
/// own wavelength (`base_wavelength_m = tile_size_m / 2`) while still
/// producing visible relief in the VII-0 proof shot.
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
        let finest_wavelength_m = base_wavelength_m / fbm.lacunarity.powi(fbm.octaves as i32 - 1);
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
///
/// PRECISION NOTE: `world_x`/`world_z` are `f32`, so this function inherits
/// ordinary `f32` precision limits at extreme magnitudes (exact only up to
/// `2^24` meters) — fine for direct/general-purpose queries near a
/// reasonable origin, and it's what the `warp_enabled` path in
/// [`tile_mesh`] still uses (warp isn't yet paid for at planetary offsets,
/// see the module doc's KNOWN LIMIT). Mesh generation's default (non-warped)
/// path uses [`height_at_grid_index`] instead, which is exact at any tile
/// magnitude.
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

/// One fBm octave's wavelength, expressed as an integer count of GRID CELLS
/// (rounded to the nearest cell, minimum `1`). This is the bridge between
/// `TerrainParams`' continuous, world-meter octave definition and the exact
/// integer-lattice sampling [`height_at_grid_index`] needs: the rounding
/// happens once, on a small params-only quantity that never scales with a
/// tile coordinate, so it costs no precision at any tile magnitude (unlike
/// rounding something derived from a huge global index would).
fn octave_wavelength_in_grid_cells(params: &TerrainParams, octave_index: u32) -> i64 {
    let wavelength_m = params.base_wavelength_m / params.fbm.lacunarity.powi(octave_index as i32);
    let cells = (wavelength_m / params.cell_size_m()).round().max(1.0);
    cells as i64
}

/// The height field sampled at an EXACT global grid index — never through a
/// float world coordinate. This is what [`tile_mesh`] uses for every vertex
/// (when warp is off, the default), and it's the fix for the precision trap
/// `height()` alone can't avoid at planetary tile magnitudes:
///
/// For each octave, `global_i`/`global_j` (exact `i64`, e.g.
/// `tile.tile_x * grid_resolution + i`) are split into a lattice cell and an
/// in-cell fraction by plain `i64` Euclidean division against that octave's
/// `octave_wavelength_in_grid_cells` — `div_euclid`/`rem_euclid` are exact
/// for any `i64` operand, so there is no large-magnitude float
/// multiplication anywhere in the split (the earlier, fixed bug: casting
/// `global_index as f64 * cell_size` to `f32` loses precision once the
/// product's magnitude passes `f32`'s ~24-bit mantissa, which is exactly
/// what collapsed vertices at `tile_x = 1_000_000`). The remainder is always
/// bounded by the (small, params-only) wavelength-in-cells, so converting it
/// to a `[0,1)` fraction is a division of two small numbers — full `f32`
/// precision, regardless of how large `global_i`/`global_j` are.
pub fn height_at_grid_index(
    world_seed: Seed,
    params: &TerrainParams,
    global_i: i64,
    global_j: i64,
) -> f32 {
    let noise = Noise::new(world_seed.0);
    let mut amplitude = 1.0f32;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for octave_index in 0..params.fbm.octaves {
        let wavelength_cells = octave_wavelength_in_grid_cells(params, octave_index);
        let cell_x = global_i.div_euclid(wavelength_cells);
        let cell_y = global_j.div_euclid(wavelength_cells);
        let frac_x = global_i.rem_euclid(wavelength_cells) as f32 / wavelength_cells as f32;
        let frac_y = global_j.rem_euclid(wavelength_cells) as f32 / wavelength_cells as f32;
        sum += amplitude * noise.value2_at_lattice(cell_x, cell_y, frac_x, frac_y);
        norm += amplitude;
        amplitude *= params.fbm.gain;
    }
    let field = if norm > 0.0 { sum / norm } else { 0.0 };
    field * params.height_amplitude
}

/// Height at a global grid index, dispatching on [`TerrainParams::warp_enabled`]:
/// the exact-integer [`height_at_grid_index`] path when warp is off (the
/// default, and the case `tile_mesh` must be exact for at any tile
/// magnitude); [`height`] via an `f32` world coordinate when warp is on (see
/// the module doc's KNOWN LIMIT).
fn height_at_global(world_seed: Seed, params: &TerrainParams, global_i: i64, global_j: i64) -> f32 {
    if params.warp_enabled {
        let cell_size = params.cell_size_m() as f64;
        let world_x = (global_i as f64 * cell_size) as f32;
        let world_z = (global_j as f64 * cell_size) as f32;
        height(world_seed, params, world_x, world_z)
    } else {
        height_at_grid_index(world_seed, params, global_i, global_j)
    }
}

/// A tile's world-space origin (its `(i=0, j=0)` grid corner), in meters,
/// kept in `f64`/`i64` (never truncated to `f32`) for a consumer to place
/// the tile without baking a large-magnitude value into a small type. See
/// the module doc for why this must stay separate from the mesh's
/// tile-local `f32` vertex positions rather than being added into them.
pub fn tile_origin_m(tile: TerrainTile, params: &TerrainParams) -> (f64, f64) {
    let n = params.grid_resolution as i64;
    let cell_size = params.cell_size_m() as f64;
    (
        (tile.tile_x * n) as f64 * cell_size,
        (tile.tile_y * n) as f64 * cell_size,
    )
}

/// Analytic-ish normal at a global grid index via central differences,
/// stepping by ONE ADJACENT GRID INDEX (`global ± 1`) rather than an `f32`
/// epsilon in meters — exact at any tile magnitude for the same reason
/// [`height_at_grid_index`] is, and it's the natural step besides: the mesh
/// can only ever display slope resolved at grid resolution, so sampling
/// anywhere other than the adjacent grid vertex would measure detail the
/// triangles can't show (too fine) or blur across further cells (too
/// coarse). `step_m = cell_size_m` converts the resulting per-grid-step
/// slope to per-meter; `cell_size_m` doesn't scale with the tile coordinate,
/// so this multiplication is exact-enough regardless of tile magnitude.
fn normal_at_global(
    world_seed: Seed,
    params: &TerrainParams,
    global_i: i64,
    global_j: i64,
) -> [f32; 3] {
    let h_neg_x = height_at_global(world_seed, params, global_i - 1, global_j);
    let h_pos_x = height_at_global(world_seed, params, global_i + 1, global_j);
    let h_neg_z = height_at_global(world_seed, params, global_i, global_j - 1);
    let h_pos_z = height_at_global(world_seed, params, global_i, global_j + 1);
    let step_m = params.cell_size_m();
    let slope_x = (h_pos_x - h_neg_x) / (2.0 * step_m);
    let slope_z = (h_pos_z - h_neg_z) / (2.0 * step_m);
    // Height field is y = h(x, z); surface normal = normalize(-dh/dx, 1, -dh/dz).
    let n = Vec3::new(-slope_x, 1.0, -slope_z).normalize();
    [n.x, n.y, n.z]
}

/// Build one tile's mesh: `(grid_resolution+1)^2` vertices, two triangles per
/// cell, winding/index-emission convention mirrored from transmute's
/// `add_grid_face` (row-major, `[a, c, b, b, c, d]` per quad).
///
/// Vertex positions are TILE-LOCAL `f32`, derived from the LOCAL index alone
/// (`i as f64 * cell_size`, magnitude bounded by `tile_size_m` — see the
/// module doc for why this, and not a world-position subtraction, is what
/// keeps geometry exact at any tile coordinate). `Mesh`/`Vertex` come
/// straight from `transmutation` (see `packages/seed/Cargo.toml` for why
/// `seed -> transmutation` is a legal forward edge, not a cycle).
pub fn tile_mesh(world_seed: Seed, tile: TerrainTile, params: &TerrainParams) -> Mesh {
    let n = params.grid_resolution;
    let n_i64 = n as i64;
    let row = n + 1;
    let cell_size = params.cell_size_m() as f64;

    let mut vertices = Vec::with_capacity((row * row) as usize);
    for j in 0..=n {
        for i in 0..=n {
            let global_i = tile.tile_x * n_i64 + i as i64;
            let global_j = tile.tile_y * n_i64 + j as i64;

            let h = height_at_global(world_seed, params, global_i, global_j);
            // LOCAL index only — never the (potentially huge) global index —
            // so this cast to f32 is always exact (magnitude <= tile_size_m).
            let local_x = (i as f64 * cell_size) as f32;
            let local_z = (j as f64 * cell_size) as f32;
            let normal = normal_at_global(world_seed, params, global_i, global_j);
            let uv = [i as f32 / n as f32, j as f32 / n as f32];
            vertices.push(Vertex::new([local_x, h, local_z], normal, uv));
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
