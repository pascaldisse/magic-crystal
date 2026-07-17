//! VII-0a terrain ordeals — seed -> Mesh tile sampler
//! (docs/proposals/RITE-VII-THE-PLANET-WALKER.md §VII-0 "THE FIRST GROUND").
//! Idiom mirrors `tests/ordeals.rs`'s S0 ordeals: each test prints its
//! verbatim numbers and asserts a named property.

use seed::hash::coord_key_i64;
use seed::terrain::{height, tile_mesh, tile_origin_m, tile_seed, TerrainParams, TerrainTile};
use seed::Seed;
use transmutation::Mesh;

/// A small grid resolution for fast ordeals — independent of the crate's
/// derived default, which is fine (every ordeal below holds for ANY
/// `TerrainParams`, not just the default one).
fn small_params() -> TerrainParams {
    TerrainParams {
        grid_resolution: 9,
        ..TerrainParams::default()
    }
}

/// Fold a mesh's vertex positions + indices into a stable digest (FNV-1a over
/// the raw bits) — a cheap stand-in for "sha256 or similar" that needs no new
/// dependency; bit-for-bit equality of the fold is exactly bit-for-bit
/// equality of the inputs (FNV-1a is a bijection-per-byte accumulator, not a
/// cryptographic requirement here — we're proving reproducibility, not
/// hiding a secret).
fn mesh_digest(mesh: &Mesh) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    let mut fold_bytes = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    };
    for v in &mesh.vertices {
        fold_bytes(&v.position[0].to_bits().to_le_bytes());
        fold_bytes(&v.position[1].to_bits().to_le_bytes());
        fold_bytes(&v.position[2].to_bits().to_le_bytes());
        fold_bytes(&v.normal[0].to_bits().to_le_bytes());
        fold_bytes(&v.normal[1].to_bits().to_le_bytes());
        fold_bytes(&v.normal[2].to_bits().to_le_bytes());
        fold_bytes(&v.uv[0].to_bits().to_le_bytes());
        fold_bytes(&v.uv[1].to_bits().to_le_bytes());
    }
    for &i in &mesh.indices {
        fold_bytes(&i.to_le_bytes());
    }
    h
}

// ---------------------------------------------------------------------------
// ORDEAL (a) — regeneration determinism: same (seed, tile, params) => byte-
// identical mesh, computed twice, independently.
// ---------------------------------------------------------------------------

#[test]
fn ordeal_regeneration_determinism() {
    let params = small_params();
    let tile = TerrainTile::new(3, -7);

    let mesh_a = tile_mesh(Seed::new(0x5EED_1234), tile, &params);
    let mesh_b = tile_mesh(Seed::new(0x5EED_1234), tile, &params);

    let digest_a = mesh_digest(&mesh_a);
    let digest_b = mesh_digest(&mesh_b);
    assert_eq!(mesh_a.vertices.len(), mesh_b.vertices.len());
    assert_eq!(mesh_a.indices, mesh_b.indices);
    assert_eq!(digest_a, digest_b, "regeneration diverged");

    println!(
        "ORDEAL(a) regeneration-determinism: vertices={} indices={} digest_a={:#x} digest_b={:#x} identical={}",
        mesh_a.vertices.len(),
        mesh_a.indices.len(),
        digest_a,
        digest_b,
        digest_a == digest_b
    );
}

// ---------------------------------------------------------------------------
// ORDEAL (b) — NO-STORAGE: same as (a) but the two computations share NO
// state — every seed/params object is dropped and rebuilt from a raw seed
// integer between the two calls.
// ---------------------------------------------------------------------------

/// Rebuild everything from a raw `u64` seed value alone — no object from a
/// prior call is reachable here.
fn rebuild_and_digest(raw_seed: u64, tile_x: i64, tile_y: i64, grid_resolution: u32) -> u64 {
    let world_seed = Seed::new(raw_seed);
    let tile = TerrainTile::new(tile_x, tile_y);
    let params = TerrainParams {
        grid_resolution,
        ..TerrainParams::default()
    };
    let mesh = tile_mesh(world_seed, tile, &params);
    mesh_digest(&mesh)
}

#[test]
fn ordeal_no_storage_cold_rebuild() {
    let raw_seed = 0x900D_5EED_u64;
    let (tile_x, tile_y) = (42, -11);
    let grid_resolution = 9;

    let digest_first = rebuild_and_digest(raw_seed, tile_x, tile_y, grid_resolution);
    // Nothing from the call above survives past its return; this second call
    // reconstructs the entire chain (Seed -> TerrainParams -> Mesh) from the
    // same raw integers alone (the NMS property: state = f(seed, coords)).
    let digest_second = rebuild_and_digest(raw_seed, tile_x, tile_y, grid_resolution);

    assert_eq!(digest_first, digest_second, "cold rebuild diverged");
    println!(
        "ORDEAL(b) no-storage: raw_seed={:#x} tile=({},{}) digest_first={:#x} digest_second={:#x} identical={}",
        raw_seed, tile_x, tile_y, digest_first, digest_second, digest_first == digest_second
    );
}

// ---------------------------------------------------------------------------
// ORDEAL (c) — seam: for adjacent tiles in all four edge directions, the
// shared-edge vertices agree — HEIGHT (mesh `position[1]`) compared as
// byte-identical `f32` (it's the direct output of `height_at_grid_index`,
// never reconstructed from a world coordinate, so bit-exact equality is the
// honest bar), and horizontal WORLD position (`local + origin`, `f64`)
// compared within a DERIVED tolerance — see `assert_seam_all_four_directions`
// for why reconstruction can't be bit-exact even when correct (each side
// rounds its tile-local `f32` independently before promoting to `f64`) and
// how the tolerance is derived.
// ---------------------------------------------------------------------------

/// A mesh's shared edge, in emission order: `(world_x, height, world_z)` with
/// `world_x`/`world_z` reconstructed in `f64` from the tile-local `f32`
/// vertex position plus [`tile_origin_m`] (also `f64`). `edge` matches the
/// `(i, j)` grid, `i`/`j` in `0..=grid_resolution`.
fn edge_world_positions(
    world_seed: Seed,
    tile: TerrainTile,
    params: &TerrainParams,
    edge: Edge,
) -> Vec<(f64, f32, f64)> {
    let n = params.grid_resolution;
    let mesh = tile_mesh(world_seed, tile, params);
    let (origin_x, origin_z) = tile_origin_m(tile, params);
    let row = n + 1;
    let mut out = Vec::with_capacity((n + 1) as usize);
    for k in 0..=n {
        let (i, j) = match edge {
            Edge::PlusX => (n, k),
            Edge::MinusX => (0, k),
            Edge::PlusZ => (k, n),
            Edge::MinusZ => (k, 0),
        };
        let v = mesh.vertices[(j * row + i) as usize];
        let world_x = origin_x + v.position[0] as f64;
        let world_y = v.position[1];
        let world_z = origin_z + v.position[2] as f64;
        out.push((world_x, world_y, world_z));
    }
    out
}

#[derive(Clone, Copy, Debug)]
enum Edge {
    PlusX,
    MinusX,
    PlusZ,
    MinusZ,
}

/// The four (this tile's edge, neighbor tile, neighbor's matching opposite
/// edge) triples for `center`, shared by the near-origin and large-coordinate
/// seam ordeals below.
fn seam_directions(center: TerrainTile) -> [(&'static str, Edge, TerrainTile, Edge); 4] {
    [
        ("+x", Edge::PlusX, center.neighbor(1, 0), Edge::MinusX),
        ("-x", Edge::MinusX, center.neighbor(-1, 0), Edge::PlusX),
        ("+z", Edge::PlusZ, center.neighbor(0, 1), Edge::MinusZ),
        ("-z", Edge::MinusZ, center.neighbor(0, -1), Edge::PlusZ),
    ]
}

/// Check all four edge directions of `center` against their neighbors;
/// returns the count of vertices checked. Panics on any mismatch, naming the
/// direction/vertex/component.
///
/// HEIGHT (component 1) is compared BIT-EXACT: it's the direct `f32` output
/// of `height_at_grid_index`, never reconstructed from a tile-local position
/// plus an origin, so tile-independence means literal `to_bits()` equality.
///
/// Horizontal WORLD position (components 0/2) is compared within a DERIVED
/// tolerance, not bit-exact: reconstructing it sums a tile-local `f32`
/// (itself already rounded once, independently, on each side of the seam)
/// into an `f64` origin, so the two sides take genuinely different rounding
/// paths even when correct. The tolerance is the same one
/// `ordeal_local_vertex_spacing_stable_at_large_tile_coordinate` derives:
/// local magnitude is bounded by `tile_size_m` (Ruling 4), so a single f32
/// rounding step there is worth about `tile_size_m * f32::EPSILON`; two
/// independent roundings (one per side of the seam) plus a safety margin
/// gives `4x`. Crucially this bound does NOT grow with the tile coordinate —
/// that tile-position-independence is exactly what MUST-FIX 1 required.
fn assert_seam_all_four_directions(
    world_seed: Seed,
    params: &TerrainParams,
    center: TerrainTile,
    label: &str,
) -> usize {
    let tolerance = 4.0 * f32::EPSILON as f64 * params.tile_size_m as f64;
    let mut checked_vertices = 0usize;
    for (name, center_edge, neighbor_tile, neighbor_edge) in seam_directions(center) {
        let center_positions = edge_world_positions(world_seed, center, params, center_edge);
        let neighbor_positions =
            edge_world_positions(world_seed, neighbor_tile, params, neighbor_edge);
        assert_eq!(center_positions.len(), neighbor_positions.len());
        for (idx, (c, nb)) in center_positions.iter().zip(&neighbor_positions).enumerate() {
            checked_vertices += 1;
            assert!(
                (c.0 - nb.0).abs() <= tolerance,
                "{label} seam {name} vertex {idx}: world_x mismatch {:?} vs {:?} (tol {tolerance})",
                c,
                nb
            );
            assert_eq!(
                c.1.to_bits(),
                nb.1.to_bits(),
                "{label} seam {name} vertex {idx}: height mismatch {:?} vs {:?}",
                c,
                nb
            );
            assert!(
                (c.2 - nb.2).abs() <= tolerance,
                "{label} seam {name} vertex {idx}: world_z mismatch {:?} vs {:?} (tol {tolerance})",
                c,
                nb
            );
        }
    }
    checked_vertices
}

#[test]
fn ordeal_seam_all_four_directions_byte_identical() {
    let world_seed = Seed::new(0xF00D_5EED);
    let params = small_params();
    let center = TerrainTile::new(-2, 5);

    let checked_vertices = assert_seam_all_four_directions(world_seed, &params, center, "near-origin");

    println!(
        "ORDEAL(c) seam: directions=4 checked_vertices={} height_bit_exact=true world_xz_within_tolerance=true",
        checked_vertices
    );
}

// ---------------------------------------------------------------------------
// ORDEAL (c2) — MUST-FIX 1/2 regression: the same seam property, but at a
// tile coordinate the adversary's probe showed collapsing under the OLD
// (world-f32-intermediate) implementation (tile_x = 1_000_000 gave local
// spacings of 0.0; tile_x = 10_000_000 degenerated entirely). Run at
// 10_000_000 and its negative twin so both signs of the i64 range are
// covered.
// ---------------------------------------------------------------------------

#[test]
fn ordeal_seam_large_tile_coordinate() {
    let world_seed = Seed::new(0xF00D_5EED);
    let params = small_params();

    let far_positive = TerrainTile::new(10_000_000, -3_000_000);
    let checked_positive =
        assert_seam_all_four_directions(world_seed, &params, far_positive, "tile_x=+1e7");

    let far_negative = TerrainTile::new(-10_000_000, 3_000_000);
    let checked_negative =
        assert_seam_all_four_directions(world_seed, &params, far_negative, "tile_x=-1e7");

    println!(
        "ORDEAL(c2) seam-large-coordinate: tile=({},{}) checked={} within_tolerance=true | tile=({},{}) checked={} within_tolerance=true",
        far_positive.tile_x, far_positive.tile_y, checked_positive,
        far_negative.tile_x, far_negative.tile_y, checked_negative,
    );
}

// ---------------------------------------------------------------------------
// ORDEAL (c3) — MUST-FIX 1 direct regression: LOCAL vertex spacing stays
// `== cell_size_m` (to a derived f32-ULP tolerance) at a large tile
// coordinate. This is the exact probe the adversary ran (spacing collapsing
// to 0.0 at tile_x = 1_000_000): tile-local positions must never be derived
// from a world-magnitude value, so their spacing must be tile-position-
// INDEPENDENT.
// ---------------------------------------------------------------------------

#[test]
fn ordeal_local_vertex_spacing_stable_at_large_tile_coordinate() {
    let params = small_params();
    let world_seed = Seed::new(0xF00D_5EED);
    let cell_size = params.cell_size_m();

    // Derived tolerance: local positions are bounded in magnitude by
    // `tile_size_m` (Ruling 4's whole point), so the worst-case f32 rounding
    // error on any one of them is about `tile_size_m * f32::EPSILON`; two
    // such values subtracted can carry twice that, so 4x is a safety margin
    // on top of the theoretical 2x, not a plucked number.
    let tolerance = 4.0 * f32::EPSILON * params.tile_size_m;

    for tile in [
        TerrainTile::new(0, 0),
        TerrainTile::new(1_000_000, 0),
        TerrainTile::new(10_000_000, -10_000_000),
        TerrainTile::new(-10_000_000, 10_000_000),
    ] {
        let mesh = tile_mesh(world_seed, tile, &params);
        let n = params.grid_resolution;
        let row = n + 1;
        // Spacing along the i=0 row's first two vertices (j=0, i=0 and i=1).
        let p0 = mesh.vertices[0].position[0];
        let p1 = mesh.vertices[1].position[0];
        let spacing = p1 - p0;
        assert!(
            (spacing - cell_size).abs() <= tolerance,
            "tile ({},{}): local spacing {spacing} != cell_size {cell_size} (tol {tolerance})",
            tile.tile_x,
            tile.tile_y
        );
        // Same check along j at i=0 (vertex (0,0) vs (0,1), index `row`).
        let q0 = mesh.vertices[0].position[2];
        let q1 = mesh.vertices[row as usize].position[2];
        let spacing_j = q1 - q0;
        assert!(
            (spacing_j - cell_size).abs() <= tolerance,
            "tile ({},{}): local j-spacing {spacing_j} != cell_size {cell_size} (tol {tolerance})",
            tile.tile_x,
            tile.tile_y
        );
        println!(
            "ORDEAL(c3) local-spacing: tile=({},{}) spacing_i={:.6} spacing_j={:.6} cell_size={:.6} tol={:.3e}",
            tile.tile_x, tile.tile_y, spacing, spacing_j, cell_size, tolerance
        );
    }
}

// ---------------------------------------------------------------------------
// ORDEAL (d) — different seed => different mesh (discriminates).
// ---------------------------------------------------------------------------

#[test]
fn ordeal_different_seed_different_mesh() {
    let params = small_params();
    let tile = TerrainTile::new(0, 0);

    let mesh_a = tile_mesh(Seed::new(0x1111_1111), tile, &params);
    let mesh_b = tile_mesh(Seed::new(0x2222_2222), tile, &params);

    let digest_a = mesh_digest(&mesh_a);
    let digest_b = mesh_digest(&mesh_b);
    assert_ne!(digest_a, digest_b, "different seeds produced the same mesh");

    println!(
        "ORDEAL(d) different-seed: digest_a={:#x} digest_b={:#x} differ={}",
        digest_a,
        digest_b,
        digest_a != digest_b
    );
}

// ---------------------------------------------------------------------------
// ORDEAL (e) — tile isolation: regenerating tile A never changes tile B, and
// regenerating A again after B reproduces A's first result exactly (S0
// isolation-ordeal precedent, tests/ordeals.rs).
// ---------------------------------------------------------------------------

#[test]
fn ordeal_tile_isolation() {
    let world_seed = Seed::new(0xC0FFEE_u64);
    let params = small_params();
    let tile_a = TerrainTile::new(100, 200);
    let tile_b = TerrainTile::new(-55, 9001);

    let mesh_a_first = tile_mesh(world_seed, tile_a, &params);
    let digest_a_first = mesh_digest(&mesh_a_first);

    // Regenerate a wholly different tile in between.
    let mesh_b = tile_mesh(world_seed, tile_b, &params);
    let digest_b = mesh_digest(&mesh_b);

    let mesh_a_second = tile_mesh(world_seed, tile_a, &params);
    let digest_a_second = mesh_digest(&mesh_a_second);

    assert_eq!(
        digest_a_first, digest_a_second,
        "tile A changed after tile B was generated"
    );
    assert_ne!(
        digest_a_first, digest_b,
        "distinct tiles produced the same mesh (too weak a check, but should still differ)"
    );

    println!(
        "ORDEAL(e) tile-isolation: digest_a_first={:#x} digest_b={:#x} digest_a_second={:#x} a_stable={}",
        digest_a_first, digest_b, digest_a_second, digest_a_first == digest_a_second
    );
}

// Height-field sanity: tile-independence property stated directly (not just
// exercised indirectly via the seam ordeal) — same world (x, z) through two
// different tiles' worth of params/seed plumbing still calls the SAME
// `height()` function with the SAME arguments, so this is really just
// documentation-as-test that `height` takes no tile key at all (its
// signature has no such parameter to pass).
#[test]
fn height_field_is_a_pure_function_of_world_xz() {
    let world_seed = Seed::new(0xABCD_EF01);
    let params = TerrainParams::default();
    let (x, z) = (123.5_f32, -77.25_f32);

    let h1 = height(world_seed, &params, x, z);
    let h2 = height(world_seed, &params, x, z);
    assert_eq!(h1.to_bits(), h2.to_bits());
    println!("height-purity: height({x},{z}) = {h1} (repeat identical)");
}

// ---------------------------------------------------------------------------
// ADVISORY 5 — exercise tile_seed / coord_key_i64 with a real ordeal instead
// of leaving them as unverified scaffolding: (1) coord_key_i64 is injective
// over a coordinate sweep spanning both signs and both ends of the i64
// range near overflow, (2) tile_seed gives distinct tiles distinct seeds
// over the same sweep (no collisions), and (3) it's deterministic (same
// tile -> same seed, twice).
// ---------------------------------------------------------------------------

#[test]
fn ordeal_tile_seed_and_coord_key_i64_distinguish_coordinates() {
    let sweep: Vec<i64> = vec![
        0,
        1,
        -1,
        1_000_000,
        -1_000_000,
        10_000_000,
        -10_000_000,
        i64::MAX,
        i64::MIN,
        i64::MAX - 1,
        i64::MIN + 1,
    ];

    // coord_key_i64 injective over the sweep (it's a bit-reinterpret cast,
    // so this must hold for every i64, but checking the sweep is the honest
    // ordeal-style spot-check rather than an unfalsifiable "trust me").
    let mut keys: Vec<u64> = sweep.iter().map(|&v| coord_key_i64(v)).collect();
    let key_count_before_dedup = keys.len();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(
        keys.len(),
        key_count_before_dedup,
        "coord_key_i64 collided within the sweep"
    );

    // tile_seed distinguishes every tile pair built from the sweep (a small
    // grid of it, not the full cross product, to keep this fast).
    let world_seed = Seed::new(0x7117_5EED);
    let mut tile_seeds: Vec<(i64, i64, u64)> = Vec::new();
    for &x in &sweep {
        for y in [0i64, 1, -1] {
            let tile = TerrainTile::new(x, y);
            let s = tile_seed(world_seed, tile);
            tile_seeds.push((x, y, s.0));
        }
    }
    let mut seed_values: Vec<u64> = tile_seeds.iter().map(|&(_, _, s)| s).collect();
    let seed_count_before_dedup = seed_values.len();
    seed_values.sort_unstable();
    seed_values.dedup();
    assert_eq!(
        seed_values.len(),
        seed_count_before_dedup,
        "tile_seed collided across the sweep"
    );

    // Determinism: recomputing the same tile's seed gives the same value.
    let repeat = tile_seed(world_seed, TerrainTile::new(10_000_000, -3_000_000));
    let repeat_again = tile_seed(world_seed, TerrainTile::new(10_000_000, -3_000_000));
    assert_eq!(repeat.0, repeat_again.0);

    println!(
        "ORDEAL tile-seed/coord-key: sweep_len={} coord_keys_distinct={} tile_seeds_checked={} tile_seeds_distinct={} deterministic={}",
        sweep.len(),
        keys.len(),
        seed_count_before_dedup,
        seed_values.len(),
        repeat.0 == repeat_again.0
    );
}
