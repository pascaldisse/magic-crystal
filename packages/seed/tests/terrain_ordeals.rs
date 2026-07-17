//! VII-0a terrain ordeals — seed -> Mesh tile sampler
//! (docs/proposals/RITE-VII-THE-PLANET-WALKER.md §VII-0 "THE FIRST GROUND").
//! Idiom mirrors `tests/ordeals.rs`'s S0 ordeals: each test prints its
//! verbatim numbers and asserts a named property.

use seed::terrain::{height, tile_mesh, tile_origin_m, TerrainParams, TerrainTile};
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
// shared-edge vertices' WORLD positions (tile-local + reconstructed origin)
// are byte-identical f32 values, vertex by vertex.
// ---------------------------------------------------------------------------

/// World-space positions of a mesh's shared edge, in emission order.
/// `edge` selects which side of the tile: matches `grid_vertex_world_position`'s
/// `(i, j)` grid, `i`/`j` in `0..=grid_resolution`.
fn edge_world_positions(
    world_seed: Seed,
    tile: TerrainTile,
    params: &TerrainParams,
    edge: Edge,
) -> Vec<(f32, f32, f32)> {
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
        let world_x = v.position[0] + origin_x as f32;
        let world_y = v.position[1];
        let world_z = v.position[2] + origin_z as f32;
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

#[test]
fn ordeal_seam_all_four_directions_byte_identical() {
    let world_seed = Seed::new(0xF00D_5EED);
    let params = small_params();
    let center = TerrainTile::new(-2, 5);

    // (this tile's edge, this tile's edge direction, neighbor tile, neighbor's
    // matching opposite edge) — the neighbor's `-x`/`-z` etc face meets the
    // center tile's `+x`/`+z` etc face.
    let directions: [(&str, Edge, TerrainTile, Edge); 4] = [
        ("+x", Edge::PlusX, center.neighbor(1, 0), Edge::MinusX),
        ("-x", Edge::MinusX, center.neighbor(-1, 0), Edge::PlusX),
        ("+z", Edge::PlusZ, center.neighbor(0, 1), Edge::MinusZ),
        ("-z", Edge::MinusZ, center.neighbor(0, -1), Edge::PlusZ),
    ];

    let mut all_identical = true;
    let mut checked_vertices = 0usize;
    for (name, center_edge, neighbor_tile, neighbor_edge) in directions {
        let center_positions = edge_world_positions(world_seed, center, &params, center_edge);
        let neighbor_positions =
            edge_world_positions(world_seed, neighbor_tile, &params, neighbor_edge);
        assert_eq!(center_positions.len(), neighbor_positions.len());
        for (idx, (c, nb)) in center_positions.iter().zip(&neighbor_positions).enumerate() {
            checked_vertices += 1;
            let identical = c.0.to_bits() == nb.0.to_bits()
                && c.1.to_bits() == nb.1.to_bits()
                && c.2.to_bits() == nb.2.to_bits();
            if !identical {
                all_identical = false;
            }
            assert_eq!(
                c.0.to_bits(),
                nb.0.to_bits(),
                "seam {name} vertex {idx}: world_x mismatch {:?} vs {:?}",
                c,
                nb
            );
            assert_eq!(
                c.1.to_bits(),
                nb.1.to_bits(),
                "seam {name} vertex {idx}: world_y (height) mismatch {:?} vs {:?}",
                c,
                nb
            );
            assert_eq!(
                c.2.to_bits(),
                nb.2.to_bits(),
                "seam {name} vertex {idx}: world_z mismatch {:?} vs {:?}",
                c,
                nb
            );
        }
    }

    println!(
        "ORDEAL(c) seam: directions=4 checked_vertices={} all_identical={}",
        checked_vertices, all_identical
    );
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
