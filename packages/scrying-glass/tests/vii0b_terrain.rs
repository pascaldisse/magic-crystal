//! RITE VII · VII-0b — THE FIRST GROUND, part (b): the render weld ordeals.
//! A terrain patch is authored ONLY as a sigil (seed + tile coords + optional
//! dial overrides) — NO stored geometry — generated at load through VII-0a's
//! `seed::tile_mesh` and sealed through the SAME Great Chain path every other
//! static part rides. These ordeals build synthetic realms on disk and drive
//! the REAL `from_ecs`/`from_ecs_at` weld — the exact path a realm loads
//! through (the `body_sigil.rs` precedent).

use crystal::{EcsWorld, load_world_dir};
use scrying_glass::player::Ground;
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults};
use seed::terrain::height_at_grid_index;
use seed::{TerrainParams, TerrainTile};

fn params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: 55.0,
        near: 0.1,
        far: 4_000.0,
        sky_top: "#20152f".into(),
        sky_horizon: "#9a627d".into(),
        mesh_color: "#9aa0a6".into(),
        radial_segments: 24,
        camera_position: [0.0, 2.0, 22.0],
        camera_yaw: 0.0,
        camera_pitch: 0.0,
        cluster_error_threshold: 1.0,
        tick_dt: 1.0 / 60.0,
        sun: SunDefaults {
            sun_color: "#ffe2b0".into(),
            sun_intensity: 1.1,
            sun_position: [60.0, 90.0, 30.0],
            ambient_intensity: 0.32,
        },
        emission_intensity: 2.5,
    }
}

/// Write a scratch world dir containing ONLY a `terrain` sigil entity, load
/// it fresh, drive the real `from_ecs` weld. Returns the dir (for a caller
/// that wants to inspect the raw JSON too) and the built scene.
fn terrain_only_world(tag: &str, terrain_json: &str) -> (std::path::PathBuf, RenderScene) {
    let dir = std::env::temp_dir().join(format!("gaia_vii0b_{tag}_{}", std::process::id()));
    let scenes = dir.join("scenes");
    std::fs::create_dir_all(&scenes).expect("create temp scenes dir");
    let scene = format!(r#"{{ "naruko_first_ground": {{ "terrain": {terrain_json} }} }}"#);
    std::fs::write(scenes.join("main.json"), scene).expect("write temp scene");
    let mut world = EcsWorld::default();
    load_world_dir(&dir, &mut world).expect("load the terrain-only realm");
    let render = RenderScene::from_ecs(world, &params()).expect("weld the terrain patch");
    (dir, render)
}

/// ORDEAL (a) — NO-STORAGE at the scene level. Loading the SAME terrain
/// sigil realm cold TWICE (fresh `Core`/`EcsWorld` each time, fresh temp dir
/// read from disk each time) produces byte-identical leaf triangles — the
/// patch is regenerated from the sigil alone, never cached/stored geometry
/// smuggled in some other way. Also asserts the authored JSON itself carries
/// no mesh/vertex data for the patch: the sigil is the ONLY authored
/// artifact.
#[test]
fn no_storage_cold_double_load_is_byte_identical_and_json_carries_no_geometry() {
    let terrain_json = r#"{ "seed": 42, "tile_x": 3, "tile_y": -2 }"#;
    let (dir_a, scene_a) = terrain_only_world("nostorage_a", terrain_json);
    let (dir_b, scene_b) = terrain_only_world("nostorage_b", terrain_json);

    let tris_a = scene_a.leaf_triangles();
    let tris_b = scene_b.leaf_triangles();
    assert!(!tris_a.is_empty(), "the patch produced real geometry");
    assert_eq!(
        tris_a.len(),
        tris_b.len(),
        "two cold loads must generate the same triangle count"
    );
    let positions_a: Vec<[[f32; 3]; 3]> = tris_a.iter().map(|t| t.positions).collect();
    let positions_b: Vec<[[f32; 3]; 3]> = tris_b.iter().map(|t| t.positions).collect();
    assert_eq!(
        positions_a, positions_b,
        "two cold loads of the SAME sigil must be byte-identical (NO-STORAGE: \
         the patch is regenerated from the sigil, not cached)"
    );

    // The authored JSON is the sigil ALONE: no key that could carry stored
    // geometry (vertices/positions/indices/mesh) rides along with it.
    let raw = std::fs::read_to_string(dir_a.join("scenes").join("main.json"))
        .expect("read the authored realm");
    let doc: serde_json::Value = serde_json::from_str(&raw).expect("parse the authored realm");
    let entity = &doc["naruko_first_ground"];
    let keys: Vec<&str> = entity
        .as_object()
        .expect("entity is an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        vec!["terrain"],
        "the patch entity authors ONLY the terrain sigil"
    );
    for forbidden in ["vertices", "positions", "indices", "mesh"] {
        assert!(
            !raw.contains(forbidden),
            "authored realm must never carry geometry data ({forbidden:?} found)"
        );
    }

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

/// ORDEAL (e) — THE WALKER FLOOR SEES IT. The generated patch's triangles
/// enter `Ground` exactly like every other static part
/// (`leaf_positions_of(&chains)` -> `Ground::from_positions`, VII-0b's scene
/// seam doc). A height query at the tile's exact center must land ON the
/// generated field, AT the field's own analytic value — not merely inside
/// some loose envelope — proving the floor a future walker (RITE VII-1) will
/// step on already sees THIS field, not just some plausible-looking surface.
///
/// The query point is chosen to land EXACTLY on a mesh grid vertex (the tile
/// center: `tile_size_m/2` is an integer multiple of `cell_size_m` for the
/// default params), so the true reference is `seed::height_at_grid_index` at
/// that vertex's own global index — the SAME analytic field
/// `seed::tile_mesh` sampled to build the mesh in the first place (no mesh
/// built here to get the reference; read straight off `params()`, never a
/// frozen literal). The floor's raycast/interpolation at an exact vertex
/// should reproduce that value to within ordinary f32 arithmetic noise, not
/// a half-cell-size envelope — DERIVED tolerance below (an analytic ULP
/// budget; measured on this geometry the discrepancy is exactly 0.0, see the
/// tolerance derivation at the assertion site), never a plucked epsilon.
#[test]
fn generated_patch_enters_the_walker_floor_at_the_analytic_field_value() {
    let params = TerrainParams::default();
    let tile = TerrainTile::new(5, 0);
    let world_seed = seed::Seed(7);
    let (origin_x, origin_z) = seed::tile_origin_m(tile, &params);
    let cell_size_m = params.cell_size_m();

    // tile_size_m/2 / cell_size_m must be an exact integer grid index for
    // this "on-vertex" query to be valid; assert it rather than assume it,
    // so a future params change that breaks the assumption fails loudly here
    // instead of silently degrading to an interpolated (looser) comparison.
    let half_cells = params.tile_size_m / 2.0 / cell_size_m;
    assert!(
        (half_cells - half_cells.round()).abs() < 1e-6,
        "tile center must land on an exact grid vertex for this ordeal's on-vertex \
         comparison (tile_size_m/2/cell_size_m = {half_cells}, not an integer)"
    );
    let local_i = half_cells.round() as i64;
    let world_x = origin_x + local_i as f64 * cell_size_m as f64;
    let world_z = origin_z + local_i as f64 * cell_size_m as f64;
    let n = params.grid_resolution as i64;
    let global_i = tile.tile_x * n + local_i;
    let global_j = tile.tile_y * n + local_i;
    let analytic_height = height_at_grid_index(world_seed, &params, global_i, global_j);

    let terrain_json = format!(
        r#"{{ "seed": 7, "tile_x": {}, "tile_y": {} }}"#,
        tile.tile_x, tile.tile_y
    );
    let (dir, scene) = terrain_only_world("floor", &terrain_json);

    let floor = Ground::from_positions(&scene.leaf_positions());
    let height = floor
        .height_at(world_x as f32, world_z as f32, f32::INFINITY)
        .expect("the patch center must be found on the walker floor");

    // DERIVED tolerance: an ANALYTIC ULP budget, not a plucked epsilon. A
    // ray-triangle height query at an exact shared vertex chains roughly a
    // dozen f32 multiply/add/divide steps (edge vectors, a cross product, a
    // barycentric division) over coordinates scaled to ~hundreds of meters;
    // 16 ULPs at that magnitude is a standard generous bound for a chain
    // that short. Measured on this exact geometry the live floor query vs
    // the analytic field discrepancy is EXACTLY 0.0 (bit-identical) — this
    // gate exists as a margin against a future geometry/scale change
    // introducing genuine float noise, not because the current behavior
    // needs any slack (the "measure floor, gate ~10x" law's floor here is
    // 0.0, which can't itself be the gate, so the gate is the analytic ULP
    // bound instead — still many orders of magnitude tighter than a
    // half-cell-size envelope would have been).
    let scale = world_x.abs().max(world_z.abs()).max(1.0) as f32;
    let tolerance = scale * f32::EPSILON * 16.0;
    let measured = (height - analytic_height).abs();
    assert!(
        measured <= tolerance,
        "floor height {height} vs analytic field height {analytic_height} at \
         ({world_x},{world_z}): discrepancy {measured} exceeds derived tolerance \
         {tolerance}"
    );
    eprintln!(
        "[ordeal] walker floor vs analytic field: measured discrepancy {measured:.3e} m \
         (derived tolerance {tolerance:.3e} m)"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Write a scratch world dir containing ONLY a `terrain` sigil entity, load
/// it fresh, drive the real `from_ecs_at` weld with an EXPLICIT
/// `render_origin` — the production entry point A1 (adversary review) found
/// untested end-to-end (the earlier translation-invariance ordeal exercised
/// only the private helper functions directly, never proving `from_ecs_at`
/// actually threads `render_origin` through the sigil-parse -> tile_mesh ->
/// append_terrain chain). Otherwise identical to `terrain_only_world`.
fn terrain_only_world_at(
    tag: &str,
    terrain_json: &str,
    render_origin: [f64; 3],
) -> (std::path::PathBuf, RenderScene) {
    let dir = std::env::temp_dir().join(format!("gaia_vii0b_{tag}_{}", std::process::id()));
    let scenes = dir.join("scenes");
    std::fs::create_dir_all(&scenes).expect("create temp scenes dir");
    let scene = format!(r#"{{ "naruko_first_ground": {{ "terrain": {terrain_json} }} }}"#);
    std::fs::write(scenes.join("main.json"), scene).expect("write temp scene");
    let mut world = EcsWorld::default();
    load_world_dir(&dir, &mut world).expect("load the terrain-only realm");
    let render =
        RenderScene::from_ecs_at(world, &params(), render_origin).expect("weld the terrain patch");
    (dir, render)
}

/// Canonicalize a triangle-position list into an ORDER-INDEPENDENT form: sort
/// each triangle's 3 vertices, then sort the triangle list — so two
/// geometrically-identical-but-differently-ordered triangle soups (e.g. two
/// runs of the Great Chain's clustering, which is deterministic but not
/// input-order-preserving) compare equal. `f32::total_cmp` gives a total
/// order (never panics on the values this pipeline produces).
fn canonical_triangles(tris: &[[[f32; 3]; 3]]) -> Vec<[[f32; 3]; 3]> {
    fn cmp_vert(a: &[f32; 3], b: &[f32; 3]) -> std::cmp::Ordering {
        a[0].total_cmp(&b[0])
            .then_with(|| a[1].total_cmp(&b[1]))
            .then_with(|| a[2].total_cmp(&b[2]))
    }
    let mut out: Vec<[[f32; 3]; 3]> = tris
        .iter()
        .map(|tri| {
            let mut verts = *tri;
            verts.sort_by(cmp_vert);
            verts
        })
        .collect();
    out.sort_by(|a, b| {
        cmp_vert(&a[0], &b[0])
            .then_with(|| cmp_vert(&a[1], &b[1]))
            .then_with(|| cmp_vert(&a[2], &b[2]))
    });
    out
}

/// ORDEAL (c), A1 fix — THE COORDINATE SEAM, proven through the REAL
/// production weld (`from_ecs_at` -> the sigil parse -> `seed::tile_mesh` ->
/// `append_terrain`), not just the private helper functions in isolation.
///
/// Two tiles, both loaded with `render_origin` CO-LOCATED with their own
/// `tile_origin_m` — a NEAR tile `(0,0)` and a PLANET-SCALE FAR tile
/// `(10_000_000, -10_000_000)` (Ruling 4's own worked magnitude). In BOTH
/// regimes the residual offset (`tile_origin - render_origin`) is exactly
/// `[0,0,0]` by construction (an f64 subtraction of a value from itself is
/// exact regardless of magnitude — no precision claim needed there). So in
/// BOTH regimes the weld's world-space leaf triangles should equal that
/// tile's OWN local mesh (`seed::tile_mesh`) with NO placement offset
/// applied at all — proven independently at each magnitude by comparing
/// (order-independently, since the Great Chain's clustering doesn't
/// preserve input order) the scene's leaf triangles against a freshly-called
/// `seed::tile_mesh` for that SAME tile.
///
/// This is honestly scoped to PLACEMENT, not content (A2 correction): the
/// near and far regimes use DIFFERENT tile identities, so their generated
/// HEIGHT content legitimately differs (different global grid indices sample
/// different noise — VII-0a's per-tile independence by design). What must
/// (and does) match between regimes is the GRID SHAPE — the local x/z
/// footprint pattern, which depends only on `grid_resolution`/`cell_size_m`
/// (identical params in both regimes), never on the tile's magnitude.
#[test]
fn render_origin_at_planetary_tile_magnitude_reproduces_the_local_mesh_through_the_real_weld() {
    let params = TerrainParams::default();
    let world_seed = seed::Seed(9);

    for (tag, tile) in [
        ("near", TerrainTile::new(0, 0)),
        ("far", TerrainTile::new(10_000_000, -10_000_000)),
    ] {
        let tile_origin = seed::tile_origin_m(tile, &params);
        let render_origin = [tile_origin.0, 0.0, tile_origin.1];
        let terrain_json = format!(
            r#"{{ "seed": 9, "tile_x": {}, "tile_y": {} }}"#,
            tile.tile_x, tile.tile_y
        );
        let (dir, scene) = terrain_only_world_at(tag, &terrain_json, render_origin);

        // The REAL weld's world-space leaf triangles...
        let actual: Vec<[[f32; 3]; 3]> =
            scene.leaf_triangles().iter().map(|t| t.positions).collect();
        assert!(
            !actual.is_empty(),
            "{tag}: the patch produced real geometry"
        );

        // ...must equal this tile's own UNPLACED local mesh (offset==0 by
        // construction, so world position == local position exactly).
        let local_mesh = seed::tile_mesh(world_seed, tile, &params);
        let expected: Vec<[[f32; 3]; 3]> = local_mesh
            .indices
            .chunks_exact(3)
            .map(|tri| {
                [
                    local_mesh.vertices[tri[0] as usize].position,
                    local_mesh.vertices[tri[1] as usize].position,
                    local_mesh.vertices[tri[2] as usize].position,
                ]
            })
            .collect();

        assert_eq!(
            canonical_triangles(&actual),
            canonical_triangles(&expected),
            "{tag} regime (tile {tile:?}): the REAL from_ecs_at weld, with \
             render_origin co-located with the tile's own origin, must \
             reproduce the tile's local mesh bit-identically (order-independent) \
             — no precision lost at this tile's magnitude"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // Cross-regime: same grid params ⇒ the LOCAL x/z footprint pattern (not
    // the height content, which legitimately differs by tile identity) must
    // be identical between near and far tile magnitude — the actual
    // translation-invariance claim, honestly scoped to placement geometry.
    let near_mesh = seed::tile_mesh(world_seed, TerrainTile::new(0, 0), &params);
    let far_mesh = seed::tile_mesh(
        world_seed,
        TerrainTile::new(10_000_000, -10_000_000),
        &params,
    );
    let mut near_xz: Vec<(u32, u32)> = near_mesh
        .vertices
        .iter()
        .map(|v| (v.position[0].to_bits(), v.position[2].to_bits()))
        .collect();
    let mut far_xz: Vec<(u32, u32)> = far_mesh
        .vertices
        .iter()
        .map(|v| (v.position[0].to_bits(), v.position[2].to_bits()))
        .collect();
    near_xz.sort();
    far_xz.sort();
    assert_eq!(
        near_xz, far_xz,
        "the local x/z grid footprint must be bit-identical regardless of tile \
         magnitude — only height content (a different tile's noise) may differ"
    );
}

/// A `terrain` entity that ALSO carries a `mesh` component is a loud
/// authoring error — the sigil-only law, enforced at the scene seam.
#[test]
fn terrain_entity_with_a_mesh_component_is_refused() {
    let dir = std::env::temp_dir().join(format!("gaia_vii0b_doublegeom_{}", std::process::id()));
    let scenes = dir.join("scenes");
    std::fs::create_dir_all(&scenes).expect("create temp scenes dir");
    let scene = r##"{
      "bad_ground": {
        "terrain": { "seed": 1, "tile_x": 0, "tile_y": 0 },
        "mesh": { "parts": [ { "shape": "box", "size": [1,1,1], "color": "#808080" } ] }
      }
    }"##;
    std::fs::write(scenes.join("main.json"), scene).expect("write temp scene");
    let mut world = EcsWorld::default();
    load_world_dir(&dir, &mut world).expect("load the (deliberately bad) realm");
    let result = RenderScene::from_ecs(world, &params()).map(|_| ());
    let _ = std::fs::remove_dir_all(&dir);
    let error = result.expect_err("terrain + mesh together must be refused");
    assert!(
        error.contains("terrain") && error.contains("mesh"),
        "error should name both sigils: {error}"
    );
}
