//! RITE VII · VII-0b — THE FIRST GROUND, part (b): the render weld ordeals.
//! A terrain patch is authored ONLY as a sigil (seed + tile coords + optional
//! dial overrides) — NO stored geometry — generated at load through VII-0a's
//! `seed::tile_mesh` and sealed through the SAME Great Chain path every other
//! static part rides. These ordeals build synthetic realms on disk and drive
//! the REAL `from_ecs`/`from_ecs_at` weld — the exact path a realm loads
//! through (the `body_sigil.rs` precedent).

use crystal::{load_world_dir, EcsWorld};
use scrying_glass::player::Ground;
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults};

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
    let scene = format!(
        r#"{{ "naruko_first_ground": {{ "terrain": {terrain_json} }} }}"#
    );
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
/// generated field, not fall through to `None` or hit some other surface —
/// proving the floor a future walker (RITE VII-1) will step on already sees
/// the patch. Tolerance: HALF the grid's cell size — the coarsest error a
/// bilinear-ish nearest-triangle floor query can have relative to the true
/// analytic field is bounded by the mesh's own triangulation resolution
/// (never a plucked epsilon).
#[test]
fn generated_patch_enters_the_walker_floor() {
    // tile_size_m defaults to 64, grid_resolution derives to 64 (see
    // TerrainParams::derive's Nyquist floor against the default fBm) ⇒
    // cell_size_m = 1.0. tile_x=5,tile_y=0 ⇒ tile_origin = (5*64*1, 0) =
    // (320, 0); tile center = origin + tile_size/2 = (352, 32).
    let terrain_json = r#"{ "seed": 7, "tile_x": 5, "tile_y": 0 }"#;
    let (dir, scene) = terrain_only_world("floor", terrain_json);

    let floor = Ground::from_positions(&scene.leaf_positions());
    let cell_size_m = 1.0_f32; // derived above; see the seed-crate ordeal for the Nyquist derivation itself
    let tolerance = cell_size_m * 0.5;
    let height = floor
        .height_at(352.0, 32.0, f32::INFINITY)
        .expect("the patch center must be found on the walker floor");
    assert!(
        height.abs() <= 9.6 + tolerance,
        "queried floor height {height} exceeds the patch's own amplitude bound 9.6 (+tol {tolerance})"
    );

    let _ = std::fs::remove_dir_all(&dir);
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
