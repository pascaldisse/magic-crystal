//! RITE VII · VII-0b — THE FIRST GROUND. Adversary review A3: the proposal's
//! words are "gaze returns terrain" — this file makes them LITERALLY true.
//! Every other VII-0b oracle ordeal (`canon.rs::canon_terrain_patch_bounds_
//! and_range_are_derived`) reads `World::geometry` directly; none of them
//! actually drove `look()` with an eye that SEES the patch and checked it
//! comes back in the caption list. This does exactly that, against a
//! dedicated synthetic world (isolated from the real canon realm's own
//! caption-ordering ordeals — the `vi2_break`-realm-isolation precedent) so
//! the assertion is about THIS entity, not incidentally true because of
//! unrelated canon churn.

use oracle::{look, EyePose, Glance, Layers, LookParams, World};

/// A world containing ONLY the canon terrain sigil
/// (`{seed:20260717, tile_x:0, tile_y:2}`, matching `worlds/naruko`'s
/// `naruko_first_ground`) — tile_origin (0,128), tile_size_m=64,
/// height_amplitude=9.6 (see `canon.rs`'s header derivation for these
/// numbers), so world AABB x[0,64] y[-9.6,9.6] z[128,192], center [32,0,160].
fn terrain_only_world() -> World {
    let dir = std::env::temp_dir().join(format!("gaia_vii0b_gaze_{}", std::process::id()));
    let scenes = dir.join("scenes");
    std::fs::create_dir_all(&scenes).expect("create temp scenes dir");
    std::fs::write(
        scenes.join("main.json"),
        r#"{ "naruko_first_ground": { "terrain": { "seed": 20260717, "tile_x": 0, "tile_y": 2 } } }"#,
    )
    .expect("write temp scene");
    let world = World::load(&dir).expect("load the terrain-only realm");
    let _ = std::fs::remove_dir_all(&dir);
    world
}

fn caption_ids(g: &Glance) -> Vec<String> {
    g.nearest.iter().map(|n| n.id.clone()).collect()
}

/// ADVERSARY A3 — an eye placed FACING the patch (above and behind it on the
/// +z side, looking down -z toward the patch center) must have `look()`
/// return `naruko_first_ground` in its captions, at the HAND-DERIVED range
/// from that eye.
///
/// DERIVATION: eye = [32, 40, 260], yaw=0 pitch=0 (fwd=(0,0,-1) at yaw 0,
/// the canon eye-basis convention — `canon.rs`'s header). Patch center
/// [32,0,160] ⇒ d = center - eye = [0,-40,-100], range = √(0²+40²+100²) =
/// √11600 = 107.7033.
///
/// FRUSTUM FIT (FOV 60°, aspect 1, half=30°): the eye shares the patch's own
/// x=32, so the bearing is dead-ahead (0° horizontal to the CENTER); the
/// patch's x half-extent (32 m, tile_size_m/2) at |dz|=100 subtends
/// atan(32/100) = 17.75° < 30° ⇒ both side edges are inside. Vertically, the
/// patch's y extremes are ±9.6 (height_amplitude) around y=0; relative to
/// the eye (y=40, dz=100) the top edge subtends atan((9.6-40)/100) = -16.91°
/// and the bottom edge atan((-9.6-40)/100) = -26.38°, both inside [-30°,30°]
/// ⇒ the WHOLE AABB projects inside the frustum, not just its center.
///
/// `max_extent` (64 m, the tile footprint) is well under
/// `support_ratio(8.0) × range(107.7) ≈ 861.6`, so the patch is NOT demoted
/// to world-support (`include_support` stays the default `false`) — it must
/// appear as an ordinary caption.
#[test]
fn gaze_facing_the_patch_returns_naruko_first_ground_at_the_derived_range() {
    let world = terrain_only_world();
    let eye = EyePose {
        position: [32.0, 40.0, 260.0],
        yaw: 0.0,
        pitch: 0.0,
    };
    let g = look(
        &world,
        eye,
        LookParams {
            nearest_n: 8,
            layers: Layers::NONE,
            ..Default::default()
        },
    )
    .expect("look succeeds");

    let caps = caption_ids(&g);
    assert!(
        caps.contains(&"naruko_first_ground".to_string()),
        "an eye facing the patch must see it in captions — got {caps:?}"
    );
    assert_eq!(
        g.entity_count, 1,
        "the only entity in this world is the patch"
    );

    let entry = g
        .nearest
        .iter()
        .find(|n| n.id == "naruko_first_ground")
        .expect("naruko_first_ground is captioned");
    const RANGE_TOL: f32 = 1e-3; // same derived tolerance as canon.rs CANON #2/#8
    assert!(
        (entry.range - 107.7033).abs() < RANGE_TOL,
        "range: live {} != derived 107.7033 (tol {RANGE_TOL})",
        entry.range
    );
    assert!(
        !entry.support,
        "the patch (64 m max_extent) must not be demoted to world-support at range 107.7"
    );
}

/// A NEGATIVE control: an eye facing AWAY from the patch (yaw 180°, so
/// fwd=(0,0,+1)) must NOT see it — proving the previous test's success is
/// about real frustum geometry, not a `look()` bug that captions everything
/// regardless of gaze direction.
#[test]
fn gaze_facing_away_from_the_patch_does_not_return_it() {
    let world = terrain_only_world();
    let eye = EyePose {
        position: [32.0, 40.0, 260.0],
        yaw: 180.0_f32.to_radians(),
        pitch: 0.0,
    };
    let g = look(
        &world,
        eye,
        LookParams {
            nearest_n: 8,
            layers: Layers::NONE,
            ..Default::default()
        },
    )
    .expect("look succeeds");
    assert_eq!(
        g.entity_count, 0,
        "facing away from the ONLY entity in this world must caption nothing"
    );
}
