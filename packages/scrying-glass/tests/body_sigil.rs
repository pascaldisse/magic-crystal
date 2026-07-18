//! F2 — BODY-SIGIL FORK GUARD ordeals. A `body` is EITHER a skinned vessel
//! (`preset`, driven by the RITE V weld) OR a rigid solver body (`shape`),
//! NEVER both. The adversary's latent double-driver — a `{preset}` body on a
//! MESHED entity becoming BOTH a default rigid AND a skinned vessel — is closed:
//! a preset body is a vessel ONLY, the rigid solver never sees it, and
//! `{preset, shape}` is a LOUD authoring error that names the entity and refuses
//! the realm. An unknown body field (typo'd dial) is likewise loud
//! (`deny_unknown_fields`). These build synthetic realms on disk and drive the
//! REAL `from_ecs` weld — the exact path a realm loads through.

use crystal::{EcsWorld, load_world_dir};
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

/// Build a one-entity realm on disk: a floor slab plus a `subject` carrying the
/// given `body` JSON on a meshed, transformed entity — then drive the REAL
/// `from_ecs` weld. Returns the weld's `Result` (Err = the realm was refused).
fn scene_with_body(tag: &str, body_json: &str) -> Result<RenderScene, String> {
    let dir = std::env::temp_dir().join(format!("gaia_f2_{tag}_{}", std::process::id()));
    let scenes = dir.join("scenes");
    std::fs::create_dir_all(&scenes).expect("create temp scenes dir");
    let scene = format!(
        r##"{{
          "floor": {{
            "transform": {{ "position": [0, 0, 0] }},
            "mesh": {{ "parts": [
              {{ "shape": "box", "size": [10, 1, 10], "position": [0, -0.5, 0], "color": "#808080" }}
            ] }}
          }},
          "subject": {{
            "transform": {{ "position": [0, 2, 0] }},
            "mesh": {{ "parts": [
              {{ "shape": "box", "size": [0.5, 0.5, 0.5], "position": [0, 0, 0], "color": "#a0a0a0" }}
            ] }},
            "body": {body_json}
          }}
        }}"##
    );
    std::fs::write(scenes.join("main.json"), scene).expect("write temp scene");
    let mut world = EcsWorld::default();
    let load = load_world_dir(&dir, &mut world).map(|_| ());
    let result = load.and_then(|()| RenderScene::from_ecs(world, &params()));
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// F2 — THE DOUBLE-DRIVER, CLOSED. A `{preset}` body on a MESHED entity renders
/// as a skinned VESSEL only; the rigid solver never sees it (no shape ⇒
/// zero-physics, solver count unchanged — no latent default rigid).
#[test]
fn f2_preset_body_on_meshed_entity_is_vessel_only() {
    let scene = scene_with_body("double", r#"{ "preset": "nari" }"#).expect("the realm builds");
    assert_eq!(
        scene.bodies.len(),
        1,
        "the preset body is a skinned vessel (rendered as the vessel)"
    );
    assert!(
        scene.physics().is_none(),
        "solver count UNCHANGED: a preset body is NOT a rigid (no double-driver)"
    );
}

/// F2 — `{preset, shape}` together is a LOUD authoring error that NAMES the
/// entity and refuses the realm (never a silent both-driver).
#[test]
fn f2_both_preset_and_shape_refuses_the_realm() {
    let err = scene_with_body("both", r#"{ "preset": "nari", "shape": "box" }"#)
        .map(|_| ())
        .expect_err("a body declaring both `preset` and `shape` must refuse the realm");
    assert!(
        err.contains("subject"),
        "the error must NAME the offending entity: {err}"
    );
    assert!(
        err.contains("preset") && err.contains("shape"),
        "the error must explain the fork (both preset and shape): {err}"
    );
}

/// F2 — control: a `{shape}` body is a RIGID in the solver, NOT a skinned
/// vessel. The path preset-bodies are steered away from still works.
#[test]
fn f2_shape_body_is_a_rigid_not_a_vessel() {
    let scene = scene_with_body("shape", r#"{ "shape": "box", "size": [0.5, 0.5, 0.5] }"#)
        .expect("the realm builds");
    assert!(
        scene.bodies.is_empty(),
        "a shape body is NOT a skinned vessel"
    );
    let physics = scene
        .physics()
        .expect("a shape body IS a rigid in the solver");
    assert_eq!(
        physics.bindings().len(),
        1,
        "exactly one rigid body is bound to the solver"
    );
}

/// F2 — `deny_unknown_fields`: a typo'd body dial is a LOUD error naming the
/// entity, never a silently-defaulted body.
#[test]
fn f2_unknown_body_field_is_loud() {
    let err = scene_with_body("typo", r#"{ "shape": "box", "densty": 500 }"#)
        .map(|_| ())
        .expect_err("an unknown body field must be a loud error");
    assert!(
        err.contains("subject"),
        "the error must NAME the offending entity: {err}"
    );
    assert!(
        err.contains("densty"),
        "the error must name the unknown field: {err}"
    );
}
