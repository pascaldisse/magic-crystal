//! RITE V · V0 render-side ordeals — the embodied body enters the traced
//! world. These are CPU-only (no GPU): they prove the `body` sigil composes
//! into world-space skinned triangles that ride the DYNAMIC partition (re-fed
//! to the traced BVH every tick, like the living layer), stand on the seawall,
//! and are deterministic.

use std::path::Path;

use crystal::{Core, load_world_dir};
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults};

fn naruko_params() -> SceneParameters {
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

fn load_naruko() -> Core {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).expect("load naruko");
    core
}

/// The `body` sigil composes into a standing embodied vessel: nari, the nari
/// preset, with a full skinned triangle soup (matching the vessel's 2268-tri
/// idle mesh).
#[test]
fn v0_body_composes_from_the_sigil() {
    let core = load_naruko();
    let scene = RenderScene::from_ecs(core.world, &naruko_params()).expect("render scene");

    assert_eq!(scene.bodies.len(), 1, "exactly nari carries a body sigil");
    let nari = &scene.bodies[0];
    assert_eq!(nari.gaia_id, "nari");
    assert_eq!(nari.preset, "nari");
    // The compose skins the whole vessel (nari idle mesh = 2268 tris).
    assert_eq!(
        nari.world_tris.len(),
        2268,
        "nari body must be the full skinned vessel"
    );
    println!(
        "[v0-render] nari body composed: {} tris",
        nari.world_tris.len()
    );
}

/// The body's triangles ENTER THE DYNAMIC PARTITION — the exact soup the traced
/// BVH splices each tick, on top of the living layer — and they PERSIST across a
/// world tick (the per-tick splice cadence, kami's precedent). She also stands
/// on the seawall (feet at the seawall top y = 1.4, derived from her placement).
#[test]
fn v0_body_enters_dynamic_partition_each_tick() {
    let mut scene = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");
    let body_tris = scene.bodies[0].world_tris.len();
    let living = scene.dynamics.leaf_triangles().len();

    let dynamic = scene.dynamic_leaf_triangles();
    assert_eq!(
        dynamic.len(),
        living + body_tris,
        "dynamic partition = living layer + the embodied body"
    );

    // Feet on the seawall: the lowest body vertex is the seawall top (y=1.4).
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for t in &scene.bodies[0].world_tris {
        for p in &t.positions {
            y_min = y_min.min(p[1]);
            y_max = y_max.max(p[1]);
            // She stands within the seawall footprint in x/z.
            assert!(
                p[0].abs() < 1.0 && (p[2] - 18.0).abs() < 1.0,
                "on the seawall"
            );
        }
    }
    // Derived: feet = 2.505 + local_min_y(-1.104977) = 1.400023 (seawall top).
    assert!(
        (y_min - 1.4).abs() < 1e-2,
        "nari feet on the seawall top (y=1.4), got {y_min}"
    );
    assert!(
        y_max > 3.0,
        "nari stands ~2.1 m tall, head near y=3.5, got {y_max}"
    );

    // The splice is per-tick: after advancing the world clock the body is STILL
    // in the dynamic partition (V0 idle ⇒ same triangles; V1 drives the pose).
    scene.tick();
    let after = scene.dynamic_leaf_triangles();
    assert_eq!(
        after.len(),
        scene.dynamics.leaf_triangles().len() + body_tris,
        "the body re-enters the dynamic partition every tick"
    );
    println!(
        "[v0-render] dynamic partition = {} living + {} body = {}; persists across tick",
        living,
        body_tris,
        dynamic.len()
    );
}

/// Determinism: two independent loads compose the SAME world-space body — the
/// skinned triangle positions are byte-identical (the ENTROPY law through the
/// whole render-side weld).
#[test]
fn v0_body_render_is_deterministic() {
    let a = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("a");
    let b = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("b");
    let bytes = |scene: &RenderScene| -> Vec<u8> {
        let mut out = Vec::new();
        for t in &scene.bodies[0].world_tris {
            for p in &t.positions {
                for c in p {
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
            for c in &t.albedo {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out
    };
    assert_eq!(
        bytes(&a),
        bytes(&b),
        "the embodied body must compose byte-identically"
    );
    println!("[v0-render] body render bytes identical across loads");
}
