//! MEASUREMENT SCAFFOLD — will be rewritten into the real ordeal once numbers
//! are read. Prints rested positions/AABBs/ranges and the rest-tick.

use crystal::{load_world_dir, Core, EcsWorld};
use oracle::World;
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults};
use std::path::{Path, PathBuf};

fn naruko_world() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko")
}

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

const CANON_EYE: [f32; 3] = [0.0, 7.0, 44.0];
const IDS: [&str; 4] = [
    "naruko_crate",
    "naruko_stack_crate_0",
    "naruko_stack_crate_1",
    "naruko_stack_crate_2",
];

fn scene() -> RenderScene {
    let mut world = EcsWorld::default();
    load_world_dir(naruko_world(), &mut world).expect("load naruko");
    RenderScene::from_ecs(world, &params()).expect("scene")
}

/// Tick to kinetic-floor rest for ALL bodies; return the first tick at which
/// every body's per-tick |Δy| < 1e-6 (and tick > 60), plus the ticked world.
fn tick_to_rest() -> (u64, u64) {
    let mut scene = scene();
    let mut last: Vec<f64> = IDS.iter().map(|id| scene.body_position(id).unwrap()[1]).collect();
    let mut rest_tick = None;
    for tick in 0..600u64 {
        scene.tick();
        let mut all_floor = true;
        for (i, id) in IDS.iter().enumerate() {
            let y = scene.body_position(id).unwrap()[1];
            if (y - last[i]).abs() >= 1.0e-6 {
                all_floor = false;
            }
            last[i] = y;
        }
        if rest_tick.is_none() && all_floor && tick > 60 {
            rest_tick = Some(tick);
        }
    }
    (rest_tick.unwrap_or(u64::MAX), 600)
}

#[test]
fn measure() {
    let (rt_a, _) = tick_to_rest();
    let (rt_b, _) = tick_to_rest();
    eprintln!("[measure] rest_tick a={rt_a} b={rt_b}");

    // Tick a fresh run to full 600, then gaze.
    let mut sc = scene();
    for _ in 0..600u64 {
        sc.tick();
    }
    for id in IDS {
        eprintln!("[measure] {id} body_position={:?}", sc.body_position(id).unwrap());
    }
    let ticked: EcsWorld = sc.dynamics.into_world();
    let tcid = ticked.component_id("transform").unwrap();
    // Load oracle the normal way (correct {"v"} envelope + full registry), then
    // inject the solver-rested transform into each crate.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko").canonicalize().unwrap();
    let mut w = World::load(dir).unwrap();
    let wcid = w.core.world.component_id("transform").unwrap();
    for id in IDS {
        let src = ticked.entity_for_gaia(id).unwrap();
        let rested = ticked.get_component(src, tcid).unwrap();
        let dst = w.core.world.entity_for_gaia(id).unwrap();
        w.core.world.set_component(dst, wcid, serde_json::json!({ "v": rested })).unwrap();
    }
    for id in IDS {
        let g = w.geometry(id).expect("geom");
        let b = g.bounds.expect("bounds");
        let c = b.center();
        let dx = c[0] - CANON_EYE[0];
        let dy = c[1] - CANON_EYE[1];
        let dz = c[2] - CANON_EYE[2];
        let range = (dx * dx + dy * dy + dz * dz).sqrt();
        eprintln!(
            "[measure] {id} min={:?} max={:?} center={:?} size={:?} range={:.4}",
            b.min, b.max, c, b.size(), range
        );
    }
}

#[test]
fn probe_pier() {
    let mut world = EcsWorld::default();
    load_world_dir(naruko_world(), &mut world).unwrap();
    let pier = scrying_glass::scene::top_flat_surface_y(&world, "naruko_pier").unwrap().unwrap();
    let mut sc = RenderScene::from_ecs(world, &params()).unwrap();
    let ph = sc.physics().unwrap();
    let b = &ph.bindings()[0];
    eprintln!("[probe] pier_top={pier:.10} half={:.6} radius={:.6}", b.half_height, b.contact_radius);
    let _ = &mut sc;
}
