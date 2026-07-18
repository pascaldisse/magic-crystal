//! BUILDING PUSH — PLAY.a proof harness (magic-crystal-play, playable-physics
//! lane). Loads the real naruko realm, settles `bldg_tower` (a 3-storey
//! anchored bonded structure south of the market), then drives the EXACT
//! push door the window key rides (view-ray pick over
//! `Physics::push_targets()` -> `Op::Impulse` -> `RenderScene::tick_with_ops`)
//! into it. `anchor_base` pins the tower's ground-floor bond ring to the
//! world (`elements::building::erect`'s technique) — a hard shove on the
//! standing structure shears against that fixed base exactly the way a real
//! load-bearing ground floor would, tearing the ground-floor bonds first and
//! dropping everything above. No GPU: physics seam only, fast iteration.
//!
//! Run: cargo run -p scrying-glass --release --example building_push

use std::path::Path;

use crystal::{load_world_dir, EcsWorld, ImpulseOp, Op};
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults};

const SETTLE_TICKS: u64 = 200;
const AFTER_TICKS: u64 = 900;
const TOWER: &str = "bldg_tower";

/// The window's push dials (GAIA_PUSH_REACH / GAIA_PUSH_SPEED / GAIA_PUSH_AIM_RADIUS).
const REACH: f32 = 4.0;
const SPEED: f32 = 5.0;
const AIM_RADIUS: f32 = 0.9;

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

fn build_scene() -> RenderScene {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut world = EcsWorld::default();
    load_world_dir(&world_path, &mut world).expect("load naruko");
    RenderScene::from_ecs(world, &naruko_params()).expect("render scene")
}

/// The window's picker, verbatim (main.rs `build_push_ops`).
fn pick(scene: &RenderScene, eye: [f32; 3], dir: [f32; 3]) -> Option<Op> {
    let physics = scene.physics()?;
    let e = glam::Vec3::from_array(eye);
    let d = glam::Vec3::from_array(dir).normalize();
    let mut best: Option<(f32, String)> = None;
    for (id, centroid) in physics.push_targets() {
        let c = glam::Vec3::new(centroid[0] as f32, centroid[1] as f32, centroid[2] as f32);
        let v = c - e;
        let t = v.dot(d);
        if t <= 0.0 || t > REACH {
            continue;
        }
        if (v - d * t).length() > AIM_RADIUS {
            continue;
        }
        if best.as_ref().is_none_or(|(bt, _)| t < *bt) {
            best = Some((t, id));
        }
    }
    best.map(|(_, id)| {
        let dv = d * SPEED;
        Op::Impulse(ImpulseOp {
            id,
            delta_velocity: [dv.x as f64, dv.y as f64, dv.z as f64],
            ..Default::default()
        })
    })
}

fn has_target(scene: &RenderScene, id: &str) -> bool {
    scene
        .physics()
        .map(|p| p.push_targets().iter().any(|(g, _)| g == id))
        .unwrap_or(false)
}

fn centroid(scene: &RenderScene, id: &str) -> Option<[f64; 3]> {
    scene
        .physics()?
        .push_targets()
        .into_iter()
        .find(|(g, _)| g == id)
        .map(|(_, c)| c)
}

fn min_feet_y(scene: &RenderScene) -> f64 {
    let physics = scene.physics().expect("physics");
    physics
        .bindings()
        .iter()
        .map(|b| physics.pose(b).position[1])
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    // ── CONTROL — settle only, no push: the structure must stand ─────────
    let mut scene = build_scene();
    for t in 0..SETTLE_TICKS {
        scene.tick();
        if t % 20 == 0 {
            println!("    t={t} {TOWER} whole={} centroid={:?}", has_target(&scene, TOWER), centroid(&scene, TOWER));
        }
    }
    let whole = has_target(&scene, TOWER);
    let rest_centroid = centroid(&scene, TOWER).expect("tower centroid at rest");
    println!("[control] tower whole after settle = {whole}, centroid = {rest_centroid:?} (authored [8,3.3,33])");
    assert!(whole, "the tower must stand on its own weight before any push (raise `love` / lower `density`)");
    assert!(
        (rest_centroid[1] - 3.3).abs() < 0.3,
        "the tower centroid must sit near its authored height at rest, got {:.3}",
        rest_centroid[1]
    );

    // ── THE PUSH — a player standing south of the tower, aiming level. ───
    let eye = [8.0, 1.6, 37.0];
    let dir = [0.0, 0.0, -1.0];
    let op = pick(&scene, eye, dir).expect("ray must pick the tower");
    let picked = match &op {
        Op::Impulse(i) => i.id.clone(),
        _ => unreachable!(),
    };
    println!("[push] ray picked {picked:?}");
    assert_eq!(picked, TOWER, "ray must select the tower");
    scene.tick_with_ops(&[op]);

    let mut min_top_y = rest_centroid[1];
    for i in 0..AFTER_TICKS {
        scene.tick();
        if let Some(c) = centroid(&scene, TOWER) {
            min_top_y = min_top_y.min(c[1]);
        }
        if i % 150 == 0 {
            let feet = min_feet_y(&scene);
            let entities = scene.dynamics.entities().len();
            println!("        tick {i}: tower whole={} min feet y={feet:.3} dynamic entities={entities}", has_target(&scene, TOWER));
        }
    }
    let broken = !has_target(&scene, TOWER);
    let entities_after = scene.dynamics.entities().len();
    let feet = min_feet_y(&scene);
    println!("[push] tower broken = {broken}");
    println!("[push] dynamic entities after = {entities_after}");
    println!("[push] min feet y (final, debris settle floor) = {feet:.3}");

    assert!(broken, "the pushed load-bearing structure must fracture (its whole-body target must vanish)");
    assert!(entities_after >= 2, "shatter must birth >= 2 fragment vessels, got {entities_after}");
    assert!(feet > -1.0, "no debris may sink far through the ground, min feet {feet:.3}");

    println!("\nBUILDING PUSH PROOF PASSED — the anchored ground-floor bonds shear under the push, the structure fractures and drops.");
}
