//! PLAYGROUND PROOF — the Architect's hands, offline. Loads the real naruko
//! realm (the delivered scene data), settles the plaza toys, then drives the
//! EXACT push door the window key rides: a view-ray pick over
//! `Physics::push_targets()` → one `Op::Impulse` → `RenderScene::tick_with_ops`
//! (the same solver seam an agent op would take — no test-only physics path).
//!
//!   1. THE STACK topples: aim at the 5-crate tower, shove, and the pushed
//!      crate tilts far past level and the stack scatters — crates settle
//!      without sinking through the pier.
//!   2. THE BONDED crate shatters: aim at it, shove hard, its bonds tear and
//!      it fractures into MULTIPLE fragment vessels (the dynamic entity count
//!      jumps; the whole-body target disappears — the oracle now sees shards).
//!
//! No GPU: this is the physics seam only. Run:
//!   cargo run -p scrying-glass --release --example playground_push

use std::path::Path;

use crystal::{EcsWorld, ImpulseOp, Op, load_world_dir};
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults};

const STACK_TOP: &str = "playground_stack_4";
const BONDED: &str = "playground_break_crate";
const SETTLE_TICKS: u64 = 150;
const AFTER_TICKS: u64 = 600;

/// Push dials — the window's defaults (GAIA_PUSH_REACH / GAIA_PUSH_SPEED).
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

/// The window's picker, verbatim (main.rs `build_push_ops`): the nearest
/// pushable body a view ray from `eye` along `dir` is aimed at, within reach.
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

fn tilt_of(scene: &RenderScene, id: &str) -> f64 {
    let physics = scene.physics().expect("physics");
    let Some(binding) = physics.bindings().iter().find(|b| b.gaia_id == id) else {
        return f64::NAN; // gone (e.g. fractured)
    };
    let pose = physics.pose(binding);
    let up = pose.rotation_columns[1];
    let len = up.iter().map(|c| c * c).sum::<f64>().sqrt();
    (up[1] / len).clamp(-1.0, 1.0).acos()
}

fn has_target(scene: &RenderScene, id: &str) -> bool {
    scene
        .physics()
        .map(|p| p.push_targets().iter().any(|(g, _)| g == id))
        .unwrap_or(false)
}

fn min_feet_y(scene: &RenderScene) -> f64 {
    let physics = scene.physics().expect("physics");
    physics
        .bindings()
        .iter()
        .map(|b| physics.pose(b).position[1] - 0.4) // half-height 0.4
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    // ── TOY 1 — THE STACK TOPPLES ────────────────────────────────────────
    let mut scene = build_scene();
    for _ in 0..SETTLE_TICKS {
        scene.tick();
    }
    let top_rest = scene.body_position(STACK_TOP).expect("stack top");
    let tilt_rest = tilt_of(&scene, STACK_TOP);

    // A player standing 4 m south of the tower, looking at it (level gaze).
    let eye = [-3.0, 1.7, 39.0];
    let dir = [0.0, 0.05, -1.0];
    let op = pick(&scene, eye, dir).expect("ray must pick a stack crate");
    let picked = match &op {
        Op::Impulse(i) => i.id.clone(),
        _ => unreachable!(),
    };
    println!("[stack] ray picked {picked:?}, impulse fired");
    scene.tick_with_ops(&[op]);
    let mut max_tilt: f64 = tilt_rest;
    for _ in 0..AFTER_TICKS {
        scene.tick();
        max_tilt = max_tilt.max(tilt_of(&scene, STACK_TOP));
    }
    let top_now = scene.body_position(STACK_TOP).expect("stack top after");
    let drop = top_rest[1] - top_now[1];
    let displaced = ((top_now[0] - top_rest[0]).powi(2) + (top_now[2] - top_rest[2]).powi(2)).sqrt();
    let feet = min_feet_y(&scene);
    println!(
        "[stack] tilt_rest={tilt_rest:.3} max_tilt={max_tilt:.3} rad ({:.0}°); top crate fell {drop:.2} m (xz {displaced:.2} m); min feet y={feet:.3}",
        max_tilt.to_degrees()
    );
    for i in 0..5 {
        let id = format!("playground_stack_{i}");
        let p = scene.body_position(&id).expect("crate");
        println!("        {id}: pos=[{:.2},{:.2},{:.2}] tilt={:.0}°", p[0], p[1], p[2], tilt_of(&scene, &id).to_degrees());
    }
    assert!(picked.starts_with("playground_stack_"), "picked the tower");
    assert!(max_tilt > 0.6, "the tower must topple (max tilt > 0.6 rad), got {max_tilt:.3}");
    assert!(drop > 1.0, "the top crate must fall (> 1 m drop), got {drop:.2} m");
    assert!(feet > -0.2, "no crate may sink through the pier, min feet {feet:.3}");

    // ── TOY 2 — THE BONDED CRATE SHATTERS ────────────────────────────────
    let mut scene = build_scene();
    for _ in 0..SETTLE_TICKS {
        scene.tick();
    }
    let entities_before = scene.dynamics.entities().len();
    let whole_before = has_target(&scene, BONDED);
    println!("[bonded] whole after settle (before push)={whole_before}");
    assert!(whole_before, "bonded crate must survive settling WHOLE before the push (raise `love`)");

    // The crate sits on the pier at eye-below level, so the player looks DOWN
    // at it to punch it (a natural aim at a floor-level box).
    let eye = [-0.8, 1.7, 37.3];
    let dir = [0.0, -0.40, -0.91];
    let op = pick(&scene, eye, dir).expect("ray must pick the bonded crate");
    let picked = match &op {
        Op::Impulse(i) => i.id.clone(),
        _ => unreachable!(),
    };
    println!("[bonded] ray picked {picked:?}, hard shove fired");
    assert_eq!(picked, BONDED, "ray must select the bonded crate");
    scene.tick_with_ops(&[op]);
    for _ in 0..AFTER_TICKS {
        scene.tick();
    }
    let entities_after = scene.dynamics.entities().len();
    let still_whole = has_target(&scene, BONDED);
    let fragments = entities_after as i64 - entities_before as i64;
    println!(
        "[bonded] whole-body target present after={still_whole}; dynamic entities {entities_before} -> {entities_after} (+{fragments} fragments)"
    );
    assert!(!still_whole, "the bonded crate must break (its whole-body target must vanish)");
    assert!(fragments >= 2, "shatter must birth >= 2 fragment vessels, got +{fragments}");

    println!("\nPLAYGROUND PROOF PASSED — the stack topples, the bonded crate shatters, all on the Op::Impulse door.");
}
