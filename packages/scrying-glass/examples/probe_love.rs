//! PROBE (throwaway tuning aid) — sweep the bonded break-crate's `love`:
//! load the real naruko realm, settle, report whether the crate survives
//! settling WHOLE, then fire the exact window push door and report the
//! fragment count. Non-asserting — prints numbers so a `love` sweep can find
//! the value that rests whole yet shatters on a hard push. Reads the live
//! `love` from the scene JSON (edit it between runs).
//!
//!   cargo run -p scrying-glass --release --example probe_love

use std::path::Path;

use crystal::{EcsWorld, ImpulseOp, Op, load_world_dir};
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults};

const BONDED: &str = "playground_break_crate";
const SETTLE_TICKS: u64 = 150;
const AFTER_TICKS: u64 = 600;
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

fn has_target(scene: &RenderScene, id: &str) -> bool {
    scene
        .physics()
        .map(|p| p.push_targets().iter().any(|(g, _)| g == id))
        .unwrap_or(false)
}

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

fn main() {
    let mut scene = build_scene();
    for _ in 0..SETTLE_TICKS {
        scene.tick();
    }
    let whole_after_settle = has_target(&scene, BONDED);
    let entities_before = scene.dynamics.entities().len();
    println!("[probe] whole after {SETTLE_TICKS} settle ticks = {whole_after_settle}");

    if !whole_after_settle {
        println!("[probe] VERDICT: love too LOW — crate breaks at rest. Raise love.");
        return;
    }

    let eye = [-0.8, 1.7, 37.3];
    let dir = [0.0, -0.40, -0.91];
    match pick(&scene, eye, dir) {
        Some(op) => {
            scene.tick_with_ops(&[op]);
            for _ in 0..AFTER_TICKS {
                scene.tick();
            }
            let still_whole = has_target(&scene, BONDED);
            let fragments = scene.dynamics.entities().len() as i64 - entities_before as i64;
            println!("[probe] after push: still_whole={still_whole} fragments=+{fragments}");
            if still_whole {
                println!("[probe] VERDICT: love too HIGH — push does not shatter. Lower love.");
            } else if fragments >= 2 {
                println!("[probe] VERDICT: GOOD — rests whole, shatters into >=2 fragments.");
            } else {
                println!("[probe] VERDICT: marginal — broke but <2 fragments.");
            }
        }
        None => println!("[probe] ray missed the crate"),
    }
}
