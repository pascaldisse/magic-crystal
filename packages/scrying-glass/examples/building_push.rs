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

const SETTLE_TICKS: u64 = 40;
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

/// Live top-of-tower height (max world-y of any still-standing particle in
/// the tower's lattice group) — the honest "does it stand" measure. Unlike
/// `group_centroid` (which drops anchored/inv_mass==0 particles from its
/// weighted average, so it reads ~0.33 m ABOVE the authored 3.3 m centroid
/// even at tick 0, before any physics runs — an accounting artifact, not
/// movement), the top height starts at exactly the authored span top
/// (`3.3 + 6.6/2 = 6.6`) and only drops if the structure actually sags.
fn top_y(scene: &RenderScene) -> f64 {
    let physics = scene.physics().expect("physics");
    let solver = physics.solver();
    let group = solver.bonded_groups.iter().find(|g| g.len() == 6 * 11 * 6).expect("tower group (N=396)");
    group.iter().map(|&i| solver.particles.pos[i].y).fold(f64::NEG_INFINITY, f64::max)
}

fn main() {
    // ── CONTROL — settle only, no push: the structure must stand ─────────
    let mut scene = build_scene();
    let spawn_top = top_y(&scene);
    println!("[control] spawn top y = {spawn_top:.3} (authored top = 6.6)");
    for t in 0..SETTLE_TICKS {
        scene.tick();
        if t % 5 == 0 {
            println!("    t={t} {TOWER} whole={} centroid={:?} top_y={:.3}", has_target(&scene, TOWER), centroid(&scene, TOWER), top_y(&scene));
        }
    }
    let whole = has_target(&scene, TOWER);
    let rest_centroid = centroid(&scene, TOWER).expect("tower centroid at rest");
    let rest_top = top_y(&scene);
    println!("[control] tower whole after settle = {whole}, centroid = {rest_centroid:?}, top_y = {rest_top:.3} (authored [8,3.3,33], top 6.6)");
    assert!(whole, "the tower must stand on its own weight before any push (raise `love` / lower `density`)");
    assert!(
        (rest_top - spawn_top).abs() < 0.3,
        "the tower top must hold near its spawn height at rest (no self-weight sag), spawn={spawn_top:.3} rest={rest_top:.3}"
    );

    // ── THE PUSH — a player standing south of the tower (2 m off, within
    // the window's REACH=4.0), aiming AT the tower's live rest centroid
    // (~3.6 m up — a real player looks UP at a 3-storey tower, not level
    // from the ground). Computed from the measured `rest_centroid`, never
    // hardcoded against a stale authored height. ───────────────────────────
    let eye = [8.0, 1.6, 35.0];
    let target = glam::Vec3::from_array(rest_centroid.map(|v| v as f32));
    let raw_dir = target - glam::Vec3::from_array(eye);
    let dir = [raw_dir.x, raw_dir.y, raw_dir.z];
    println!("[push] eye={eye:?} dir={dir:?} (aimed at rest centroid {rest_centroid:?})");
    let op = pick(&scene, eye, dir).expect("ray must pick the tower");
    let picked = match &op {
        Op::Impulse(i) => i.id.clone(),
        _ => unreachable!(),
    };
    println!("[push] ray picked {picked:?}");
    assert_eq!(picked, TOWER, "ray must select the tower");
    scene.tick_with_ops(&[op]);

    // `min_top_y` tracks `top_y()` (live max particle height of the tower's
    // ORIGINAL lattice group), NOT `centroid()` — `centroid` reads through
    // `push_targets()`, which drops out the instant the whole-body target
    // breaks (fragments are no longer "the tower"), so a centroid-based
    // tracker freezes at the break tick and can never see the actual storey
    // drop that follows. `top_y` stays valid post-break (it walks the same
    // particle indices regardless of fragment status), so it is the honest
    // basis for a progressive-collapse number.
    let mut min_top_y = rest_top;
    let mut break_tick: Option<u64> = None;
    let mut entity_history: Vec<(u64, usize)> = Vec::new();
    let mut prev_entities = 0usize;
    for i in 0..AFTER_TICKS {
        scene.tick();
        min_top_y = min_top_y.min(top_y(&scene));
        let whole_now = has_target(&scene, TOWER);
        if break_tick.is_none() && !whole_now {
            break_tick = Some(i);
        }
        let entities = scene.dynamics.entities().len();
        if entities != prev_entities {
            entity_history.push((i, entities));
            prev_entities = entities;
        }
        if i % 150 == 0 {
            let feet = min_feet_y(&scene);
            println!("        tick {i}: tower whole={whole_now} top_y={:.3} min feet y={feet:.3} dynamic entities={entities}", top_y(&scene));
        }
    }
    // Diagnostic (not an assertion): the y-height of the first N fractured
    // bonds' particles — evidence for WHERE the break started (ground-floor
    // bonds first vs. an arbitrary layer), read straight off the solver's
    // fracture journal (never re-derived).
    {
        let physics = scene.physics().expect("physics");
        let solver = physics.solver();
        let sample: Vec<(u64, f64, f64)> = solver
            .fractures
            .iter()
            .take(12)
            .map(|f| (f.tick, solver.particles.pos[f.a].y, solver.particles.pos[f.b].y))
            .collect();
        println!("[push] first fractures (tick, y_a, y_b), base row ~ y=0.0..0.6: {sample:?}");
        println!("[push] total fracture events = {}", solver.fractures.len());
    }

    let broken = !has_target(&scene, TOWER);
    let entities_after = scene.dynamics.entities().len();
    let feet = min_feet_y(&scene);
    let collapse = rest_top - min_top_y;
    println!("[push] tower broken = {broken}, first break tick = {break_tick:?}");
    println!("[push] dynamic entities after = {entities_after}");
    println!("[push] entity-count history (tick, count) at each change: {entity_history:?}");
    println!("[push] min feet y (final, debris settle floor) = {feet:.3}");
    println!("[push] top_y: rest {rest_top:.3} -> min {min_top_y:.3} m (collapse drop {collapse:.3} m)");

    assert!(broken, "the pushed load-bearing structure must fracture (its whole-body target must vanish)");
    assert!(entities_after >= 2, "shatter must birth >= 2 fragment vessels, got {entities_after}");
    assert!(feet > -1.0, "no debris may sink far through the ground, min feet {feet:.3}");
    assert!(
        collapse > 0.5,
        "a load-bearing tower sheared at its anchored base must show a real storey drop (>0.5 m), got {collapse:.3} m"
    );

    println!("\nBUILDING PUSH PROOF PASSED — the anchored ground-floor bonds shear under the push, the structure fractures and drops {collapse:.3} m.");
}
