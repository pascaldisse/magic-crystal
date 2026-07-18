//! SHIFT 16 — INSTRUMENT INSIDE solver_step: the sub-table.
//!
//! N0.k named `elements::Solver::step()` the 5.36ms thief but stopped at the
//! door. This splits ONE tick of the REAL settled naruko solver to the leaf —
//! integrate vs solve_distance vs shape_matching vs collision_static vs
//! collision_body (the O(n²) pass) vs cluster_floodfill vs velocity_passes vs
//! fracture — via the already-existing `step_profiled` (`PhaseProfile`), plus
//! the body/particle/pair COUNTS that drive each cost. The N0.k lesson twice
//! over: never cut before the sub-table says WHERE.
//!
//! With `GAIA_NATIVE_SLEEP=1` (+ optional GAIA_NATIVE_SLEEP_VEL /
//! GAIA_NATIVE_SLEEP_FRAMES) it also prints the sleep A/B: same warmed solver,
//! sleep OFF vs ON, median per-phase ms + asleep-island/particle counts.
//!
//! Run: cargo run -p scrying-glass --release --example naruko_solver_substages

use std::path::Path;

use crystal::{Core, load_world_dir};
use elements::Solver;
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults};

fn naruko_params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: 60.0,
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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct SubTable {
    integrate: f64,
    solve_distance: f64,
    shape_matching: f64,
    collision_static: f64,
    collision_body: f64,
    cluster_floodfill: f64,
    velocity_passes: f64,
    fracture_pass: f64,
    total: f64,
    particles: usize,
    bonds: usize,
    clustered: usize,
    pair_checks: u64,
    asleep_particles: usize,
    asleep_islands: usize,
}

fn measure(s: &mut Solver, n: usize) -> SubTable {
    let mut integ = Vec::new();
    let mut sdist = Vec::new();
    let mut shape = Vec::new();
    let mut cstat = Vec::new();
    let mut cbody = Vec::new();
    let mut flood = Vec::new();
    let mut vel = Vec::new();
    let mut frac = Vec::new();
    let mut total = Vec::new();
    let mut last = elements::PhaseProfile::default();
    for _ in 0..n {
        let p = s.step_profiled();
        integ.push(p.integrate.as_secs_f64() * 1e3);
        sdist.push(p.solve_distance.as_secs_f64() * 1e3);
        shape.push(p.shape_matching.as_secs_f64() * 1e3);
        cstat.push(p.collision_static.as_secs_f64() * 1e3);
        cbody.push(p.collision_body.as_secs_f64() * 1e3);
        flood.push(p.cluster_floodfill.as_secs_f64() * 1e3);
        vel.push(p.velocity_passes.as_secs_f64() * 1e3);
        frac.push(p.fracture_pass.as_secs_f64() * 1e3);
        total.push(p.total.as_secs_f64() * 1e3);
        last = p;
    }
    let (ap, ai) = s.sleep_counts();
    SubTable {
        integrate: median(integ),
        solve_distance: median(sdist),
        shape_matching: median(shape),
        collision_static: median(cstat),
        collision_body: median(cbody),
        cluster_floodfill: median(flood),
        velocity_passes: median(vel),
        fracture_pass: median(frac),
        total: median(total),
        particles: last.particles,
        bonds: last.bonds,
        clustered: last.clustered_particles,
        pair_checks: last.body_pair_checks,
        asleep_particles: ap,
        asleep_islands: ai,
    }
}

fn print_table(label: &str, t: &SubTable) {
    println!("\n── {label} ──");
    println!("  particles {}   bonds {}   clustered {}   O(n²) pair-checks/tick {}",
        t.particles, t.bonds, t.clustered, t.pair_checks);
    println!("  asleep: {} particles across {} islands", t.asleep_particles, t.asleep_islands);
    println!("  phase              median ms");
    println!("  integrate          {:>9.4}", t.integrate);
    println!("  solve_distance     {:>9.4}", t.solve_distance);
    println!("  shape_matching     {:>9.4}", t.shape_matching);
    println!("  collision_static   {:>9.4}", t.collision_static);
    println!("  collision_body     {:>9.4}", t.collision_body);
    println!("  cluster_floodfill  {:>9.4}", t.cluster_floodfill);
    println!("  velocity_passes    {:>9.4}", t.velocity_passes);
    println!("  fracture_pass      {:>9.4}", t.fracture_pass);
    println!("  ── TOTAL           {:>9.4}", t.total);
}

fn main() {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();
    let mut scene =
        RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("render scene");

    // Settle the crates on the REAL mesh — the resting arrangement the live
    // loop pays for.
    for _ in 0..120 {
        scene.tick();
    }

    let base = scene
        .physics()
        .expect("naruko declares physics bodies")
        .solver()
        .clone();

    let tris = base.collider.as_ref().map(|c| c.triangles.len()).unwrap_or(0);
    println!("SHIFT 16 — solver_step SUB-TABLE on the REAL settled naruko");
    println!("collider triangles {tris}   rigids {}   bonded-lattices n/a", base.rigids.len());

    // BEFORE — sleep OFF (byte-unchanged default). Fresh warmed clone, brief
    // re-settle, then time.
    let mut off = base.clone();
    off.set_sleep(false);
    for _ in 0..30 { off.step(); }
    let before = measure(&mut off, 240);
    print_table("BEFORE — sleep OFF (N0.k baseline)", &before);

    // AFTER — sleep ON. Same warmed start; let islands settle & sleep, then time.
    let vel = std::env::var("GAIA_NATIVE_SLEEP_VEL").ok().and_then(|s| s.parse().ok());
    let frames = std::env::var("GAIA_NATIVE_SLEEP_FRAMES").ok().and_then(|s| s.parse().ok());
    let mut on = base.clone();
    on.set_sleep(true);
    if let Some(v) = vel { on.config.sleep_vel = v; }
    if let Some(f) = frames { on.config.sleep_frames = f; }
    for _ in 0..120 { on.step(); } // let quiet counters cross the threshold
    let after = measure(&mut on, 240);
    print_table(
        &format!("AFTER — sleep ON (vel {:.4} frames {})", on.config.sleep_vel, on.config.sleep_frames),
        &after,
    );

    let spd = if after.total > 0.0 { before.total / after.total } else { f64::INFINITY };
    println!("\nsolver_step TOTAL: {:.4} ms → {:.4} ms   ({:.2}x)", before.total, after.total, spd);
}
