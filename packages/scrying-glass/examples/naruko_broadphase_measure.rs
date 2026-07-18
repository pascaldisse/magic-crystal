//! P-SCALE — EXACT BROADPHASE on the REAL naruko mesh.
//!
//! Closes the adversary's real-mesh gap: `pscale_broadphase_measure` times the
//! broadphase on a SYNTHETIC tiled ground sized to naruko's triangle COUNT.
//! This one loads the ACTUAL `worlds/naruko` static soup (the exact triangles
//! `RenderScene::from_ecs` feeds `Physics::install`) and the ACTUAL declared
//! crate bodies at their settled rest, then:
//!
//!   1. BYTE-IDENTITY — steps two clones of the warmed solver, broadphase ON
//!      vs the brute sweep OFF, and asserts per-tick `state_hash` bit-equal
//!      (the real-mesh analogue of the pscale_broadphase replay ordeal).
//!   2. TICK COST — `step_profiled` median `collision_static` + `total` ms,
//!      brute vs broadphase, on the real resting arrangement. That total is
//!      the Architect's HUD projection.
//!
//! Run: cargo run -p scrying-glass --release --example naruko_broadphase_measure

use std::path::Path;
use std::time::Instant;

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

/// Step `s` `n` times with `step_profiled`, returning median (total, static) ms.
fn measure(s: &mut Solver, n: usize) -> (f64, f64) {
    let mut totals = Vec::with_capacity(n);
    let mut statics = Vec::with_capacity(n);
    for _ in 0..n {
        let p = s.step_profiled();
        totals.push(p.total.as_secs_f64() * 1e3);
        statics.push(p.collision_static.as_secs_f64() * 1e3);
    }
    (median(totals), median(statics))
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

    let tris = base
        .collider
        .as_ref()
        .map(|c| c.triangles.len())
        .unwrap_or(0);
    let particles = base.particles.pos.len();

    // Two warmed clones: broadphase ON vs brute OFF, same start state.
    let mut on = base.clone();
    on.set_collision_broadphase(true);
    on.build_collision_grid();
    let mut off = base.clone();
    off.set_collision_broadphase(false);

    let stats = on.collision_grid_stats();

    println!("P-SCALE broadphase on the REAL naruko mesh\n");
    println!("collider triangles: {tris}   solver particles: {particles}");
    if let Some((gt, (nx, ny, nz), cell)) = stats {
        println!("broadphase grid: {gt} tris, {nx}x{ny}x{nz} cells @ {cell:.3} m");
    }

    // ── 1. BYTE-IDENTITY over the real mesh ────────────────────────────────
    let mut diverged: Option<u64> = None;
    for tick in 1..=240u64 {
        on.step();
        off.step();
        if on.state_hash() != off.state_hash() && diverged.is_none() {
            diverged = Some(tick);
        }
    }
    match diverged {
        None => println!("\nBYTE-IDENTITY: 240 ticks ON==OFF bit-equal on the real naruko soup ✓"),
        Some(t) => {
            println!("\nBYTE-IDENTITY: DIVERGED at tick {t} — broadphase != brute on real mesh ✗");
            std::process::exit(1);
        }
    }

    // ── 2. TICK COST — brute vs broadphase on the real resting scene ───────
    // Fresh warmed clones so the timing state matches (post-settle rest).
    let mut on_t = base.clone();
    on_t.set_collision_broadphase(true);
    on_t.build_collision_grid();
    let mut off_t = base.clone();
    off_t.set_collision_broadphase(false);
    // brief settle so both are at steady rest before timing
    for _ in 0..30 {
        on_t.step();
        off_t.step();
    }
    let t0 = Instant::now();
    let (t_on, s_on) = measure(&mut on_t, 160);
    let (t_off, s_off) = measure(&mut off_t, 160);
    let _ = t0;

    println!("\nphase            BEFORE (brute)      AFTER (broadphase)   speedup");
    println!(
        "collision_static  {:>9.3} ms       {:>9.3} ms        {:>6.1}x",
        s_off,
        s_on,
        if s_on > 0.0 {
            s_off / s_on
        } else {
            f64::INFINITY
        }
    );
    println!(
        "TOTAL tick        {:>9.3} ms       {:>9.3} ms        {:>6.1}x",
        t_off,
        t_on,
        if t_on > 0.0 {
            t_off / t_on
        } else {
            f64::INFINITY
        }
    );
    println!(
        "\nREAL-MESH HUD tick: {:.3} ms broadphase  (was {:.3} ms brute)",
        t_on, t_off
    );
}
