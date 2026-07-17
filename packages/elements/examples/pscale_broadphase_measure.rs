//! P-SCALE — EXACT BROADPHASE, the perf evidence.
//!
//! Reproduces the live-frame audit's physics shape (docs/perf/
//! 2026-07-17-live-frame-audit.md): a ~108-particle bonded crate resting on a
//! ~11,684-triangle STATIC collider, 8 substeps, and times ONE `step_profiled`
//! tick median — with the brute-force sweep (BEFORE) and with the conservative
//! broadphase (AFTER). The `collision_static` phase is the one the broadphase
//! touches; `total` is the whole-tick number the live loop pays.
//!
//! HONEST SCOPE: the collider here is a synthetic dense tiled ground sized to
//! the audit's triangle COUNT, not the exact naruko world mesh — the brute
//! cost (108 × tris × 8 substep contact tests) is arrangement-independent, so
//! BEFORE reproduces faithfully; AFTER models the real "resting crate on a big
//! static world" (the crate only ever reaches its local floor tiles, exactly
//! as in the live scene). Run:
//!   cargo run -p elements --release --example pscale_broadphase_measure

use elements::collision::{Collider, ContactMaterial, Triangle};
use elements::{Solver, SolverConfig, Vec3};

/// Dense tiled ground at y=0 over [-half,half]^2, `tiles×tiles×2` triangles,
/// all normal +y.
fn tiled_ground(half: f64, tiles: usize) -> Collider {
    let up = Vec3::new(0.0, 1.0, 0.0);
    let n = tiles.max(1);
    let step = (2.0 * half) / n as f64;
    let mut triangles = Vec::with_capacity(n * n * 2);
    for ix in 0..n {
        for iz in 0..n {
            let x0 = -half + ix as f64 * step;
            let x1 = x0 + step;
            let z0 = -half + iz as f64 * step;
            let z1 = z0 + step;
            let a = Vec3::new(x0, 0.0, z0);
            let b = Vec3::new(x1, 0.0, z0);
            let c = Vec3::new(x1, 0.0, z1);
            let d = Vec3::new(x0, 0.0, z1);
            triangles.push(Triangle::with_normal(a, b, c, up));
            triangles.push(Triangle::with_normal(a, c, d, up));
        }
    }
    Collider {
        triangles,
        material: ContactMaterial::default(),
    }
}

/// The audit scene: a 108-particle bonded crate (6×6×3) resting on the tiled
/// world. `broadphase` selects the sweep.
fn build(broadphase: bool, tiles: usize) -> (Solver, usize) {
    let cfg = SolverConfig {
        dt: 1.0 / 60.0,
        substeps: 8,
        ..SolverConfig::default()
    };
    let mut s = Solver::new(cfg);
    // ~11,684 triangles: 76×77×2 = 11,704 (report the actual count).
    let ground = tiled_ground(60.0, tiles);
    let tri_count = ground.triangles.len();
    s.collider = Some(ground);
    // 6×6×3 = 108 particles, ~1 m crate, resting just above the floor.
    s.spawn_bonded_box(
        Vec3::new(0.0, 0.55, 0.0),
        Vec3::new(1.0, 1.0, 0.5),
        (6, 6, 3),
        400.0,
        0.5,
        1.0e-7,
        0.08,
    );
    s.set_collision_broadphase(broadphase);
    (s, tri_count)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn measure(broadphase: bool, tiles: usize) -> (f64, f64, usize, u64, Option<(usize, (usize, usize, usize), f64)>) {
    let (mut s, tris) = build(broadphase, tiles);
    // Settle to rest, then measure.
    for _ in 0..90 {
        s.step();
    }
    let mut totals = Vec::new();
    let mut statics = Vec::new();
    let mut particles = 0usize;
    for _ in 0..160 {
        let p = s.step_profiled();
        totals.push(p.total.as_secs_f64() * 1e3);
        statics.push(p.collision_static.as_secs_f64() * 1e3);
        particles = p.particles;
    }
    let stats = s.collision_grid_stats();
    (median(totals), median(statics), particles, tris as u64, stats)
}

fn main() {
    let tiles = 76; // 76×76×2 = 11,552 ; bump to hit ~11,684
    let tiles = tiles + 1; // 77×77×2 = 11,858 — near the audit's 11,684

    println!("P-SCALE broadphase perf — 108-particle crate on a big static world, 8 substeps\n");

    let (t_off, s_off, n_off, tris, _) = measure(false, tiles);
    let (t_on, s_on, n_on, _, stats) = measure(true, tiles);

    println!("collider triangles: {tris}   crate particles: {n_off} (on) / {n_on} (off)");
    if let Some((gt, (nx, ny, nz), cell)) = stats {
        println!("broadphase grid: {gt} tris, {nx}x{ny}x{nz} cells @ {cell:.3} m");
    }
    println!();
    println!("phase            BEFORE (brute)      AFTER (broadphase)   speedup");
    println!(
        "collision_static  {:>9.3} ms       {:>9.3} ms        {:>6.1}x",
        s_off,
        s_on,
        if s_on > 0.0 { s_off / s_on } else { f64::INFINITY }
    );
    println!(
        "TOTAL tick        {:>9.3} ms       {:>9.3} ms        {:>6.1}x",
        t_off,
        t_on,
        if t_on > 0.0 { t_off / t_on } else { f64::INFINITY }
    );
    println!();
    // Serial-frame projection (audit's non-tick phases, median ms).
    let non_tick = 0.536 + 0.225 + 0.457 + 10.948 + 0.583; // skin+splice+upload+trace+blit
    println!(
        "projected serial frame with AFTER tick: {:.3} + {:.3} = {:.3} ms  ({:.1} fps)",
        non_tick,
        t_on,
        non_tick + t_on,
        1000.0 / (non_tick + t_on)
    );
    println!(
        "  (BEFORE tick would give {:.3} ms = {:.1} fps)",
        non_tick + t_off,
        1000.0 / (non_tick + t_off)
    );
}
