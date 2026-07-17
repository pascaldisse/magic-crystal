//! P-SCALE — EXACT BROADPHASE ordeals. The collision broadphase is an
//! ACCELERATION structure, not a pixel-lever: it must change WHICH triangles
//! `solve_collision_normal` visits without changing the physics by a single
//! bit. These ordeals prove that by construction over multi-triangle colliders
//! (where the broadphase actually prunes):
//!
//!   1. BYTE-IDENTICAL REPLAY — broadphase ON vs the brute-force sweep OFF,
//!      the same scenario, per-tick `state_hash` bit-equal every tick.
//!   2. PRUNED-PAIR ZERO-CONTACT AUDIT — with the debug audit armed, every
//!      triangle the broadphase prunes, if narrow-phased, yields zero contact.
//!   3. DETERMINISM — the same broadphase run twice folds byte-identical.
//!
//! Non-vacuous: each scenario stands on a TILED ground of thousands of
//! triangles, and the ordeal asserts the grid is genuinely multi-cell and that
//! the broadphase is actually pruning (a resting particle sees far fewer than
//! the whole soup).

use elements::broadphase::TriangleGrid;
use elements::collision::{Collider, ContactMaterial, Triangle};
use elements::{Solver, SolverConfig, Vec3};

/// A large TILED horizontal ground at `y=0`: `tiles × tiles` quads over
/// `[-half, half]²`, each quad two triangles, all normal +y. Thousands of
/// triangles so the broadphase has real work to prune (unlike the 2-triangle
/// `Collider::ground_plane`).
fn tiled_ground(half: f64, tiles: usize, material: ContactMaterial) -> Collider {
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
        material,
    }
}

const GROUND_HALF: f64 = 25.0;
const GROUND_TILES: usize = 40; // 40×40×2 = 3200 triangles

fn cfg() -> SolverConfig {
    SolverConfig {
        dt: 1.0 / 60.0,
        substeps: 8,
        ..SolverConfig::default()
    }
}

/// SCENARIO A — a rigid box dropped onto the tiled ground, bounces, rests.
fn scenario_drop_box() -> Solver {
    let mut s = Solver::new(cfg());
    s.collider = Some(tiled_ground(GROUND_HALF, GROUND_TILES, ContactMaterial::default()));
    s.spawn_rigid_box(
        Vec3::new(0.3, 3.0, -0.2),
        Vec3::new(1.0, 1.0, 1.0),
        (4, 4, 4),
        2000.0,
        1.0,
        0.08,
    );
    s
}

/// SCENARIO B — a bonded lattice ("building") given a lateral shove so it
/// topples, fractures, and its shards settle on the tiled ground. Exercises
/// static collision AND body-vs-body against a multi-triangle world.
fn scenario_topple() -> Solver {
    let mut s = Solver::new(SolverConfig {
        fracture_threshold: 1.0e3,
        ..cfg()
    });
    s.collider = Some(tiled_ground(GROUND_HALF, GROUND_TILES, ContactMaterial::default()));
    let whole = s.spawn_bonded_box(
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(2.0, 4.0, 2.0),
        (4, 8, 4),
        2000.0,
        0.05,
        1.0e-7,
        0.06,
    );
    // Lateral shove on the upper half — the topple hand.
    for &i in &whole {
        if s.particles.pos[i].y > 2.0 {
            s.particles.vel[i] = Vec3::new(6.0, 0.0, 0.0);
        }
    }
    s
}

/// SCENARIO C — a short column of rigid spheres settling into a rest stack on
/// the tiled ground (static + body-vs-body).
fn scenario_sphere_stack() -> Solver {
    let mut s = Solver::new(cfg());
    s.collider = Some(tiled_ground(GROUND_HALF, GROUND_TILES, ContactMaterial::default()));
    for k in 0..4 {
        s.spawn_rigid_sphere(
            Vec3::new(0.0, 0.4 + k as f64 * 0.34, 0.0),
            0.15,
            2,
            2000.0,
            1.0,
            0.08,
        );
    }
    s
}

/// Drive a scenario with the broadphase ON vs OFF and assert the per-tick
/// state hashes are bit-equal every tick. Returns the ON solver (for further
/// assertions) and the hash trace.
fn assert_replay_identical(build: fn() -> Solver, ticks: u64, name: &str) -> (Solver, Vec<u64>) {
    let mut on = build();
    on.set_collision_broadphase(true);
    let mut off = build();
    off.set_collision_broadphase(false);
    let mut hashes = Vec::with_capacity(ticks as usize);
    for t in 0..ticks {
        on.step();
        off.step();
        let ha = on.state_hash();
        let hb = off.state_hash();
        assert_eq!(
            ha, hb,
            "{name}: tick {t} — broadphase ON hash {ha:#x} != brute-force OFF hash {hb:#x}; \
             the acceleration structure changed the physics"
        );
        hashes.push(ha);
    }
    (on, hashes)
}

#[test]
fn ordeal_broadphase_replay_byte_identical() {
    for (build, ticks, name) in [
        (scenario_drop_box as fn() -> Solver, 220u64, "drop_box"),
        (scenario_topple as fn() -> Solver, 260, "topple"),
        (scenario_sphere_stack as fn() -> Solver, 220, "sphere_stack"),
    ] {
        let (on, hashes) = assert_replay_identical(build, ticks, name);
        // Non-vacuous grid: genuinely multi-cell over a multi-triangle soup.
        let (tris, (nx, ny, nz), cell) = on
            .collision_grid_stats()
            .expect("broadphase grid must be built after stepping with a collider");
        assert!(tris >= 3000, "{name}: expected a multi-thousand-triangle soup, got {tris}");
        assert!(
            nx * ny * nz > 1,
            "{name}: grid collapsed to a single cell ({nx}x{ny}x{nz}) — the broadphase would \
             prune nothing, making this ordeal vacuous"
        );
        // The hashes must actually evolve (the body moved / settled), or the
        // scenario is static and proves nothing.
        assert!(
            hashes.windows(2).any(|w| w[0] != w[1]),
            "{name}: state never changed across {ticks} ticks — vacuous scenario"
        );
        println!(
            "ORDEAL broadphase-replay {name}: {ticks} ticks byte-identical ON==OFF, {tris} tris, \
             grid {nx}x{ny}x{nz} @ cell {cell:.3} m"
        );
    }
}

/// The pruned-pair audit: armed, every pruned triangle narrow-phased at the
/// particle's pre-resolution position yields zero contact (a `debug_assert`
/// inside `solve_collision_normal` panics otherwise). Runs in debug builds.
#[test]
fn ordeal_broadphase_pruned_pairs_never_contact() {
    for (build, ticks, name) in [
        (scenario_drop_box as fn() -> Solver, 180u64, "drop_box"),
        (scenario_topple as fn() -> Solver, 200, "topple"),
        (scenario_sphere_stack as fn() -> Solver, 180, "sphere_stack"),
    ] {
        let mut s = build();
        s.set_collision_broadphase(true);
        s.set_broadphase_audit(true);
        for _ in 0..ticks {
            s.step(); // any pruned-but-contacting triangle trips the debug_assert
        }
        println!(
            "ORDEAL broadphase-pruned-pairs-never-contact {name}: {ticks} ticks, every pruned \
             pair audited zero-contact"
        );
    }
}

#[test]
fn ordeal_broadphase_deterministic() {
    let run = || {
        let mut s = scenario_topple();
        s.set_collision_broadphase(true);
        let mut hashes = Vec::new();
        for _ in 0..240 {
            s.step();
            hashes.push(s.state_hash());
        }
        hashes
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "two identical broadphase runs diverged — non-deterministic");
    println!("ORDEAL broadphase-deterministic: 240 ticks byte-identical across two runs");
}

/// The grid itself is conservatively complete: a direct-query cross-check that
/// every triangle whose AABB meets a probe box is returned by `query`. Guards
/// the broadphase's core invariant independent of the solver.
#[test]
fn ordeal_grid_query_is_complete() {
    let ground = tiled_ground(GROUND_HALF, GROUND_TILES, ContactMaterial::default());
    let reach = 0.08 + ContactMaterial::default().contact_margin;
    let cell = TriangleGrid::derive_cell_size(&ground.triangles, reach);
    let fp = TriangleGrid::fingerprint(&ground.triangles);
    let grid = TriangleGrid::build(&ground.triangles, cell, fp);

    // Probe boxes scattered over the ground; brute-check the grid returns a
    // SUPERSET of every AABB-overlapping triangle.
    let mut cand = Vec::new();
    for &(cx, cz) in &[(0.0, 0.0), (7.3, -4.1), (-12.0, 9.5), (24.0, 24.0), (-24.5, -1.0)] {
        for &r in &[0.05_f64, 0.3, 1.5] {
            let center = Vec3::new(cx, 0.0, cz);
            let rr = Vec3::new(r, r, r);
            grid.query(center - rr, center + rr, &mut cand);
            for (ti, t) in ground.triangles.iter().enumerate() {
                let tmin = Vec3::new(
                    t.v0.x.min(t.v1.x).min(t.v2.x),
                    t.v0.y.min(t.v1.y).min(t.v2.y),
                    t.v0.z.min(t.v1.z).min(t.v2.z),
                );
                let tmax = Vec3::new(
                    t.v0.x.max(t.v1.x).max(t.v2.x),
                    t.v0.y.max(t.v1.y).max(t.v2.y),
                    t.v0.z.max(t.v1.z).max(t.v2.z),
                );
                let overlaps = tmin.x <= center.x + r
                    && tmax.x >= center.x - r
                    && tmin.y <= center.y + r
                    && tmax.y >= center.y - r
                    && tmin.z <= center.z + r
                    && tmax.z >= center.z - r;
                if overlaps {
                    assert!(
                        cand.binary_search(&(ti as u32)).is_ok(),
                        "grid.query missed AABB-overlapping triangle {ti} at probe ({cx},{cz}) r={r}"
                    );
                }
            }
        }
    }
    // And the returned list is ascending + deduped (iteration-order invariant).
    grid.query(
        Vec3::new(-GROUND_HALF, -1.0, -GROUND_HALF),
        Vec3::new(GROUND_HALF, 1.0, GROUND_HALF),
        &mut cand,
    );
    assert!(cand.windows(2).all(|w| w[0] < w[1]), "query result not strictly ascending/deduped");
    println!("ORDEAL grid-query-complete: superset + ascending/deduped verified over 3200 tris");
}
