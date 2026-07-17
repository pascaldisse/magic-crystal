//! ADVERSARY — margin hunt on the broadphase query reach.
//!
//! The query reach is `radius + travel + radius`: contact radius, measured
//! travel, ONE push-chain hop of size `radius`. But `Triangle::contact_depth`
//! is a TWO-SIDED shell (`signed ∈ (−r, r)`, depth = `r − signed`), so a
//! single push can be up to `2r` — twice the budgeted hop. A particle just
//! behind wall A's plane is pushed `≈2r` forward, landing within `r` of a
//! wall C the grid never returned. Brute contacts C; broadphase cannot.

use elements::broadphase::TriangleGrid;
use elements::collision::{Collider, ContactMaterial, Triangle};
use elements::{Solver, SolverConfig, Vec3};

/// A square wall tile in the y-z plane at `x`, side `side`, normal +x,
/// centered at (x, cy, cz). Two triangles.
fn wall(x: f64, cy: f64, cz: f64, side: f64) -> Vec<Triangle> {
    let h = side / 2.0;
    let n = Vec3::new(1.0, 0.0, 0.0);
    let a = Vec3::new(x, cy - h, cz - h);
    let b = Vec3::new(x, cy + h, cz - h);
    let c = Vec3::new(x, cy + h, cz + h);
    let d = Vec3::new(x, cy - h, cz + h);
    vec![
        Triangle::with_normal(a, b, c, n),
        Triangle::with_normal(a, c, d, n),
    ]
}

/// Two-sided-shell overshoot: wall A at x=0.5 (particle signed = −0.5, just
/// inside the BACK of the shell, r_eff = 0.501), wall C at x=1.3. Tile side
/// 0.7014 forces grid cell = 0.7014 (mean extent > reach), so the query box
/// [−1.002, 1.002] covers cell ix=0 only; wall C bins at ix=1. Brute: A
/// pushes depth 1.001 → center x = 1.001 → C signed = −0.299 → C pushes.
/// Broadphase: C never visited. Divergence expected at tick 1.
#[test]
fn adv_two_sided_overshoot() {
    let build = |broadphase: bool| -> Solver {
        let mut s = Solver::new(SolverConfig::default());
        let mut triangles = wall(0.5, 0.0, 0.0, 0.7014);
        triangles.extend(wall(1.3, 0.0, 0.0, 0.7014));
        s.collider = Some(Collider {
            triangles,
            material: ContactMaterial::default(),
        });
        s.particles.add_with_radius(Vec3::new(0.0, 0.0, 0.0), 1.0, 0.5);
        s.set_collision_broadphase(broadphase);
        s
    };
    let mut on = build(true);
    let mut off = build(false);
    on.build_collision_grid();
    let stats = on.collision_grid_stats().expect("grid built");
    println!("grid stats (tris, (nx,ny,nz), cell): {stats:?}");

    let mut diverged = None;
    for tick in 1..=10 {
        on.step();
        off.step();
        let (h_on, h_off) = (on.state_hash(), off.state_hash());
        println!(
            "tick {tick}: on={:016x} off={:016x} pos_on={:?} pos_off={:?}",
            h_on, h_off, on.particles.pos[0], off.particles.pos[0]
        );
        if h_on != h_off && diverged.is_none() {
            diverged = Some(tick);
        }
    }
    assert!(
        diverged.is_none(),
        "BROADPHASE != BRUTE: state_hash diverged at tick {} — query reach \
         does not cover the two-sided contact shell's 2r push",
        diverged.unwrap()
    );
}

/// Grid completeness under edge cases: a triangle whose AABB min sits EXACTLY
/// on a cell boundary, a long sliver spanning many cells diagonally, and a
/// degenerate zero-area sliver. For 200 deterministic query AABBs, the grid's
/// answer must be a superset of the brute AABB-overlap set.
#[test]
fn adv_grid_boundary_superset() {
    let mut tris: Vec<Triangle> = Vec::new();
    // Base: small tiles so the cell is small and boundaries are dense.
    for i in 0..20 {
        for j in 0..20 {
            let x0 = i as f64 * 0.5;
            let z0 = j as f64 * 0.5;
            tris.push(Triangle::new(
                Vec3::new(x0, 0.0, z0),
                Vec3::new(x0 + 0.5, 0.0, z0),
                Vec3::new(x0 + 0.5, 0.0, z0 + 0.5),
            ));
        }
    }
    // (a) AABB min exactly on a cell boundary (origin is world min = 0).
    tris.push(Triangle::new(
        Vec3::new(2.0, 0.0, 2.0),
        Vec3::new(2.5, 1.0, 2.0),
        Vec3::new(2.0, 1.0, 2.5),
    ));
    // (b) long sliver spanning many cells diagonally.
    tris.push(Triangle::new(
        Vec3::new(0.1, 0.05, 0.1),
        Vec3::new(9.7, 0.06, 9.6),
        Vec3::new(9.7, 0.07, 9.7),
    ));
    // (c) degenerate zero-area sliver (collinear).
    tris.push(Triangle::new(
        Vec3::new(1.0, 0.5, 1.0),
        Vec3::new(3.0, 0.5, 3.0),
        Vec3::new(5.0, 0.5, 5.0),
    ));

    let aabb = |t: &Triangle| -> (Vec3, Vec3) {
        (
            Vec3::new(
                t.v0.x.min(t.v1.x).min(t.v2.x),
                t.v0.y.min(t.v1.y).min(t.v2.y),
                t.v0.z.min(t.v1.z).min(t.v2.z),
            ),
            Vec3::new(
                t.v0.x.max(t.v1.x).max(t.v2.x),
                t.v0.y.max(t.v1.y).max(t.v2.y),
                t.v0.z.max(t.v1.z).max(t.v2.z),
            ),
        )
    };

    let fp = TriangleGrid::fingerprint(&tris);
    let cell = TriangleGrid::derive_cell_size(&tris, 0.501);
    let grid = TriangleGrid::build(&tris, cell, fp);
    println!("cell={} res={:?}", grid.cell_size(), grid.resolution());

    // Deterministic LCG.
    let mut x: u64 = 0x9E3779B97F4A7C15;
    let mut rng = move || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::new();
    let mut violations = 0usize;
    for q in 0..200 {
        let cx = rng() * 12.0 - 1.0;
        let cy = rng() * 3.0 - 1.0;
        let cz = rng() * 12.0 - 1.0;
        let hx = rng() * 2.0 + 1e-6;
        let hy = rng() * 2.0 + 1e-6;
        let hz = rng() * 2.0 + 1e-6;
        let qmin = Vec3::new(cx - hx, cy - hy, cz - hz);
        let qmax = Vec3::new(cx + hx, cy + hy, cz + hz);
        grid.query(qmin, qmax, &mut out);
        for (ti, t) in tris.iter().enumerate() {
            let (lo, hi) = aabb(t);
            let overlaps = lo.x <= qmax.x
                && hi.x >= qmin.x
                && lo.y <= qmax.y
                && hi.y >= qmin.y
                && lo.z <= qmax.z
                && hi.z >= qmin.z;
            if overlaps && out.binary_search(&(ti as u32)).is_err() {
                println!("VIOLATION query {q}: tri {ti} overlaps but not returned");
                violations += 1;
            }
        }
    }
    assert_eq!(violations, 0, "grid query dropped overlapping triangles");
}
