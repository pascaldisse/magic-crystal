//! ADVERSARY vii1 (nari) — three-wall push chain, forced small cells.
//! Each hop is up to 2r; three walls make total displacement ~6r, far past
//! any fixed per-hop budget. Cell size 0.6 separates the walls into distinct
//! cells (w1 ix=0, w2 ix=1, w3 ix=3), so only the fixpoint re-query can
//! reach walls 2 and 3. Brute vs broadphase state_hash per tick.

use elements::collision::{Collider, ContactMaterial, Triangle};
use elements::{Solver, SolverConfig, Vec3};

fn wall(x: f64, side: f64) -> Vec<Triangle> {
    let h = side / 2.0;
    let n = Vec3::new(1.0, 0.0, 0.0);
    let a = Vec3::new(x, -h, -h);
    let b = Vec3::new(x, h, -h);
    let c = Vec3::new(x, h, h);
    let d = Vec3::new(x, -h, h);
    vec![
        Triangle::with_normal(a, b, c, n),
        Triangle::with_normal(a, c, d, n),
    ]
}

#[test]
fn adv_chain_three_walls() {
    let build = |broadphase: bool| -> Solver {
        let mut s = Solver::new(SolverConfig::default());
        let mut triangles = wall(0.5, 0.6);
        triangles.extend(wall(1.5, 0.6));
        triangles.extend(wall(2.5, 0.6));
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
        "BROADPHASE != BRUTE: diverged at tick {} — fixpoint failed the chain",
        diverged.unwrap()
    );
}
