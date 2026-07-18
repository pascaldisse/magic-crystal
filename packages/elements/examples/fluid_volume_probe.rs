//! FLUID VOLUME PROBE — round-7 follow-up. `fluid_measure` shows the default
//! pool's SURFACE (max-y) dropping from spawn 0.58 m to rest ~0.29 m — a
//! ~50% height loss with the footprint physically capped near its spawn
//! width by the walls (can't spread enough to explain that via footprint
//! growth alone). This probe checks whether that is TRUE volume loss
//! (particles genuinely closer together in 3-D, i.e. clustering/soup) or an
//! artifact of the surface-height metric, by measuring the mean nearest-
//! neighbour distance (a real geometric packing witness, independent of the
//! SPH kernel's smoothed density estimate) at spawn vs at rest.
//!
//! Run: cargo run -p elements --release --example fluid_volume_probe

use elements::fluid::{fill, surface_height, FluidPoolSpec};
use elements::pointgrid::PointGrid;
use elements::Solver;

fn mean_nn_dist(s: &Solver, search_r: f64) -> (f64, f64) {
    let grid = PointGrid::build(&s.particles.pos, &s.fluid_particles, PointGrid::cell_size(search_r));
    let mut cand = Vec::new();
    let mut sum = 0.0_f64;
    let mut n = 0usize;
    let mut worst_min = f64::INFINITY;
    for &i in &s.fluid_particles {
        grid.query_ball(s.particles.pos[i], search_r, &mut cand);
        let mut best = f64::INFINITY;
        for &jc in &cand {
            let j = jc as usize;
            if j == i {
                continue;
            }
            let d = (s.particles.pos[i] - s.particles.pos[j]).length();
            if d < best {
                best = d;
            }
        }
        if best.is_finite() {
            sum += best;
            n += 1;
            worst_min = worst_min.min(best);
        }
    }
    (sum / n.max(1) as f64, worst_min)
}

fn bbox(s: &Solver) -> (elements::Vec3, elements::Vec3) {
    let mut lo = elements::Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut hi = elements::Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &i in &s.fluid_particles {
        let p = s.particles.pos[i];
        lo.x = lo.x.min(p.x);
        lo.y = lo.y.min(p.y);
        lo.z = lo.z.min(p.z);
        hi.x = hi.x.max(p.x);
        hi.y = hi.y.max(p.y);
        hi.z = hi.z.max(p.z);
    }
    (lo, hi)
}

fn main() {
    // argv[1] inner xz, argv[2] fill_height, argv[3] spacing — probe any pool scale.
    let inner: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1.2);
    let fill_h: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0.6);
    let sp_arg: f64 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0.08);
    let spec = FluidPoolSpec { inner: (inner, inner), fill_height: fill_h, spacing: sp_arg, wall_height: (fill_h + 0.3).max(0.6), ..FluidPoolSpec::default() };
    let sp = spec.spacing;
    let mut pool = fill(spec);
    let (nn0, min0) = mean_nn_dist(&pool.solver, sp * 2.0);
    let (lo0, hi0) = bbox(&pool.solver);
    println!(
        "SPAWN: N={}, spacing={sp:.4}, mean NN dist {nn0:.4} m (spawn lattice reference), min NN {min0:.4}, bbox {:.3},{:.3},{:.3} .. {:.3},{:.3},{:.3}",
        pool.fluid.len(), lo0.x, lo0.y, lo0.z, hi0.x, hi0.y, hi0.z
    );

    for t in 0..120 {
        pool.solver.step();
        if t % 20 == 19 {
            let (nn, minn) = mean_nn_dist(&pool.solver, sp * 2.0);
            let (lo, hi) = bbox(&pool.solver);
            println!(
                "tick {:3}: surface {:.4} m, mean NN {:.4} m ({:+.1}% vs spawn), min NN {minn:.4}, bbox y [{:.3},{:.3}], bbox xz [{:.3},{:.3}]x[{:.3},{:.3}]",
                pool.solver.tick, surface_height(&pool), nn, (nn / nn0 - 1.0) * 100.0,
                lo.y, hi.y, lo.x, hi.x, lo.z, hi.z
            );
        }
    }
}
