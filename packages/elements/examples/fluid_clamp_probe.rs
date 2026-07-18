//! FLUID CLAMP HUNT — minimal reproduction of the compression-only explosion.
//!
//! Round-4 diagnostic: compression_only=TRUE explodes, compression_only=FALSE
//! domes. The physics read says compression-only CANNOT explode (it only ever
//! pushes apart) — so an exploding compression-only is an IMPLEMENTATION BUG.
//!
//! This probe isolates the culprit. Two axes:
//!   A) pool-scale A/B: compression_only=true with tensile_k on vs off.
//!   B) a 3x3x3 mini-cube: settle under gravity, watch it stay bounded/flat.
//!
//! Run: cargo run -p elements --release --example fluid_clamp_probe

use elements::fluid::{fill, surface_height, FluidPoolSpec};
use elements::fluid_kernel::poly6;
use elements::pointgrid::PointGrid;
use elements::Solver;

fn max_speed(s: &Solver) -> f64 {
    s.fluid_particles
        .iter()
        .map(|&i| s.particles.vel[i].length())
        .fold(0.0, f64::max)
}

fn max_overdensity(s: &Solver) -> f64 {
    let cfg = s.fluid.unwrap();
    let (h, rho0) = (cfg.h, cfg.rest_density);
    let grid = PointGrid::build(&s.particles.pos, &s.fluid_particles, PointGrid::cell_size(h));
    let mut cand = Vec::new();
    let mut worst = f64::NEG_INFINITY;
    for &i in &s.fluid_particles {
        grid.query_ball(s.particles.pos[i], h, &mut cand);
        let mut density = 0.0;
        for &jc in &cand {
            let j = jc as usize;
            let mj = 1.0 / s.particles.inv_mass[j];
            density += mj * poly6((s.particles.pos[i] - s.particles.pos[j]).length(), h);
        }
        worst = worst.max(density / rho0 - 1.0);
    }
    worst
}

/// Surface flatness: stddev of the top-layer particle heights (proxy: report
/// the surface height and the spread as (max - median) of the highest decile).
fn surface_spread(s: &Solver) -> (f64, f64) {
    let mut ys: Vec<f64> = s.fluid_particles.iter().map(|&i| s.particles.pos[i].y).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let top = &ys[ys.len() * 9 / 10..];
    let max = *top.last().unwrap();
    let med = top[top.len() / 2];
    (max, max - med)
}

fn run_pool_cfg(label: &str, tensile_k: f64, compression_only: bool, iters: usize, relax: f64, ticks: u64) {
    let spec = FluidPoolSpec { spacing: 0.08, ..FluidPoolSpec::default() };
    let mut pool = fill(spec);
    {
        let cfg = pool.solver.fluid.as_mut().unwrap();
        cfg.tensile_k = tensile_k;
        cfg.compression_only = compression_only;
        cfg.solver_iterations = iters;
        cfg.relax = relax;
    }
    let h0 = surface_height(&pool);
    println!(
        "\n== {label} ==  (compression_only={compression_only}, tensile_k={tensile_k}, iters={iters}, relax={relax})"
    );
    println!("  spawn surface {h0:.4} m, N={}", pool.fluid.len());
    let mut exploded = false;
    for t in 0..ticks {
        pool.solver.step();
        let sp = max_speed(&pool.solver);
        if t % 20 == 19 {
            let (surf, spread) = surface_spread(&pool.solver);
            println!(
                "  tick {:3}: surface {:.4} m, spread(top10%) {:.4} m, max overdensity {:+.4}, max speed {:.4} m/s",
                pool.solver.tick, surf, spread, max_overdensity(&pool.solver), sp
            );
        }
        if sp > 5.0 || surface_height(&pool) > 3.0 {
            println!("  >>> EXPLODED at tick {} (speed {:.2} m/s, surface {:.2} m)", pool.solver.tick, sp, surface_height(&pool));
            exploded = true;
            break;
        }
    }
    if !exploded {
        let (surf, spread) = surface_spread(&pool.solver);
        println!("  RESULT: stable. rest surface {surf:.4} m, flatness spread {spread:.4} m");
    }
}

fn main() {
    println!("FLUID CLAMP HUNT — compression-only explosion reproduction\n");

    // A) The inherited default: compression_only + tensile_k=0.1 -> EXPLODES tick 1.
    run_pool_cfg("A. compression_only, tensile_k=0.1, iters=4 (INHERITED DEFAULT)", 0.1, true, 4, 0.1, 120);
    // B) s_corr OFF, iters=4 -> delayed explosion (tick ~34).
    run_pool_cfg("B. compression_only, tensile_k=0.0, iters=4", 0.0, true, 4, 0.1, 200);
    // Sweep iters/relax with s_corr OFF to locate the stability edge.
    run_pool_cfg("D. compression_only, tensile_k=0.0, iters=1, relax=0.1", 0.0, true, 1, 0.1, 200);
    run_pool_cfg("E. compression_only, tensile_k=0.0, iters=2, relax=0.1", 0.0, true, 2, 0.1, 200);
    run_pool_cfg("F. compression_only, tensile_k=0.0, iters=4, relax=0.05", 0.0, true, 4, 0.05, 200);
    run_pool_cfg("G. compression_only, tensile_k=0.0, iters=10, relax=0.02", 0.0, true, 10, 0.02, 200);
}
