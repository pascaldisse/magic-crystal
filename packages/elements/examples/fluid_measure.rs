//! FLUID — honest per-tick cost + incompressibility readout of the PBF pool
//! diorama, at the default resolution. Prints the phase breakdown (like
//! P-SCALE), the density error at rest, and the crate-drop float/sink outcome.
//!
//! Run: cargo run -p elements --release --example fluid_measure

use elements::fluid::{body_center_y, drop_crate, fill, settle, surface_height, FluidPoolSpec};
use elements::fluid_kernel::poly6;
use elements::pointgrid::PointGrid;
use elements::{Solver, Vec3};

/// Max relative density error |ρ_i/ρ0 − 1| over the fluid, at current state.
fn max_density_error(s: &Solver) -> f64 {
    let cfg = s.fluid.unwrap();
    let h = cfg.h;
    let rho0 = cfg.rest_density;
    let grid = PointGrid::build(&s.particles.pos, &s.fluid_particles, PointGrid::cell_size(h));
    let mut cand = Vec::new();
    let mut worst = 0.0_f64;
    for &i in &s.fluid_particles {
        grid.query_ball(s.particles.pos[i], h, &mut cand);
        let mut density = 0.0;
        for &jc in &cand {
            let j = jc as usize;
            let mj = 1.0 / s.particles.inv_mass[j];
            density += mj * poly6((s.particles.pos[i] - s.particles.pos[j]).length(), h);
        }
        // Free-surface particles are UNDER-dense by design (fewer neighbours);
        // incompressibility bounds only OVER-density (compression). Report the
        // signed max over-density and the worst |error| separately.
        worst = worst.max((density / rho0 - 1.0).abs());
    }
    worst
}

fn max_overdensity(s: &Solver) -> f64 {
    let cfg = s.fluid.unwrap();
    let h = cfg.h;
    let rho0 = cfg.rest_density;
    let grid = PointGrid::build(&s.particles.pos, &s.fluid_particles, PointGrid::cell_size(h));
    let mut cand = Vec::new();
    let mut worst = 0.0_f64;
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

fn max_speed(s: &Solver) -> f64 {
    s.fluid_particles
        .iter()
        .map(|&i| s.particles.vel[i].length())
        .fold(0.0, f64::max)
}

fn main() {
    // Use a bench-tractable spacing (coarser than the render's).
    let spec = FluidPoolSpec {
        spacing: 0.08,
        ..FluidPoolSpec::default()
    };
    println!(
        "FLUID pool: inner {:?} m, fill {} m, spacing {} m, ~{} fluid particles, substeps {}",
        spec.inner, spec.fill_height, spec.spacing, spec.fluid_count(), spec.substeps
    );
    let mut pool = fill(spec);
    let cfg = pool.solver.fluid.unwrap();
    println!(
        "calibrated: h={:.4} m, rho0(SPH)={:.4}, cfm_eps={:.4e}, N_fluid={}",
        cfg.h, cfg.rest_density, cfg.cfm_epsilon, pool.fluid.len()
    );
    let h0 = surface_height(&pool);

    // Settle to hydrostatic rest.
    println!("\n-- SETTLE (120 ticks) --");
    for t in 0..120 {
        pool.solver.step();
        if t % 20 == 19 {
            println!(
                "tick {:3}: surface {:.4} m, max overdensity {:.4}, max |err| {:.4}, max speed {:.4} m/s",
                pool.solver.tick,
                surface_height(&pool),
                max_overdensity(&pool.solver),
                max_density_error(&pool.solver),
                max_speed(&pool.solver),
            );
        }
    }
    let h_rest = surface_height(&pool);
    println!("surface: spawn {h0:.4} -> rest {h_rest:.4} m");

    // Phase cost at rest (steady state).
    println!("\n-- PER-TICK COST (median of 30 rest ticks) --");
    let mut totals = Vec::new();
    let mut fluids = Vec::new();
    let mut statics = Vec::new();
    let mut bodies = Vec::new();
    for _ in 0..30 {
        let p = pool.solver.step_profiled();
        totals.push(p.total.as_secs_f64() * 1e3);
        fluids.push(p.solve_fluid.as_secs_f64() * 1e3);
        statics.push(p.collision_static.as_secs_f64() * 1e3);
        bodies.push(p.collision_body.as_secs_f64() * 1e3);
    }
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!("  solve_fluid (PBF):   {:.3} ms", med(fluids));
    println!("  collision_static:    {:.3} ms", med(statics));
    println!("  collision_body:      {:.3} ms", med(bodies));
    println!("  WHOLE TICK:          {:.3} ms  (N_fluid={})", med(totals), pool.fluid.len());

    // Drop a DENSE crate (should sink) and a LIGHT crate scenario reported.
    println!("\n-- CRATE DROP (dense, rho=1600 > water) --");
    let crate_dims = Vec3::new(0.36, 0.36, 0.36);
    let idx = drop_crate(&mut pool, crate_dims, (5, 5, 5), 1600.0, 1.0, 2.0, 0.036);
    let start_cy = body_center_y(&pool, idx);
    let mut peak_splash = surface_height(&pool);
    for t in 0..200 {
        pool.solver.step();
        peak_splash = peak_splash.max(surface_height(&pool));
        if t % 40 == 39 {
            println!(
                "tick {:3}: crate cy {:.4} m, surface {:.4} m, max overdensity {:.4}",
                pool.solver.tick,
                body_center_y(&pool, idx),
                surface_height(&pool),
                max_overdensity(&pool.solver),
            );
        }
    }
    let end_cy = body_center_y(&pool, idx);
    println!(
        "dense crate: cy {start_cy:.4} -> {end_cy:.4} m (floor≈{:.3}); peak splash {peak_splash:.4} m (rest {h_rest:.4})",
        crate_dims.y * 0.5
    );

    // A light crate in a fresh pool (should float).
    println!("\n-- CRATE DROP (light, rho=400 < water) --");
    let mut pool2 = fill(FluidPoolSpec { spacing: 0.08, ..FluidPoolSpec::default() });
    settle(&mut pool2, 120);
    let hr2 = surface_height(&pool2);
    let idx2 = drop_crate(&mut pool2, crate_dims, (5, 5, 5), 400.0, 1.0, 2.0, 0.036);
    for _ in 0..250 {
        pool2.solver.step();
    }
    let cy2 = body_center_y(&pool2, idx2);
    println!(
        "light crate: settled cy {cy2:.4} m vs surface {:.4} m (floats if cy near/above submerged eq.)",
        surface_height(&pool2)
    );
    let _ = hr2;
}
