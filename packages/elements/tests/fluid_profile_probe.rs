//! ROUND-9 INSTRUMENTATION — measure the hydrostatic density/pressure profile
//! of the SETTLED column and sweep the ρ₀-below-packing factor (lever 1). The
//! handoff's law: measure first, then choose the lever. Prints per-depth-bin
//! mean SPH overdensity `C = ρ/ρ₀ − 1` (what compression_only acts on) and the
//! rise of a light submerged box, for each factor. Run:
//!   cargo test -p elements --test fluid_profile_probe -- --nocapture --ignored

use elements::fluid::{body_center_y, drop_crate, fill, surface_height, FluidPool, FluidPoolSpec};
use elements::fluid_kernel::poly6;
use elements::pointgrid::PointGrid;
use elements::Solver;

fn small_spec() -> FluidPoolSpec {
    FluidPoolSpec {
        inner: (0.24, 0.24),
        wall_height: 0.55,
        fill_height: 0.48,
        spacing: 0.06,
        substeps: 4,
        ..FluidPoolSpec::default()
    }
}

fn density_at(s: &Solver, i: usize, grid: &PointGrid, h: f64, cand: &mut Vec<u32>) -> f64 {
    grid.query_ball(s.particles.pos[i], h, cand);
    let mut density = 0.0;
    for &jc in cand.iter() {
        let j = jc as usize;
        let mj = if s.particles.inv_mass[j] > 0.0 { 1.0 / s.particles.inv_mass[j] } else { 0.0 };
        density += mj * poly6((s.particles.pos[i] - s.particles.pos[j]).length(), h);
    }
    density
}

fn min_nn(s: &Solver, search_r: f64) -> f64 {
    let grid = PointGrid::build(&s.particles.pos, &s.fluid_particles, PointGrid::cell_size(search_r));
    let mut cand = Vec::new();
    let mut worst = f64::INFINITY;
    for &i in &s.fluid_particles {
        grid.query_ball(s.particles.pos[i], search_r, &mut cand);
        for &jc in &cand {
            let j = jc as usize;
            if j == i { continue; }
            worst = worst.min((s.particles.pos[i] - s.particles.pos[j]).length());
        }
    }
    worst
}

fn recalibrate(pool: &mut FluidPool, factor: f64) {
    pool.solver.fluid.as_mut().unwrap().rest_density_factor = factor;
    pool.solver.calibrate_fluid_rest_density();
}

fn print_profile(pool: &FluidPool) {
    let cfg = pool.solver.fluid.unwrap();
    let (h, rho0) = (cfg.h, cfg.rest_density);
    let surf = surface_height(pool);
    let floor_y = pool.fluid.iter().map(|&i| pool.solver.particles.pos[i].y).fold(f64::INFINITY, f64::min);
    let span = (surf - floor_y).max(1e-6);
    let nbins = 6usize;
    let grid = PointGrid::build(&pool.solver.particles.pos, &pool.solver.fluid_particles, PointGrid::cell_size(h));
    let mut cand = Vec::new();
    let mut sum = vec![0.0_f64; nbins];
    let mut cnt = vec![0usize; nbins];
    for &i in &pool.solver.fluid_particles {
        let y = pool.solver.particles.pos[i].y;
        let mut b = (((surf - y) / span) * nbins as f64) as usize;
        if b >= nbins { b = nbins - 1; }
        sum[b] += density_at(&pool.solver, i, &grid, h, &mut cand) / rho0 - 1.0;
        cnt[b] += 1;
    }
    print!("  rho0={rho0:7.2}  surf={surf:.4} span={span:.4}  meanC by depth(top→bot):");
    for b in 0..nbins {
        if cnt[b] == 0 { print!("   --   "); } else { print!(" {:+.4}", sum[b] / cnt[b] as f64); }
    }
    println!();
}

// ROUND-10 lean discrimination probe: container-boundary Akinci ON (default).
// No per-tick nn_stats — just settle, drop, tail-mean rest depth of light vs
// heavy. Prints per-stage progress so it can never look hung.
#[test]
#[ignore = "round-10 container-boundary discrimination probe; run explicitly --ignored --nocapture"]
fn container_discrimination_probe() {
    let spec = small_spec();
    let crate_dims = elements::Vec3::new(0.09, 0.09, 0.09);
    let settle_depth = |density: f64, container: bool| -> (f64, f64, f64) {
        let mut pool = fill(spec);
        pool.solver.fluid.as_mut().unwrap().container_boundary = container;
        pool.solver.calibrate_fluid_rest_density();
        for _ in 0..260 { pool.solver.step(); }
        let surf = surface_height(&pool);
        let submerge_y = (surf * 0.30).max(crate_dims.y * 0.5 + spec.spacing);
        let idx = drop_crate(&mut pool, crate_dims, (3, 3, 3), density, submerge_y, 0.0, 0.01);
        let y0 = body_center_y(&pool, idx);
        let (mut tail_sum, mut tail_n) = (0.0_f64, 0usize);
        let total = 400;
        for t in 0..total {
            pool.solver.step();
            if t >= total - 80 { tail_sum += body_center_y(&pool, idx); tail_n += 1; }
        }
        (y0, tail_sum / tail_n as f64, surf)
    };
    for &container in &[true, false] {
        let (ls, lr, surf) = settle_depth(200.0, container);
        eprintln!("[container={container}] light(200) done: start {ls:.4} rest {lr:.4} surf {surf:.4}");
        let (hs, hr, _) = settle_depth(2000.0, container);
        eprintln!("[container={container}] heavy(2000) done: start {hs:.4} rest {hr:.4}");
        println!(
            "container={container}: light rest {lr:.4}  heavy rest {hr:.4}  DISCRIMINATION light-heavy = {:+.4} m  (need > one spacing {:.4})",
            lr - hr, spec.spacing
        );
    }
}

#[test]
#[ignore = "instrumentation probe (round-9): sweeps rho0 factor, prints profile + robust net box rise; run explicitly with --ignored --nocapture"]
fn profile_sweep() {
    let spec = small_spec();
    let search_r = spec.spacing * 2.0;
    // Robust witness: light box (density 200, cork-like), released at rest deep
    // in the settled column; net rise = mean-y over the LAST window minus the
    // release y (averages out the transient bob / slosh noise).
    // Physical water density = spec.rest_density = 1000 (particle mass =
    // 1000·spacing³). Archimedes DISCRIMINATION TEST: release each density at
    // BOTH a LOW start and a HIGH start. Real buoyancy → each density converges
    // to its OWN equilibrium depth (light high, heavy low) regardless of where
    // it was released. Artifact → all converge to the same depth.
    for &(factor, density, start_frac) in &[
        (0.92_f64, 200.0_f64, 0.15_f64), (0.92, 200.0, 0.85),
        (0.92, 2000.0, 0.15), (0.92, 2000.0, 0.85),
        (0.92, 1000.0, 0.15), (0.92, 1000.0, 0.85),
    ] {
        let mut pool = fill(spec);
        recalibrate(&mut pool, factor);
        for _ in 0..260 { pool.solver.step(); }
        let surf = surface_height(&pool);
        let crate_dims = elements::Vec3::new(0.09, 0.09, 0.09);
        let lo = crate_dims.y * 0.5 + spec.spacing;
        let submerge_y = lo + (surf - crate_dims.y - lo).max(0.0) * start_frac;
        let _ = start_frac;
        let idx = drop_crate(&mut pool, crate_dims, (3, 3, 3), density, submerge_y, 0.0, 0.01);
        let y0 = body_center_y(&pool, idx);
        let mut ypeak = y0;
        let mut worst_nn = f64::INFINITY;
        let (mut tail_sum, mut tail_n) = (0.0_f64, 0usize);
        let total = 400;
        for t in 0..total {
            pool.solver.step();
            let y = body_center_y(&pool, idx);
            ypeak = ypeak.max(y);
            worst_nn = worst_nn.min(min_nn(&pool.solver, search_r));
            if t >= total - 80 { tail_sum += y; tail_n += 1; }
        }
        let tail_mean = tail_sum / tail_n as f64;
        let _ = ypeak;
        println!(
            "factor {factor:.2} density {density:5.0} start {start_frac:.2}: y0={y0:.4} surf={surf:.4} -> EQUILIBRIUM tailMeanY={tail_mean:.4}  (net {:+.4})  worst_nn={worst_nn:.4} (floor≈{:.4})",
            tail_mean - y0, 0.85 * spec.spacing,
        );
    }
    // One profile print for the chosen factor.
    let mut pool = fill(spec);
    recalibrate(&mut pool, 0.92);
    for _ in 0..260 { pool.solver.step(); }
    print!("factor 0.92 settled profile:");
    print_profile(&pool);
}
