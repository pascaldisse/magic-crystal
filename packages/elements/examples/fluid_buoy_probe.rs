//! BUOYANCY PROBE (round-9 scratch) — sweep the SUBMERGED-body coupling to
//! find whether a sealed (non-porous) light body rises. The round-8 buoyancy
//! ordeal released a crate whose particle radius (0.01 m) was a fraction of
//! the fluid spacing (0.06 m): the body was POROUS — fluid slipped between its
//! particles and it displaced nothing, so "pressureless fluid" may have been a
//! confounded read. Here we vary crate lattice resolution + particle radius
//! (so the body actually SEALS against the fluid) and density, and measure the
//! net rise of a body released mid-column. Tick-capped; prints a table.

use elements::fluid::{body_center_y, fill, surface_height, FluidPool, FluidPoolSpec};
use elements::pointgrid::PointGrid;
use elements::{Solver, Vec3};

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

fn nn_min(s: &Solver, search_r: f64) -> f64 {
    let grid = PointGrid::build(&s.particles.pos, &s.fluid_particles, PointGrid::cell_size(search_r));
    let mut cand = Vec::new();
    let mut worst = f64::INFINITY;
    for &i in &s.fluid_particles {
        grid.query_ball(s.particles.pos[i], search_r, &mut cand);
        let mut best = f64::INFINITY;
        for &jc in &cand {
            let j = jc as usize;
            if j == i {
                continue;
            }
            best = best.min((s.particles.pos[i] - s.particles.pos[j]).length());
        }
        if best.is_finite() {
            worst = worst.min(best);
        }
    }
    worst
}

fn run(spec: FluidPoolSpec, lattice: (usize, usize, usize), r_factor: f64, density: f64) -> (f64, f64, f64, f64) {
    let mut pool: FluidPool = fill(spec);
    for _ in 0..200 {
        pool.solver.step();
    }
    let surf = surface_height(&pool);
    // A cube sized to the crate lattice so its particles seal at r_factor×spacing.
    let s = spec.spacing;
    let cs = r_factor * s; // crate particle radius
    // crate side = (n-1) × crate particle spacing; choose crate spacing so the
    // particles overlap enough to seal: crate spacing = 2×radius (touching).
    let crate_pspacing = 2.0 * cs;
    let side = crate_pspacing * (lattice.0 as f64 - 1.0);
    let dims = Vec3::new(side, side, side);
    // Release fully submerged, centred low in the column.
    let submerge_y = (surf * 0.4).max(dims.y * 0.5 + s);
    let idx = pool.solver.spawn_rigid_box(Vec3::new(0.0, submerge_y, 0.0), dims, lattice, density, 1.0, cs);
    let y0 = body_center_y(&pool, idx);
    let search_r = s * 2.0;
    let mut y_peak = y0;
    let mut worst_min = f64::INFINITY;
    let mut y_final = y0;
    for _ in 0..300 {
        pool.solver.step();
        let y = body_center_y(&pool, idx);
        y_peak = y_peak.max(y);
        worst_min = worst_min.min(nn_min(&pool.solver, search_r));
        y_final = y;
    }
    (y0, y_peak, y_final, worst_min)
}

fn main() {
    let spec = small_spec();
    println!("BUOYANCY PROBE — fluid rest_density {}, spacing {}", spec.rest_density, spec.spacing);
    println!("lattice  r_fac  density   y0      y_peak   y_final  rise     min_nn");
    for &(lat, rf) in &[((3usize, 3usize, 3usize), 0.5), ((4, 4, 4), 0.5), ((5, 5, 5), 0.5)] {
        for &density in &[50.0, 200.0, 400.0, 700.0] {
            let (y0, yp, yf, mn) = run(spec, lat, rf, density);
            println!(
                "{:?}  {:.2}   {:6.0}   {:.4}  {:.4}  {:.4}  {:+.4}  {:.4}",
                lat, rf, density, y0, yp, yf, yp - y0, mn
            );
        }
    }
}
