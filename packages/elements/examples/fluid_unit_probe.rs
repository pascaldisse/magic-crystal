//! FLUID UNIT PROBE — the density constraint in ISOLATION (no gravity, no
//! walls, no collision). The pool-scale churn mixes the density solve with
//! gravity + floor restitution; this strips all of that so ONLY the PBF
//! density projection acts. Catches the clamp / sign / gate bugs at the
//! smallest scale the physics read demands.
//!
//! Run: cargo run -p elements --release --example fluid_unit_probe

use elements::fluid_kernel::poly6;
use elements::pointgrid::PointGrid;
use elements::{Solver, SolverConfig, Vec3};

fn max_speed(s: &Solver) -> f64 {
    s.fluid_particles
        .iter()
        .map(|&i| s.particles.vel[i].length())
        .fold(0.0, f64::max)
}

fn density_stats(s: &Solver) -> (f64, f64) {
    // (max overdensity C, min C) over fluid particles.
    let cfg = s.fluid.unwrap();
    let (h, rho0) = (cfg.h, cfg.rest_density);
    let grid = PointGrid::build(&s.particles.pos, &s.fluid_particles, PointGrid::cell_size(h));
    let mut cand = Vec::new();
    let (mut hi, mut lo) = (f64::NEG_INFINITY, f64::INFINITY);
    for &i in &s.fluid_particles {
        grid.query_ball(s.particles.pos[i], h, &mut cand);
        let mut density = 0.0;
        for &jc in &cand {
            let j = jc as usize;
            let mj = 1.0 / s.particles.inv_mass[j];
            density += mj * poly6((s.particles.pos[i] - s.particles.pos[j]).length(), h);
        }
        let c = density / rho0 - 1.0;
        hi = hi.max(c);
        lo = lo.min(c);
    }
    (hi, lo)
}

/// Build an n×n×n fluid cube, NO gravity, NO collider, calibrate rest density.
fn build_cube(n: usize, spacing: f64) -> Solver {
    let cfg = SolverConfig { dt: 1.0 / 60.0, substeps: 1, gravity: Vec3::ZERO, ..SolverConfig::default() };
    let mut s = Solver::new(cfg);
    let span = (n - 1) as f64 * spacing;
    let center = Vec3::ZERO;
    let dims = Vec3::new(span, span, span);
    let rest_density = 1000.0;
    let radius = 0.5 * spacing;
    s.spawn_fluid_box(center, dims, spacing, rest_density, 3.0, radius);
    s.calibrate_fluid_rest_density();
    s
}

/// Scale every fluid particle's position about the centroid by `factor`
/// (<1 compress, >1 stretch), zero velocities.
fn scale_positions(s: &mut Solver, factor: f64) {
    let c = {
        let mut sum = Vec3::ZERO;
        for &i in &s.fluid_particles { sum = sum + s.particles.pos[i]; }
        sum.scale(1.0 / s.fluid_particles.len() as f64)
    };
    for &i in &s.fluid_particles {
        let p = c + (s.particles.pos[i] - c).scale(factor);
        s.particles.pos[i] = p;
        s.particles.prev[i] = p;
        s.particles.vel[i] = Vec3::ZERO;
    }
}

fn run(label: &str, factor: f64, compression_only: bool, tensile_k: f64, ticks: u64) {
    let mut s = build_cube(5, 0.08);
    {
        let cfg = s.fluid.as_mut().unwrap();
        cfg.compression_only = compression_only;
        cfg.tensile_k = tensile_k;
        cfg.solver_iterations = 4;
    }
    scale_positions(&mut s, factor);
    let (hi0, lo0) = density_stats(&s);
    println!(
        "\n== {label} ==\n  factor={factor} compression_only={compression_only} tensile_k={tensile_k}"
    );
    println!("  START: C in [{lo0:+.4}, {hi0:+.4}], speed 0");
    for t in 0..ticks {
        s.step();
        if t % 10 == 9 || t == 0 {
            let (hi, lo) = density_stats(&s);
            println!("  tick {:3}: C in [{lo:+.4}, {hi:+.4}], max speed {:.5} m/s", s.tick, max_speed(&s));
        }
    }
}

fn main() {
    println!("FLUID UNIT PROBE — density constraint isolated (no gravity/walls)\n");
    println!("Rest lattice: interior C=0, surface C<0 (fewer neighbours).");
    println!("EXPECTATIONS under compression_only:");
    println!("  * COMPRESSED cube (factor<1, C>0 everywhere): must PUSH APART, C->~0, settle (speed->0).");
    println!("  * STRETCHED cube (factor>1, C<0 everywhere): with s_corr OFF must do NOTHING (speed stays 0).");
    println!("  * REST cube (factor=1): must stay at rest (speed ~0).");

    // Compression-only, s_corr OFF.
    run("A. COMPRESS 0.9, compression_only, s_corr OFF", 0.9, true, 0.0, 60);
    run("B. STRETCH 1.1, compression_only, s_corr OFF", 1.1, true, 0.0, 60);
    run("C. REST 1.0, compression_only, s_corr OFF", 1.0, true, 0.0, 60);
    // s_corr ON, but STRETCH/REST under compression_only clamp C to 0
    // everywhere (C<=0 -> lambda=0 on every particle), so BOTH the old
    // global-disable gate (`!compression_only`) and the current per-pair
    // gate (`li!=0 || lj!=0`) leave s_corr at exactly zero here -- no tick-1
    // kick to expose under either semantics. These two only prove s_corr
    // stays inert when nothing is compressed; the pair-gate itself (some
    // pairs live, some not, in the SAME run) is exercised by F below.
    run("D. STRETCH 1.1, compression_only, s_corr ON (0.1)", 1.1, true, 0.1, 30);
    run("E. REST 1.0, compression_only, s_corr ON (0.1)", 1.0, true, 0.1, 30);
    // Pair-gate exercise: compression_only WITH s_corr ON. Some particles
    // overdense (lambda>0, s_corr active on their pairs), others rest/under-
    // dense (lambda=0, s_corr must stay gated OFF on those pairs). Must NOT
    // explode -- settles like case A, s_corr only adds anti-clustering push
    // in the still-compressed neighbourhoods.
    run("F. COMPRESS 0.9, compression_only, s_corr ON (0.1) -- pair-gate exercise", 0.9, true, 0.1, 60);
}
