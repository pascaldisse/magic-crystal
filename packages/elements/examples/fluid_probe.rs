use elements::fluid::{fill, FluidPoolSpec};
use elements::fluid_kernel::{poly6, FluidConfig};
use elements::pointgrid::PointGrid;
use elements::Solver;

// Surface flatness: max-y in centre columns vs edge columns, and global max.
fn flatness(s: &Solver) -> (f64, f64, f64) {
    let (mut cmax, mut emax, mut gmax) =
        (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &i in &s.fluid_particles {
        let q = s.particles.pos[i];
        gmax = gmax.max(q.y);
        if q.x.abs() < 0.12 && q.z.abs() < 0.12 {
            cmax = cmax.max(q.y);
        }
        if q.x.abs() > 0.45 || q.z.abs() > 0.45 {
            emax = emax.max(q.y);
        }
    }
    (cmax, emax, gmax)
}

fn maxspd(s: &Solver) -> f64 {
    s.fluid_particles
        .iter()
        .map(|&i| s.particles.vel[i].length())
        .fold(0.0, f64::max)
}

// Density compression stats: min/mean/max of rho_i/rho0 over interior particles.
fn density_stats(s: &Solver) -> (f64, f64, f64) {
    let cfg: FluidConfig = s.fluid.unwrap();
    let h = cfg.h;
    let rho0 = cfg.rest_density;
    let grid = PointGrid::build(&s.particles.pos, &s.fluid_particles, PointGrid::cell_size(h));
    let mut cand = Vec::new();
    let (mut mn, mut mx, mut sum, mut cnt) = (f64::INFINITY, 0.0_f64, 0.0_f64, 0usize);
    for &i in &s.fluid_particles {
        let pi = s.particles.pos[i];
        grid.query_ball(pi, h, &mut cand);
        let mut density = 0.0;
        for &jc in &cand {
            let j = jc as usize;
            let mj = if s.particles.inv_mass[j] > 0.0 { 1.0 / s.particles.inv_mass[j] } else { 0.0 };
            density += mj * poly6((pi - s.particles.pos[j]).length(), h);
        }
        let ratio = density / rho0;
        mn = mn.min(ratio);
        mx = mx.max(ratio);
        sum += ratio;
        cnt += 1;
    }
    (mn, sum / cnt as f64, mx)
}

fn run(spec: FluidPoolSpec, tune: impl Fn(&mut FluidConfig), ticks: usize, label: &str) {
    let mut p = fill(spec);
    let mut cfg = p.solver.fluid.unwrap();
    tune(&mut cfg);
    p.solver.fluid = Some(cfg);
    for _ in 0..ticks {
        p.solver.step();
    }
    let (c, e, g) = flatness(&p.solver);
    let (dmn, dmean, dmx) = density_stats(&p.solver);
    println!(
        "{label}: center={c:.3} edge={e:.3} dome={:.3} gmax={g:.3} spd={:.3} | rho/rho0 min={dmn:.3} mean={dmean:.3} max={dmx:.3}",
        c - e,
        maxspd(&p.solver),
    );
}

fn main() {
    let base = FluidPoolSpec { spacing: 0.08, ..Default::default() };
    println!("== bilateral, sweep tensile_k (400 ticks) ==");
    for &k in &[0.0, 0.005, 0.01, 0.02, 0.05, 0.1] {
        run(base, |c| { c.compression_only = false; c.tensile_k = k; c.relax = 0.1; c.solver_iterations = 4; }, 400,
            &format!("k={k:.3}"));
    }
    println!("== bilateral, sweep relax at k=0.01 ==");
    for &r in &[0.05, 0.1, 0.15, 0.2] {
        run(base, |c| { c.compression_only = false; c.tensile_k = 0.01; c.relax = r; c.solver_iterations = 4; }, 400,
            &format!("relax={r:.2}"));
    }
}
