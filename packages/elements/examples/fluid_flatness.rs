//! FLUID FLATNESS — settle the default pool and measure the free-surface
//! deviation against the DERIVED discretization bound.
//!
//! DERIVED BOUND. The free surface is sampled by discrete particles on a
//! lattice of pitch `spacing`. The topmost particle of each x-z column marks
//! the surface there; its height is quantized to the lattice layer it settled
//! on, so two adjacent columns' surface heights can legitimately differ by up
//! to one particle layer = `spacing`. A particle-discretized surface therefore
//! CANNOT be flatter than this sampling granularity: the column-top heights are
//! uniform over a band of width ~`spacing`, giving
//!   * peak deviation (max-min of column tops)  <=  k_peak * spacing, k_peak = 1
//!   * RMS deviation                              ~   spacing / sqrt(12) = 0.29*spacing
//! We accept a rest pool as FLAT when the interior column-top RMS deviation is
//! within the discretization band, RMS <= k_rms * spacing with k_rms = 0.5
//! (half a layer — comfortably inside one-layer sampling, and well under the
//! cohesive-dome signature which is many layers of systematic central mounding).
//!
//! Run: cargo run -p elements --release --example fluid_flatness

use elements::fluid::{fill, surface_height, FluidPoolSpec};
use elements::Solver;

fn max_speed(s: &Solver) -> f64 {
    s.fluid_particles.iter().map(|&i| s.particles.vel[i].length()).fold(0.0, f64::max)
}

/// Column-binned surface: bin interior fluid particles into an x-z grid of
/// pitch `spacing`, take each occupied column's TOP particle y. Exclude a
/// one-cell margin at each wall (the meniscus climbs the wall — not the free
/// surface). Returns (mean, rms_dev, peak_dev, n_columns).
fn surface_flatness(s: &Solver, spec: &FluidPoolSpec) -> (f64, f64, f64, usize) {
    let sp = spec.spacing;
    let hx = spec.inner.0 * 0.5 - sp; // interior margin (skip wall meniscus)
    let hz = spec.inner.1 * 0.5 - sp;
    use std::collections::HashMap;
    let mut col: HashMap<(i64, i64), f64> = HashMap::new();
    for &i in &s.fluid_particles {
        let p = s.particles.pos[i];
        if p.x.abs() > hx || p.z.abs() > hz {
            continue;
        }
        let key = ((p.x / sp).round() as i64, (p.z / sp).round() as i64);
        let e = col.entry(key).or_insert(f64::NEG_INFINITY);
        if p.y > *e {
            *e = p.y;
        }
    }
    let tops: Vec<f64> = col.values().copied().collect();
    let n = tops.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0);
    }
    let mean = tops.iter().sum::<f64>() / n as f64;
    let var = tops.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / n as f64;
    let rms = var.sqrt();
    let peak = tops.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - tops.iter().cloned().fold(f64::INFINITY, f64::min);
    (mean, rms, peak, n)
}

fn run(label: &str, restitution: f64, viscosity_c: f64, rho0_scale: f64, compression_only: bool, ticks: u64) {
    let spec = FluidPoolSpec { spacing: 0.08, ..FluidPoolSpec::default() };
    let sp = spec.spacing;
    let mut pool = fill(spec);
    if let Some(col) = pool.solver.collider.as_mut() {
        col.material.restitution = restitution;
    }
    {
        let cfg = pool.solver.fluid.as_mut().unwrap();
        cfg.viscosity_c = viscosity_c;
        cfg.rest_density *= rho0_scale; // <1 -> more particles read C>0 at rest
        cfg.compression_only = compression_only;
    }
    println!("\n== {label} ==  restitution={restitution} viscosity_c={viscosity_c} rho0_scale={rho0_scale} compression_only={compression_only}");
    let t0 = std::time::Instant::now();
    for t in 0..ticks {
        pool.solver.step();
        if t % 100 == 99 {
            let (mean, rms, peak, n) = surface_flatness(&pool.solver, &spec);
            println!(
                "  tick {:3}: col-mean {:.4} m, RMS dev {:.4} m, peak dev {:.4} m, cols {n}, max speed {:.4} m/s",
                pool.solver.tick, mean, rms, peak, max_speed(&pool.solver)
            );
        }
    }
    let per_tick = t0.elapsed().as_secs_f64() * 1000.0 / ticks as f64;
    let (mean, rms, peak, n) = surface_flatness(&pool.solver, &spec);
    println!("  REST: mean {mean:.4} m over {n} cols | RMS {rms:.4} (bound {:.4} {}) | peak {peak:.4} (bound {:.4} {}) | speed {:.4} | {:.2} ms/tick",
        0.5 * sp, if rms <= 0.5 * sp { "OK" } else { "OVER" },
        sp, if peak <= sp { "OK" } else { "OVER" }, max_speed(&pool.solver), per_tick);
}

fn main() {
    let sp = 0.08;
    println!("FLUID FLATNESS — default pool, spacing {sp} m");
    println!("Derived bound: RMS <= 0.5*spacing = {:.4} m ; peak <= 1.0*spacing = {:.4} m", 0.5 * sp, sp);
    // Select a single case via argv[1] (index) so each can run under its own
    // `timeout 300` wall clock; no arg -> run everything (slow, may exceed a
    // single 300s budget at 500 ticks x 5 cases).
    let cases: Vec<(&str, f64, f64, f64, bool, u64)> = vec![
        ("C. restitution 0.0, viscosity 0.20, rho0x1.00 (baseline, compression_only)", 0.0, 0.20, 1.00, true, 500),
        ("E. restitution 0.0, viscosity 0.20, rho0x0.95, compression_only", 0.0, 0.20, 0.95, true, 500),
        ("F. restitution 0.0, viscosity 0.20, rho0x0.90, compression_only", 0.0, 0.20, 0.90, true, 500),
        ("G. restitution 0.0, viscosity 0.20, rho0x0.85, compression_only", 0.0, 0.20, 0.85, true, 500),
        ("H. restitution 0.0, viscosity 0.20, rho0x1.00, BILATERAL (compression_only=false) -- comparison", 0.0, 0.20, 1.00, false, 500),
    ];
    let arg: Option<usize> = std::env::args().nth(1).and_then(|s| s.parse().ok());
    match arg {
        Some(i) => {
            let (label, r, v, rho, co, ticks) = cases[i];
            run(label, r, v, rho, co, ticks);
        }
        None => {
            for (label, r, v, rho, co, ticks) in cases {
                run(label, r, v, rho, co, ticks);
            }
        }
    }
}
