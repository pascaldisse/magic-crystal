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
//! Run: cargo run -p elements --release --example fluid_flatness -- [case] [ticks] [tensile_k] [wall-margin/spacing] [spacing]
//! argv[1] case index (default: run all 5, may exceed one 300s budget).
//! argv[2] tick-count override (hold-window extension beyond the case's
//! baked default, chunked by the caller's own `timeout`).
//! argv[3] tensile_k override (comparison probe, e.g. `0.1` reproduces the
//! round-6 hydrostatic detonation for contrast against the round-7 default).
//! argv[4] interior wall-exclusion margin in units of `spacing` (default
//! `1.0`; the corner of a box pool sees TWO walls at once and its meniscus
//! climbs more than a straight wall's — `2.0` excludes that ring too).
//! argv[5] spacing override (coarser = fewer particles = faster; used to
//! probe the long-horizon trend cheaply).
//!
//! ROUND-7 FINDING (spacing 0.08, tensile_k=0, compression_only=true,
//! margin=1): RMS stays inside bound at every tick sampled (500..1600,
//! 0.03-0.04m vs 0.04m bound) and center-vs-edge stays within ~1cm with NO
//! systematic sign (not the dome signature — a cohesive dome shows center
//! persistently and increasingly ABOVE edge by many layers). Peak deviation
//! persistently EXCEEDS bound (0.14-0.26m vs 0.08m), traced to two disclosed,
//! non-dome causes: (1) a corner-meniscus ring (the 4 vertical edges of the
//! box, confirmed by `margin=2.0` still leaving other outlier columns, so
//! corners are A cause, not the only one) and (2) a persistent low-amplitude
//! boundary jitter — mean fluid speed stays low (~0.01-0.03 m/s, the BULK is
//! quiet) but a few particles near the walls keep taking speed kicks up to
//! ~1.1 m/s that do not decay to zero over 1600 ticks (unlike the SAME
//! config at coarser spacing 0.14, which locks to an EXACT flat, exactly
//! zero-speed lattice by tick 200 and holds it for 3000 ticks straight —
//! proving the underlying density constraint itself has no residual drift;
//! the jitter is boundary/collision-tolerance noise at finer sampling, not a
//! s_corr- or tensile-instability effect, since tensile_k=0 throughout).
//! Comparison: the SAME config at tensile_k=0.1 (round-6's value) detonates
//! within 60 ticks at this pool scale (surface 6.3m from a 0.6m fill, particle
//! speeds >170 m/s) — re-confirming round-6's hydrostatic-detonation finding
//! and that tensile_k=0 is what keeps this bounded.

use elements::fluid::{fill, FluidPoolSpec};
use elements::Solver;

fn max_speed(s: &Solver) -> f64 {
    s.fluid_particles.iter().map(|&i| s.particles.vel[i].length()).fold(0.0, f64::max)
}

fn mean_speed(s: &Solver) -> f64 {
    let n = s.fluid_particles.len().max(1);
    s.fluid_particles.iter().map(|&i| s.particles.vel[i].length()).sum::<f64>() / n as f64
}

/// Center-column top mean vs edge-column top mean (dome-shape witness: a
/// cohesive dome shows center systematically ABOVE edge by many layers; noisy
/// churn does not have that sign/pattern).
fn center_vs_edge(s: &Solver, spec: &FluidPoolSpec) -> (f64, f64) {
    let sp = spec.spacing;
    let hx = spec.inner.0 * 0.5 - sp;
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
    let center_r = (hx.min(hz) * 0.35) / sp; // inner 35% radius in cell units
    let (mut csum, mut cn, mut esum, mut en) = (0.0, 0usize, 0.0, 0usize);
    for (&(cx, cz), &y) in &col {
        let r = ((cx * cx + cz * cz) as f64).sqrt();
        if r <= center_r {
            csum += y;
            cn += 1;
        } else {
            esum += y;
            en += 1;
        }
    }
    (
        if cn > 0 { csum / cn as f64 } else { f64::NAN },
        if en > 0 { esum / en as f64 } else { f64::NAN },
    )
}

/// Column-binned surface: bin interior fluid particles into an x-z grid of
/// pitch `spacing`, take each occupied column's TOP particle y. Exclude a
/// one-cell margin at each wall (the meniscus climbs the wall — not the free
/// surface). Returns (mean, rms_dev, peak_dev, n_columns).
fn surface_flatness(s: &Solver, spec: &FluidPoolSpec) -> (f64, f64, f64, usize) {
    let sp = spec.spacing;
    let margin = std::env::args().nth(4).and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0);
    let hx = spec.inner.0 * 0.5 - margin * sp; // interior margin (skip wall meniscus)
    let hz = spec.inner.1 * 0.5 - margin * sp;
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

#[allow(clippy::too_many_arguments)]
fn run_tk(
    label: &str,
    restitution: f64,
    viscosity_c: f64,
    rho0_scale: f64,
    compression_only: bool,
    ticks: u64,
    tensile_k_override: Option<f64>,
) {
    let spacing = std::env::args().nth(5).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.08);
    let spec = FluidPoolSpec { spacing, ..FluidPoolSpec::default() };
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
        if let Some(tk) = tensile_k_override {
            cfg.tensile_k = tk;
        }
    }
    let tk_shown = tensile_k_override.unwrap_or(pool.solver.fluid.unwrap().tensile_k);
    println!("\n== {label} ==  restitution={restitution} viscosity_c={viscosity_c} rho0_scale={rho0_scale} compression_only={compression_only} tensile_k={tk_shown}");
    let t0 = std::time::Instant::now();
    for t in 0..ticks {
        pool.solver.step();
        if t % 100 == 99 {
            let (mean, rms, peak, n) = surface_flatness(&pool.solver, &spec);
            let (cy, ey) = center_vs_edge(&pool.solver, &spec);
            println!(
                "  tick {:3}: col-mean {:.4} m, RMS dev {:.4} m, peak dev {:.4} m, cols {n}, center {cy:.4} vs edge {ey:.4} (Δ{:+.4}), max speed {:.4} m/s, mean speed {:.4} m/s",
                pool.solver.tick, mean, rms, peak, cy - ey, max_speed(&pool.solver), mean_speed(&pool.solver)
            );
        }
    }
    let per_tick = t0.elapsed().as_secs_f64() * 1000.0 / ticks as f64;
    let (mean, rms, peak, n) = surface_flatness(&pool.solver, &spec);
    println!("  REST: mean {mean:.4} m over {n} cols | RMS {rms:.4} (bound {:.4} {}) | peak {peak:.4} (bound {:.4} {}) | speed {:.4} | {:.2} ms/tick",
        0.5 * sp, if rms <= 0.5 * sp { "OK" } else { "OVER" },
        sp, if peak <= sp { "OK" } else { "OVER" }, max_speed(&pool.solver), per_tick);
    outlier_report(&pool.solver, &spec, mean);
}

/// Dump the extreme column tops (highest/lowest N) + which fluid particles
/// carry the top-3 speeds — tells outlier-particle noise apart from a
/// systemic dome/detonation.
fn outlier_report(s: &Solver, spec: &FluidPoolSpec, mean: f64) {
    let sp = spec.spacing;
    let margin = std::env::args().nth(4).and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0);
    let hx = spec.inner.0 * 0.5 - margin * sp;
    let hz = spec.inner.1 * 0.5 - margin * sp;
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
    let mut tops: Vec<((i64, i64), f64)> = col.into_iter().collect();
    tops.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let n_over = tops.iter().filter(|(_, y)| (y - mean).abs() > 2.0 * sp).count();
    print!("  OUTLIERS (mean {mean:.4}): top5 ");
    for (k, y) in tops.iter().take(5) {
        print!("{:?}={y:.4} ", k);
    }
    print!("| bottom5 ");
    for (k, y) in tops.iter().rev().take(5) {
        print!("{:?}={y:.4} ", k);
    }
    println!("| cols >2*spacing from mean: {n_over}/{}", tops.len());
    let mut spds: Vec<f64> = s.fluid_particles.iter().map(|&i| s.particles.vel[i].length()).collect();
    spds.sort_by(|a, b| b.partial_cmp(a).unwrap());
    println!("  top-5 particle speeds: {:.4} {:.4} {:.4} {:.4} {:.4} m/s", spds[0], spds[1], spds[2], spds[3], spds[4]);
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
    // Optional argv[2]: override the case's tick count (a longer hold window
    // than the case's baked default, chunked by the caller's own `timeout`).
    let tick_override: Option<u64> = std::env::args().nth(2).and_then(|s| s.parse().ok());
    // Optional argv[3]: override tensile_k (comparison probe — e.g. re-run
    // case 0 at the paper's k=0.1 to separate a corner/wall artifact from a
    // tensile-instability-dependent one).
    let tk_override: Option<f64> = std::env::args().nth(3).and_then(|s| s.parse().ok());
    match arg {
        Some(i) => {
            let (label, r, v, rho, co, ticks) = cases[i];
            run_tk(label, r, v, rho, co, tick_override.unwrap_or(ticks), tk_override);
        }
        None => {
            for (label, r, v, rho, co, ticks) in cases {
                run_tk(label, r, v, rho, co, tick_override.unwrap_or(ticks), tk_override);
            }
        }
    }
}
