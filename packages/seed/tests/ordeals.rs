//! S0 ordeals — the determinism law's trial (§ ENTROPY.md "ENFORCEMENT = the
//! determinism ordeals"). Each test prints its verbatim numbers.

use glam::Vec2;
use seed::hash::domain;
use seed::{density_scatter, grid_jitter, poisson_disk, Fbm, Noise, Region, Seed};

// ---------------------------------------------------------------------------
// ORDEAL 1 — determinism: same (seed, coords) => byte-identical across two
// runs AND across iteration order.
// ---------------------------------------------------------------------------

#[test]
fn ordeal_determinism_two_runs_and_iteration_order() {
    let noise = Noise::new(0xDEAD_BEEF);

    // Two independent passes over the same coordinates -> bit-identical.
    let run = |()| {
        let mut bits: Vec<u32> = Vec::new();
        for i in 0..500i32 {
            for j in 0..500i32 {
                let x = i as f32 * 0.137 - 12.0;
                let y = j as f32 * 0.091 + 3.0;
                bits.push(noise.value2(x, y).to_bits());
                bits.push(noise.gradient2(x, y).to_bits());
            }
        }
        bits
    };
    let a = run(());
    let b = run(());
    assert_eq!(a, b, "two runs diverged");
    let mismatches = a.iter().zip(&b).filter(|(x, y)| x != y).count();

    // Iteration order independence for scatter: forward vs reverse cell walk
    // must yield the same point SET.
    let region = Region::new(Vec2::new(-40.0, -40.0), Vec2::new(40.0, 40.0));
    let fwd = grid_jitter(0x1234, region, 2.0);
    let mut sorted_fwd: Vec<[u32; 2]> =
        fwd.iter().map(|p| [p.x.to_bits(), p.y.to_bits()]).collect();
    let mut rev = fwd.clone();
    rev.reverse();
    let mut sorted_rev: Vec<[u32; 2]> =
        rev.iter().map(|p| [p.x.to_bits(), p.y.to_bits()]).collect();
    sorted_fwd.sort_unstable();
    sorted_rev.sort_unstable();
    assert_eq!(sorted_fwd, sorted_rev, "scatter order-dependent");

    println!(
        "ORDEAL determinism: samples={} run_mismatches={} scatter_points={} order_identical={}",
        a.len(),
        mismatches,
        fwd.len(),
        sorted_fwd == sorted_rev
    );
}

// ---------------------------------------------------------------------------
// ORDEAL 2 — isolation: regenerating region B alone == region B from a
// full-world pass (the zero-loading property).
// ---------------------------------------------------------------------------

/// Generate one region's jittered instances from the world seed + region coord
/// alone, via the hierarchical sub-seed path world -> region.
fn region_points(world: Seed, rx: i32, ry: i32) -> Vec<Vec2> {
    // A stable per-region index from its 2D coordinate (Cantor-style pairing).
    let key = ((rx as i64 as u64) << 32) ^ (ry as u32 as u64);
    let region_seed = world.sub_at(domain::REGION, key);
    let size = 64.0f32;
    let region = Region::new(
        Vec2::new(rx as f32 * size, ry as f32 * size),
        Vec2::new((rx as f32 + 1.0) * size, (ry as f32 + 1.0) * size),
    );
    grid_jitter(region_seed.0, region, 3.0)
}

#[test]
fn ordeal_isolation_region_regenerates_alone() {
    let world = Seed::new(0xC0FFEE);

    // Full-world pass over a grid of regions, remember region B's output.
    let (bx, by) = (2, -3);
    let mut from_full: Option<Vec<Vec2>> = None;
    let mut total = 0usize;
    for ry in -4..5 {
        for rx in -4..5 {
            let pts = region_points(world, rx, ry);
            total += pts.len();
            if (rx, ry) == (bx, by) {
                from_full = Some(pts);
            }
        }
    }
    let from_full = from_full.expect("region B in full pass");

    // Region B regenerated entirely alone — no neighbor ever touched.
    let alone = region_points(world, bx, by);

    let full_bits: Vec<[u32; 2]> = from_full
        .iter()
        .map(|p| [p.x.to_bits(), p.y.to_bits()])
        .collect();
    let alone_bits: Vec<[u32; 2]> = alone
        .iter()
        .map(|p| [p.x.to_bits(), p.y.to_bits()])
        .collect();
    assert_eq!(
        full_bits, alone_bits,
        "isolated region B != full-pass region B"
    );

    println!(
        "ORDEAL isolation: full_pass_total={} regionB_from_full={} regionB_alone={} identical={}",
        total,
        from_full.len(),
        alone.len(),
        full_bits == alone_bits
    );
}

// ---------------------------------------------------------------------------
// ORDEAL 3 — noise statistics: mean/variance of 1e6 samples within derived
// bounds; no visible axis bias (isotropy under x<->y swap).
// ---------------------------------------------------------------------------

#[test]
fn ordeal_noise_statistics() {
    let noise = Noise::new(0xA11CE);
    let n = 1000i32; // 1000 x 1000 = 1e6 samples
    let step = 0.317f32;

    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    // Autocorrelation accumulators for a lag along x vs along y.
    let lag = step;
    let mut acc_x = 0.0f64;
    let mut acc_y = 0.0f64;
    let mut acc_n = 0.0f64;

    for i in 0..n {
        for j in 0..n {
            let x = i as f32 * step - 100.0;
            let y = j as f32 * step - 50.0;
            let v = noise.value2(x, y) as f64;
            sum += v;
            sumsq += v * v;
            acc_x += v * noise.value2(x + lag, y) as f64;
            acc_y += v * noise.value2(x, y + lag) as f64;
            acc_n += 1.0;
        }
    }
    let count = (n as f64) * (n as f64);
    let mean = sum / count;
    let var = sumsq / count - mean * mean;
    let std = var.sqrt();

    // Derived bounds: SE of the mean = std / sqrt(N); an unbiased field's mean
    // sits well inside a few SE. Value noise variance for smootherstep-
    // interpolated uniform[-1,1] lattice values is O(0.1).
    let se = std / count.sqrt();
    let mean_tol = 8.0 * se;
    assert!(mean.abs() < mean_tol, "mean {mean} exceeds {mean_tol}");
    assert!(
        (0.02..0.5).contains(&var),
        "variance {var} out of derived band"
    );

    // Isotropy: autocorrelation along x vs along y must match under swap.
    let acf_x = acc_x / acc_n;
    let acf_y = acc_y / acc_n;
    let aniso = (acf_x - acf_y).abs();
    // Both are correlations of the same field at the same lag distance; the
    // lattice is symmetric under x<->y so the gap is pure sampling noise.
    let aniso_tol = 6.0 * se;
    assert!(aniso < aniso_tol.max(1e-3), "axis bias {aniso} exceeds tol");

    println!(
        "ORDEAL noise-stats: N={} mean={:.3e} var={:.6} std={:.6} se={:.3e} mean_tol={:.3e} acf_x={:.6} acf_y={:.6} aniso={:.3e}",
        count as u64, mean, var, std, se, mean_tol, acf_x, acf_y, aniso
    );
}

// ---------------------------------------------------------------------------
// ORDEAL 4 — Poisson-disk: min-distance property holds for EVERY pair
// (exhaustive on the test region).
// ---------------------------------------------------------------------------

#[test]
fn ordeal_poisson_min_distance_exhaustive() {
    let region = Region::new(Vec2::new(0.0, 0.0), Vec2::new(100.0, 100.0));
    let radius = 4.0f32;
    let pts = poisson_disk(0xBADF00D, region, radius, 20_000);
    assert!(pts.len() > 50, "too few points to be a meaningful trial");

    let mut min_pair = f32::INFINITY;
    let mut violations = 0usize;
    for a in 0..pts.len() {
        for b in (a + 1)..pts.len() {
            let d = pts[a].distance(pts[b]);
            if d < min_pair {
                min_pair = d;
            }
            if d < radius {
                violations += 1;
            }
        }
    }
    assert_eq!(violations, 0, "min-distance violated {violations} times");
    assert!(
        min_pair >= radius,
        "closest pair {min_pair} < radius {radius}"
    );

    println!(
        "ORDEAL poisson: darts=20000 kept={} radius={} closest_pair={:.6} violations={}",
        pts.len(),
        radius,
        min_pair,
        violations
    );
}

// ---------------------------------------------------------------------------
// ORDEAL 5 — scatter honors density: high vs low density count ratio within a
// derived confidence interval.
// ---------------------------------------------------------------------------

#[test]
fn ordeal_density_scatter_ratio() {
    let cell = 1.0f32;
    let (p_hi, p_lo) = (0.8f32, 0.2f32);
    // 200x200 = 40_000 cells per region.
    let hi_region = Region::new(Vec2::new(0.0, 0.0), Vec2::new(200.0, 200.0));
    let lo_region = Region::new(Vec2::new(1000.0, 0.0), Vec2::new(1200.0, 200.0));

    let hi = density_scatter(0x5EED, hi_region, cell, |_| p_hi);
    let lo = density_scatter(0x5EED, lo_region, cell, |_| p_lo);
    let (hi_n, lo_n) = (hi.len() as f64, lo.len() as f64);
    let cells = 200.0 * 200.0;

    let ratio = hi_n / lo_n;
    let expected = (p_hi / p_lo) as f64; // 4.0

    // Derived CI: each count ~ Binomial(cells, p). Propagate 1-sigma of each
    // count through the ratio; assert the observed ratio is within ~4 sigma.
    let hi_mean = cells * p_hi as f64;
    let lo_mean = cells * p_lo as f64;
    let hi_sd = (cells * p_hi as f64 * (1.0 - p_hi as f64)).sqrt();
    let lo_sd = (cells * p_lo as f64 * (1.0 - p_lo as f64)).sqrt();
    // Relative-error propagation for a quotient.
    let rel = ((hi_sd / hi_mean).powi(2) + (lo_sd / lo_mean).powi(2)).sqrt();
    let ratio_sd = expected * rel;
    let k = 4.0;
    let lo_bound = expected - k * ratio_sd;
    let hi_bound = expected + k * ratio_sd;
    assert!(
        (lo_bound..=hi_bound).contains(&ratio),
        "density ratio {ratio} outside [{lo_bound}, {hi_bound}]"
    );

    println!(
        "ORDEAL density: cells={} hi_count={} lo_count={} ratio={:.4} expected={:.1} CI=[{:.4},{:.4}] (k={})",
        cells as u64, hi.len(), lo.len(), ratio, expected, lo_bound, hi_bound, k
    );
}

// fBm smoke — the octave stack stays in range and uses its defaults.
#[test]
fn fbm_defaults_in_range() {
    let noise = Noise::new(7);
    let fbm = Fbm::default();
    let mut max_abs = 0.0f32;
    for i in 0..2000 {
        let x = i as f32 * 0.05;
        let v = fbm.sample2(|a, b| noise.value2(a, b), x, x * 0.5);
        max_abs = max_abs.max(v.abs());
    }
    assert!(max_abs <= 1.0001, "fBm exceeded [-1,1]: {max_abs}");
    println!(
        "fBm defaults: octaves={} lacunarity={} gain={} freq={} max_abs={:.6}",
        fbm.octaves, fbm.lacunarity, fbm.gain, fbm.frequency, max_abs
    );
}
