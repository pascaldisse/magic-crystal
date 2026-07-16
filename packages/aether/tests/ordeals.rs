//! The A0 ordeals — the trials the participating-media substrate must survive.
//!
//! Run with `cargo test -p aether -- --nocapture` to see the per-ordeal
//! numbers printed. Each tolerance is *derived* (documented at its assertion),
//! never a magic number tuned to pass.

use aether::{
    in_scattered_energy, optical_depth, single_scatter, transmittance, vec3, CloudPuff, Constant,
    DensityGrid, HomogeneousMedium, Light, SteamColumn,
};
use std::f64::consts::PI;

/// ORDEAL 1 — analytic slab. Transmittance through a homogeneous slab of
/// constant density must equal `exp(−sigma_t · d)`.
///
/// Derivation of the tolerance: for constant density the midpoint quadrature
/// of `∫ sigma_t ds` is exact (`Σ sigma_t·ds = sigma_t·(t1−t0)`) regardless of
/// step count, so the only error is floating-point accumulation over `steps`
/// additions — bounded well under `1e-12`. We assert `1e-12`.
#[test]
fn ordeal_1_analytic_slab() {
    let medium = HomogeneousMedium::new(0.3, 0.7, 0.0); // sigma_t = 1.0
    let field = Constant(1.0);
    let d = 2.5;
    let mut worst = 0.0f64;
    for &steps in &[1usize, 8, 64, 512] {
        let t = transmittance(
            &medium,
            &field,
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            0.0,
            d,
            steps,
        );
        let analytic = (-medium.sigma_t() * d).exp();
        let err = (t - analytic).abs();
        worst = worst.max(err);
        eprintln!(
            "ORDEAL 1  slab  steps={steps:>4}  march={t:.17}  analytic={analytic:.17}  |err|={err:.3e}"
        );
    }
    // Also a non-unit sigma_t to prove the law, not the constant.
    let m2 = HomogeneousMedium::new(0.0, 2.3, 0.0);
    let t2 = transmittance(
        &m2,
        &field,
        vec3(-1.0, 2.0, 0.5),
        vec3(0.0, 1.0, 0.0),
        0.0,
        1.7,
        33,
    );
    let a2 = (-2.3 * 1.7f64).exp();
    let e2 = (t2 - a2).abs();
    worst = worst.max(e2);
    eprintln!(
        "ORDEAL 1  slab  sigma_t=2.3 d=1.7  march={t2:.17}  analytic={a2:.17}  |err|={e2:.3e}"
    );
    eprintln!("ORDEAL 1  worst |err| = {worst:.3e}  (tol 1e-12)");
    assert!(
        worst < 1e-12,
        "slab transmittance error {worst:.3e} exceeds 1e-12"
    );
}

/// ORDEAL 2 — the Henyey-Greenstein phase function integrates to 1 over the
/// sphere: `∮ p(μ) dω = 2π ∫_{-1}^{1} p(μ) dμ = 1`.
///
/// Derivation of the tolerance: Simpson's rule on a smooth integrand
/// (`g ≤ 0.8`, no pole reached) converges as `O(h⁴)`. With `N = 200_000`
/// panels the residual is far below `1e-6`; we assert `1e-6`.
#[test]
fn ordeal_2_hg_phase_normalized() {
    let n = 200_000usize; // even → Simpson
    let mut worst = 0.0f64;
    for &g in &[-0.8, -0.4, 0.0, 0.3, 0.6, 0.8] {
        let medium = HomogeneousMedium::new(0.0, 1.0, g);
        // Simpson over μ ∈ [-1, 1]; multiply by 2π for the azimuth.
        let h = 2.0 / n as f64;
        let mut sum = medium.phase(-1.0) + medium.phase(1.0);
        for i in 1..n {
            let mu = -1.0 + i as f64 * h;
            let w = if i % 2 == 1 { 4.0 } else { 2.0 };
            sum += w * medium.phase(mu);
        }
        let integral = 2.0 * PI * (h / 3.0) * sum;
        let err = (integral - 1.0).abs();
        worst = worst.max(err);
        eprintln!("ORDEAL 2  HG  g={g:+.2}  ∮p dω = {integral:.15}  |err|={err:.3e}");
    }
    eprintln!("ORDEAL 2  worst |err| = {worst:.3e}  (tol 1e-6)");
    assert!(
        worst < 1e-6,
        "HG phase normalization error {worst:.3e} exceeds 1e-6"
    );
}

/// ORDEAL 3 — energy. First-order in-scattered energy along a beam must never
/// exceed the extinction budget `1 − T`, and (unoccluded, unit light) equals
/// `albedo · (1 − T)`.
///
/// Derivation: with density 1, no shadow (`Tl = 1`), unit radiance, the
/// in-scatter integral is `sigma_s ∫₀ᵈ exp(−sigma_t t) dt = albedo·(1−T)`.
/// The march approximates `∫ Tc dt` with midpoint rule, error `O(ds²)`; at
/// `steps = 4096` over `d = 3` the residual is below `1e-6`, we assert `1e-5`.
#[test]
fn ordeal_3_energy_budget() {
    let medium = HomogeneousMedium::new(0.2, 0.8, 0.5); // sigma_t=1.0, albedo=0.8
    let field = Constant(1.0);
    let d = 3.0;
    let origin = vec3(0.0, 0.0, 0.0);
    let dir = vec3(1.0, 0.0, 0.0);
    let steps = 4096;

    let t_through = transmittance(&medium, &field, origin, dir, 0.0, d, steps);
    let budget = 1.0 - t_through;

    // Case A — no occlusion (shadow_dist = 0 ⇒ Tl = 1), unit directional light.
    let light = Light::Directional {
        to_light: vec3(0.0, 1.0, 0.0),
        radiance: 1.0,
    };
    let energy_a = in_scattered_energy(&medium, &field, origin, dir, 0.0, d, steps, &light, 0.0, 1);
    let expected = medium.albedo() * budget;
    let err_a = (energy_a - expected).abs();
    eprintln!(
        "ORDEAL 3  budget(1-T)={budget:.15}  energy_A={energy_a:.15}  albedo*(1-T)={expected:.15}  |err|={err_a:.3e}"
    );
    assert!(
        err_a < 1e-5,
        "energy vs albedo*(1-T) error {err_a:.3e} exceeds 1e-5"
    );
    assert!(
        energy_a <= budget + 1e-12,
        "energy_A {energy_a} exceeds budget {budget}"
    );

    // Case B — with self-shadowing occlusion (shadow_dist = 4). Still ≤ budget.
    let energy_b = in_scattered_energy(
        &medium, &field, origin, dir, 0.0, d, steps, &light, 4.0, 512,
    );
    let ratio = energy_b / budget;
    eprintln!("ORDEAL 3  occluded energy_B={energy_b:.15}  energy_B/budget={ratio:.15}  (≤ 1)");
    assert!(
        energy_b <= budget + 1e-12,
        "occluded energy_B {energy_b} exceeds budget {budget}"
    );

    // Single-scatter radiance toward the camera is likewise finite & bounded.
    let ss = single_scatter(
        &medium, &field, origin, dir, 0.0, d, steps, &light, 4.0, 512,
    );
    eprintln!("ORDEAL 3  single_scatter radiance toward camera = {ss:.15}");
    assert!(ss.is_finite() && ss >= 0.0);
}

/// ORDEAL 4 — determinism. Same seed ⇒ byte-identical grid AND byte-identical
/// march results. No randomness anywhere (ENTROPY law).
#[test]
fn ordeal_4_determinism() {
    let dims = [24usize, 40, 24];
    let vsize = 0.25;
    let origin = vec3(-3.0, 0.0, -3.0);
    let source = SteamColumn::default(); // seed baked into the preset

    let g1 = DensityGrid::rasterize(dims, vsize, origin, &source);
    let g2 = DensityGrid::rasterize(dims, vsize, origin, &source);

    // Byte-identical density buffers.
    let bits1: Vec<u32> = g1.data().iter().map(|v| v.to_bits()).collect();
    let bits2: Vec<u32> = g2.data().iter().map(|v| v.to_bits()).collect();
    assert_eq!(bits1, bits2, "grid data not byte-identical");

    // f16 round-trip is stable and deterministic.
    let h1 = g1.to_f16();
    let h2 = g2.to_f16();
    assert_eq!(h1, h2, "f16 conversion not deterministic");

    // Byte-identical march results through the grid.
    let medium = HomogeneousMedium::new(0.1, 0.9, 0.4);
    let light = Light::Point {
        position: vec3(2.0, 6.0, 1.0),
        intensity: 20.0,
    };
    let ss1 = single_scatter(
        &medium,
        &g1,
        vec3(-2.5, 4.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        0.0,
        5.0,
        256,
        &light,
        8.0,
        64,
    );
    let ss2 = single_scatter(
        &medium,
        &g2,
        vec3(-2.5, 4.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        0.0,
        5.0,
        256,
        &light,
        8.0,
        64,
    );
    eprintln!("ORDEAL 4  grid f32 buffer bits match: {}", bits1 == bits2);
    eprintln!(
        "ORDEAL 4  single_scatter run1 bits = {:#018x}",
        ss1.to_bits()
    );
    eprintln!(
        "ORDEAL 4  single_scatter run2 bits = {:#018x}",
        ss2.to_bits()
    );
    assert_eq!(
        ss1.to_bits(),
        ss2.to_bits(),
        "march result not byte-identical"
    );
    eprintln!("ORDEAL 4  determinism: grid + f16 + march byte-identical (PASS)");
}

/// ORDEAL 5 — grid vs analytic homogeneous agreement. A grid rasterized from a
/// constant field must transmit identically to the analytic constant field,
/// where the ray stays inside the grid box.
///
/// Derivation of the tolerance: trilinear interpolation of a constant grid
/// returns exactly that constant inside the box, so there is *no* march error
/// — the only disagreement is the grid's storage precision. The grid is `f32`
/// (the `f16`-convertible GPU-upload buffer), so a density `d` is stored as
/// `fl(d)` with relative rounding ≤ `f32::EPSILON` (2⁻²³). The optical depth
/// is `sigma_t·d·L`, so the bound is
/// `tol = sigma_t · d · L · f32::EPSILON` — the substrate agrees with the
/// analytic field to within its own storage resolution, no tighter claim is
/// honest. The analytic-vs-closed-form leg (pure `f64`) still asserts `1e-12`.
#[test]
fn ordeal_5_grid_vs_analytic() {
    let density = 1.3;
    let field = Constant(density);
    let dims = [8usize, 8, 8];
    let vsize = 1.0;
    let origin = vec3(0.0, 0.0, 0.0); // box [0,8]³
    let grid = DensityGrid::rasterize(dims, vsize, origin, &field);

    let medium = HomogeneousMedium::new(0.4, 0.6, 0.0); // sigma_t = 1.0
                                                        // Ray fully inside the box: from x=1 to x=7 at fixed y,z cell centers.
    let o = vec3(1.0, 4.0, 4.0);
    let dir = vec3(1.0, 0.0, 0.0);
    let (t0, t1, steps) = (0.0, 6.0, 128);

    let tau_grid = optical_depth(&medium, &grid, o, dir, t0, t1, steps);
    let tau_analytic = optical_depth(&medium, &field, o, dir, t0, t1, steps);
    let err = (tau_grid - tau_analytic).abs();

    // Derived tolerance: grid is f32-stored, so agreement is bounded by the
    // storage rounding of the density value carried along the path.
    let tol = medium.sigma_t() * density * (t1 - t0) * f64::from(f32::EPSILON);

    // Sanity: analytic leg matches the closed form to f64 accumulation error.
    let closed = medium.sigma_t() * density * (t1 - t0);
    eprintln!(
        "ORDEAL 5  tau_grid={tau_grid:.17}  tau_analytic={tau_analytic:.17}  closed={closed:.17}  |err|={err:.3e}  (tol {tol:.3e})"
    );
    assert!(
        err < tol,
        "grid vs analytic optical-depth error {err:.3e} exceeds f32-storage tol {tol:.3e}"
    );
    assert!(
        (tau_analytic - closed).abs() < 1e-12,
        "analytic vs closed form exceeds 1e-12"
    );

    // And a shaped source rasterizes without NaN/negatives (grid sanity).
    let puff = CloudPuff::default();
    let pgrid = DensityGrid::rasterize([16, 16, 16], 0.25, vec3(-2.0, -2.0, -2.0), &puff);
    let max = pgrid.data().iter().cloned().fold(0.0f64 as f32, f32::max);
    assert!(pgrid.data().iter().all(|v| v.is_finite() && *v >= 0.0));
    eprintln!("ORDEAL 5  cloud-puff grid: all finite & ≥0, max density = {max:.6}");
}
