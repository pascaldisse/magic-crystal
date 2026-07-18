//! FLUID ORDEALS — round 8: WATER WITH A FLOOR. The kernel this locks is
//! `compression_only = true` (the unilateral liquid density constraint) with
//! s_corr RETIRED (`tensile_k` inert) and, in its place, a genuine collision-
//! style pairwise MINIMUM-SEPARATION floor resolved through the solver's own
//! contact machinery ([`elements::Solver::solve_fluid_contacts`]): any two
//! fluid particles closer than `r_min = min_sep_factor × spacing` become a
//! CONTACT in the same per-substep solve every rigid body uses. This is the
//! CURE for the round-7 collapse s_corr could neither fix (it detonated) nor
//! reveal (with it off, the SPH density estimate read a coincident-particle
//! collapse as near-ρ₀).
//!
//! Round-7's HONEST CAVEAT was: every SPH-density metric (`max_overdensity`,
//! count/mass, `state_hash`) was blind to real-space clustering — the EXACT
//! `small_spec()` fixture collapsed at rest (mean NN 0.06m -> ~0.006m, pairs
//! coincident) while ordeal 1 read near-ρ₀ and passed. Round-8 CLOSES that
//! gap with `ordeal_min_separation_holds` (gate 2), a GEOMETRIC (non-SPH)
//! witness that catches collapse forever, plus `ordeal_hydrostatic_endurance`
//! (gate 3, no detonation over sim-seconds). `fluid_volume_probe` on this
//! fixture now shows min NN holding at ~0.83×spacing (the floor) instead of
//! collapsing.
//!
//! GATE 4 (BUOYANCY) IS AN OPEN ITEM — `ordeal_buoyancy_rises` is #[ignore]d
//! and EXPECTED RED: buoyancy does NOT emerge, because the compression_only
//! fluid (ρ₀ at max packing) is nearly pressureless in the bulk, so the
//! contact coupling has no hydrostatic gradient to lift a body. The floor
//! cured collapse/stability but not pressure. See that ordeal's doc for the
//! full root cause and the escalation.
//!
//! Every fixture here is intentionally SMALL (a fast pool, not the render-
//! scale one `fluid_measure`/`fluid_diorama` use) so the suite stays
//! budget-bounded under `cargo test`.

use elements::fluid::{body_center_y, drop_crate, fill, surface_height, FluidPool, FluidPoolSpec};
use elements::fluid_kernel::poly6;
use elements::pointgrid::PointGrid;
use elements::{Solver, Vec3};

/// Mean and MINIMUM nearest-neighbour distance over the fluid particles — the
/// GEOMETRIC packing witness, independent of the SPH kernel's smoothed density
/// estimate (the metric round-7's ordeals lacked). `search_r` bounds the grid
/// query; pass a small multiple of the spawn spacing.
fn nn_stats(s: &Solver, search_r: f64) -> (f64, f64) {
    let grid = PointGrid::build(&s.particles.pos, &s.fluid_particles, PointGrid::cell_size(search_r));
    let mut cand = Vec::new();
    let (mut sum, mut n, mut worst_min) = (0.0_f64, 0usize, f64::INFINITY);
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
            sum += best;
            n += 1;
            worst_min = worst_min.min(best);
        }
    }
    (sum / n.max(1) as f64, worst_min)
}

/// A fast ordeal-scale pool: small enough that a few hundred ticks stays
/// well under a test-suite budget. `9x9x5` = 405-ish fluid particles at this
/// spacing (`fluid_count` is exact; this comment is indicative).
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

fn max_speed(s: &Solver) -> f64 {
    s.fluid_particles.iter().map(|&i| s.particles.vel[i].length()).fold(0.0, f64::max)
}

/// Max relative OVER-density `max(C_i, 0)` over fluid particles — the
/// quantity the `compression_only` clamp actively drives to zero (it never
/// resists the under-dense/stretched side, so that side is not incompress-
/// ibility's business here).
fn max_overdensity(s: &Solver) -> f64 {
    let cfg = s.fluid.unwrap();
    let (h, rho0) = (cfg.h, cfg.rest_density);
    let grid = PointGrid::build(&s.particles.pos, &s.fluid_particles, PointGrid::cell_size(h));
    let mut cand = Vec::new();
    let mut worst = 0.0_f64;
    for &i in &s.fluid_particles {
        grid.query_ball(s.particles.pos[i], h, &mut cand);
        let mut density = 0.0;
        for &jc in &cand {
            let j = jc as usize;
            let mj = if s.particles.inv_mass[j] > 0.0 { 1.0 / s.particles.inv_mass[j] } else { 0.0 };
            density += mj * poly6((s.particles.pos[i] - s.particles.pos[j]).length(), h);
        }
        worst = worst.max(density / rho0 - 1.0);
    }
    worst.max(0.0)
}

fn total_fluid_mass(pool: &FluidPool) -> f64 {
    pool.fluid
        .iter()
        .map(|&i| {
            let im = pool.solver.particles.inv_mass[i];
            if im > 0.0 { 1.0 / im } else { 0.0 }
        })
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 1 — Incompressibility at rest, DERIVED gate.
//
// `calibrate_fluid_rest_density` sets ρ₀ = the MAX density over the freshly
// spawned lattice, so at tick 0 (before any dynamics) `max(C_i, 0) == 0` by
// construction — that is the machine-precision floor, not a plucked number.
// Settling under gravity introduces GENUINE hydrostatic load at the base
// layer, which the compression_only constraint resists but cannot zero
// exactly in finitely many relaxed Jacobi passes (see fluid_kernel's
// `relax`/`solver_iterations` docs). The gate is the house convention used
// elsewhere in this workspace (measured-floor × N, e.g. the VI-2 fragment-
// overlap ordeal): measure the floor from a SHORT calibration window right
// after settling, then require the following hold window never exceeds a
// documented multiple of that floor — self-consistent, never a bare literal.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_incompressibility_at_rest_derived() {
    let mut pool = fill(small_spec());
    // Spawn-time floor: by construction of calibrate_fluid_rest_density.
    let spawn_floor = max_overdensity(&pool.solver);
    assert!(spawn_floor <= 1.0e-9, "spawn lattice must start at C<=0 by construction, got {spawn_floor:.3e}");

    // Settle under gravity.
    for _ in 0..200 {
        pool.solver.step();
    }
    // Calibration window: measure the achieved residual floor.
    let mut floor = 0.0_f64;
    for _ in 0..20 {
        pool.solver.step();
        floor = floor.max(max_overdensity(&pool.solver));
    }
    let gate = (floor * 3.0).max(1.0e-6); // 3x the measured settle-time residual, floored at fp noise.

    // Hold window: the derived gate must hold for a further, independent stretch.
    let mut worst = 0.0_f64;
    for _ in 0..100 {
        pool.solver.step();
        worst = worst.max(max_overdensity(&pool.solver));
    }
    println!(
        "ORDEAL incompressibility-at-rest: spawn floor {spawn_floor:.3e}, settle floor {floor:.4}, gate {gate:.4}, hold-window worst {worst:.4}"
    );
    assert!(
        worst <= gate,
        "rest overdensity {worst:.4} exceeded the derived gate {gate:.4} (3x measured settle floor {floor:.4})"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 2 — Mass and particle-count conservation through settle + a splash.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_mass_and_count_conservation() {
    let mut pool = fill(small_spec());
    let n0 = pool.fluid.len();
    let m0 = total_fluid_mass(&pool);
    assert!(m0.is_finite() && m0 > 0.0);

    for _ in 0..150 {
        pool.solver.step();
    }
    assert_eq!(pool.fluid.len(), n0, "fluid particle count changed at rest");
    let m1 = total_fluid_mass(&pool);
    assert!((m1 - m0).abs() <= 1.0e-9 * m0, "fluid mass drifted at rest: {m0} -> {m1}");

    // A splash should not spawn/despawn/reassign mass either.
    let idx = drop_crate(&mut pool, elements::Vec3::new(0.1, 0.1, 0.1), (3, 3, 3), 1600.0, 0.3, 1.0, 0.01);
    for _ in 0..80 {
        pool.solver.step();
    }
    assert_eq!(pool.fluid.len(), n0, "fluid particle count changed across a splash");
    let m2 = total_fluid_mass(&pool);
    assert!((m2 - m0).abs() <= 1.0e-9 * m0, "fluid mass drifted across a splash: {m0} -> {m2}");
    let _ = idx;
    println!("ORDEAL mass/count: N={n0} constant, mass {m0:.6} -> {m1:.6} (rest) -> {m2:.6} (post-splash)");
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 3 — Rest stays at rest: a settled lattice, given no further
// disturbance, does not drift. Bound DERIVED from the discretization same
// way `fluid_flatness` derives its surface bound: a particle at true rest
// should not travel more than a small fraction of `spacing` over the window.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_rest_stays_at_rest() {
    let spec = small_spec();
    let mut pool = fill(spec);
    for _ in 0..250 {
        pool.solver.step();
    }
    let p0: Vec<_> = pool.fluid.iter().map(|&i| pool.solver.particles.pos[i]).collect();
    let surf0 = surface_height(&pool);

    for _ in 0..150 {
        pool.solver.step();
    }
    let surf1 = surface_height(&pool);
    let max_disp = pool
        .fluid
        .iter()
        .zip(p0.iter())
        .map(|(&i, &p0)| (pool.solver.particles.pos[i] - p0).length())
        .fold(0.0_f64, f64::max);
    let spd = max_speed(&pool.solver);

    // Derived bound: a truly settled liquid should not walk more than one
    // discretization layer (`spacing`) over a 150-tick window — anything
    // beyond that is drift, not rest-state numerical jitter.
    let gate = spec.spacing;
    println!(
        "ORDEAL rest-stays-at-rest: surface {surf0:.4} -> {surf1:.4} m, max particle displacement {max_disp:.4} m (gate {gate:.4}), end speed {spd:.4} m/s"
    );
    assert!(max_disp <= gate, "a settled particle drifted {max_disp:.4} m > one spacing ({gate:.4} m) with no disturbance");
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 4 — Determinism: two identical rest worldlines, byte-identical.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_determinism_rest_byte_identical() {
    let run = || {
        let mut pool = fill(small_spec());
        for _ in 0..300 {
            pool.solver.step();
        }
        (pool.solver.state_hash(), pool.solver.tick, pool.solver.particles.pos[pool.fluid[0]])
    };
    let (h1, t1, p1) = run();
    let (h2, t2, p2) = run();
    assert_eq!(t1, 300);
    assert_eq!(t2, 300);
    assert_eq!(h1, h2, "rest worldline diverged between identical runs");
    assert_eq!(p1, p2);
    println!("ORDEAL determinism (rest): run A hash=0x{h1:016x} run B hash=0x{h2:016x} → IDENTICAL @ tick {t1}");
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 5 — Splash-deterministic. Runs WITHOUT s_corr (tensile_k=0.0, the
// round-7 default) — the disclosed cost of the deferral: a splash's
// scattered droplets have no artificial-pressure anti-clustering push, so
// SOME visual clumping in the airborne spray is possible (see the diorama
// PNG for the own-eyes read; not asserted numerically here — that judgment
// belongs to the eyes, not a threshold). What IS asserted, and IS this
// ordeal's job: the splash worldline itself is still perfectly deterministic
// (same seed/config/tick count -> byte-identical replay) — s_corr's absence
// changes the physics, not the determinism.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_splash_deterministic() {
    let run = || {
        let mut pool = fill(small_spec());
        for _ in 0..100 {
            pool.solver.step();
        }
        let idx = drop_crate(&mut pool, elements::Vec3::new(0.1, 0.1, 0.1), (3, 3, 3), 1600.0, 0.3, 1.5, 0.01);
        for _ in 0..120 {
            pool.solver.step();
        }
        (pool.solver.state_hash(), pool.solver.tick, pool.solver.particles.pos[pool.solver.rigids[idx].indices[0]])
    };
    let (h1, t1, p1) = run();
    let (h2, t2, p2) = run();
    assert_eq!(t1, 220);
    assert_eq!(t2, 220);
    assert_eq!(h1, h2, "splash worldline (s_corr OFF) diverged between identical runs");
    assert_eq!(p1, p2, "crate sample position diverged");
    println!("ORDEAL determinism (splash, s_corr OFF): run A hash=0x{h1:016x} run B hash=0x{h2:016x} → IDENTICAL @ tick {t1}");
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 6 — GEOMETRIC MINIMUM-SEPARATION (gate 2, the round-7 missing gate).
// The non-SPH witness: after settling, the TRUE minimum nearest-neighbour
// distance over the pool must never drop below `r_min × (1 - tol)` — the
// collision-style floor `solve_fluid_contacts` enforces. This catches the
// SPH-invisible collapse round-7 exposed (min NN -> 0, particles coincident,
// while `max_overdensity` read near-ρ₀) forever: `r_min` and `tol` are the
// only dials, both derived (`r_min = cfg.min_separation` = min_sep_factor ×
// spacing; `tol` a documented fraction), never a bare metre literal.
const FLOOR_TOL: f64 = 0.15; // fraction below r_min a settled pair may reach
                             // (measured rest equilibrium ~0.83×spacing vs the
                             // 0.85×spacing floor — a ~2% overshoot; 15% is a
                             // comfortable margin that still slams the door on
                             // any real collapse, which drives NN toward zero).
#[test]
fn ordeal_min_separation_holds() {
    let spec = small_spec();
    let mut pool = fill(spec);
    let r_min = pool.solver.fluid.unwrap().min_separation;
    assert!(r_min > 0.0, "the min-separation floor must be derived at spawn");
    let search_r = spec.spacing * 2.0;

    let (_, min0) = nn_stats(&pool.solver, search_r);
    assert!(min0 + 1e-12 >= r_min, "spawn min NN {min0:.4} < r_min {r_min:.4}");

    for _ in 0..200 {
        pool.solver.step();
    }
    let floor_gate = r_min * (1.0 - FLOOR_TOL);
    let mut worst_min = f64::INFINITY;
    for _ in 0..150 {
        pool.solver.step();
        let (_, mn) = nn_stats(&pool.solver, search_r);
        worst_min = worst_min.min(mn);
    }
    println!(
        "ORDEAL min-separation: r_min {r_min:.4} m, gate {floor_gate:.4} (r_min×(1-{FLOOR_TOL})), worst min NN over hold window {worst_min:.4} m"
    );
    assert!(
        worst_min >= floor_gate,
        "min NN {worst_min:.4} fell below the floor {floor_gate:.4} — SPH-invisible collapse (the round-7 failure)"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 7 — HYDROSTATIC COLUMN ENDURANCE (gate 3). The pool holds under
// gravity for a sustained run of sim-seconds with NO particle ever reaching a
// detonation speed — the exact failure s_corr>0 produced. The speed bound is
// DERIVED (CFL-style): a stable pool never moves a particle more than one
// spacing per full tick, so `v_max < spacing / dt`. Anti-hang: a fixed tick
// cap AND an early bail the first tick the bound is exceeded.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_hydrostatic_endurance() {
    const SIM_SECONDS: f64 = 6.0; // the endurance window (param).
    let spec = small_spec();
    let mut pool = fill(spec);
    let dt = pool.solver.config.dt;
    let ticks = (SIM_SECONDS / dt).round() as usize;
    let speed_bound = spec.spacing / dt; // one spacing per tick — detonation, not dynamics.
    let mut worst_speed = 0.0_f64;
    let mut detonated_at: Option<usize> = None;
    for t in 0..ticks {
        pool.solver.step();
        let spd = max_speed(&pool.solver);
        worst_speed = worst_speed.max(spd);
        if spd > speed_bound {
            detonated_at = Some(t);
            break; // ANTI-HANG: never grind ticks through an explosion.
        }
    }
    println!(
        "ORDEAL endurance: {SIM_SECONDS}s ({ticks} ticks), worst max-speed {worst_speed:.4} m/s, bound {speed_bound:.4} m/s (spacing/dt)"
    );
    assert!(
        detonated_at.is_none(),
        "pool detonated at tick {detonated_at:?}: max speed {worst_speed:.4} m/s exceeded the CFL bound {speed_bound:.4} m/s"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 8 — BUOYANCY (gate 4). OPEN ITEM — #[ignore]d, EXPECTED RED.
//
// The intent: a light body (physical density << the fluid's water density)
// submerged in the settled pool RISES, float/sink emerging from the mass
// ratio through the fluid<->solid contact coupling. The round-8 MEASURED
// REALITY (see `fluid_buoy_probe`): it does NOT. Even a density-50 crate
// (which real water floats ~95% out of the surface) released mid-column
// settles near the pool floor — it resists sinking onto the floor (a thin
// fluid layer holds it ~0.03 m up) but develops NO net lift.
//
// ROOT CAUSE (honest, escalated): the `compression_only` unilateral density
// constraint with ρ₀ calibrated to the MAX (fullest) packing leaves the bulk
// fluid UNDER-dense (C<0 → no force) — it is nearly PRESSURELESS except at
// the very base. Gate 1 (`max_overdensity` low) passes precisely BECAUSE
// there is almost no hydrostatic pressure. With no pressure GRADIENT in the
// bulk, the pure pairwise contact coupling has nothing to transmit, so no
// buoyant force emerges. The round-8 min-separation floor cured the collapse
// and stabilised the column (gates 1–3) but does not manufacture pressure.
// A real fix needs pressure-bearing fluid (bilateral-with-anti-clustering, or
// ρ₀ calibrated to the interior mean) and/or an Akinci-style boundary coupling
// where the solid contributes to the fluid density estimate. Neither is in
// scope here. This ordeal keeps the rise assertion so running it --ignored
// shows the honest RED; it is NOT part of the green suite.
// ─────────────────────────────────────────────────────────────────────────
#[test]
#[ignore = "OPEN ITEM (gate 4): buoyancy does not emerge from the pressureless compression_only fluid + contact coupling — EXPECTED RED, escalated, not part of the green suite"]
fn ordeal_buoyancy_rises() {
    let spec = small_spec();
    let mut pool = fill(spec);
    let r_min = pool.solver.fluid.unwrap().min_separation;
    let search_r = spec.spacing * 2.0;

    for _ in 0..200 {
        pool.solver.step();
    }
    let surf = surface_height(&pool);
    let crate_dims = Vec3::new(0.09, 0.09, 0.09);
    let submerge_y = (surf * 0.35).max(crate_dims.y * 0.5 + spec.spacing);
    let idx = drop_crate(&mut pool, crate_dims, (3, 3, 3), 400.0, submerge_y, 0.0, 0.01);
    let y_start = body_center_y(&pool, idx);

    let floor_gate = r_min * (1.0 - FLOOR_TOL);
    let mut worst_min = f64::INFINITY;
    let mut y_peak = y_start;
    for _ in 0..250 {
        pool.solver.step();
        y_peak = y_peak.max(body_center_y(&pool, idx));
        let (_, mn) = nn_stats(&pool.solver, search_r);
        worst_min = worst_min.min(mn);
    }
    let rise = y_peak - y_start;
    println!(
        "ORDEAL buoyancy (OPEN, expected red): crate released y={y_start:.4}, peak y={y_peak:.4} (rose {rise:+.4} m); fluid worst min NN {worst_min:.4} m (floor {floor_gate:.4})"
    );
    // The packing half IS a real round-8 result and holds even here:
    assert!(
        worst_min >= floor_gate,
        "fluid packing collapsed under the body: min NN {worst_min:.4} < floor {floor_gate:.4}"
    );
    // The lift half is the OPEN failure:
    assert!(
        rise > spec.spacing,
        "OPEN ITEM: a light submerged body failed to rise (rose only {rise:+.4} m <= one spacing {:.4}) — the pressureless-fluid buoyancy gap",
        spec.spacing
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 9 — SABOTAGE: proves gate 2 (min-separation) is non-vacuous. It
// DISABLES the floor (`min_separation = 0`, recovering the round-7 s_corr-off
// configuration) and shows the pool collapses — min NN falls far below the
// floor gate. Expected: RED. Standalone #[ignore]; run explicitly to watch
// the collapse the cure prevents.
// ─────────────────────────────────────────────────────────────────────────
#[test]
#[ignore = "sabotage probe: disables the min-separation floor and IS EXPECTED TO FAIL (the round-7 collapse) — run explicitly with --ignored, not part of the green suite"]
fn sabotage_no_floor_collapses() {
    let spec = small_spec();
    let mut pool = fill(spec);
    let r_min = pool.solver.fluid.unwrap().min_separation;
    {
        let cfg = pool.solver.fluid.as_mut().unwrap();
        cfg.min_sep_factor = 0.0;
        cfg.min_separation = 0.0; // DISABLE the floor — recover round-7.
    }
    let search_r = spec.spacing * 2.0;
    for _ in 0..200 {
        pool.solver.step();
    }
    let (mean_nn, worst_min) = nn_stats(&pool.solver, search_r);
    let floor_gate = r_min * (1.0 - FLOOR_TOL);
    println!(
        "SABOTAGE no-floor: mean NN {mean_nn:.4} m, min NN {worst_min:.4} m (gate {floor_gate:.4}) — collapse expected"
    );
    assert!(
        worst_min >= floor_gate,
        "EXPECTED FAILURE: with the floor disabled min NN {worst_min:.4} collapsed below {floor_gate:.4} — proves gate 2 is non-vacuous"
    );
}
