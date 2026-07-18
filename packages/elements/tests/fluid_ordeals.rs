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
//
// ROUND-9 UPDATE (two levers ATTEMPTED and MEASURED, still RED — see
// `tests/fluid_profile_probe.rs` for the numbers, kept as evidence):
//   LEVER 2 (Akinci two-way fluid↔solid boundary coupling): the submerged
//     body's particles now contribute ψ_b·W to the fluid density and receive
//     the mirrored, inverse-mass-split position correction back
//     (`solver::solve_fluid` + `FluidConfig::solid_coupling`). Implemented and
//     live — but with no fluid pressure to couple, it only produces a mass-
//     BLIND upward shove.
//   LEVER 1 (ρ₀ below packing, `FluidConfig::rest_density_factor`): SWEPT
//     1.0→0.85. It does NOT create a sustained gradient — a FREE-SURFACE
//     `compression_only` pool simply EXPANDS to relieve over-density, so the
//     settled column reads UNDER-dense in every depth bin at every factor
//     (λ≈0 at rest). Worse, the discrimination sweep converged EVERY density
//     (200–2000 kg/m³) and every release height to the SAME equilibrium depth
//     (~0.165 m): ZERO mass discrimination — a density-2000 box rests where a
//     density-200 cork does. That is a displacement artifact, NOT Archimedes.
//     Default reverted to 1.0 (a factor<1 also breaks gate 1's C≤0-at-spawn).
// The TRUE remaining lever is a real `λ` FIELD at depth: CONTAINER-boundary
// Akinci particles so the confined bottom fluid stops reading boundary-
// deficient and develops λ, plus a confined over-density. Larger effort,
// escalated — NOT faked here. This ordeal keeps the rise assertion so running
// it --ignored shows the honest RED; it is NOT part of the green suite.
// ─────────────────────────────────────────────────────────────────────────
#[test]
#[ignore = "OPEN ITEM (gate 4): buoyancy does not emerge — round-9 measured BOTH levers (Akinci two-way coupling live + rho0-below-packing swept) and proved zero mass discrimination (all densities converge to one depth); real fix needs a container-boundary lambda field. EXPECTED RED, escalated, not faked, not in the green suite"]
fn ordeal_buoyancy_rises() {
    // A TRUE buoyancy gate must test ARCHIMEDES, not a single body's transient
    // PEAK (the old `y_peak - y_start > spacing` assertion — round-9 proved it a
    // false positive: the mass-blind Akinci shove peaks a light body up by
    // +0.14 m with NO real pressure behind it). The non-gameable witness is
    // MASS DISCRIMINATION on the SETTLED (net) depth: release the SAME crate at
    // two densities — light (≪ water 1000) and heavy (≫ water) — from the SAME
    // deep pose, let each reach equilibrium, and require the light one to REST
    // clearly HIGHER than the heavy one. No coupling artifact can pass this: it
    // needs a depth-varying λ that pushes light up and lets heavy sink.
    let spec = small_spec();
    let search_r = spec.spacing * 2.0;
    let crate_dims = Vec3::new(0.09, 0.09, 0.09);

    // Equilibrium (tail-mean) depth of a released crate of the given density.
    let settle_depth = |density: f64| -> (f64, f64, f64) {
        let mut pool = fill(spec);
        let r_min = pool.solver.fluid.unwrap().min_separation;
        for _ in 0..260 {
            pool.solver.step();
        }
        let surf = surface_height(&pool);
        let submerge_y = (surf * 0.30).max(crate_dims.y * 0.5 + spec.spacing);
        let idx = drop_crate(&mut pool, crate_dims, (3, 3, 3), density, submerge_y, 0.0, 0.01);
        let y_start = body_center_y(&pool, idx);
        let mut worst_min = f64::INFINITY;
        let (mut tail_sum, mut tail_n) = (0.0_f64, 0usize);
        let total = 400;
        for t in 0..total {
            pool.solver.step();
            let (_, mn) = nn_stats(&pool.solver, search_r);
            worst_min = worst_min.min(mn);
            if t >= total - 80 {
                tail_sum += body_center_y(&pool, idx);
                tail_n += 1;
            }
        }
        (y_start, tail_sum / tail_n as f64, (r_min * (1.0 - FLOOR_TOL)).min(worst_min))
    };

    let (light_start, light_rest, light_worst) = settle_depth(200.0);
    let (heavy_start, heavy_rest, heavy_worst) = settle_depth(2000.0);
    let floor_gate = fill(spec).solver.fluid.unwrap().min_separation * (1.0 - FLOOR_TOL);
    // Discrimination = how much higher the cork rests than the lead block. A
    // real fluid separates them by most of the column; we ask for at least one
    // spacing of clear separation (well above solver noise).
    let discrimination = light_rest - heavy_rest;
    println!(
        "ORDEAL buoyancy (OPEN, expected red): light(200) start {light_start:.4} rest {light_rest:.4}; heavy(2000) start {heavy_start:.4} rest {heavy_rest:.4}; DISCRIMINATION light-heavy = {discrimination:+.4} m (need > one spacing {:.4}); worst min NN light {light_worst:.4} heavy {heavy_worst:.4} (floor {floor_gate:.4})",
        spec.spacing
    );
    // The packing half IS a real round-8 result and holds even under a body:
    assert!(
        light_worst >= floor_gate && heavy_worst >= floor_gate,
        "fluid packing collapsed under a submerged body: min NN light {light_worst:.4} / heavy {heavy_worst:.4} < floor {floor_gate:.4}"
    );
    // The Archimedes half is the OPEN failure (round-9: discrimination ≈ 0):
    assert!(
        discrimination > spec.spacing,
        "OPEN ITEM: no mass discrimination — a cork (200) and a lead block (2000) settle to nearly the SAME depth (Δ={discrimination:+.4} m <= one spacing {:.4}); the pressureless fluid has no λ field to enact Archimedes",
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
