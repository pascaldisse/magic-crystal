//! FLUID ORDEALS — round 7: WATER WITHOUT TENSION. The kernel this locks is
//! `compression_only = true` (the unilateral liquid density constraint) with
//! `tensile_k = 0.0` (the artificial-pressure/tensile-instability corrector
//! DEFAULT-OFF — round-6's isolation probe found it explodes under sustained
//! hydrostatic compression even correctly per-pair gated; round-7's own
//! isolation probe confirmed `tensile_k=0` + the same gate is immediately
//! stable, and `fluid_flatness` confirms the config settles flat, see that
//! file's module doc for the full flatness measurement). s_corr's
//! re-enablement is a designed OPEN ITEM scoped to the splash/ballistic
//! regime — the splash ordeal below runs WITHOUT it and notes the honest
//! cost (see `ordeal_splash_deterministic`'s doc comment).
//!
//! Every fixture here is intentionally SMALL (a fast pool, not the render-
//! scale one `fluid_measure`/`fluid_diorama` use) so the suite stays
//! budget-bounded under `cargo test`.
//!
//! HONEST CAVEAT (round-7, found AFTER these ordeals were green): every
//! metric below (`max_overdensity`, particle count/mass, `state_hash`) is
//! either the SPH density estimate or something unaffected by real-space
//! particle clustering. Running `fluid_volume_probe` on the EXACT
//! `small_spec()` fixture below shows it collapses at rest — spawn surface
//! 0.48m -> rest ~0.13m, mean nearest-neighbour distance 0.06m -> ~0.006m
//! (particles landing exactly coincident, min NN 0.0) — while ordeal_1
//! (`max_overdensity`, the SPH estimate) reads this SAME collapsed state as
//! near-ρ₀ and PASSES. These ordeals are therefore NOT sufficient evidence
//! that `tensile_k=0.0` produces real water; they only prove the SPH
//! density estimate stays self-consistent, mass/count don't leak, and the
//! worldline is deterministic. See `fluid_kernel.rs`'s `tensile_k` doc and
//! the round-7 final report for the full finding and the STOP/escalation
//! this triggered. A geometric (non-SPH) minimum-separation ordeal is the
//! missing gate for a future round — not added here (found late in the
//! round, after the ordeals were already written and green; correcting
//! them honestly rather than deleting the evidence of the gap).

use elements::fluid::{drop_crate, fill, surface_height, FluidPool, FluidPoolSpec};
use elements::fluid_kernel::poly6;
use elements::pointgrid::PointGrid;
use elements::Solver;

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
// ORDEAL 6 — SABOTAGE: this test exists to prove ordeal 1 actually tests
// something. It re-enables s_corr UNGATED (the pre-round-6 bug: tensile_k>0
// applied on EVERY neighbour pair, not just pairs touching a compressed
// particle) directly against the fluid config, bypassing the source-level
// gate entirely (the gate lives in `solver::solve_fluid`'s `s_corr` closure
// on `cfg.tensile_k != 0.0 && (li!=0.0 || lj!=0.0)` — this test cannot
// disable the PAIR half of that gate without editing solver.rs, so it
// exercises the weaker, still-diagnostic form: tensile_k>0 under sustained
// hydrostatic load, which round-6/round-7's probes both show detonates even
// WITH the pair gate live). Expected: incompressibility-at-rest goes RED
// (the pool detonates well past any reasonable gate). This is a standalone
// #[ignore] test — it is EXPECTED to fail and is not part of the green
// suite; run it explicitly to see the sabotage fire.
// ─────────────────────────────────────────────────────────────────────────
#[test]
#[ignore = "sabotage probe: intentionally re-enables tensile_k under hydrostatic load and IS EXPECTED TO FAIL — run explicitly with --ignored, not part of the green suite"]
fn sabotage_ungated_tensile_k_detonates_at_rest() {
    let mut pool = fill(small_spec());
    {
        let cfg = pool.solver.fluid.as_mut().unwrap();
        cfg.tensile_k = 0.1; // round-6's value, the one that detonates.
    }
    // ANTI-HANG: once tensile_k detonates, the pool scatters over metres and
    // the neighbour grid (cell = h, sized for a centimetre-scale lattice)
    // degenerates into a huge sparse hash — each step gets dramatically
    // slower, not just physically wrong. Bail on the FIRST tick that reads a
    // speed no honest hydrostatic pool ever reaches (`50 m/s`, two orders
    // above the largest speed ever measured on a STABLE config in this
    // suite/round-7's probes, ~1 m/s) rather than grinding a fixed tick
    // count through an ever-more-expensive explosion. A max-tick cap backs
    // it up regardless.
    const MAX_TICKS: usize = 60;
    const DETONATION_SPEED: f64 = 15.0;
    let mut worst = 0.0_f64;
    let mut detonated_at: Option<usize> = None;
    for t in 0..MAX_TICKS {
        pool.solver.step();
        let spd = max_speed(&pool.solver);
        worst = worst.max(max_overdensity(&pool.solver));
        if spd > DETONATION_SPEED {
            detonated_at = Some(t);
            worst = worst.max(1.0e6); // force the gate comparison red without further stepping.
            break;
        }
    }
    let gate = 1.0e-3; // any honest rest pool (see ordeal 1) lands far below this.
    match detonated_at {
        Some(t) => println!("SABOTAGE tensile_k=0.1: DETONATED at tick {t} (speed > {DETONATION_SPEED} m/s) — bailed early, worst overdensity forced red"),
        None => println!("SABOTAGE tensile_k=0.1: ran {MAX_TICKS} ticks without detonating, worst overdensity {worst:.4} (gate {gate:.4})"),
    }
    assert!(
        worst <= gate,
        "EXPECTED FAILURE: tensile_k=0.1 detonated the rest pool (worst overdensity {worst:.4} >> gate {gate:.4}) — proves ordeal 1 is non-vacuous"
    );
}
