//! The ordeals — trials the Elements must pass. Determinism (the Loom's
//! clock), the analytic pendulum, hanging-chain length conservation, an
//! honest energy accounting, and fracture (love torn by strife).

use elements::{DistanceConstraint, Solver, SolverConfig, Vec3, LOVE};

/// Build a hanging chain: particle 0 anchored at `origin`, `n` links of
/// `seg` length descending in `-x`, each link a rigid-ish distance bond.
fn hanging_chain(cfg: SolverConfig, n: usize, seg: f64, mass: f64, compliance: f64) -> Solver {
    let mut s = Solver::new(cfg);
    // Anchor at origin (infinite mass).
    s.particles.add_mass(Vec3::new(0.0, 0.0, 0.0), 0.0);
    for i in 1..=n {
        let x = -(i as f64) * seg;
        s.particles.add_mass(Vec3::new(x, 0.0, 0.0), mass);
        s.constraints
            .push(DistanceConstraint::new(i - 1, i, seg, compliance, LOVE));
    }
    s
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 1 — Determinism: two runs of a 500-particle chain, byte-identical.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_determinism_byte_identical() {
    let run = || {
        let cfg = SolverConfig {
            seed: 12345,
            ..SolverConfig::default()
        };
        // 1 anchor + 499 links = 500 particles.
        let mut s = hanging_chain(cfg, 499, 0.1, 1.0, 1.0e-6);
        for _ in 0..1000 {
            s.step();
        }
        (s.state_hash(), s.tick, s.particles.pos[250])
    };
    let (h1, t1, p1) = run();
    let (h2, t2, p2) = run();
    assert_eq!(t1, 1000);
    assert_eq!(t2, 1000);
    assert_eq!(h1, h2, "state hash diverged between identical worldlines");
    assert_eq!(p1, p2, "sample position diverged");
    println!(
        "ORDEAL determinism: run A hash=0x{h1:016x} run B hash=0x{h2:016x} → IDENTICAL @ tick {t1}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 2 — Single pendulum period vs analytic small-angle T = 2π√(L/g).
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_pendulum_period() {
    // High substep count for a faithful stiff (rigid) bond; small angle so
    // the analytic small-angle formula is the ground truth.
    let g = 9.81;
    let l = 1.0;
    let cfg = SolverConfig {
        dt: 1.0 / 240.0,
        substeps: 16,
        iterations: 1,
        gravity: Vec3::new(0.0, -g, 0.0),
        ..SolverConfig::default()
    };
    let mut s = Solver::new(cfg);
    // Anchor at origin; bob released from small angle θ0 in the x–y plane.
    let theta0 = 0.05_f64; // rad — small; amplitude correction ≈ θ0²/16 ≈ 1.6e-4
    let bx = l * theta0.sin();
    let by = -l * theta0.cos();
    s.particles.add_mass(Vec3::new(0.0, 0.0, 0.0), 0.0);
    s.particles.add_mass(Vec3::new(bx, by, 0.0), 1.0);
    s.constraints
        .push(DistanceConstraint::new(0, 1, l, 0.0, LOVE)); // rigid rod

    // Detect a full period by counting x zero-crossings (bob swings through
    // the bottom). Two crossings in the same direction = one period.
    let dt = cfg.dt;
    let mut prev_x = s.particles.pos[1].x;
    let mut crossings: Vec<f64> = Vec::new();
    let max_ticks = 2000;
    for k in 1..=max_ticks {
        s.step();
        let x = s.particles.pos[1].x;
        // Rising-through-zero crossing (negative → positive).
        if prev_x < 0.0 && x >= 0.0 {
            // Linear-interpolate the crossing time within the tick.
            let frac = -prev_x / (x - prev_x);
            crossings.push((k as f64 - 1.0 + frac) * dt);
        }
        prev_x = x;
        if crossings.len() >= 2 {
            break;
        }
    }
    assert!(crossings.len() >= 2, "did not observe a full period");
    let measured = crossings[1] - crossings[0];
    let analytic = 2.0 * std::f64::consts::PI * (l / g).sqrt();
    let rel_err = (measured - analytic).abs() / analytic;

    // TOLERANCE — derived + documented:
    //   * XPBD substep discretization (dt_sub = 1/(240·16) ≈ 260 µs) leaves an
    //     O(dt_sub²) period error, empirically < 0.3 %.
    //   * PBD velocity read-back adds slight numerical damping, negligible
    //     over one period for period *timing* (it bleeds amplitude, not rate).
    //   * Finite-angle correction of the true pendulum, θ0²/16 ≈ 1.6e-4,
    //     which we deliberately keep below the tolerance by choosing θ0=0.05.
    // Budget: 1 %.
    let tol = 0.01;
    assert!(
        rel_err < tol,
        "pendulum period off by {:.4}% (measured {:.5}s vs analytic {:.5}s)",
        rel_err * 100.0,
        measured,
        analytic
    );
    println!(
        "ORDEAL pendulum: measured T={:.5}s  analytic T={:.5}s  rel_err={:.4}%  (tol {:.1}%)",
        measured,
        analytic,
        rel_err * 100.0,
        tol * 100.0
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 3 — Hanging chain length conservation: each rigid link holds its
// rest length once settled.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_chain_length_conservation() {
    let seg = 0.25;
    let n = 20;
    let cfg = SolverConfig {
        dt: 1.0 / 120.0,
        substeps: 20,
        iterations: 1,
        ..SolverConfig::default()
    };
    // Nearly-rigid bonds (tiny compliance) so stretch is bounded and small.
    let mut s = hanging_chain(cfg, n, seg, 1.0, 1.0e-7);
    for _ in 0..3000 {
        s.step();
    }
    // Measure worst-case segment stretch and total-length drift.
    let mut max_stretch = 0.0_f64;
    let mut total = 0.0_f64;
    for c in &s.constraints {
        let len = (s.particles.pos[c.a] - s.particles.pos[c.b]).length();
        total += len;
        let stretch = (len - c.rest).abs() / c.rest;
        max_stretch = max_stretch.max(stretch);
    }
    let ideal = n as f64 * seg;
    let drift = (total - ideal).abs() / ideal;
    // Compliant bonds stretch a little under load — bounded by compliance.
    // Budget: worst link < 1 % stretch, total length within 0.5 %.
    assert!(
        max_stretch < 0.01,
        "worst link stretched {:.4}% (> 1% budget)",
        max_stretch * 100.0
    );
    assert!(
        drift < 0.005,
        "total length drifted {:.4}% (> 0.5% budget)",
        drift * 100.0
    );
    println!(
        "ORDEAL chain length: {} links, worst stretch {:.4}%, total {:.5}m vs ideal {:.5}m (drift {:.4}%)",
        n,
        max_stretch * 100.0,
        total,
        ideal,
        drift * 100.0
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 4 — Energy accounting: honest report of XPBD damping (NOT asserted
// zero). We measure the total mechanical energy of a swinging pendulum over
// many periods and report the decay.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_energy_behavior_documented() {
    let g = 9.81;
    let l = 1.0;
    let m = 1.0;
    let cfg = SolverConfig {
        dt: 1.0 / 240.0,
        substeps: 16,
        iterations: 1,
        gravity: Vec3::new(0.0, -g, 0.0),
        ..SolverConfig::default()
    };
    let mut s = Solver::new(cfg);
    let theta0 = 0.3_f64; // larger swing to have appreciable energy
    s.particles.add_mass(Vec3::new(0.0, 0.0, 0.0), 0.0);
    s.particles
        .add_mass(Vec3::new(l * theta0.sin(), -l * theta0.cos(), 0.0), m);
    s.constraints
        .push(DistanceConstraint::new(0, 1, l, 0.0, LOVE));

    let energy = |s: &Solver| -> f64 {
        let p = s.particles.pos[1];
        let v = s.particles.vel[1];
        // Reference potential from the anchor (y=0). KE + PE.
        0.5 * m * v.dot(v) + m * g * p.y
    };
    let e0 = energy(&s);
    // The pendulum's OSCILLATION-ENERGY budget: the energy above the lowest
    // point (pure PE at the bottom, y = -l). This is the quantity XPBD
    // damping actually bleeds — total mechanical energy is (ideally) constant,
    // so drift must be measured against the swing budget, not the total.
    let swing_budget = e0 - (m * g * (-l));
    // ~10 periods: T ≈ 2.006s, dt = 1/240 → ~4800 ticks.
    let ticks = 4800;
    let periods = ticks as f64 * cfg.dt / (2.0 * std::f64::consts::PI * (l / g).sqrt());
    for _ in 0..ticks {
        s.step();
    }
    let e_end = energy(&s);
    let lost = e0 - e_end; // positive == energy bled off (damping)
    let frac = lost / swing_budget; // fraction of the swing budget lost
    let per_period = frac / periods;
    println!(
        "ORDEAL energy (XPBD damping, honest): E0={:.6}J  Eend={:.6}J  \
         swing budget={:.6}J.  Over {:.2} periods the read-back bled \
         {:.4}J = {:.2}% of the swing budget ({:.3}%/period).  \
         XPBD/PBD velocity read-back is NOT symplectic — it damps \
         MONOTONICALLY; the loss is small at 16 substeps but REAL, not zero.",
        e0,
        e_end,
        swing_budget,
        periods,
        lost,
        frac * 100.0,
        per_period * 100.0
    );
    // Assert only the physically-required facts: energy does NOT spuriously
    // grow (no blow-up), and the damping stays bounded (stable, not runaway).
    assert!(
        lost >= -1e-4,
        "energy GREW — solver injected energy (blow-up), E0={e0} Eend={e_end}"
    );
    assert!(
        frac < 0.5,
        "lost {:.1}% of swing budget over {:.1} periods — damping too aggressive",
        frac * 100.0,
        periods
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 5 — Fracture: a chain with ONE weak link (love 0.1) breaks at the
// weak link under load; the strong links (love 1.0) hold.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_fracture_weak_link_breaks() {
    let seg = 0.2;
    let n = 8;
    let cfg = SolverConfig {
        dt: 1.0 / 120.0,
        substeps: 12,
        iterations: 1,
        // Threshold tuned so a love-0.1 bond (breaks at 0.1·threshold) gives
        // way under the chain's hanging load, while love-1.0 bonds never do.
        fracture_threshold: 200.0,
        ..SolverConfig::default()
    };
    let mut s = Solver::new(cfg);
    s.particles.add_mass(Vec3::new(0.0, 0.0, 0.0), 0.0);
    for i in 1..=n {
        s.particles
            .add_mass(Vec3::new(-(i as f64) * seg, 0.0, 0.0), 1.0);
        // The 4th link (between particle 3 and 4) is the weak one.
        let love = if i == 4 { 0.1 } else { LOVE };
        s.constraints
            .push(DistanceConstraint::new(i - 1, i, seg, 1.0e-6, love));
    }
    let weak = (3usize, 4usize);
    let n_before = s.constraints.len();
    let mut broke_at: Option<u64> = None;
    for _ in 0..600 {
        s.step();
        if !s.fractures.is_empty() && broke_at.is_none() {
            broke_at = Some(s.fractures[0].tick);
        }
        if s.constraints.len() < n_before {
            break;
        }
    }
    // Exactly the weak link fractured; strong links survive.
    assert_eq!(
        s.fractures.len(),
        1,
        "expected exactly one fracture, got {}",
        s.fractures.len()
    );
    let ev = s.fractures[0];
    assert_eq!(
        (ev.a, ev.b),
        weak,
        "the WRONG bond broke: {:?}",
        (ev.a, ev.b)
    );
    assert_eq!(ev.love, 0.1, "the broken bond was not the weak one");
    // The weak bond is gone; all remaining bonds are strong (love 1.0).
    assert_eq!(s.constraints.len(), n_before - 1);
    for c in &s.constraints {
        assert_eq!(c.bond.love, LOVE, "a strong bond was torn");
    }
    println!(
        "ORDEAL fracture: weak link ({},{}) love={} broke at tick {} under strife {:.1}; \
         {} strong bonds (love=1.0) held.",
        ev.a,
        ev.b,
        ev.love,
        broke_at.unwrap(),
        ev.strife,
        s.constraints.len()
    );
}
