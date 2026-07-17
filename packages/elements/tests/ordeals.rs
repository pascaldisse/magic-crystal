//! The ordeals — trials the Elements must pass. Determinism (the Loom's
//! clock), the analytic pendulum, hanging-chain length conservation, an
//! honest energy accounting, and fracture (love torn by strife).

use elements::{
    Collider, ContactMaterial, DistanceConstraint, RigidBody, Solver, SolverConfig, Vec3, LOVE,
};

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

// ═════════════════════════════════════════════════════════════════════════
// P2 ORDEALS — rigid bodies on the particle substrate.
// ═════════════════════════════════════════════════════════════════════════

/// Max magnitude of any body-particle velocity — the "still?" witness.
fn max_body_speed(s: &Solver, body: &RigidBody) -> f64 {
    body.indices
        .iter()
        .map(|&i| s.particles.vel[i].length())
        .fold(0.0_f64, f64::max)
}

/// Angle (radians) between the body's rotated up-axis and world up — the
/// "level?" witness. Zero == perfectly level.
fn tilt_from_level(body: &RigidBody) -> f64 {
    let up = Vec3::new(0.0, 1.0, 0.0);
    let mapped = body.rotation.mul_vec(up);
    (mapped.dot(up) / mapped.length()).clamp(-1.0, 1.0).acos()
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 6 — Box drop: a rigid box dropped onto a ground plane settles LEVEL
// and STILL (velocity → ~0, no jitter after settle), all measured.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_box_drop_settles_level_and_still() {
    let cfg = SolverConfig {
        dt: 1.0 / 120.0,
        substeps: 12,
        iterations: 1,
        ..SolverConfig::default()
    };
    let mut s = Solver::new(cfg);
    // Fully plastic ground so the drop settles quickly (restitution 0).
    let mat = ContactMaterial {
        restitution: 0.0,
        ..ContactMaterial::default()
    };
    s.collider = Some(Collider::ground_plane(0.0, 5.0, mat));
    let radius = 0.02;
    let body = s.spawn_rigid_box(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.5, 0.5, 0.5),
        (3, 3, 3),
        500.0,
        1.0,
        radius,
    );
    // 4 s of settling.
    for _ in 0..480 {
        s.step();
    }
    let speed = max_body_speed(&s, &s.rigids[body]);
    let tilt = tilt_from_level(&s.rigids[body]);
    // Bottom-layer particles should rest at ground + radius; measure the
    // spread of the lowest layer's heights (levelness of the resting face).
    let ys: Vec<f64> = s.rigids[body]
        .indices
        .iter()
        .map(|&i| s.particles.pos[i].y)
        .collect();
    let min_y = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let bottom_spread = ys
        .iter()
        .filter(|&&y| y < min_y + 0.05)
        .map(|&y| (y - (0.0 + radius)).abs())
        .fold(0.0_f64, f64::max);

    // TOLERANCE — derived + documented:
    //  * Residual jitter: at 12 substeps the read-back leaves each contact a
    //    sub-mm/s tremor; a plastic (e=0) ground bleeds it out. Budget 1e-3 m/s.
    //  * Levelness: the box enters axis-aligned and never tips; the only tilt
    //    is polar-fit round-off, ~1e-6 rad. Budget 1e-2 rad (0.57°) — generous.
    //  * Resting height: bottom particles held at ground+radius within the
    //    per-substep penetration depth. Budget 2 mm.
    assert!(
        speed < 1.0e-3,
        "box still jittering: max particle speed {speed:.3e} m/s (> 1e-3)"
    );
    assert!(
        tilt < 1.0e-2,
        "box not level: up-axis tilted {:.4}° (> 0.57°)",
        tilt.to_degrees()
    );
    assert!(
        bottom_spread < 2.0e-3,
        "resting face off ground+radius by {:.4} mm (> 2 mm)",
        bottom_spread * 1000.0
    );
    println!(
        "ORDEAL box-drop: settled  max speed={speed:.3e} m/s  tilt={:.5}°  \
         bottom-face height error={:.4} mm  (level & still)",
        tilt.to_degrees(),
        bottom_spread * 1000.0
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 7 — Rigid invariance: a tumbling box (stiffness 1.0) preserves every
// pairwise particle distance within tolerance — it does not deform.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_rigid_invariance_under_tumbling() {
    let cfg = SolverConfig {
        dt: 1.0 / 120.0,
        substeps: 20,
        iterations: 1,
        gravity: Vec3::ZERO, // pure tumble — isolate the shape-match fidelity
        ..SolverConfig::default()
    };
    let mut s = Solver::new(cfg);
    let body_idx = s.spawn_rigid_box(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.6, 0.4, 0.5),
        (3, 3, 3),
        800.0,
        1.0, // perfectly rigid
        0.02,
    );
    // Seed a tumble: v_i = ω × r_i + v_linear, about a tilted axis.
    let omega = Vec3::new(1.5, 2.5, -1.0); // rad/s
    let v_lin = Vec3::new(0.3, 0.0, -0.2);
    let centroid = s.rigids[body_idx].centroid;
    for &i in &s.rigids[body_idx].indices.clone() {
        let r = s.particles.pos[i] - centroid;
        s.particles.vel[i] = omega.cross(r) + v_lin;
    }
    // Record rest pairwise distances.
    let idx = s.rigids[body_idx].indices.clone();
    let rest_dist =
        |s: &Solver, a: usize, b: usize| (s.particles.pos[a] - s.particles.pos[b]).length();
    let rest: Vec<f64> = (0..idx.len())
        .flat_map(|a| (a + 1..idx.len()).map(move |b| (a, b)))
        .map(|(a, b)| rest_dist(&s, idx[a], idx[b]))
        .collect();

    let mut worst = 0.0_f64;
    for _ in 0..600 {
        s.step();
        let mut k = 0;
        for a in 0..idx.len() {
            for b in a + 1..idx.len() {
                let d = rest_dist(&s, idx[a], idx[b]);
                let rel = (d - rest[k]).abs() / rest[k];
                worst = worst.max(rel);
                k += 1;
            }
        }
    }
    // TOLERANCE — derived + documented:
    //  * Shape matching at 1 iteration/substep leaves an O((ω·dt_sub)²)
    //    residual before re-fit. Here max ω≈3.1 rad/s, dt_sub=1/2400 →
    //    (ω·dt_sub)² ≈ 1.7e-6. Budget 5e-4 (0.05 %) — orders above the bound,
    //    covering fit round-off across 600 ticks.
    assert!(
        worst < 5.0e-4,
        "rigid body deformed: worst pairwise distance drift {:.4}% (> 0.05%)",
        worst * 100.0
    );
    println!(
        "ORDEAL rigid-invariance: 27-particle box tumbling (|ω|≈{:.2} rad/s) \
         over 600 ticks — worst pairwise distance drift {:.5}% (rigid held)",
        omega.length(),
        worst * 100.0
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 8 — Determinism with rigids + collision: byte-identical state hash
// at tick 1000 across two runs.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_determinism_rigid_collision() {
    let run = || {
        let cfg = SolverConfig {
            dt: 1.0 / 120.0,
            substeps: 10,
            iterations: 1,
            seed: 777,
            ..SolverConfig::default()
        };
        let mut s = Solver::new(cfg);
        s.collider = Some(Collider::ground_plane(0.0, 8.0, ContactMaterial::default()));
        // Two bodies, dropped from an angle-inducing offset so they tumble,
        // bounce and settle — exercising every P2 path.
        s.spawn_rigid_box(
            Vec3::new(-0.3, 1.2, 0.1),
            Vec3::new(0.4, 0.4, 0.4),
            (3, 3, 3),
            600.0,
            1.0,
            0.03,
        );
        s.spawn_rigid_sphere(Vec3::new(0.5, 1.6, -0.2), 0.25, 3, 700.0, 1.0, 0.03);
        // Tilt the box so it lands on an edge and tumbles.
        let idx = s.rigids[0].indices.clone();
        for &i in &idx {
            s.particles.vel[i] = Vec3::new(0.7, 0.0, 0.4);
        }
        for _ in 0..1000 {
            s.step();
        }
        (s.state_hash(), s.tick, s.rigids[0].centroid)
    };
    let (h1, t1, c1) = run();
    let (h2, t2, c2) = run();
    assert_eq!(t1, 1000);
    assert_eq!(t2, 1000);
    assert_eq!(h1, h2, "state hash diverged with rigids + collision");
    assert_eq!(c1, c2, "body centroid diverged");
    println!(
        "ORDEAL determinism (rigid+collision): run A=0x{h1:016x} run B=0x{h2:016x} \
         → IDENTICAL @ tick {t1}, centroid=({:.5},{:.5},{:.5})",
        c1.x, c1.y, c1.z
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 9 — Restitution: a dropped body's bounce apex ratio matches the
// restitution parameter (apex/drop height ratio ≈ e²), within derived tol.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_restitution_apex_matches_param() {
    let e = 0.5;
    let g = 9.81;
    let cfg = SolverConfig {
        dt: 1.0 / 240.0,
        substeps: 8,
        iterations: 1,
        gravity: Vec3::new(0.0, -g, 0.0),
        ..SolverConfig::default()
    };
    let mut s = Solver::new(cfg);
    let mat = ContactMaterial {
        restitution: e,
        friction_static: 0.0,
        friction_dynamic: 0.0,
        ..ContactMaterial::default()
    };
    s.collider = Some(Collider::ground_plane(0.0, 5.0, mat));
    let r = 0.1;
    let y0 = 2.0;
    // A one-particle "sphere" (point mass) — clean restitution, no internal
    // modes to launder energy through.
    let body = s.spawn_rigid_sphere(Vec3::new(0.0, y0, 0.0), r, 1, 500.0, 1.0, r);
    let ci = s.rigids[body].indices[0];

    let mut bounced = false;
    let mut peak = r;
    for _ in 0..4000 {
        s.step();
        let y = s.particles.pos[ci].y;
        let vy = s.particles.vel[ci].y;
        if !bounced && vy > 0.0 && y < r + 0.05 {
            bounced = true;
        }
        if bounced {
            if y > peak {
                peak = y;
            }
            if vy < 0.0 {
                break; // apex passed
            }
        }
    }
    // Apex height above the contact point vs the original drop height above
    // contact — the energy ratio e².
    let measured = (peak - r) / (y0 - r);
    let expected = e * e;
    let rel_err = (measured - expected).abs() / expected;
    // TOLERANCE — derived + documented:
    //  * Restitution is applied on the contact substep using the pre-substep
    //    incoming speed; up to one substep of gravity (g·dt_sub ≈ 5.1e-3 m/s
    //    vs v_in ≈ 6.1 m/s → ~8e-4 rel) is unaccounted. Apex sampling adds
    //    ≤ ½g·dt² of height error. Budget 3 %.
    assert!(
        rel_err < 0.03,
        "restitution apex ratio {measured:.4} vs e²={expected:.4} off by {:.2}% (> 3%)",
        rel_err * 100.0
    );
    println!(
        "ORDEAL restitution: e={e}  apex/drop ratio measured={measured:.4}  \
         expected e²={expected:.4}  rel_err={:.2}%",
        rel_err * 100.0
    );
}

/// Lay a rigid box FLUSH on a plane defined by `(along_x, up_slope, normal)`
/// through the origin: a `counts` lattice filling `dims` (x along `along_x`,
/// y along `normal`/thickness, z along `up_slope`), its base sitting at
/// `normal·radius`. Returns the new body index.
#[allow(clippy::too_many_arguments)]
fn place_box_on_plane(
    s: &mut Solver,
    along_x: Vec3,
    up_slope: Vec3,
    normal: Vec3,
    dims: Vec3,
    counts: (usize, usize, usize),
    density: f64,
    radius: f64,
    margin: f64,
) -> usize {
    let (nx, ny, nz) = (counts.0.max(1), counts.1.max(1), counts.2.max(1));
    let n_total = nx * ny * nz;
    let particle_mass = density * dims.x * dims.y * dims.z / n_total as f64;
    let inv_mass = 1.0 / particle_mass;
    let step = Vec3::new(
        if nx > 1 {
            dims.x / (nx - 1) as f64
        } else {
            0.0
        },
        if ny > 1 {
            dims.y / (ny - 1) as f64
        } else {
            0.0
        },
        if nz > 1 {
            dims.z / (nz - 1) as f64
        } else {
            0.0
        },
    );
    let mut indices = Vec::with_capacity(n_total);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let local_x = -dims.x * 0.5 + step.x * ix as f64;
                // Base layer resting at `radius + margin` above the plane — the
                // effective contact distance, so the contact is live from tick
                // 1 with no initial penetration (a deeper overlap reads back as
                // a pop) and no empty substep for velocity to leak through.
                let local_y = radius + margin + step.y * iy as f64;
                let local_z = -dims.z * 0.5 + step.z * iz as f64;
                let pos = along_x.scale(local_x) + normal.scale(local_y) + up_slope.scale(local_z);
                indices.push(s.particles.add_with_radius(pos, inv_mass, radius));
            }
        }
    }
    let body = RigidBody::from_indices(&s.particles, indices, 1.0, s.config.polar);
    s.rigids.push(body);
    s.rigids.len() - 1
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 10 — Friction on an incline: below the critical repose angle
// (tan θ_c = μ_s) the box HOLDS; above it, the box SLIDES. Both sides tested.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_friction_incline_critical_angle() {
    let mu_s = 0.6_f64;
    let mu_d = 0.4_f64;
    let critical = mu_s.atan(); // tan θ_c = μ_s
    let g = 9.81;

    // Run a box on an incline of `angle`; return down-slope centroid travel
    // over the LATE window (after a 0.75 s settle) — the slide signature.
    let travel_on_incline = |angle: f64| -> f64 {
        let cfg = SolverConfig {
            dt: 1.0 / 240.0,
            substeps: 16,
            iterations: 8,
            gravity: Vec3::new(0.0, -g, 0.0),
            ..SolverConfig::default()
        };
        let mut s = Solver::new(cfg);
        let (sin, cos) = (angle.sin(), angle.cos());
        let normal = Vec3::new(0.0, cos, sin);
        let along_x = Vec3::new(1.0, 0.0, 0.0);
        let up_slope = Vec3::new(0.0, sin, -cos); // ⟂ normal & x
        let down_slope = up_slope.scale(-1.0); // gravity's in-plane pull
        let mat = ContactMaterial {
            friction_static: mu_s,
            friction_dynamic: mu_d,
            restitution: 0.0,
            ..ContactMaterial::default()
        };
        s.collider = Some(Collider::incline(angle, 20.0, mat));
        let radius = 0.03;
        let margin = ContactMaterial::default().contact_margin;
        // A 3-D box aligned FLUSH to the ramp face (its lattice frame is the
        // ramp's along/normal/up-slope), so it rests on a full face and does
        // not tumble. Friction is enforced at the body level (one contact
        // supports the whole weight), so the hold condition is the clean
        // geometric Coulomb law tanθ < μ_s regardless of the layer count.
        let body = place_box_on_plane(
            &mut s,
            along_x,
            up_slope,
            normal,
            Vec3::new(0.3, 0.3, 0.3),
            (3, 3, 3),
            600.0,
            radius,
            margin,
        );
        // Settle 0.75 s.
        for _ in 0..180 {
            s.step();
        }
        let start = s.rigids[body].centroid;
        // Then measure 1.25 s of down-slope travel.
        for _ in 0..300 {
            s.step();
        }
        let end = s.rigids[body].centroid;
        (end - start).dot(down_slope)
    };

    // Below critical (holds): 6° under.
    let below = critical - 6.0_f64.to_radians();
    let travel_hold = travel_on_incline(below);
    // Above critical (slides): 6° over.
    let above = critical + 6.0_f64.to_radians();
    let travel_slide = travel_on_incline(above);

    // TOLERANCE — derived + documented:
    //  * The Coulomb hold condition is exactly tan θ < μ_s (the dt_sub²
    //    factors on drive and stiction depth cancel), so the sign flips AT
    //    θ_c = atan(μ_s). A held box stays within per-substep round-off:
    //    budget 5 mm of drift over 1.25 s. A sliding box accelerates at
    //    g(sin θ − μ_d cos θ) ≈ 1.8 m/s² here → ≫ 0.1 m over the window.
    assert!(
        travel_hold < 5.0e-3,
        "box on {:.1}° (< θ_c={:.1}°) slid {:.4} m — static friction failed",
        below.to_degrees(),
        critical.to_degrees(),
        travel_hold
    );
    assert!(
        travel_slide > 0.1,
        "box on {:.1}° (> θ_c={:.1}°) only moved {:.4} m — did not slide",
        above.to_degrees(),
        critical.to_degrees(),
        travel_slide
    );
    println!(
        "ORDEAL friction: μ_s={mu_s} → θ_c={:.2}°.  {:.1}° HELD (down-slope travel \
         {:.4} m)  ·  {:.1}° SLID (travel {:.4} m).  Coulomb critical angle confirmed.",
        critical.to_degrees(),
        below.to_degrees(),
        travel_hold,
        above.to_degrees(),
        travel_slide
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL 11 — Energy honesty (rigid path): a tumbling box in zero gravity
// bleeds kinetic energy through shape-match re-fit + PBD read-back. Measure
// and REPORT the damping; assert only that energy is not injected and stays
// bounded — never assert zero.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_rigid_energy_honest() {
    let cfg = SolverConfig {
        dt: 1.0 / 240.0,
        substeps: 16,
        iterations: 1,
        gravity: Vec3::ZERO,
        ..SolverConfig::default()
    };
    let mut s = Solver::new(cfg);
    let body = s.spawn_rigid_box(
        Vec3::ZERO,
        Vec3::new(0.6, 0.4, 0.5),
        (3, 3, 3),
        800.0,
        1.0,
        0.02,
    );
    let omega = Vec3::new(2.0, 3.0, -1.5);
    let idx = s.rigids[body].indices.clone();
    let centroid = s.rigids[body].centroid;
    for &i in &idx {
        let r = s.particles.pos[i] - centroid;
        s.particles.vel[i] = omega.cross(r);
    }
    let kinetic = |s: &Solver| -> f64 {
        idx.iter()
            .map(|&i| {
                let m = 1.0 / s.particles.inv_mass[i];
                0.5 * m * s.particles.vel[i].dot(s.particles.vel[i])
            })
            .sum()
    };
    let e0 = kinetic(&s);
    let ticks = 2400; // 10 s
    for _ in 0..ticks {
        s.step();
    }
    let e_end = kinetic(&s);
    let lost = e0 - e_end;
    let frac = lost / e0;
    println!(
        "ORDEAL rigid-energy (honest): KE0={e0:.6}J  KEend={e_end:.6}J over {ticks} ticks (10 s).  \
         Shape-match re-fit + PBD read-back bled {lost:.6}J = {:.3}% of the rotational KE \
         ({:.4}%/s).  The rigid path damps MONOTONICALLY — small at 16 substeps but REAL, \
         NOT zero.",
        frac * 100.0,
        frac * 100.0 / 10.0
    );
    assert!(
        lost >= -1.0e-6,
        "energy GREW — solver injected energy (blow-up): KE0={e0} KEend={e_end}"
    );
    assert!(
        frac < 0.5,
        "lost {:.1}% of rotational KE in 10 s — damping too aggressive",
        frac * 100.0
    );
}

// ═════════════════════════════════════════════════════════════════════════
// VI-1 ORDEALS — THE STACK TOPPLES. N rigid boxes stacked on a ground plane;
// an impulse (op data, chosen by the test, never a hardcoded engine constant)
// pushes the top box; the stack topples and settles.
// ═════════════════════════════════════════════════════════════════════════

/// Build a stack of `n` rigid boxes resting on the ground plane at `y=0`,
/// each box directly atop the one below (SAME derivation the realm authors
/// use — `tests/physics.rs:86` / `worlds/naruko/scenes/main.json`):
/// `rest_y[0] = ground + half_extent + radius`,
/// `rest_y[k] = rest_y[k-1] + 2*half_extent + radius`. Returns the solver and
/// the rigid indices, bottom to top.
fn build_stack(
    cfg: SolverConfig,
    n: usize,
    half_extent: f64,
    density: f64,
    stiffness: f64,
    radius: f64,
) -> (Solver, Vec<usize>) {
    let mut s = Solver::new(cfg);
    let mat = ContactMaterial {
        restitution: 0.0,
        ..ContactMaterial::default()
    };
    s.collider = Some(Collider::ground_plane(0.0, 50.0, mat));
    let dims = Vec3::new(2.0 * half_extent, 2.0 * half_extent, 2.0 * half_extent);
    let mut rigids = Vec::with_capacity(n);
    let mut y = radius + half_extent;
    for _ in 0..n {
        let idx = s.spawn_rigid_box(
            Vec3::new(0.0, y, 0.0),
            dims,
            (3, 3, 3),
            density,
            stiffness,
            radius,
        );
        rigids.push(idx);
        y += 2.0 * half_extent + radius;
    }
    (s, rigids)
}

/// Total linear momentum of the system: `Σ mass_i · vel_i` over every free
/// (non-anchor) particle.
fn total_momentum(s: &Solver) -> Vec3 {
    let mut p = Vec3::ZERO;
    for i in 0..s.particles.pos.len() {
        let inv_m = s.particles.inv_mass[i];
        if inv_m == 0.0 {
            continue; // anchors carry no momentum (infinite mass)
        }
        p = p + s.particles.vel[i].scale(1.0 / inv_m);
    }
    p
}

/// The stack-topple test config: matches the realm's tick rate (`worlds/
/// naruko` authors `tick_dt = 1/60`) with the solver default substep count.
fn topple_cfg() -> SolverConfig {
    SolverConfig {
        dt: 1.0 / 60.0,
        seed: 777,
        ..SolverConfig::default()
    }
}

/// Drive a fresh 3-box stack through settle → impulse → re-settle, returning
/// the per-tick state hashes and (for the caller to inspect) the momentum
/// trace. Shared by the replay and momentum ordeals so both trials watch the
/// exact same choreography.
fn run_topple(settle_ticks: u64, topple_ticks: u64, impulse: Vec3) -> (Solver, Vec<u64>) {
    let (mut s, rigids) = build_stack(topple_cfg(), 3, 0.4, 500.0, 1.0, 0.05);
    let mut hashes = Vec::with_capacity((settle_ticks + topple_ticks) as usize);
    for _ in 0..settle_ticks {
        s.step();
        hashes.push(s.state_hash());
    }
    let top = *rigids.last().unwrap();
    s.apply_impulse(top, impulse);
    for _ in 0..topple_ticks {
        s.step();
        hashes.push(s.state_hash());
    }
    (s, hashes)
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL VI-1a — replay determinism: two full topples from the same seed
// fold to byte-identical state hashes at EVERY tick (settle + impulse +
// re-settle, not just the end state).
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_stack_topple_replay_is_byte_identical() {
    let settle_ticks = 300u64; // 5 s — the stack is at rest well before this
    let topple_ticks = 600u64; // 10 s — settles again well before this
    let impulse = Vec3::new(3.0, 0.0, 0.0); // op data: the test's pick, not an engine constant
    let (_, hashes_a) = run_topple(settle_ticks, topple_ticks, impulse);
    let (_, hashes_b) = run_topple(settle_ticks, topple_ticks, impulse);
    assert_eq!(
        hashes_a.len(),
        (settle_ticks + topple_ticks) as usize,
        "ticked the full topple"
    );
    assert_eq!(
        hashes_a, hashes_b,
        "two identical topples must fold to identical state hashes at every tick"
    );
    println!(
        "ORDEAL stack-topple replay: {} ticks x 2 runs, byte-identical every tick, final hash {:#018x}",
        hashes_a.len(),
        hashes_a.last().unwrap()
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL VI-1b — momentum drift, DERIVED. The floor is measured on the SAME
// stack at rest (no impulse, no net external force — gravity is exactly
// balanced by the ground contact, so any tick-to-tick momentum change is
// pure numerical noise). The topple then starts and ends at rest (momentum
// ~0 both ends, by construction — the read-back velocity of an unmoving
// particle is exactly zero): the ground is an infinite-mass anchor, so the
// applied impulse's momentum, and every contact/friction force the topple
// rides through, is absorbed into it and never carried by the tracked
// system once it re-settles. The honest "total drift" is therefore just the
// system's net momentum change from before-impulse to after-resettle — it
// should itself be ~0, gated at ~10x the resting floor times the tick count
// (the floor scaled up for the many more ticks, and the substantially larger
// transient forces, the topple's contact solve rides through).
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_stack_topple_momentum_drift_bounded() {
    let (mut s, rigids) = build_stack(topple_cfg(), 3, 0.4, 500.0, 1.0, 0.05);
    let settle_ticks = 300u64;
    for _ in 0..settle_ticks {
        s.step();
    }
    // FLOOR — measured at rest, no impulse: the per-tick momentum-change
    // magnitude with zero net external force.
    let floor_ticks = 60u64; // 1 s
    let mut floor = 0.0_f64;
    let mut prev = total_momentum(&s);
    for _ in 0..floor_ticks {
        s.step();
        let now = total_momentum(&s);
        floor = floor.max((now - prev).length());
        prev = now;
    }
    // A fully-settled stack's read-back velocity lands on bit-exact 0.0 (the
    // friction bleed clamps to the residual, XPBD read-back is exact
    // subtraction) — the measured floor can legitimately BE zero. Floor it at
    // the f64 unit-in-last-place scaled to this system's momentum magnitude
    // (~10^2-10^3 kg*m/s, from the stack's ~50 kg total mass and the m/s-scale
    // impulse below) so the derived gate is never vacuously zero.
    let floor = floor.max(f64::EPSILON * 1.0e3);
    let momentum_before = total_momentum(&s);

    // THE IMPULSE — op data, the test's choice.
    let top = *rigids.last().unwrap();
    let delta_velocity = Vec3::new(3.0, 0.0, 0.0);
    s.apply_impulse(top, delta_velocity);

    let topple_ticks = 600u64; // 10 s — the stack topples and re-settles well inside this
    for _ in 0..topple_ticks {
        s.step();
    }
    let momentum_after = total_momentum(&s);

    // Confirm the episode actually ends at rest (the "starts and ends at
    // rest" premise the drift derivation leans on).
    let end_speed = rigids
        .iter()
        .map(|&r| max_body_speed(&s, &s.rigids[r]))
        .fold(0.0_f64, f64::max);
    assert!(
        end_speed < 1.0e-2,
        "stack did not re-settle within {topple_ticks} ticks: max speed {end_speed:.4}"
    );

    let unexplained = momentum_after - momentum_before;
    let total_drift = unexplained.length();
    let gate = 10.0 * floor * topple_ticks as f64;
    println!(
        "ORDEAL stack-topple momentum: resting floor={floor:.3e} kg*m/s/tick over {floor_ticks} ticks; \
         topple={topple_ticks} ticks; gate=10x floor x ticks={gate:.3e}; \
         momentum before={:?} after={:?} (impulse applied={:?} kg*m/s, absorbed by the ground on re-settle); \
         unexplained drift={total_drift:.3e} kg*m/s (< gate: {})",
        momentum_before, momentum_after, delta_velocity.scale(s.rigids[top].total_mass),
        total_drift < gate
    );
    assert!(
        total_drift < gate,
        "unexplained momentum drift {total_drift:.4} kg*m/s exceeds derived gate {gate:.4} \
         (floor {floor:.3e} x 10 x {topple_ticks} ticks)"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL VI-1c — stack-at-rest stability: once the topple settles, M
// consecutive ticks (M = round(1/dt), one second's worth) show every body's
// max particle speed below the SAME rest floor `ordeal_box_drop_settles_
// level_and_still` uses (1.0e-3 m/s — residual XPBD read-back tremor on a
// plastic contact) — no jitter, no reinvented tolerance.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_stack_at_rest_stability_after_topple() {
    const REST_SPEED_FLOOR: f64 = 1.0e-3; // ordeal_box_drop_settles_level_and_still's floor
    let cfg = topple_cfg();
    let (mut s, rigids) = build_stack(cfg, 3, 0.4, 500.0, 1.0, 0.05);
    let settle_ticks = 300u64;
    for _ in 0..settle_ticks {
        s.step();
    }
    let top = *rigids.last().unwrap();
    s.apply_impulse(top, Vec3::new(3.0, 0.0, 0.0));
    let topple_ticks = 600u64;
    for _ in 0..topple_ticks {
        s.step();
    }
    // M ticks derived from the tick rate — one second's worth.
    let m = (1.0 / cfg.dt).round() as u64;
    let mut max_speed_over_window = 0.0_f64;
    for _ in 0..m {
        s.step();
        let speed = rigids
            .iter()
            .map(|&r| max_body_speed(&s, &s.rigids[r]))
            .fold(0.0_f64, f64::max);
        max_speed_over_window = max_speed_over_window.max(speed);
    }
    println!(
        "ORDEAL stack-at-rest: M={m} ticks (1/dt={:.4}) post-topple, max particle speed \
         {max_speed_over_window:.3e} m/s (< floor {REST_SPEED_FLOOR:.1e})",
        1.0 / cfg.dt
    );
    assert!(
        max_speed_over_window < REST_SPEED_FLOOR,
        "stack still jittering {m} ticks after the topple: max speed {max_speed_over_window:.3e} \
         m/s (>= floor {REST_SPEED_FLOOR:.1e})"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// ORDEAL VI-1e — F2/F3: MIXED-RADIUS body-vs-body rest. Two single-particle
// rigid bodies with DIFFERENT collision radii (`r_bottom != r_top`) stack on
// a ground plane; the derived rest gap between their centres is
// `mean(r_bottom, r_top) + contact_margin` (solver.rs' `solve_body_
// collisions` doc-comment) — NOT the bare mean (pre-F2 bug: no margin) and
// NOT the sum (pre-F3 bug: double-counts relative to the static single-
// radius convention). Deliberately unequal radii so a formula regressing to
// either wrong convention is caught: a bare-mean bug undershoots the gap by
// exactly `contact_margin`; a sum bug overshoots it by
// `mean(r_bottom, r_top)` — both far outside the derived tolerance below.
// ─────────────────────────────────────────────────────────────────────────
#[test]
fn ordeal_mixed_radius_bodies_rest_at_derived_gap() {
    let cfg = SolverConfig {
        dt: 1.0 / 120.0,
        substeps: 12,
        seed: 999,
        ..SolverConfig::default()
    };
    let mut s = Solver::new(cfg);
    let mat = ContactMaterial {
        restitution: 0.0,
        ..ContactMaterial::default()
    };
    s.collider = Some(Collider::ground_plane(0.0, 50.0, mat));
    let contact_margin = mat.contact_margin;

    let r_bottom = 0.04; // collision radius, bottom body
    let r_top = 0.09; // collision radius, top body — deliberately != r_bottom
    let density = 500.0;
    let sphere_radius = 0.05; // mass-bearing volume radius (both bodies)

    // Bottom body authored already near its static rest height (radius +
    // margin above the ground, the SAME single-radius convention the static
    // pass uses) so it settles fast; top body dropped from a few cm above
    // where the derived formula predicts it should come to rest.
    let bottom_y = r_bottom + contact_margin;
    let expected_gap = (r_bottom + r_top) * 0.5 + contact_margin; // F2/F3 formula
    let drop_height = 0.05;
    let bottom = s.spawn_rigid_sphere(
        Vec3::new(0.0, bottom_y, 0.0),
        sphere_radius,
        1,
        density,
        1.0,
        r_bottom,
    );
    let top = s.spawn_rigid_sphere(
        Vec3::new(0.0, bottom_y + expected_gap + drop_height, 0.0),
        sphere_radius,
        1,
        density,
        1.0,
        r_top,
    );

    let settle_ticks = 300u64; // 2.5 s at dt=1/120 — well past the drop_height's fall
    for _ in 0..settle_ticks {
        s.step();
    }
    let speed = max_body_speed(&s, &s.rigids[bottom]).max(max_body_speed(&s, &s.rigids[top]));
    assert!(
        speed < 1.0e-2,
        "mixed-radius pair did not settle within {settle_ticks} ticks: max speed {speed:.4}"
    );

    let bottom_pos = s.particles.pos[s.rigids[bottom].indices[0]];
    let top_pos = s.particles.pos[s.rigids[top].indices[0]];
    let measured_gap = (top_pos - bottom_pos).length();

    // Derived tolerance: `physics.rs::crate_rest_at_derived_analytic_height`
    // measured the settle residual on a resting contact at ≈ contact_margin
    // (the particle hovers within the surface skin) and set its tolerance to
    // 6x that — tight enough to fail a wrong-convention bug by an order of
    // magnitude, loose enough never to flap on ordinary settle jitter. Same
    // derivation, reused (not reinvented) here.
    let tol = 6.0 * contact_margin;
    println!(
        "ORDEAL mixed-radius rest: r_bottom={r_bottom} r_top={r_top} contact_margin={contact_margin:.1e}; \
         measured gap={measured_gap:.6} expected mean(r_i,r_j)+margin={expected_gap:.6} \
         (bare-mean would predict {:.6}, sum would predict {:.6}); tol={tol:.1e}",
        (r_bottom + r_top) * 0.5,
        r_bottom + r_top + contact_margin,
    );
    assert!(
        (measured_gap - expected_gap).abs() < tol,
        "mixed-radius rest gap {measured_gap:.6} != derived mean(r_i,r_j)+margin {expected_gap:.6} \
         (tol {tol:.1e}) — bare-mean would give {:.6}, sum would give {:.6}",
        (r_bottom + r_top) * 0.5,
        r_bottom + r_top + contact_margin,
    );
}
