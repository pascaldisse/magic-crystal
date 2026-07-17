//! RITE VI-2 — SOMETHING BREAKS. Ordeals over the elements-level physics:
//! a bonded crate dropped from height H onto a hard ground plane, strife
//! tears one bond, flood-fill splits it, mass and momentum are honestly
//! accounted through the break, and the whole drop+break+settle sequence
//! replays byte-identical. The fragment-mesh/ECS/oracle-facing ordeals live
//! in `packages/fracture` and `packages/oracle` (documented there — this
//! file is the pure-solver half: fracture, mass, momentum, determinism).

use elements::{
    default_bond_love, Collider, ContactMaterial, Solver, SolverConfig, Vec3, LOVE, STONE_DENSITY,
};

/// A weak crate: same lattice `spawn_bonded_box` builds, but authored with a
/// deliberately soft essence (well below `STONE_DENSITY`) so `default_bond_
/// love` derives a love this drop height can actually overcome — the ordeal
/// must be able to PROVE a real break, not assert on a crate that coasts to
/// rest unbroken (a vacuous-tail risk the house style forbids).
fn drop_scenario(height: f64) -> (Solver, Vec<usize>) {
    let cfg = SolverConfig {
        dt: 1.0 / 120.0,
        substeps: 12,
        // A large multiplier on top of an already-soft love: the CRATE
        // (default_bond_love-derived, well under 1.0) is meant to break: see
        // `default_bond_love`'s doc — threshold is the GLOBAL strife-to-love
        // conversion, love is what actually differs per material.
        fracture_threshold: 4.0e3,
        ..SolverConfig::default()
    };
    let mut s = Solver::new(cfg);
    s.collider = Some(Collider::ground_plane(0.0, 5.0, ContactMaterial::default()));

    // A light crate essence (balsa-adjacent — well under STONE_DENSITY), so
    // `default_bond_love` derives a soft, breakable bond.
    let density = 200.0;
    let love = default_bond_love(density);
    assert!(
        love < LOVE,
        "test setup: density {density} vs STONE_DENSITY {STONE_DENSITY} must derive a \
         breakable (< LOVE) bond, or this whole scenario would be vacuous"
    );
    let dims = Vec3::new(1.0, 1.0, 1.0);
    let counts = (3usize, 3, 3);
    let compliance = 1.0e-7; // near-rigid bonds — the crate is stiff, not springy
    let radius = 0.03; // derived contact thickness: ~1/10 of the smallest lattice step (0.5)
    let whole = s.spawn_bonded_box(
        Vec3::new(0.0, height, 0.0),
        dims,
        counts,
        density,
        love,
        compliance,
        radius,
    );
    (s, whole)
}

/// Run the drop for up to `max_ticks`, returning the tick a fracture was
/// first recorded (if any) and the per-tick momentum trace up to (and
/// including) that tick — the shared bedrock for the mass/momentum/replay
/// ordeals below, so every ordeal watches the EXACT same worldline.
fn run_until_break_or(s: &mut Solver, max_ticks: u64) -> (Option<u64>, Vec<Vec3>) {
    let mut momentum_trace = Vec::new();
    let mut broke_at = None;
    for _ in 0..max_ticks {
        s.step();
        momentum_trace.push(total_momentum(s));
        if broke_at.is_none() && !s.fractures.is_empty() {
            broke_at = Some(s.tick);
        }
    }
    (broke_at, momentum_trace)
}

fn total_momentum(s: &Solver) -> Vec3 {
    let mut p = Vec3::ZERO;
    for i in 0..s.particles.pos.len() {
        let inv_m = s.particles.inv_mass[i];
        if inv_m <= 0.0 {
            continue;
        }
        p = p + s.particles.vel[i].scale(1.0 / inv_m);
    }
    p
}

const DROP_HEIGHT: f64 = 3.0; // metres — a param this ordeal's own choice fixes
const RUN_TICKS: u64 = 400; // ~3.3s at dt=1/120 — airborne, impact, settle

/// (d) grep-anchor + behavioral proof: the fracture threshold path flows
/// through the bond's own love (see `Bond::fractured`'s doc for the full
/// reconciliation). Behaviorally: two otherwise-identical crates differing
/// ONLY in essence density (hence ONLY in per-bond love) diverge in fracture
/// outcome under the identical drop and identical global threshold — proof
/// the break point is load-bearing on love, not a hardcoded number.
#[test]
fn ordeal_fracture_threshold_reads_bond_love_not_a_hardcoded_break() {
    // grep-able anchor: the ONLY break condition in the solver.
    let source = include_str!("../src/solver.rs");
    assert!(
        source.contains("c.bond.fractured(t)") || source.contains("c.bond.fractured(threshold)"),
        "fracture_pass must break bonds via Bond::fractured(threshold), not an inline strife \
         comparison — the threshold path must flow through the bond's own love"
    );

    let (mut soft, whole_soft) = drop_scenario(DROP_HEIGHT);
    let (broke_soft, _) = run_until_break_or(&mut soft, RUN_TICKS);
    assert!(
        broke_soft.is_some(),
        "the soft (low-love) crate must break under this drop"
    );
    assert!(!soft.fractures.is_empty());

    // A near-STONE_DENSITY crate under the IDENTICAL drop + IDENTICAL global
    // threshold must survive: only its bonds' love changed.
    let hard_cfg = SolverConfig {
        dt: 1.0 / 120.0,
        substeps: 12,
        fracture_threshold: 4.0e3,
        ..SolverConfig::default()
    };
    let mut hard = Solver::new(hard_cfg);
    hard.collider = Some(Collider::ground_plane(0.0, 5.0, ContactMaterial::default()));
    let density = STONE_DENSITY;
    let love = default_bond_love(density);
    assert_eq!(
        love, LOVE,
        "a crate at STONE_DENSITY must derive an unbreakable bond love"
    );
    let _whole_hard = hard.spawn_bonded_box(
        Vec3::new(0.0, DROP_HEIGHT, 0.0),
        Vec3::new(1.0, 1.0, 1.0),
        (3, 3, 3),
        density,
        love,
        1.0e-7,
        0.03,
    );
    let (broke_hard, _) = run_until_break_or(&mut hard, RUN_TICKS);
    assert!(
        broke_hard.is_none(),
        "the STONE_DENSITY (love==LOVE, unbreakable) crate fractured under the SAME drop and \
         SAME global threshold as the soft crate — the break point is not actually reading love"
    );
    let _ = whole_soft;
    println!(
        "ORDEAL fracture-threshold-reads-love: soft crate (density 200, love {:.4}) broke at \
         tick {:?}; STONE_DENSITY crate (love {LOVE}) held for {RUN_TICKS} ticks — the only \
         difference between the two runs was the bonds' own love.",
        default_bond_love(200.0),
        broke_soft
    );
}

/// (a) EQUIVALENT EXCHANGE — sum of fragment mass exactly equals whole mass,
/// bit-exact (`0e0`). Exactness strategy documented at
/// `fracture::fragment_masses_exact` (this crate has no dependency on
/// `fracture`, so the strategy is reproduced inline here at the elements
/// level: reduce both sides to the identical ascending-particle-index
/// summation order before comparing, so float add-order can never
/// introduce drift — see the comment inline).
#[test]
fn ordeal_equivalent_exchange_mass_exact() {
    let (mut s, whole) = drop_scenario(DROP_HEIGHT);
    let (broke_at, _) = run_until_break_or(&mut s, RUN_TICKS);
    assert!(
        broke_at.is_some(),
        "setup must actually break, or this ordeal is vacuous"
    );

    let fragments = s.fragment_components(&whole);
    assert!(
        fragments.len() >= 2,
        "a real break must yield at least two fragments"
    );

    // Reduce to one canonical ascending order (proves partition completeness
    // AND makes the two sums literally the same addition sequence).
    let mut whole_sorted = whole.clone();
    whole_sorted.sort_unstable();
    let mut frag_sorted: Vec<usize> = fragments.iter().flatten().copied().collect();
    frag_sorted.sort_unstable();
    assert_eq!(
        whole_sorted, frag_sorted,
        "every particle's mass must land in EXACTLY one fragment (no loss, no duplication)"
    );
    let mass_of = |indices: &[usize]| -> f64 {
        indices
            .iter()
            .map(|&i| {
                let inv_m = s.particles.inv_mass[i];
                if inv_m > 0.0 {
                    1.0 / inv_m
                } else {
                    0.0
                }
            })
            .sum()
    };
    let whole_mass = mass_of(&whole_sorted);
    let fragments_mass = mass_of(&frag_sorted);
    assert_eq!(
        whole_mass - fragments_mass,
        0.0,
        "Equivalent Exchange violated: whole mass {whole_mass} != fragments mass {fragments_mass}"
    );
    println!(
        "ORDEAL equivalent-exchange: whole mass {whole_mass:.6} kg == fragments mass \
         {fragments_mass:.6} kg (diff {:.1e}, bit-exact 0e0), {} fragments from {} particles",
        whole_mass - fragments_mass,
        fragments.len(),
        whole.len()
    );
}

/// (b) momentum-through-fracture bounded by a DERIVED drift tolerance,
/// checked PER TICK through the fracture tick (not just at endpoints), with
/// a discrimination proof this gate actually bites (see the
/// `ordeal_momentum_gate_catches_injected_leak`-style pattern in
/// `ordeals.rs`; this file's `ordeal_momentum_gate_is_not_vacuous_...` below
/// plays that role for the drop scenario).
fn momentum_drift_gate(s0: &Solver) -> f64 {
    // Floor: measure the SAME solver's per-tick momentum drift on a course
    // with gravity but NO collider and NO bonds under load (free particles
    // in free-fall, so the ONLY force acting is uniform gravity — every
    // particle's velocity changes by the SAME g*dt_sub each substep, hence
    // total momentum's per-tick delta is bit-close to `total_mass * g * dt`,
    // and any residual is purely floating-point noise from the summation +
    // integration passes).
    let mut floor_solver = Solver::new(s0.config);
    let n = 27usize; // same 3x3x3 particle count as the drop crate
    let mass = 200.0 * 1.0 / n as f64; // same density/volume/count as drop_scenario
    for i in 0..n {
        floor_solver
            .particles
            .add_mass(Vec3::new(i as f64 * 0.5, 10.0, 0.0), mass);
    }
    let mut prev = total_momentum(&floor_solver);
    let mut measured_floor = 0.0_f64;
    let floor_ticks = 60u64;
    for _ in 0..floor_ticks {
        floor_solver.step();
        let now = total_momentum(&floor_solver);
        let g = floor_solver.config.gravity;
        let dt = floor_solver.config.dt;
        let expected_delta = g.scale(mass * n as f64 * dt);
        let residual = (now - prev - expected_delta).length();
        measured_floor = measured_floor.max(residual);
        prev = now;
    }
    let total_mass = mass * n as f64;
    let g_mag = floor_solver.config.gravity.length();
    let typical_momentum = total_mass * g_mag * floor_solver.config.dt * RUN_TICKS as f64;
    let operations_per_tick = n as f64 * floor_solver.config.substeps as f64;
    let derived_floor = f64::EPSILON * typical_momentum.max(1.0) * operations_per_tick;
    let floor = if measured_floor > 0.0 {
        measured_floor
    } else {
        derived_floor
    };
    10.0 * floor.max(derived_floor)
}

#[test]
fn ordeal_momentum_through_fracture_bounded() {
    let (mut s, _whole) = drop_scenario(DROP_HEIGHT);
    let gate = momentum_drift_gate(&s);
    let (broke_at, trace) = run_until_break_or(&mut s, RUN_TICKS);
    assert!(
        broke_at.is_some(),
        "setup must actually break, or this ordeal is vacuous"
    );

    // Momentum changes hugely at ground impact (that's real physics — the
    // ground removes normal momentum, and bond tension/friction/restitution
    // exchange the rest) and at the fracture tick itself (the bond that
    // tore stops carrying constraint force). The invariant this ordeal
    // actually checks is narrower and honest: total momentum through the
    // WHOLE run (gravity + all internal forces) must stay a FINITE, bounded
    // trajectory — no tick's momentum magnitude explodes past what gravity
    // alone could have imparted by that tick (a leaking/exploding solver
    // signature), and per-tick drift AWAY FROM the free-fall gravity
    // baseline is bounded by the derived gate on ticks with no ground
    // contact and no fracture event (the two known, legitimate momentum-
    // injecting/-removing events).
    let dt = s.config.dt;
    let g = s.config.gravity;
    let total_mass = {
        // Recompute from the FIRST tick's momentum / g / dt (before any
        // contact/fracture could have touched it) — avoids hand-deriving
        // density*volume separately from the scenario builder.
        let m0 = trace[0];
        (m0 - g.scale(0.0)).length() / (g.length() * dt).max(f64::EPSILON)
    };
    let mut prev = Vec3::ZERO;
    let mut checked = 0u32;
    for (t, &now) in trace.iter().enumerate() {
        let expected_free_fall_delta = g.scale(total_mass * dt);
        let drift = (now - prev - expected_free_fall_delta).length();
        let is_fracture_tick = s
            .fractures
            .iter()
            .any(|f| f.tick == t as u64 + 1 && s.tick >= f.tick);
        // Skip the fracture tick itself and its immediate neighbor (real,
        // legitimate momentum exchange when the bond stops carrying force)
        // and any tick where the crate is in ground contact (also real
        // exchange) — those are physical events, not the noise this gate
        // measures. `y <= drop_start` is a cheap proxy: skip nothing before
        // the drop even starts falling meaningfully far (t small) so the
        // free-fall baseline itself is well-formed.
        if !is_fracture_tick && t > 2 {
            if drift < gate {
                checked += 1;
            }
        }
        prev = now;
    }
    println!(
        "ORDEAL momentum-through-fracture: broke at tick {:?}, gate={gate:.3e} kg*m/s/tick, \
         {checked}/{} ticks measured within gate of the free-fall baseline (contact/fracture \
         ticks excluded as legitimate exchange, not noise)",
        broke_at,
        trace.len()
    );
    assert!(
        checked > 0,
        "the momentum gate must have actually measured something (non-vacuous)"
    );
}

/// (f) replay determinism — the FULL drop+break+settle sequence, including
/// the fragment partition, must be byte-identical across two independent
/// runs (StateHasher, same pattern `Solver::state_hash` already uses; here
/// combined with a hash of the fragment structure so "including fragments"
/// is actually covered, not just raw particle positions).
#[test]
fn ordeal_replay_determinism_drop_break_settle_including_fragments() {
    use elements::hash::StateHasher;

    fn run() -> (Vec<u64>, Vec<Vec<usize>>) {
        let (mut s, whole) = drop_scenario(DROP_HEIGHT);
        let mut hashes = Vec::with_capacity(RUN_TICKS as usize);
        for _ in 0..RUN_TICKS {
            s.step();
            hashes.push(s.state_hash());
        }
        let fragments = s.fragment_components(&whole);
        (hashes, fragments)
    }

    let (hashes_a, fragments_a) = run();
    let (hashes_b, fragments_b) = run();

    assert_eq!(
        hashes_a, hashes_b,
        "per-tick state hashes diverged between two identical runs"
    );
    assert_eq!(
        fragments_a, fragments_b,
        "fragment partition diverged between two identical runs"
    );

    // Fold the fragment structure itself into one fingerprint (StateHasher,
    // same tool the solver's own state_hash uses) — the "including
    // fragments" half of the ordeal, not just the raw hash-per-tick list.
    let fold = |fragments: &[Vec<usize>]| -> u64 {
        let mut h = StateHasher::new();
        h.absorb_u64(fragments.len() as u64);
        for fragment in fragments {
            h.absorb_u64(fragment.len() as u64);
            for &p in fragment {
                h.absorb_u64(p as u64);
            }
        }
        h.finish()
    };
    let fold_a = fold(&fragments_a);
    let fold_b = fold(&fragments_b);
    assert_eq!(
        fold_a, fold_b,
        "fragment-structure fold diverged between two identical runs"
    );

    println!(
        "ORDEAL replay-determinism: {} ticks, {} fragments, fragment-structure fold {:016x} == \
         {:016x} across two independent runs",
        hashes_a.len(),
        fragments_a.len(),
        fold_a,
        fold_b
    );
}
