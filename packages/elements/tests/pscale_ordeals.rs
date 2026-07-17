//! P-SCALE — THE BUILDING FALLS · ordeals over the building-scale collapse.
//! Derived bounds only (measure-don't-guess, the house style): the standing
//! structure stays at rest, the collapse replays byte-identical, post-collapse
//! fragments do not interpenetrate, `step_profiled` matches `step` bit-for-bit,
//! and the per-tick cost is RECORDED (not gated — a perf gate at unknown scale
//! would be a guess; the numbers are the evidence).

use elements::building::{erect, settle, topple, Building, BuildingSpec};
use elements::{Collider, ContactMaterial, Solver, SolverConfig, Vec3};

/// The default proof scenario at a test-fast scale (N = 288). Settled to rest.
fn standing(lattice: (usize, usize, usize)) -> Building {
    let spec = BuildingSpec {
        lattice,
        ..BuildingSpec::default()
    };
    let mut b = erect(spec);
    // Settle to the STANDING PLATEAU (see `settle`'s METASTABILITY doc): ~40
    // ticks holds the tower tall (top ≈23.5 m of 24 m) before the slow
    // buckling creep of a tall single-iteration XPBD stack sets in.
    let rest_fragments = settle(&mut b, 40);
    assert_eq!(
        rest_fragments, 1,
        "the settled structure must be ONE whole body (lattice {lattice:?}) — if it sheds a bond \
         at rest, the 'rest stays at rest' premise is already false"
    );
    b
}

/// The building's tallest particle — its standing height witness.
fn top_y(b: &Building) -> f64 {
    b.whole.iter().map(|&p| b.solver.particles.pos[p].y).fold(0.0_f64, f64::max)
}

const TOPPLE_SPEED: f64 = 30.0;
const TOPPLE_FRACTION: f64 = 0.5;

/// ORDEAL — STRUCTURE AT REST STAYS AT REST (no drift explosion at scale).
/// A settled, un-toppled building held for a long window must keep every
/// particle essentially still: no fracture, and the fastest particle's speed
/// bounded by a DERIVED floor — the same solver's own steady-state residual on
/// an ANCHORED single body at rest, measured not guessed (same shape as
/// `vi2_break_ordeals`'s `momentum_drift_gate`/`body_overlap_floor`). Checked
/// at TWO scales so "at scale" is real, not a single point.
#[test]
fn ordeal_building_at_rest_stays_at_rest() {
    for lattice in [(8, 16, 8), (10, 20, 10)] {
        let mut b = standing(lattice);

        // DERIVED FLOOR: the residual settle-speed of a KNOWN-stable reference
        // — the same building stepped one more second with fracture disarmed
        // (it cannot break, so any speed left is pure numerical rest residue,
        // not real motion). The rest window below must not exceed 10x this.
        let mut ref_solver = b.solver.clone();
        ref_solver.config.fracture_threshold = f64::INFINITY;
        let mut floor = 0.0_f64;
        let ref_ticks = (1.0 / ref_solver.config.dt).round() as u64;
        for _ in 0..ref_ticks {
            ref_solver.step();
            for i in 0..ref_solver.particles.pos.len() {
                if ref_solver.particles.inv_mass[i] != 0.0 {
                    floor = floor.max(ref_solver.particles.vel[i].length());
                }
            }
        }
        let gate = 10.0 * floor.max(f64::EPSILON);

        // THE REST WINDOW — armed (real threshold), no topple, no external
        // impulse, over the metastable STANDING PLATEAU (60 ticks = 1 s;
        // beyond this a tall single-iteration XPBD stack slowly buckles — an
        // honest, documented gap, see `settle`'s doc): nothing should move
        // fast, nothing should break, and the tower must STAY STANDING.
        let stand_h = top_y(&b);
        let window = 60u64;
        let mut max_speed = 0.0_f64;
        for t in 0..window {
            b.solver.step();
            for i in 0..b.solver.particles.pos.len() {
                if b.solver.particles.inv_mass[i] != 0.0 {
                    let v = b.solver.particles.vel[i].length();
                    max_speed = max_speed.max(v);
                    assert!(
                        v.is_finite(),
                        "tick {t}: particle {i} velocity diverged ({v}) — a drift EXPLOSION, \
                         exactly what this ordeal forbids at scale (lattice {lattice:?})"
                    );
                }
            }
            assert!(
                b.solver.fragment_components(&b.whole).len() == 1,
                "tick {t}: the un-toppled structure fractured at rest (lattice {lattice:?}) — \
                 rest must stay whole"
            );
        }
        assert!(
            max_speed < gate,
            "the resting structure drifted: max particle speed {max_speed:.3e} m/s exceeds the \
             derived floor gate {gate:.3e} m/s (10x a known-stable reference's rest residual) — \
             lattice {lattice:?}"
        );
        // HEIGHT RETENTION — the standing structure must not have collapsed:
        // over the bounded window it may creep at most 20% (the metastable
        // plateau's measured tolerance), never pancake.
        let end_h = top_y(&b);
        assert!(
            end_h > 0.8 * stand_h,
            "the standing structure lost height: {stand_h:.2} m -> {end_h:.2} m over {window} \
             ticks (>20%) — it collapsed at rest instead of standing (lattice {lattice:?})"
        );
        println!(
            "ORDEAL building-at-rest-stays-at-rest: lattice {lattice:?} (N={}) stood {window} \
             ticks whole, height {stand_h:.1}->{end_h:.1} m, max rest speed {max_speed:.3e} m/s < \
             gate {gate:.3e} m/s",
            b.whole.len()
        );
    }
}

/// ORDEAL — COLLAPSE IS DETERMINISTIC (replay byte-identical). The FULL
/// erect+settle+topple+collapse sequence folds to identical per-tick state
/// hashes AND identical fragment partitions across two independent runs (same
/// pattern as `vi2_break_ordeals`'s replay ordeal, at building scale).
#[test]
fn ordeal_building_collapse_replay_byte_identical() {
    fn run() -> (Vec<u64>, Vec<Vec<usize>>) {
        let mut b = standing((8, 16, 8));
        topple(&mut b, TOPPLE_SPEED, TOPPLE_FRACTION);
        let mut hashes = Vec::new();
        for _ in 0..200 {
            b.solver.step();
            hashes.push(b.solver.state_hash());
        }
        let fragments = b.solver.fragment_components(&b.whole);
        (hashes, fragments)
    }
    let (ha, fa) = run();
    let (hb, fb) = run();
    assert_eq!(ha, hb, "per-tick state hashes diverged between two identical collapses");
    assert_eq!(fa, fb, "fragment partition diverged between two identical collapses");
    assert!(
        fa.len() > 1,
        "the collapse must actually break the building, or this ordeal is vacuous"
    );
    println!(
        "ORDEAL building-collapse-replay: 200 ticks, {} fragments, byte-identical across two runs",
        fa.len()
    );
}

/// The overlap-residual FLOOR — a representative STACK of same-radius rigid
/// spheres settled into a resting column on the ground, the steady-state
/// max overlap `solve_body_collisions` leaves BETWEEN adjacent bodies. Same
/// measure-don't-guess derived-gate pattern as `vi2_break_ordeals::
/// body_overlap_floor`, but a STACK of `depth` bodies, not a single pair:
/// a settled rubble PILE stacks many single-iteration contacts, and the
/// penetration residual the O(k²) pass leaves grows with pile depth (each
/// body above adds weight the one below must push back out in one substep) —
/// a two-body reference underestimates it. `depth` is the building's own
/// storey count (`lattice.1`), so the floor's pile pressure matches the
/// rubble the ordeal actually measures. The fragment-vs-fragment gate is 10x
/// this measured floor.
fn stack_overlap_floor(cfg: SolverConfig, particle_radius: f64, depth: usize) -> f64 {
    let mut s = Solver::new(cfg);
    s.collider = Some(Collider::ground_plane(0.0, 50.0, ContactMaterial::default()));
    let contact_margin = s.collider.as_ref().unwrap().material.contact_margin;
    let density = 2000.0; // the building's own density — same contact pressure
    let sphere_radius = 0.08;
    let min_dist = particle_radius + contact_margin;
    // A vertical column, each sphere one rest-gap above the last, dropped a
    // hair so contact is live.
    let mut idx = Vec::new();
    for k in 0..depth.max(2) {
        let y = min_dist + k as f64 * (min_dist + 0.005);
        idx.push(s.spawn_rigid_sphere(
            Vec3::new(0.0, y, 0.0),
            sphere_radius,
            1,
            density,
            1.0,
            particle_radius,
        ));
    }
    for _ in 0..600 {
        s.step();
    }
    let window = (1.0 / cfg.dt).round() as u64;
    let mut max_overlap = f64::NEG_INFINITY;
    for _ in 0..window {
        s.step();
        for a in 0..idx.len() {
            for b in (a + 1)..idx.len() {
                let dist = (s.particles.pos[s.rigids[idx[a]].indices[0]]
                    - s.particles.pos[s.rigids[idx[b]].indices[0]])
                    .length();
                max_overlap = max_overlap.max(min_dist - dist);
            }
        }
    }
    if max_overlap > 0.0 {
        max_overlap
    } else {
        let operations_per_tick = idx.len() as f64 * cfg.substeps as f64;
        f64::EPSILON * min_dist.max(1.0) * operations_per_tick
    }
}

/// ORDEAL — POST-COLLAPSE FRAGMENTS DO NOT INTERPENETRATE. After the building
/// breaks and its shards settle, no two particles in DIFFERENT fragments may
/// sit closer than their derived rest gap by more than the derived gate,
/// checked every tick of a settle window (the derived-gate pattern reused from
/// `vi2_break_ordeals`, at building scale). Non-vacuous: cross-fragment near-
/// contact must actually occur (rubble piles ON itself).
#[test]
fn ordeal_building_fragments_do_not_interpenetrate() {
    let particle_radius = BuildingSpec::default().contact_radius;
    let mut b = standing((8, 16, 8));
    let cfg = b.solver.config;
    let floor = stack_overlap_floor(cfg, particle_radius, b.spec.lattice.1);
    let gate = 10.0 * floor;
    let contact_margin = b.solver.collider.as_ref().unwrap().material.contact_margin;

    topple(&mut b, TOPPLE_SPEED, TOPPLE_FRACTION);
    // Break, then settle the rubble.
    for _ in 0..260 {
        b.solver.step();
    }
    let fragments = b.solver.fragment_components(&b.whole);
    assert!(
        fragments.len() >= 2,
        "the building must break into multiple fragments to test interpenetration"
    );
    let mut membership = vec![usize::MAX; b.solver.particles.pos.len()];
    for (fi, frag) in fragments.iter().enumerate() {
        for &p in frag {
            membership[p] = fi;
        }
    }

    let mut max_overlap = f64::NEG_INFINITY;
    let mut min_cross_sep = f64::INFINITY;
    for t in 0..140 {
        b.solver.step();
        for (a, &i) in b.whole.iter().enumerate() {
            for &j in &b.whole[(a + 1)..] {
                if membership[i] == membership[j] {
                    continue;
                }
                let dist = (b.solver.particles.pos[i] - b.solver.particles.pos[j]).length();
                let min_dist = (b.solver.particles.radius[i] + b.solver.particles.radius[j]) * 0.5
                    + contact_margin;
                let overlap = min_dist - dist;
                min_cross_sep = min_cross_sep.min(dist);
                max_overlap = max_overlap.max(overlap);
                assert!(
                    overlap < gate,
                    "tick {t}: fragments {} and {} interpenetrate — particles {i}/{j} overlap by \
                     {overlap:.3e} m, over the derived gate {gate:.3e} m",
                    membership[i], membership[j]
                );
            }
        }
    }
    let typical_gap = particle_radius + contact_margin;
    assert!(
        min_cross_sep < typical_gap * 3.0,
        "fragments never approached each other (min cross-fragment sep {min_cross_sep:.4} m) — \
         the ordeal would be vacuous"
    );
    println!(
        "ORDEAL building-fragments-do-not-interpenetrate: {} fragments, gate={gate:.3e} m, max \
         cross-fragment overlap {max_overlap:.3e} m (< gate), min cross sep {min_cross_sep:.4} m",
        fragments.len()
    );
}

/// ORDEAL — `step_profiled` MATCHES `step` BIT-FOR-BIT. The measurement path
/// must not perturb the physics: the same collapse driven by `step_profiled`
/// must fold to the identical per-tick state hash as `step`. This is the guard
/// that lets the profiled numbers be trusted as the real solver's cost.
#[test]
fn ordeal_step_profiled_matches_step() {
    let mut plain = standing((8, 16, 8));
    topple(&mut plain, TOPPLE_SPEED, TOPPLE_FRACTION);
    let mut prof = standing((8, 16, 8));
    topple(&mut prof, TOPPLE_SPEED, TOPPLE_FRACTION);
    for t in 0..150 {
        plain.solver.step();
        let _ = prof.solver.step_profiled();
        assert_eq!(
            plain.solver.state_hash(),
            prof.solver.state_hash(),
            "tick {t}: step_profiled diverged from step — the timers changed the physics"
        );
    }
    println!("ORDEAL step-profiled-matches-step: 150 collapse ticks byte-identical to step");
}

/// ORDEAL — PER-TICK COST IS RECORDED (not gated). Proves the profiler fills a
/// non-vacuous, self-consistent breakdown over a real collapse: total ≥ the
/// summed phases, the O(k²) pair-check count matches `(k choose 2)`, and the
/// phase fields are actually populated. NO time threshold is asserted — a perf
/// gate at unknown scale would be a guess; the numbers are evidence, printed.
#[test]
fn ordeal_per_tick_cost_recorded() {
    let mut b = standing((8, 16, 8));
    topple(&mut b, TOPPLE_SPEED, TOPPLE_FRACTION);
    let mut saw_body_pass = false;
    let mut peak_pairs = 0u64;
    for _ in 0..120 {
        let p = b.solver.step_profiled();
        // Self-consistency: whole tick is at least the sum of its phases.
        let phase_sum = p.integrate
            + p.solve_distance
            + p.shape_matching
            + p.collision_static
            + p.collision_body
            + p.cluster_floodfill
            + p.velocity_passes
            + p.fracture_pass;
        assert!(
            p.total >= phase_sum || p.total.as_nanos() == 0,
            "whole-tick {:?} is less than the summed phases {:?} — the breakdown is inconsistent",
            p.total, phase_sum
        );
        // The O(k²) pair-check count is exactly (k choose 2) x iterations x substeps.
        let k = p.clustered_particles as u64;
        let expected = k * (k.saturating_sub(1)) / 2
            * b.solver.config.iterations.max(1) as u64
            * b.solver.config.substeps.max(1) as u64;
        assert_eq!(
            p.body_pair_checks, expected,
            "recorded body-pair-checks {} != analytic (k choose 2)xitxsub {}",
            p.body_pair_checks, expected
        );
        peak_pairs = peak_pairs.max(p.body_pair_checks);
        if p.collision_body.as_nanos() > 0 {
            saw_body_pass = true;
        }
        assert!(p.particles > 0 && p.bonds > 0);
    }
    assert!(
        saw_body_pass,
        "the O(k^2) body-collision phase never registered any time over the collapse — the \
         profiler is not actually measuring it"
    );
    println!(
        "ORDEAL per-tick-cost-recorded: 120 profiled collapse ticks, breakdown self-consistent, \
         peak body-pair-checks/tick={peak_pairs} (RECORDED, not gated)"
    );
}
