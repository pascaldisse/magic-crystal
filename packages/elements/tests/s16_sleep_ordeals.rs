//! S16 ISLAND SLEEP — the ordeals guarding per-island rest.
//!
//! The charter's law: a settled island costs ZERO, sleep is DETERMINISTIC,
//! and a slept island WAKES when pushed (silent-freeze is the one failure
//! mode). These trials prove each on the elements solver directly.

use elements::{Collider, ContactMaterial, Solver, SolverConfig, Vec3};

/// A rigid box dropped just above the ground, warmed until it settles. With
/// `sleep` on and a low `sleep_frames`, it should sleep after settling.
fn settled_box(sleep: bool, frames: u32) -> Solver {
    let cfg = SolverConfig {
        dt: 1.0 / 60.0,
        substeps: 8,
        iterations: 1,
        seed: 42,
        sleep_frames: frames,
        sleep_vel: 0.03,
        ..SolverConfig::default()
    };
    let mut s = Solver::new(cfg);
    s.collider = Some(Collider::ground_plane(0.0, 8.0, ContactMaterial::default()));
    s.spawn_rigid_box(
        Vec3::new(0.0, 0.35, 0.0),
        Vec3::new(0.4, 0.4, 0.4),
        (3, 3, 3),
        600.0,
        1.0,
        0.03,
    );
    if sleep {
        s.set_sleep(true);
    }
    s
}

// ── 1. DETERMINISM with sleep ON — two runs byte-identical ─────────────────
#[test]
fn ordeal_sleep_is_byte_identical_across_runs() {
    let run = || {
        let mut s = settled_box(true, 12);
        for _ in 0..600 {
            s.step();
        }
        // A wake + re-settle mid-run exercises the wake path too.
        let idx = s.rigids[0].indices.clone();
        s.apply_impulse_to_particles(&idx, Vec3::new(1.5, 0.0, 0.7));
        for _ in 0..600 {
            s.step();
        }
        (s.state_hash(), s.rigids[0].centroid)
    };
    let (h1, c1) = run();
    let (h2, c2) = run();
    assert_eq!(h1, h2, "sleep diverged between identical worldlines");
    assert_eq!(c1, c2, "centroid diverged with sleep on");
    println!("ORDEAL sleep determinism: 0x{h1:016x} == 0x{h2:016x} ✓");
}

// ── 2. A SETTLED island actually SLEEPS (costs zero) ──────────────────────
#[test]
fn ordeal_settled_box_sleeps() {
    let mut s = settled_box(true, 12);
    for _ in 0..300 {
        s.step();
    }
    let (asleep, islands) = s.sleep_counts();
    assert!(asleep > 0, "a settled box never slept ({asleep} asleep)");
    assert_eq!(islands, 1, "one resting box should be exactly one island");
    // Frozen: its centroid must not drift once asleep.
    let c0 = s.rigids[0].centroid;
    for _ in 0..120 {
        s.step();
    }
    let c1 = s.rigids[0].centroid;
    let drift = (c1 - c0).length();
    assert!(drift < 1e-9, "asleep box drifted {drift} (should be frozen)");
    println!("ORDEAL settled sleeps: {asleep} particles / {islands} island, drift {drift:.2e} ✓");
}

// ── 3. THE WAKE-TEST — a slept box, PUSHED, wakes and MOVES (no freeze) ────
#[test]
fn ordeal_pushed_sleeper_wakes_and_moves() {
    let mut s = settled_box(true, 12);
    for _ in 0..300 {
        s.step();
    }
    let (asleep_before, _) = s.sleep_counts();
    assert!(asleep_before > 0, "precondition: the box must be asleep");
    let c_rest = s.rigids[0].centroid;

    // The op is the hand — a lateral shove (a door push / thrown body).
    let idx = s.rigids[0].indices.clone();
    s.apply_impulse_to_particles(&idx, Vec3::new(2.0, 0.0, 0.0));

    // It must be AWAKE the instant it is pushed.
    let (asleep_after_push, _) = s.sleep_counts();
    assert_eq!(asleep_after_push, 0, "push did not wake the sleeper (SILENT FREEZE)");

    // And it must MOVE over the next ticks (not silently freeze).
    for _ in 0..30 {
        s.step();
    }
    let c_moved = s.rigids[0].centroid;
    let travel = (c_moved - c_rest).length();
    assert!(
        travel > 0.05,
        "pushed sleeper did not move (travel {travel:.4}) — SILENT FREEZE"
    );
    println!("ORDEAL wake-test: pushed sleeper travelled {travel:.4} m ✓ (no freeze)");
}

// ── 4. SLEEP vs NO-SLEEP settle to NEARLY the same rest pose ──────────────
// Freezing at rest must not teleport the body; the slept centroid should sit
// within a hair of where the always-solving body rests.
#[test]
fn ordeal_sleep_rest_pose_matches_no_sleep() {
    let mut on = settled_box(true, 12);
    let mut off = settled_box(false, 12);
    for _ in 0..400 {
        on.step();
        off.step();
    }
    let d = (on.rigids[0].centroid - off.rigids[0].centroid).length();
    assert!(
        d < 5e-3,
        "slept rest pose diverged from always-solve by {d:.4} m (too far)"
    );
    let (asleep, _) = on.sleep_counts();
    assert!(asleep > 0, "sleep arm never slept");
    println!("ORDEAL rest-pose parity: |Δcentroid| {d:.2e} m, slept {asleep} particles ✓");
}
