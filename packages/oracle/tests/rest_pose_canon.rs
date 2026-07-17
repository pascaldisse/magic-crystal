//! REST-POSE CANON ORDEALS — GUARDIAN RULING F6 ("senses read SOLVER TRUTH —
//! the world as it is, not as authored", NARUKO.md · GUARDIAN RULINGS · item
//! 5).
//!
//! `packages/oracle/tests/canon.rs` gazes at the FRESHLY-LOADED realm for
//! most vessels — the AUTHORED load-pose, physics never ticked — which is
//! legitimate STATIC-scene truth for non-physics vessels. For the physics
//! `body` vessels (`naruko_crate`, `naruko_stack_crate_0/1/2`) canon.rs's
//! `canon_nearest_ordering_and_ranges_are_derived` is now MIGRATED (per this
//! ruling) to gaze the solver-rested world directly, using the SAME shared
//! machinery this file uses (`tests/rest_pose/mod.rs`) — see that test for
//! the headline canon numbers. THIS file is the dedicated F6 proof: it
//! establishes the rest tick's determinism and derives every physics
//! vessel's SOLVER-TRUTH position/AABB/range by hand, with the analytic
//! prediction shown as a cross-check (not the other way around — the
//! measured solver numbers ARE the truth ruling F6 names; the analytic chain
//! is how we know the measurement is sane, not a substitute for it).
//!
//! DERIVATION DISCIPLINE (canon.rs precedent): every asserted number is
//! hand-derived from the realm geometry / solver binding with the arithmetic
//! shown to 4 decimals, and every tolerance is derived from a measured floor,
//! never plucked.

use oracle::World;

mod rest_pose;
use rest_pose::{fresh_scene, march_to_floor, rested_canon_world, IDS, REST_MARCH, REST_TICK};

/// F6 ORDEAL A — DETERMINISM of the rest tick. Two independent fresh runs
/// march to rest and reach the kinetic floor at the EXACT SAME tick (91),
/// byte-stable. This is the derivation of `REST_TICK` (`tests/rest_pose/mod.rs`):
/// measured, then confirmed identical across two ticks before being pinned
/// for the gaze ordeal below.
#[test]
fn rest_is_deterministic() {
    let a = march_to_floor(&mut fresh_scene(), REST_MARCH);
    let b = march_to_floor(&mut fresh_scene(), REST_MARCH);
    assert_eq!(a, Some(REST_TICK), "run A reached the floor at REST_TICK");
    assert_eq!(b, Some(REST_TICK), "run B reached the floor at REST_TICK");
    assert_eq!(
        a, b,
        "the rest tick must be byte-stable across two independent runs"
    );
    eprintln!("[ordeal F6·A] rest tick = {REST_TICK}, identical across two independent runs");
}

/// F6 ORDEAL B — the senses read the SOLVER-RESTED pose. March the realm to
/// rest, inject each body's solver-rested `transform` into an oracle-loaded
/// world (`rested_canon_world`, `tests/rest_pose/mod.rs`), and gaze with
/// `World::geometry` — the SAME gaze code the oracle uses everywhere. The
/// numbers this ordeal prints ARE solver truth (ruling F6); the analytic
/// chain below is the cross-check that proves the measurement is sane, shown
/// alongside so the delta is auditable — it is NOT the canonical number.
///
/// ANALYTIC CROSS-CHECK (physics.rs `crate_falls_and_rests…` /
/// `stack_settles…`, reused verbatim via `RestDials` in
/// `tests/rest_pose/mod.rs`), with the realm's own numbers — pier_top =
/// 1.0250 m from `top_flat_surface_y("naruko_pier")`, half = 0.4000 m and
/// radius = 0.0500 m from the solver binding, contact_margin = 0.0010 m from
/// `elements::ContactMaterial::default()`:
///   y0   = pier_top + half + radius            = 1.0250+0.4000+0.0500 = 1.4750
///   step = 2·half + radius + contact_margin     = 0.8000+0.0500+0.0010 = 0.8510
///     (naruko_crate and stack_crate_0 both rest via the PIER contact — the
///      particle-vs-triangle pass, which physics.rs's own comment notes omits
///      the margin term, a pre-existing ~1 mm approximation well under
///      REST_TOL. stack_crate_1/2 rest via the BODY-VS-BODY chain, which
///      physics.rs's stack ordeal (`:274`) DOES include contact_margin in —
///      `y += 2.0 * half_height + contact_radius + contact_margin` — so this
///      cross-check includes it too, matching physics.rs exactly rather than
///      approximating.)
/// X/Z are the authored footprint (the solver never drifts them
/// horizontally); analytic Y (cross-check only — see SOLVER-MEASURED below
/// for the actual truth):
///   naruko_crate         analytic y = y0        = 1.4750
///   naruko_stack_crate_0 analytic y = y0        = 1.4750
///   naruko_stack_crate_1 analytic y = y0 + step  = 2.3260
///   naruko_stack_crate_2 analytic y = y0 + 2·step = 3.1770
/// Analytic range = |analytic_center − eye|, eye = the canon spawn pose
/// [0,7,44] (read from the realm, not hardcoded):
///   naruko_crate  [-11.15,1.4750,13] → √(124.3225+30.525625+961) = √1115.848125 = 33.4043
///   stack_crate_0 [-13.65,1.4750,13] → √(186.3225+30.525625+961) = √1177.848125 = 34.3198
///   stack_crate_1 [-13.65,2.3260,13] → √(186.3225+21.846276+961) = √1169.168776 = 34.1931
///   stack_crate_2 [-13.65,3.1770,13] → √(186.3225+14.633289+961) = √1161.955789 = 34.0872
///
/// SOLVER-MEASURED (the F6 truth this ordeal asserts, `--nocapture` verbatim):
///   naruko_crate          y=1.4759 range=33.4042  (analytic 1.4750/33.4043,
///                          residual |1.4759−1.4750|=0.0009 m)
///   naruko_stack_crate_0  y=1.4754 range=34.3197  (analytic 1.4750/34.3198,
///                          residual 0.0004 m)
///   naruko_stack_crate_1  y=2.3259 range=34.1931  (analytic 2.3260/34.1931,
///                          residual 0.0001 m)
///   naruko_stack_crate_2  y=3.1767 range=34.0872  (analytic 3.1770/34.0872,
///                          residual 0.0003 m)
///
/// AUTHORED (canon.rs's pre-F6 load-pose) → SOLVER-RESTED (the F6 truth):
///   naruko_crate  authored [-11.15,4.5,13] range 33.0390  →  rested [-11.15,1.4759,13] range 33.4042  (Δ +0.3652)
///     — the ONLY material move: the crate is authored HUNG at y=4.5 and the
///       solver drops it 3.025 m onto the pier planks (rest y≈1.4759).
///   stack_crate_0 authored range 34.3198  →  rested range 34.3197  (Δ -0.0001)
///   stack_crate_1 authored range 34.1932  →  rested range 34.1931  (Δ -0.0001)
///   stack_crate_2 authored range 34.0874  →  rested range 34.0872  (Δ -0.0002)
///     — the STACK was authored already AT its solver-rest (chained heights),
///       so its senses-truth equals its load-pose to well under REST_TOL; F6
///       confirms rather than moves it.
///
/// TOLERANCES (derived from the measured floor — the WORST body, not
/// cherry-picked):
///   REST_TOL = 0.005 m — the worst per-axis residual across all four bodies
///     is naruko_crate's 0.0009 m (|1.4759−1.4750|; the stack crates are
///     tighter still: 0.0004/0.0001/0.0003 m). Headroom = REST_TOL / worst
///     residual = 0.005 / 0.0009 ≈ 5.6× — tight enough that a wrong rest
///     height (±0.1 m) fails by ≈111×, loose enough never to flap on the
///     measured floor. Applied to each rested-center axis: the derived
///     analytic rest vs the live solver rest.
///   RANGE_TOL = 1e-3 m — the live range is √Σ(center−eye)² in f32. The
///     rested center's live-vs-analytic gap (≤ REST_TOL on Y, X/Z exact)
///     propagates through the range gradient |Δy|/range ≤ 5.525/33.4 ≈ 0.165
///     to ≤ 0.00084 m; the measured live-vs-analytic range deltas above peak
///     at 0.0001 m (naruko_crate/stack_crate_0). 1e-3 is ≥1.2× the
///     worst-case propagated budget and ≥10× the measured max — a wrong AABB
///     (±0.1 m) still fails by ≥100×.
///   BOX_TOL = 1e-4 m — the box size is 2·half; measured f32 slop 7.6e-7 m
///     (0.79999924 vs 0.8), so 1e-4 is >100× the slop.
#[test]
fn senses_read_solver_rested_pose() {
    let (oracle_world, dials) = rested_canon_world();

    let eye = oracle_world
        .spawn_pose()
        .expect("canon spawn pose")
        .position;
    assert_eq!(
        eye,
        [0.0, 7.0, 44.0],
        "canon spawn eye (read, not hardcoded)"
    );

    // The authored gaze, for X/Z footprint and the auditable Δrange print —
    // a SEPARATE fresh oracle load, never ticked, so it is unaffected by the
    // injected transforms above.
    let authored_world = World::load(rest_pose::canon_dir()).expect("load canon naruko");
    let authored_center: Vec<[f32; 3]> = IDS
        .iter()
        .map(|id| {
            authored_world
                .geometry(id)
                .unwrap()
                .bounds
                .unwrap()
                .center()
        })
        .collect();

    let y0 = dials.pier_contact_y();
    let step = dials.body_chain_step();
    let range_from = |c: [f64; 3]| -> f64 {
        let d = [
            c[0] - eye[0] as f64,
            c[1] - eye[1] as f64,
            c[2] - eye[2] as f64,
        ];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };

    const REST_TOL: f64 = 0.005;
    const RANGE_TOL: f64 = 1e-3;
    const BOX_TOL: f64 = 1e-4;

    for (i, id) in IDS.iter().enumerate() {
        // Analytic CROSS-CHECK center: authored X/Z (the solver never drifts
        // them), analytic Y from the chained dials.
        let a = authored_center[i];
        let analytic_y = if *id == "naruko_stack_crate_1" {
            y0 + step
        } else if *id == "naruko_stack_crate_2" {
            y0 + 2.0 * step
        } else {
            // naruko_crate and naruko_stack_crate_0 both rest on the pier at y0.
            y0
        };
        let analytic_center = [a[0] as f64, analytic_y, a[2] as f64];
        let analytic_range = range_from(analytic_center);

        // The senses' RESTED gaze — SOLVER TRUTH, the same code as everywhere.
        let g = oracle_world.geometry(id).expect("rested geometry");
        let b = g.bounds.expect("rested bounds");
        let c = b.center();
        let size = b.size();

        // The solver measurement must be within REST_TOL of the analytic
        // cross-check on every axis (X/Z never drift, Y is the derived rest
        // height) — this is a SANITY check on the measurement, not a
        // definition of truth: the measurement (`c`) IS the truth.
        assert!(
            (c[0] as f64 - analytic_center[0]).abs() < REST_TOL
                && (c[1] as f64 - analytic_center[1]).abs() < REST_TOL
                && (c[2] as f64 - analytic_center[2]).abs() < REST_TOL,
            "{id} solver-measured center {c:?} != analytic cross-check {analytic_center:?} (tol {REST_TOL})"
        );

        // The box is unchanged by physics: size == 2·half on every axis.
        for s in size {
            assert!(
                (s as f64 - 2.0 * dials.half).abs() < BOX_TOL,
                "{id} rested box side {s} != 2·half {} (tol {BOX_TOL})",
                2.0 * dials.half
            );
        }

        // Range: measured solver range vs the analytic cross-check range.
        let live_range = range_from([c[0] as f64, c[1] as f64, c[2] as f64]);
        assert!(
            (live_range - analytic_range).abs() < RANGE_TOL,
            "{id} solver-measured range {live_range:.4} != analytic cross-check {analytic_range:.4} (tol {RANGE_TOL})"
        );

        // Auditable authored → SOLVER-MEASURED (truth) delta.
        let authored_range = range_from([a[0] as f64, a[1] as f64, a[2] as f64]);
        eprintln!(
            "[ordeal F6·B] {id}: authored center {a:?} range {authored_range:.4} \
             -> SOLVER-TRUTH rested center [{:.4},{:.4},{:.4}] range {live_range:.4} (Δrange {:+.4}); analytic cross-check {analytic_center:?} range {analytic_range:.4}",
            c[0],
            c[1],
            c[2],
            live_range - authored_range
        );
    }
}
