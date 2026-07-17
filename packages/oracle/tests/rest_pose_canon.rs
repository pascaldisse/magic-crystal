//! REST-POSE CANON ORDEALS — GUARDIAN RULING F6 ("senses read SOLVER TRUTH —
//! the world as it is, not as authored", NARUKO.md · GUARDIAN RULINGS · item 5).
//!
//! `packages/oracle/tests/canon.rs` gazes at the FRESHLY-LOADED realm — the
//! AUTHORED load-pose, physics never ticked — which is legitimate STATIC-scene
//! truth for the non-physics vessels it covers. But for the physics `body`
//! vessels (`naruko_crate`, `naruko_stack_crate_0/1/2`) the authored drop-pose
//! is NOT what the world becomes: the solver moves them at runtime. Ruling F6
//! says the senses should read where the solver actually RESTED a body. These
//! ordeals prove exactly that, with the SAME gaze code the oracle uses
//! everywhere (`World::geometry` → `derive_geometry`), handed a world whose
//! bodies carry their post-tick, solver-rested transforms.
//!
//! HOW (the dependency direction): `oracle` and `scrying-glass` are SIBLINGS —
//! both depend only on `crystal`, which depends on neither — so `oracle` takes
//! `scrying-glass` as a DEV-dependency with zero cycle risk. The rest-detection
//! and rest-height derivations here REUSE
//! `packages/scrying-glass/tests/physics.rs` verbatim (its kinetic-floor
//! criterion, its `pier_top + half_height + contact_radius` chain); nothing is
//! re-invented. The ticked `EcsWorld` is lifted out of the render scene through
//! the F6 accessor `Dynamics::into_world` (added for exactly this: external
//! senses gazing at solver-rested state).
//!
//! WHY inject rather than wrap-and-gaze directly: `crystal::load_world_dir`
//! (scrying-glass's loader) stores components crystal-native, but the oracle's
//! own `World::load` wraps each component in a `{"v": …}` envelope that
//! `component_value` unwraps. Wrapping a `load_world_dir` world straight into an
//! `oracle::World` therefore reads `None` for every component. So we load the
//! realm the ORACLE way (correct envelopes + full id registry) and INJECT the
//! solver-rested `transform` (read out of the ticked world) into each body — the
//! oracle's documented LIVE-mutation path ("mutate `world.core` and the next
//! gaze reflects it"). Only `transform` changes under physics; `mesh`/`body`
//! are unchanged, so the fresh oracle load supplies them correctly.
//!
//! DERIVATION DISCIPLINE (canon.rs precedent): every asserted number is
//! hand-derived from the realm geometry / solver binding with the arithmetic
//! shown to 4 decimals, and every tolerance is derived from a measured floor,
//! never plucked.

use crystal::{load_world_dir, EcsWorld};
use oracle::World;
use scrying_glass::scene::{top_flat_surface_y, RenderScene, SceneParameters, SunDefaults};
use std::path::{Path, PathBuf};

/// The realm dir, uncanonicalized — physics.rs's `naruko_world()` (fed to
/// scrying-glass's `load_world_dir`).
fn naruko_world() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko")
}

/// The realm dir, canonicalized — canon.rs's `canon_dir()` (fed to the oracle's
/// own `World::load`). Same realm, both point at `<repo>/worlds/naruko`.
fn canon_dir() -> PathBuf {
    naruko_world().canonicalize().expect("canon naruko dir")
}

/// The realm's authoring dials — mirrors physics.rs `params()` (which mirrors
/// the window/example defaults). Nothing engine-hardcoded; `tick_dt` is the
/// solver's fixed tick.
fn params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: 55.0,
        near: 0.1,
        far: 4_000.0,
        sky_top: "#20152f".into(),
        sky_horizon: "#9a627d".into(),
        mesh_color: "#9aa0a6".into(),
        radial_segments: 24,
        camera_position: [0.0, 2.0, 22.0],
        camera_yaw: 0.0,
        camera_pitch: 0.0,
        cluster_error_threshold: 1.0,
        tick_dt: 1.0 / 60.0,
        sun: SunDefaults {
            sun_color: "#ffe2b0".into(),
            sun_intensity: 1.1,
            sun_position: [60.0, 90.0, 30.0],
            ambient_intensity: 0.32,
        },
        emission_intensity: 2.5,
    }
}

/// The four physics `body` vessels in `worlds/naruko`.
const IDS: [&str; 4] = [
    "naruko_crate",
    "naruko_stack_crate_0",
    "naruko_stack_crate_1",
    "naruko_stack_crate_2",
];

/// The solver's numerical kinetic-floor epsilon (a body's per-tick |Δy|): below
/// this the body is at rest. Reused verbatim from physics.rs's `settled_at`
/// check — the solver's own rest epsilon, not a canon geometry number.
const KINETIC_FLOOR: f64 = 1.0e-6;

/// Warmup ticks before the floor check counts. The FALL itself passes through
/// sub-epsilon vertical steps (apex, launch), so floor hits before this don't
/// mean "rested". physics.rs guards its `settled_at` with the same `tick > 60`.
const REST_WARMUP: u64 = 60;

/// The tick at which EVERY declared body is first at the kinetic floor.
/// DERIVATION: measured empirically as tick 91, confirmed BYTE-IDENTICAL across
/// two independent fresh ticks (`rest_is_deterministic` below asserts both runs
/// land on exactly this value). Pinned as this file's fixed rest-tick — not
/// eyeballed from physics.rs's 600/120 march lengths.
const REST_TICK: u64 = 91;

/// March length before a rested gaze. physics.rs's own stack-settle march is 120
/// ticks; 120 is ≥1.3× the observed REST_TICK (91), so the read is taken well
/// after every body has reached the floor.
const REST_MARCH: u64 = 120;

/// A fresh render scene on the real realm through the real world tick — exactly
/// physics.rs `naruko_scene()`.
fn fresh_scene() -> RenderScene {
    let mut world = EcsWorld::default();
    load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
    RenderScene::from_ecs(world, &params()).expect("build the render scene")
}

/// Tick `scene` `ticks` times; return the first tick `> REST_WARMUP` at which
/// EVERY body's per-tick |Δy| < `KINETIC_FLOOR` (the whole system at the floor).
fn march_to_floor(scene: &mut RenderScene, ticks: u64) -> Option<u64> {
    let mut last: Vec<f64> = IDS
        .iter()
        .map(|id| scene.body_position(id).unwrap()[1])
        .collect();
    let mut first_floor = None;
    for tick in 0..ticks {
        scene.tick();
        let mut all_floor = true;
        for (i, id) in IDS.iter().enumerate() {
            let y = scene.body_position(id).unwrap()[1];
            if (y - last[i]).abs() >= KINETIC_FLOOR {
                all_floor = false;
            }
            last[i] = y;
        }
        if first_floor.is_none() && all_floor && tick > REST_WARMUP {
            first_floor = Some(tick);
        }
    }
    first_floor
}

/// F6 ORDEAL A — DETERMINISM of the rest tick. Two independent fresh runs march
/// to rest and reach the kinetic floor at the EXACT SAME tick (91), byte-stable.
/// This is the derivation of `REST_TICK`: measured, then confirmed identical
/// across two ticks before being pinned for the gaze ordeal below.
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

/// F6 ORDEAL B — the senses read the SOLVER-RESTED pose. March the realm to rest,
/// inject each body's solver-rested `transform` into an oracle-loaded world, and
/// gaze with `World::geometry`. Assert the derived AABB / range against the
/// HAND-DERIVED REST pose (`pier_top + half + radius`, chained), and compare it
/// explicitly to the AUTHORED load-pose (canon.rs) so the F6 delta is auditable.
///
/// REST-POSE DERIVATION (physics.rs `crate_falls_and_rests…` /
/// `stack_settles…`), with the realm's own numbers — pier_top = 1.0250 m from
/// `top_flat_surface_y("naruko_pier")`, half = 0.4000 m and radius = 0.0500 m
/// from the solver binding:
///   y0   = pier_top + half + radius = 1.0250 + 0.4000 + 0.0500 = 1.4750
///   step = 2·half + radius          = 0.8000 + 0.0500          = 0.8500
///     (bare-radius chain; the ~0.5 mm `contact_margin` term is omitted — a
///      sub-REST_TOL approximation, exactly as physics.rs's stack ordeal notes.)
/// X/Z are the authored footprint (the solver never drifts them horizontally):
///   naruko_crate         rests at [-11.15, 1.4750, 13]  (single crate on pier)
///   naruko_stack_crate_0 rests at [-13.65, 1.4750, 13]  (= y0)
///   naruko_stack_crate_1 rests at [-13.65, 2.3250, 13]  (= y0 + step)
///   naruko_stack_crate_2 rests at [-13.65, 3.1750, 13]  (= y0 + 2·step)
/// The mesh box is unchanged by physics (only the transform moves), so the AABB
/// half-extent stays half = 0.4000 (a 0.8 box); measured residual rotation at
/// rest is ~1e-17 rad ⇒ the box stays axis-aligned.
///
/// RANGE = |rested_center − eye|, eye = the canon spawn pose [0,7,44] (read from
/// the realm, not hardcoded):
///   naruko_crate  [-11.15,1.4750,13] → √(124.3225+30.525625+961) = √1115.848125 = 33.4043
///   stack_crate_0 [-13.65,1.4750,13] → √(186.3225+30.525625+961) = √1177.848125 = 34.3198
///   stack_crate_1 [-13.65,2.3250,13] → √(186.3225+21.855625+961) = √1169.178125 = 34.1932
///   stack_crate_2 [-13.65,3.1750,13] → √(186.3225+14.630625+961) = √1161.953125 = 34.0874
///
/// AUTHORED → RESTED (the F6 delta, cross-referenced to canon.rs):
///   naruko_crate  authored [-11.15,4.5,13] range 33.0390  →  rested 33.4043  (Δ +0.3653)
///     — the ONLY material move: the crate is authored HUNG at y=4.5 and the
///       solver drops it 3.025 m onto the pier planks (rest y=1.4750).
///   stack_crate_0 authored range 34.3198  →  rested 34.3198  (Δ 0.0000)
///   stack_crate_1 authored range 34.1932  →  rested 34.1932  (Δ 0.0000)
///   stack_crate_2 authored range 34.0874  →  rested 34.0874  (Δ 0.0000)
///     — the STACK was authored already AT its solver-rest (chained heights), so
///       its senses-truth equals its load-pose; F6 confirms rather than moves it.
///
/// TOLERANCES (derived from measured floors):
///   REST_TOL = 0.005 m — physics.rs's rest tolerance (measured settle residual
///     0.00087 m, ~6× headroom). Applied to each rested-center axis: the derived
///     analytic rest vs the live solver rest.
///   RANGE_TOL = 1e-3 m — the live range is √Σ(center−eye)² in f32. The rested
///     center's live-vs-analytic gap (≤ REST_TOL) propagates through the range
///     gradient |Δy|/range ≤ 5.525/33.4 = 0.165 to ≤ 0.00084 m; measured live
///     deltas peak at 2e-4 m. 1e-3 is ≥1.2× the worst-case budget and ≥5× the
///     measured max — a wrong AABB (±0.1 m) still fails by ≥100×.
///   BOX_TOL = 1e-4 m — the box size is 2·half; measured f32 slop 7.6e-7 m
///     (0.79999924 vs 0.8), so 1e-4 is >100× the slop.
#[test]
fn senses_read_solver_rested_pose() {
    // Derive pier_top from the realm BEFORE the render scene consumes the world.
    let mut world = EcsWorld::default();
    load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
    let pier_top = top_flat_surface_y(&world, "naruko_pier")
        .expect("pier surface query")
        .expect("the pier has a flat top surface") as f64;
    let mut scene = RenderScene::from_ecs(world, &params()).expect("build the render scene");

    // Solver-binding dials — derived, not plucked (physics.rs precedent).
    let (half, radius) = {
        let physics = scene.physics().expect("bodies are declared");
        let binding = &physics.bindings()[0];
        (binding.half_height, binding.contact_radius)
    };

    // March to rest (same criterion as ORDEAL A); confirm the pinned tick.
    let first_floor = march_to_floor(&mut scene, REST_MARCH);
    assert_eq!(
        first_floor,
        Some(REST_TICK),
        "gaze run reached rest at REST_TICK"
    );

    // Lift the solver-rested ECS out through the F6 accessor.
    let ticked: EcsWorld = scene.dynamics.into_world();
    let ticked_transform = ticked
        .component_id("transform")
        .expect("transform component");

    // Load the realm the ORACLE way (correct envelopes + registry). Capture the
    // AUTHORED gaze first, then inject the rested transforms and gaze again.
    let mut oracle_world = World::load(canon_dir()).expect("load canon naruko");
    let eye = oracle_world
        .spawn_pose()
        .expect("canon spawn pose")
        .position;
    assert_eq!(
        eye,
        [0.0, 7.0, 44.0],
        "canon spawn eye (read, not hardcoded)"
    );

    let authored_center: Vec<[f32; 3]> = IDS
        .iter()
        .map(|id| oracle_world.geometry(id).unwrap().bounds.unwrap().center())
        .collect();

    let oracle_transform = oracle_world
        .core
        .world
        .component_id("transform")
        .expect("oracle transform component");
    for id in IDS {
        let src = ticked.entity_for_gaia(id).expect("ticked body");
        let rested = ticked.get_component(src, ticked_transform).unwrap();
        let dst = oracle_world
            .core
            .world
            .entity_for_gaia(id)
            .expect("oracle body");
        oracle_world
            .core
            .world
            .set_component(dst, oracle_transform, serde_json::json!({ "v": rested }))
            .expect("inject rested transform");
    }

    // Derived analytic REST centers (X/Z authored, Y chained), and the derived
    // rest range from the canon eye.
    let y0 = pier_top + half + radius;
    let step = 2.0 * half + radius;
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
        // Analytic rest center: authored X/Z (from the authored gaze), derived Y.
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

        // The senses' RESTED gaze (same code as everywhere).
        let g = oracle_world.geometry(id).expect("rested geometry");
        let b = g.bounds.expect("rested bounds");
        let c = b.center();
        let size = b.size();

        // Center reads the derived solver rest (all three axes within REST_TOL:
        // X/Z never drift, Y is the derived rest height).
        assert!(
            (c[0] as f64 - analytic_center[0]).abs() < REST_TOL
                && (c[1] as f64 - analytic_center[1]).abs() < REST_TOL
                && (c[2] as f64 - analytic_center[2]).abs() < REST_TOL,
            "{id} rested center {c:?} != derived rest {analytic_center:?} (tol {REST_TOL})"
        );

        // The box is unchanged by physics: size == 2·half on every axis.
        for s in size {
            assert!(
                (s as f64 - 2.0 * half).abs() < BOX_TOL,
                "{id} rested box side {s} != 2·half {} (tol {BOX_TOL})",
                2.0 * half
            );
        }

        // Range reads the derived rest range.
        let live_range = range_from([c[0] as f64, c[1] as f64, c[2] as f64]);
        assert!(
            (live_range - analytic_range).abs() < RANGE_TOL,
            "{id} rested range {live_range:.4} != derived {analytic_range:.4} (tol {RANGE_TOL})"
        );

        // Auditable authored → rested delta.
        let authored_range = range_from([a[0] as f64, a[1] as f64, a[2] as f64]);
        eprintln!(
            "[ordeal F6·B] {id}: authored center {a:?} range {authored_range:.4} \
             -> rested center [{:.4},{:.4},{:.4}] range {live_range:.4} (Δrange {:+.4})",
            c[0],
            c[1],
            c[2],
            live_range - authored_range
        );
    }
}
