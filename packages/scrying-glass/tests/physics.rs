//! PHYSICS-INTO-THE-WORLD ORDEALS (Elements P3) — the first thing falls.
//!
//! The realm `worlds/naruko` declares ONE wooden crate (`naruko_crate`) with a
//! `body` sigil, hung above the pier near the stall. These ordeals prove, on
//! the REAL realm through the REAL world tick:
//!
//!   1. ULTRADETERMINISM — two identical runs fold to byte-identical solver
//!      state hashes at every tick (the Loom's clock, no dice).
//!   2. REST — the crate falls and comes to rest at the DERIVED analytic height
//!      (pier plank top + crate half-height + particle contact radius), on the
//!      planks (never slid off, never sank through).
//!   3. ZERO-PHYSICS INERTNESS — with no `body` declared the physics seam is
//!      wholly absent (`physics() == None`) and the realm is byte-unchanged
//!      across ticks (the crate, now a plain mesh, never moves).

use crystal::{EcsWorld, ImpulseOp, Op, QuerySpec, load_world_dir};
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults, top_flat_surface_y};
use std::path::{Path, PathBuf};

fn naruko_world() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko")
}

/// The realm's authoring dials (mirrors the window/example defaults). Nothing
/// here is engine-hardcoded; the tick_dt is the solver's fixed tick.
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

fn naruko_scene() -> RenderScene {
    let mut world = EcsWorld::default();
    load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
    RenderScene::from_ecs(world, &params()).expect("build the render scene")
}

/// ORDEAL 1 — ULTRADETERMINISM. Two identical runs of N ticks produce the exact
/// same solver state hash at every tick (state = f(seed, entropy) — no dice).
#[test]
fn two_runs_are_byte_identical_state_hashes() {
    let run = || {
        let mut scene = naruko_scene();
        let mut hashes = Vec::new();
        for _ in 0..240u64 {
            scene.tick();
            hashes.push(scene.physics().expect("a body is declared").state_hash());
        }
        hashes
    };
    let a = run();
    let b = run();
    assert_eq!(a.len(), 240, "ticked the full run");
    assert_eq!(
        a, b,
        "two identical runs must fold to identical state hashes"
    );
    eprintln!(
        "[ordeal] ultradeterminism: 2 runs x 240 ticks, final state hash {:#018x}, byte-identical",
        a.last().unwrap()
    );
}

/// ORDEAL 2 — REST at the DERIVED analytic height. The crate falls onto the pier
/// planks and settles. The rest height is DERIVED from the geometry, never
/// authored: `pier_plank_top + crate_half_height + particle_contact_radius`.
#[test]
fn crate_falls_and_rests_on_the_planks_at_derived_height() {
    // Derive the pier plank top from the realm geometry (the flat surface the
    // crate lands on), BEFORE the render scene consumes the world.
    let mut world = EcsWorld::default();
    load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
    let pier_top = top_flat_surface_y(&world, "naruko_pier")
        .expect("pier surface query")
        .expect("the pier has a flat top surface") as f64;
    let mut scene = RenderScene::from_ecs(world, &params()).expect("build the render scene");

    // The body's own dials (half-height, contact radius) come from the solver
    // binding — derived, not plucked.
    let (half_height, contact_radius) = {
        let physics = scene.physics().expect("a body is declared");
        let binding = &physics.bindings()[0];
        (binding.half_height, binding.contact_radius)
    };
    let analytic_rest = pier_top + half_height + contact_radius;

    let start = scene.body_position("naruko_crate").expect("crate body")[1];

    // March to rest and watch the crate come down and stop.
    let mut last_y = start;
    let mut settled_at = None;
    for tick in 0..600u64 {
        scene.tick();
        let y = scene.body_position("naruko_crate").unwrap()[1];
        if settled_at.is_none() && (y - last_y).abs() < 1.0e-6 && tick > 60 {
            settled_at = Some(tick);
        }
        last_y = y;
    }
    let rest = scene.body_position("naruko_crate").unwrap();

    eprintln!(
        "[ordeal] rest: start y={start:.4}  final=[{:.4},{:.4},{:.4}]  analytic={analytic_rest:.4}  Δ={:.5}  settled≈tick {settled_at:?}",
        rest[0],
        rest[1],
        rest[2],
        (rest[1] - analytic_rest).abs()
    );

    // Fell (started well above, ended near the planks).
    assert!(
        start > analytic_rest + 2.0,
        "crate started above the planks"
    );
    // Came to rest at the derived analytic height. The measured settle residual
    // is 0.00087 m (≈ the 1 mm contact margin); REST_TOL = 0.005 m is ~6x that
    // — tight enough that a wrong rest height (±0.1 m) fails by ~20x, loose
    // enough never to flap on the margin.
    const REST_TOL: f64 = 0.005;
    assert!(
        (rest[1] - analytic_rest).abs() < REST_TOL,
        "crate rest y {} != derived analytic {analytic_rest} (tol {REST_TOL})",
        rest[1]
    );
    // Rested ON the planks — never slid off in x/z (stayed under its footprint).
    assert!(
        (rest[0] - (-11.15)).abs() < 0.2 && (rest[2] - 13.0).abs() < 0.2,
        "crate slid off its plank footprint: [{},{}]",
        rest[0],
        rest[2]
    );
}

/// ORDEAL 3 — ZERO-PHYSICS INERTNESS. Strip the `body` sigil and the physics
/// seam is wholly absent; the realm is byte-unchanged across ticks (the crate,
/// now a plain static mesh, never falls; static geometry is identical forever).
#[test]
fn no_body_declared_is_byte_unchanged() {
    let mut world = EcsWorld::default();
    load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
    // Remove every `body` — the realm now declares no physics at all.
    if let Some(body) = world.component_id("body") {
        let carriers = world.query(&QuerySpec {
            all: vec![body],
            ..Default::default()
        });
        for entity in carriers {
            world.remove_component(entity, body).unwrap();
        }
    }
    let mut scene = RenderScene::from_ecs(world, &params()).expect("build the render scene");

    // The physics seam is wholly absent.
    assert!(
        scene.physics().is_none(),
        "no body declared ⇒ no physics seam"
    );
    // The crate is now STATIC (only the behavior-carriers remain dynamic:
    // lantern + beacon + the three signal rings + the kami orb).
    assert_eq!(
        scene.dynamics.entities().len(),
        6,
        "with the body stripped only the six behavior-carriers are dynamic"
    );

    // Static geometry is byte-unchanged across 300 ticks (nothing falls).
    let before = leaf_bytes(&scene);
    for _ in 0..300u64 {
        scene.tick();
    }
    let after = leaf_bytes(&scene);
    assert_eq!(
        before, after,
        "a zero-physics realm's static geometry is byte-unchanged across ticks"
    );
    eprintln!(
        "[ordeal] zero-physics inertness: physics=None, {} static-leaf bytes byte-unchanged over 300 ticks",
        before.len()
    );
}

fn leaf_bytes(scene: &RenderScene) -> Vec<u8> {
    let mut bytes = Vec::new();
    for t in scene.leaf_triangles() {
        for p in t.positions {
            for c in p {
                bytes.extend_from_slice(&c.to_bits().to_le_bytes());
            }
        }
    }
    bytes
}

// ═════════════════════════════════════════════════════════════════════════
// VI-1 ORDEALS — THE STACK TOPPLES. `worlds/naruko` also declares a stack of
// three crates (`naruko_stack_crate_0/1/2`) authored resting directly atop
// each other on the pier planks. These ordeals prove, on the REAL realm
// through the REAL world tick:
//
//   4. STACK REST — the authored stack settles at the DERIVED heights (same
//      derivation as the single crate, chained: each box's rest height is
//      the one below's top + half-height + contact radius).
//   5. TOPPLE REPLAY — `Op::Impulse` applied through `Dynamics::tick_with_ops`
//      pushes the top crate; two identical topples fold to byte-identical
//      solver state hashes at every tick.
//   6. NOTHING FELL THROUGH THE FLOOR — after the topple, every stack body's
//      centroid stays above the pier deck (the derived deck-top Y bound), no
//      matter how it landed.
// ═════════════════════════════════════════════════════════════════════════

const STACK_IDS: [&str; 3] = [
    "naruko_stack_crate_0",
    "naruko_stack_crate_1",
    "naruko_stack_crate_2",
];

/// ORDEAL 4 — the authored stack settles at the derived chained rest heights.
#[test]
fn stack_settles_at_derived_chained_heights() {
    let mut world = EcsWorld::default();
    load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
    let pier_top = top_flat_surface_y(&world, "naruko_pier")
        .expect("pier surface query")
        .expect("the pier has a flat top surface") as f64;
    let mut scene = RenderScene::from_ecs(world, &params()).expect("build the render scene");

    let (half_height, contact_radius) = {
        let physics = scene.physics().expect("bodies are declared");
        let binding = physics
            .bindings()
            .iter()
            .find(|b| b.gaia_id == STACK_IDS[0])
            .expect("stack crate 0 binding");
        (binding.half_height, binding.contact_radius)
    };
    // Chained derivation — mirrors how the realm authored each position.
    let mut expected = Vec::with_capacity(3);
    let mut y = pier_top + half_height + contact_radius;
    for _ in 0..3 {
        expected.push(y);
        y += 2.0 * half_height + contact_radius;
    }

    // The stack was authored already AT its rest positions — a short march
    // lets any residual contact settle, it should barely move.
    for _ in 0..120u64 {
        scene.tick();
    }

    const REST_TOL: f64 = 0.005; // same tolerance the single-crate rest ordeal derives
    for (id, exp) in STACK_IDS.iter().zip(expected.iter()) {
        let pos = scene.body_position(id).unwrap();
        eprintln!("[ordeal] stack rest: {id} y={:.4} expected={exp:.4}", pos[1]);
        assert!(
            (pos[1] - exp).abs() < REST_TOL,
            "{id} rest y {} != derived chained height {exp} (tol {REST_TOL})",
            pos[1]
        );
    }
}

/// ORDEAL 5 — an `Op::Impulse` topples the stack; two identical topples fold
/// to byte-identical state hashes at every tick of the full episode.
#[test]
fn stack_topple_via_impulse_replays_byte_identical() {
    let run = || {
        let mut world = EcsWorld::default();
        load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
        let mut scene = RenderScene::from_ecs(world, &params()).expect("build the render scene");
        // Let the authored-at-rest stack finish any residual settle first.
        for _ in 0..120u64 {
            scene.tick();
        }
        let impulse = Op::Impulse(ImpulseOp {
            id: STACK_IDS[2].to_string(),
            delta_velocity: [3.0, 0.0, 0.0],
            ..Default::default()
        });
        scene.tick_with_ops(&[impulse]);
        let mut hashes = Vec::with_capacity(600);
        for _ in 0..600u64 {
            scene.tick();
            hashes.push(scene.physics().unwrap().state_hash());
        }
        hashes
    };
    let a = run();
    let b = run();
    assert_eq!(a.len(), 600, "ticked the full topple");
    assert_eq!(
        a, b,
        "two identical topples (same impulse, same seed) must fold to identical state hashes"
    );
    eprintln!(
        "[ordeal] stack topple replay: 600 ticks x 2 runs, byte-identical, final hash {:#018x}",
        a.last().unwrap()
    );
}

/// ORDEAL 6 — after the topple, nothing fell through the pier deck. The
/// bound is DERIVED from the realm's own geometry: the pier plank top,
/// minus one contact radius (the solver's own contact thickness — the most
/// a body can numerically penetrate before it counts as "through the floor").
#[test]
fn stack_topple_never_falls_through_the_pier_deck() {
    let mut world = EcsWorld::default();
    load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
    let pier_top = top_flat_surface_y(&world, "naruko_pier")
        .expect("pier surface query")
        .expect("the pier has a flat top surface") as f64;
    let mut scene = RenderScene::from_ecs(world, &params()).expect("build the render scene");

    let contact_radius = {
        let physics = scene.physics().expect("bodies are declared");
        physics.bindings()[0].contact_radius
    };
    let deck_floor_bound = pier_top - contact_radius;

    for _ in 0..120u64 {
        scene.tick();
    }
    let impulse = Op::Impulse(ImpulseOp {
        id: STACK_IDS[2].to_string(),
        delta_velocity: [3.0, 0.0, 0.0],
        ..Default::default()
    });
    scene.tick_with_ops(&[impulse]);
    let mut min_y_over_run = f64::INFINITY;
    for _ in 0..600u64 {
        scene.tick();
        for id in STACK_IDS {
            let y = scene.body_position(id).unwrap()[1];
            min_y_over_run = min_y_over_run.min(y);
        }
    }
    eprintln!(
        "[ordeal] stack never falls through: pier_top={pier_top:.4} deck_floor_bound={deck_floor_bound:.4} \
         min y over the topple={min_y_over_run:.4}"
    );
    assert!(
        min_y_over_run > deck_floor_bound,
        "a stack body sank through the pier deck: min y {min_y_over_run} <= bound {deck_floor_bound} \
         (pier top {pier_top} - contact radius {contact_radius})"
    );
}
