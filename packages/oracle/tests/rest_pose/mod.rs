//! SHARED REST-POSE MACHINERY — GUARDIAN RULING F6 ("senses read SOLVER
//! TRUTH — the world as it is, not as authored", NARUKO.md · GUARDIAN
//! RULINGS · item 5).
//!
//! One tick-to-rest + transform-injection path, reused by BOTH
//! `tests/rest_pose_canon.rs` (the dedicated F6 ordeals proving determinism
//! and the full per-vessel derivation) and `tests/canon.rs` (the migrated
//! headline canon rows for the four physics vessels in
//! `canon_nearest_ordering_and_ranges_are_derived`). Nothing is re-invented
//! between the two files, and nothing is re-invented from
//! `packages/scrying-glass/tests/physics.rs` either — the rest-detection
//! criterion and the analytic rest-height chain are the SAME arithmetic
//! physics.rs already proved (`crate_falls_and_rests_on_the_planks_at_derived_height`,
//! `stack_settles_at_derived_chained_heights`).
//!
//! HOW (dependency direction): `oracle` and `scrying-glass` are SIBLINGS —
//! both depend only on `crystal`, which depends on neither — so `oracle`
//! takes `scrying-glass` (and `elements`, for the shared `contact_margin`
//! dial) as DEV-dependencies with zero cycle risk. The ticked `EcsWorld` is
//! lifted out of the render scene through the F6 accessor
//! `Dynamics::into_world` (added in scrying-glass for exactly this: external
//! senses gazing at solver-rested state).
//!
//! WHY inject rather than wrap-and-gaze directly: `crystal::load_world_dir`
//! (scrying-glass's loader) stores components crystal-native, but the
//! oracle's own `World::load` wraps each component in a `{"v": …}` envelope
//! that `component_value` unwraps. Wrapping a `load_world_dir` world straight
//! into an `oracle::World` therefore reads `None` for every component. So we
//! load the realm the ORACLE way (correct envelopes + full id registry) and
//! INJECT the solver-rested `transform` (read out of the ticked world) into
//! each body — the oracle's documented LIVE-mutation path ("mutate
//! `world.core` and the next gaze reflects it"). Only `transform` changes
//! under physics; `mesh`/`body` are unchanged, so the fresh oracle load
//! supplies them correctly.

use crystal::{load_world_dir, EcsWorld};
use oracle::World;
use scrying_glass::scene::{top_flat_surface_y, RenderScene, SceneParameters, SunDefaults};
use std::path::{Path, PathBuf};

/// The realm dir, uncanonicalized — fed to scrying-glass's `load_world_dir`
/// (mirrors `packages/scrying-glass/tests/physics.rs` `naruko_world()`).
pub fn naruko_world() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko")
}

/// The realm dir, canonicalized — fed to the oracle's own `World::load`
/// (mirrors `canon.rs` `canon_dir()`). Same realm, both point at
/// `<repo>/worlds/naruko`.
pub fn canon_dir() -> PathBuf {
    naruko_world().canonicalize().expect("canon naruko dir")
}

/// The realm's authoring dials — mirrors physics.rs `params()` (which mirrors
/// the window/example defaults). Nothing engine-hardcoded; `tick_dt` is the
/// solver's fixed tick.
pub fn params() -> SceneParameters {
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
pub const IDS: [&str; 4] = [
    "naruko_crate",
    "naruko_stack_crate_0",
    "naruko_stack_crate_1",
    "naruko_stack_crate_2",
];

/// The solver's numerical kinetic-floor epsilon (a body's per-tick |Δy|):
/// below this the body is at rest. Reused verbatim from physics.rs's
/// `settled_at` check — the solver's own rest epsilon, not a canon geometry
/// number.
pub const KINETIC_FLOOR: f64 = 1.0e-6;

/// Warmup ticks before the floor check counts. The FALL itself passes through
/// sub-epsilon vertical steps (apex, launch), so floor hits before this don't
/// mean "rested". physics.rs guards its `settled_at` with the same
/// `tick > 60`.
pub const REST_WARMUP: u64 = 60;

/// The tick at which EVERY declared body is first at the kinetic floor.
/// DERIVATION: measured empirically as tick 91, confirmed BYTE-IDENTICAL
/// across two independent fresh ticks (`rest_pose_canon.rs::rest_is_deterministic`
/// asserts both land on exactly this value). Pinned as this file's fixed
/// rest-tick — not eyeballed from physics.rs's 600/120 march lengths.
pub const REST_TICK: u64 = 91;

/// March length before a rested gaze. physics.rs's own stack-settle march is
/// 120 ticks; 120 is ≥1.3× the observed REST_TICK (91), so the read is taken
/// well after every body has reached the floor.
pub const REST_MARCH: u64 = 120;

/// A fresh render scene on the real realm through the real world tick —
/// exactly physics.rs `naruko_scene()`.
pub fn fresh_scene() -> RenderScene {
    let mut world = EcsWorld::default();
    load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
    RenderScene::from_ecs(world, &params()).expect("build the render scene")
}

/// Tick `scene` `ticks` times; return the first tick `> REST_WARMUP` at which
/// EVERY body's per-tick |Δy| < `KINETIC_FLOOR` (the whole system at the
/// floor).
pub fn march_to_floor(scene: &mut RenderScene, ticks: u64) -> Option<u64> {
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

/// The solver dials shared by every physics vessel here — pier surface
/// height, box half-height, and contact radius (read from the realm/solver,
/// never hardcoded), plus the material's `contact_margin`. physics.rs
/// `stack_settles_at_derived_chained_heights` (`:254-274`) uses these exact
/// dials to build its chained analytic rest heights; this struct is the same
/// arithmetic, factored out so both oracle test files share one derivation.
pub struct RestDials {
    pub pier_top: f64,
    pub half: f64,
    pub radius: f64,
    pub contact_margin: f64,
}

impl RestDials {
    /// The pier-contact rest height (particle-vs-triangle pass): a crate
    /// sitting directly on the pier deck. physics.rs's own comment on this
    /// contact (`:256-259`): this pass's true rest gap is
    /// `contact_radius + contact_margin`, but the pre-existing convention
    /// omits the ~1 mm margin term here (well under REST_TOL) — reused
    /// verbatim, not re-derived.
    pub fn pier_contact_y(&self) -> f64 {
        self.pier_top + self.half + self.radius
    }

    /// The body-vs-body chain step (F2/F3 convention, physics.rs `:262-268`):
    /// `2·half + radius + contact_margin` — INCLUDING the margin term, unlike
    /// the pier contact above. This is the exact `y +=` step physics.rs's
    /// stack ordeal uses; reused verbatim here, not approximated.
    pub fn body_chain_step(&self) -> f64 {
        2.0 * self.half + self.radius + self.contact_margin
    }
}

/// Tick a fresh scene to `REST_TICK`, asserting it lands there (the pinned,
/// checked determinism), then lift the solver-rested `EcsWorld` out through
/// the F6 accessor `Dynamics::into_world`. Returns the ticked world plus the
/// solver dials read before the tick consumed the scene's owned `EcsWorld`.
pub fn tick_to_rest() -> (EcsWorld, RestDials) {
    let mut world = EcsWorld::default();
    load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
    let pier_top = top_flat_surface_y(&world, "naruko_pier")
        .expect("pier surface query")
        .expect("the pier has a flat top surface") as f64;
    let mut scene = RenderScene::from_ecs(world, &params()).expect("build the render scene");

    let (half, radius) = {
        let physics = scene.physics().expect("bodies are declared");
        let binding = &physics.bindings()[0];
        (binding.half_height, binding.contact_radius)
    };
    let contact_margin = elements::ContactMaterial::default().contact_margin;

    let first_floor = march_to_floor(&mut scene, REST_MARCH);
    assert_eq!(
        first_floor,
        Some(REST_TICK),
        "rest march reached the pinned REST_TICK"
    );

    let ticked: EcsWorld = scene.dynamics.into_world();
    (
        ticked,
        RestDials {
            pier_top,
            half,
            radius,
            contact_margin,
        },
    )
}

/// Load the oracle canon world the ORACLE way (correct envelopes +
/// registry), then INJECT each of the four physics vessels' solver-rested
/// `transform` (read out of `ticked`) — the oracle's documented LIVE-mutation
/// path. Only `transform` changes under physics; `mesh`/`body` are
/// unchanged, so the fresh oracle load supplies them correctly.
pub fn inject_rested_transforms(ticked: &EcsWorld, oracle_world: &mut World) {
    let ticked_transform = ticked
        .component_id("transform")
        .expect("transform component");
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
}

/// The one-call convenience both test files use: tick a fresh realm to REST,
/// load the oracle's canon world, and inject the four physics vessels' rested
/// transforms into it. The returned `World` gazes at SOLVER TRUTH for
/// `naruko_crate`/`naruko_stack_crate_0/1/2`; every other vessel is
/// unaffected (their load pose already IS solver truth — nothing ever moves
/// them).
pub fn rested_canon_world() -> (World, RestDials) {
    let (ticked, dials) = tick_to_rest();
    let mut oracle_world = World::load(canon_dir()).expect("load canon naruko");
    inject_rested_transforms(&ticked, &mut oracle_world);
    (oracle_world, dials)
}
