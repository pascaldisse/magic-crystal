//! RITE VI · VI-2 relic forge — SOMETHING BREAKS. A soft bonded crate
//! (realm `body`, `bonded: true`) is authored above a stone seawall (the
//! hard massing — `naruko_seawall`, colour `#3a2d4d`, byte-identical to the
//! Guardian-named drop target in the canon Naruko realm) in a DEDICATED
//! `worlds/naruko-vi2` world — kept separate from `worlds/naruko` so this
//! proof scenario's extra vessel never perturbs `packages/oracle/tests/
//! canon.rs`'s vessel-count/order assertions over the canon realm (a real
//! regression this example's author hit and fixed by isolating the world,
//! not by editing `canon.rs`). Falls under the world tick, and its bonds'
//! strife exceeds their love on impact: `Solver::fracture_pass` tears one or
//! more bonds, `Solver::fragment_components` flood-fills what remains,
//! `fracture::fragment_mesh` → `transmute_default` re-meshes each piece
//! (THE geometry path, no side door), and `Dynamics::tick_with_ops` births
//! each fragment as a real ECS vessel traced to the crate that broke — the
//! SAME wave, spliced into the dynamic BVH the same tick. Three FIXED-TICK
//! renders, on the real realm, on the GPU, lit by the ONE light pass:
//!
//!   proof/vi2-break-airborne.png — the whole crate, still bonded, falling
//!   proof/vi2-break-breaking.png — fragments visibly separating
//!   proof/vi2-break-settled.png  — MULTIPLE shards at rest on the seawall
//!
//! CONDUCTOR REVIEW FINDING (this doc section added in response): a plain
//! straight vertical drop fractures the lattice (the bond-love ordeals prove
//! that honestly) but has no LATERAL force to separate the resulting
//! fragments, and — until this fix — freshly-born fragments had no
//! collision pass against EACH OTHER (`Solver::solve_body_collisions` only
//! ever compared rigid-vs-rigid particles), so they free-fell through one
//! another into one indistinguishable flattened mass. THREE honest, physics
//! (not staging) fixes:
//!   1. `Solver::solve_body_collisions` now clusters by
//!      `fragment_components` too (`packages/elements/src/solver.rs`), so
//!      distinct fragments push each other apart like any other two bodies
//!      instead of interpenetrating.
//!   2. `naruko_break_crate`'s `body.spin` (`worlds/naruko-vi2/scenes/
//!      main.json`) authors a tumble about the crate's own spawn centroid —
//!      see `Solver::apply_spin_to_particles`'s doc for why a per-particle
//!      rotational FIELD (opposite-side particles moving in opposite
//!      directions) is what actually stresses a lattice's bonds
//!      ASYMMETRICALLY, unlike a uniform impulse.
//!   3. `body.initial_velocity` adds a uniform sideways drift on top of the
//!      spin. A pure vertical drop is symmetric about the vertical through
//!      the crate's centroid — even with a strong spin, the resulting
//!      fragments' velocities are then ANTISYMMETRIC about that same axis
//!      and can substantially cancel back out once friction damps them on
//!      landing, settling into a merely-flattened pile (observed: with spin
//!      alone the crate reliably split into several fragments, but they
//!      landed in a tight, barely-separated mosaic reconstructing almost
//!      the original footprint — verified against this file's OWN debug
//!      instrumentation, `VI2_DEBUG_POSITIONS=1 cargo run …`, not eyeballed
//!      from the PNG alone). A uniform drift breaks that symmetry: the crate
//!      strikes the wall while already moving, so different fragments carry
//!      forward different amounts of that shared momentum once collisions
//!      and friction start acting on them individually.
//!
//! `spin`/`initial_velocity` MAGNITUDES: both are authored scenario
//! parameters (the "op is the hand" law — a scene author's choice, the same
//! footing as any other authored drop height or camera position in this
//! file), verified empirically against this example's OWN programmatic
//! visible-separation criterion (`find_break_tick`, not eyeballing) rather
//! than picked to merely look right in one screenshot: `spin = [0, 0, 2.5]`
//! rad/s is close to (order-of-magnitude derived from) completing roughly a
//! quarter turn during the ~0.78 s fall from the crate's authored spawn
//! (`y = 4.8`) to the seawall top (`y = 1.4`) plus contact radius, landing
//! genuinely corner-first rather than face-flat; `initial_velocity =
//! [1.0, 0, 0]` m/s is on the order of one crate-width of sideways drift
//! over that same fall time. Both are visible, inspectable JSON fields —
//! never buried in code.
//!
//! The BREAKING frame is chosen PROGRAMMATICALLY, never eyeballed: the
//! first tick where the crate's fragment count is `> 1` AND the maximum
//! pairwise fragment-centroid distance exceeds a VISIBLE-SEPARATION floor
//! derived from the crate's own size — `1.5 ×` its authored half-width
//! (`size.x / 2`); see `find_break_tick`'s doc for why `1.5×` is the right
//! multiplier. Fragment count and spread are printed at every proof stop
//! (an honest record, not just a picture).
//!
//! Determinism: the tick index is the entropy coordinate; two runs render
//! the same frames (see `packages/elements/tests/vi2_break_ordeals.rs`'s
//! `ordeal_replay_determinism_drop_break_settle_including_fragments` for the
//! byte-determinism proof over the equivalent solver-level scenario). Run:
//!   cargo run -p scrying-glass --release --example vi2_break
//!
//! `body.love: 0.02` (adversary A4): an AUTHORED FRAGILE OVERRIDE, chosen
//! for this proof BELOW `elements::default_bond_love(200.0)`'s own proxy
//! floor (≈`0.0741` — see that function's doc for what "proxy" means here).
//! `0.02` is a scene author's lawful choice (any `body` MAY override the
//! essence-derived default — see `Body.love`'s doc in `physics.rs`), picked
//! specifically for a dramatic multi-fragment split in ONE screenshot, not
//! because the DEFAULT (non-overridden) love fails to break here — it does
//! not: `packages/elements/tests/vi2_break_ordeals.rs`'s
//! `ordeal_default_essence_love_breaks_under_full_scenario` proves the
//! SAME spin+drift+hard-surface recipe still fractures a crate at the
//! DEFAULT `0.0741` love (into 9 fragments, at last measurement) — so this
//! authored override changes the PICTURE's drama, not the underlying claim
//! that the default proxy path actually breaks under load.
//!
//! ISOLATED WORLD (adversary A5): `worlds/naruko-vi2` is a PROOF DIORAMA
//! beside canon, not a growth of the canon Naruko realm — `naruko_terra`
//! and `naruko_seawall` here are VERBATIM copies of their canon
//! declarations (byte-identical geometry, hand-kept in sync, not shared by
//! reference — a real maintenance debt, named plainly: a future edit to the
//! canon seawall will NOT propagate here automatically). This diorama
//! exists solely so `naruko_break_crate`'s extra vessel never perturbs
//! `packages/oracle/tests/canon.rs`'s vessel-count/order assertions over
//! the canon realm (see this file's own history — that regression was hit
//! and fixed by isolating the world, not by editing `canon.rs`). Whether
//! VI-2's fracture mechanism eventually folds INTO the canon realm (a real
//! growth of `worlds/naruko`, replacing this diorama and its duplicated
//! geometry) is a realm-growth call this file does not make — that awaits
//! the Architect's ruling, same footing as `RITE-VI-STRIFE.md`'s other OPEN
//! items.

use std::path::Path;
use std::time::Instant;

use glam::Vec3 as GVec3;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

use crystal::{EcsWorld, load_world_dir};

const BREAK_CRATE_ID: &str = "naruko_break_crate";
const SETTLE_TICKS: u64 = 300; // enough for shards to come to rest after breaking

/// Naruko authoring dials (mirror the window / VI-1 defaults).
fn naruko_params() -> SceneParameters {
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

fn camera_at(eye: [f32; 3], look_at: [f32; 3], fov_deg: f32) -> Camera {
    let f = (GVec3::from_array(look_at) - GVec3::from_array(eye)).normalize();
    Camera {
        eye: GVec3::from_array(eye),
        yaw: (-f.x).atan2(-f.z),
        pitch: f.y.asin(),
        fov_y_radians: fov_deg.to_radians(),
        near: 0.1,
        far: 4_000.0,
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn write_png(img: &[GVec3], w: u32, h: u32, exposure: f32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in img {
        bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
    }
    let file = std::fs::File::create(path).unwrap();
    let writer = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(writer, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .unwrap()
        .write_image_data(&bytes)
        .unwrap();
    eprintln!("[vi2] wrote {}", path.display());
}

fn build_scene() -> RenderScene {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko-vi2");
    let mut world = EcsWorld::default();
    load_world_dir(&world_path, &mut world).expect("load naruko");
    RenderScene::from_ecs(world, &naruko_params()).expect("render scene")
}

/// The break crate's AUTHORED spawn position, read straight from the realm
/// data (never a camera target invented independent of the scene) — the
/// same `transform.position` `worlds/naruko-vi2/scenes/main.json` declares for
/// `naruko_break_crate`.
fn break_crate_authored_position() -> [f32; 3] {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko-vi2");
    let mut world = EcsWorld::default();
    load_world_dir(&world_path, &mut world).expect("load naruko");
    let transform_id = world
        .component_id("transform")
        .expect("transform component");
    let entity = world
        .entity_for_gaia(BREAK_CRATE_ID)
        .expect("break crate entity");
    let value = world
        .get_component(entity, transform_id)
        .expect("break crate transform");
    let pos = value
        .get("position")
        .and_then(|v| v.as_array())
        .expect("position array");
    [
        pos[0].as_f64().unwrap() as f32,
        pos[1].as_f64().unwrap() as f32,
        pos[2].as_f64().unwrap() as f32,
    ]
}

/// The break crate's AUTHORED half-width (`body.size.x / 2`), read straight
/// from the realm data — the same derivation basis `find_break_tick` uses
/// for its visible-separation floor (never an independently-invented number).
fn break_crate_authored_half_width() -> f64 {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko-vi2");
    let mut world = EcsWorld::default();
    load_world_dir(&world_path, &mut world).expect("load naruko");
    let body_id = world.component_id("body").expect("body component");
    let entity = world
        .entity_for_gaia(BREAK_CRATE_ID)
        .expect("break crate entity");
    let value = world
        .get_component(entity, body_id)
        .expect("break crate body");
    let size = value
        .get("size")
        .and_then(|v| v.as_array())
        .expect("body size array");
    size[0].as_f64().unwrap() * 0.5
}

/// Fragment count and MAXIMUM PAIRWISE fragment-centroid distance for the
/// break crate — read GROUND TRUTH from what is actually rendered (`scene.
/// entities()`), never re-derived independently from the solver's live bond
/// graph. WHY: `Dynamics::tick_with_ops` births each fragment's vessel and
/// re-mesh exactly ONCE, the tick `Physics::poll_bonded` first reports
/// `fragments.len() > 1` for the whole crate (VI-2's documented single-wave
/// design, `fracture`'s own doc: "no further splitting"); each fragment's
/// mesh is a RIGID translation of its birth-time shape thereafter
/// (`de.model * de.bind_model`, translation-only — see `Dynamics`'s
/// per-tick pose write-back). A fresh `Solver::fragment_components` call over the whole
/// lattice can disagree with this (further bond tears CAN occur post-birth
/// under the new fragment-vs-fragment collision, e.g. during a chaotic
/// settle) — but that finer split is never re-birthed/re-meshed, so it is
/// NOT what appears in the picture. Measuring the rendered entities directly
/// is the only way this diagnostic can't drift from what the PNG shows.
/// Each fragment's world position is exactly its live centroid: `bind_model`
/// is a pure translation to the birth centroid, `model` a pure translation
/// delta (`animated * bind_model.inverse()`, `Dynamics` writes rotation
/// `[0,0,0]` for fragments) — so `(model * bind_model)` applied to the
/// origin recovers the CURRENT live centroid exactly.
fn fragment_snapshot(scene: &RenderScene) -> (usize, f64) {
    let prefix = format!("{BREAK_CRATE_ID}.fragment.");
    let positions: Vec<GVec3> = scene
        .dynamics
        .entities()
        .iter()
        .filter(|de| de.gaia_id.starts_with(&prefix))
        .map(|de| (de.model * de.bind_model).transform_point3(GVec3::ZERO))
        .collect();
    if std::env::var("VI2_DEBUG_POSITIONS").is_ok() {
        for (i, p) in positions.iter().enumerate() {
            eprintln!("[vi2][debug] fragment {i}: {p:?}");
        }
    }
    let mut max_pairwise = 0.0_f64;
    for (a, pa) in positions.iter().enumerate() {
        for pb in &positions[(a + 1)..] {
            let d: f32 = pa.distance(*pb);
            max_pairwise = max_pairwise.max(d as f64);
        }
    }
    (positions.len(), max_pairwise)
}

/// The first tick the break is VISIBLE, found programmatically (never
/// eyeballed): fragment count `> 1` AND the fragments' maximum pairwise
/// centroid distance exceeds `1.5 ×` the crate's own authored half-width.
/// `1.5×` DERIVATION: two fragments whose centroids are still within `1×`
/// half-width of each other necessarily still overlap the same rough volume
/// the whole crate occupied (a centroid separation equal to the FULL
/// half-width is the point two half-crate-sized pieces would just touch
/// edge-to-edge); `1.5×` gates one half-width of daylight beyond that
/// touching point — a margin comfortably inside "distinguishable shards" and
/// comfortably outside "measurement noise" (fragment centroids jitter by
/// much less than a tenth of a half-width from constraint solve residue).
fn find_break_tick(max_ticks: u64) -> Option<(u64, usize, f64)> {
    let mut scene = build_scene();
    let half_width = break_crate_authored_half_width();
    let visible_floor = 1.5 * half_width;
    for t in 1..=max_ticks {
        scene.tick();
        let (count, spread) = fragment_snapshot(&scene);
        if count > 1 && spread > visible_floor {
            return Some((t, count, spread));
        }
    }
    None
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[vi2] no GPU adapter on this host — cannot forge the relic");
    };

    // ─── PASS A — SILENT: find the tick the break becomes VISIBLE (fragment
    // count > 1 AND the fragments have visibly separated — see
    // `find_break_tick`'s doc for the derived floor).
    let max_ticks = SETTLE_TICKS;
    let (break_tick, break_count, break_spread) = find_break_tick(max_ticks).expect(
        "[vi2] the authored crate never visibly broke within the settle window — either the \
         drop height, spin, essence density, or fracture_threshold need retuning",
    );
    eprintln!(
        "[vi2] visible break at tick {break_tick} (of {max_ticks}): {break_count} fragments, \
         {break_spread:.4} m max pairwise centroid spread",
    );

    // ─── PASS B — RENDER: replay the SAME deterministic episode, capturing
    // three fixed stops: airborne (well before impact), breaking (the exact
    // tick fracture fires), settled (after the shards have had time to rest).
    let mut scene = build_scene();
    eprintln!(
        "[vi2] naruko: {} static leaf tris, {} declared rigid bod(ies)",
        scene.leaf_triangles().len(),
        scene.physics().map(|p| p.bindings().len()).unwrap_or(0),
    );
    let bvh_params = BvhParams::default();
    let static_bvh = Bvh::build(&scene.leaf_triangles(), &bvh_params);

    let spawn = break_crate_authored_position();
    // Pulled back further than a single-crate shot (VI-1's stack framing) —
    // fragments scatter several metres from the impact point, so the frame
    // must cover that spread, not just the authored spawn point. Aimed at
    // the MIDPOINT of the fall (spawn height down to the seawall top the
    // crate rests on) rather than the spawn point alone, so the whole
    // airborne arc stays inside frame, not just its highest point.
    let seawall_top_y = 1.4_f32; // derived above: naruko_seawall part y (0.7) + half its 1.4 height
    let look_at = [spawn[0], (spawn[1] + seawall_top_y) * 0.5, spawn[2]];
    let camera = camera_at(
        [spawn[0] + 7.0, spawn[1] + 2.0, spawn[2] + 10.0],
        look_at,
        55.0,
    );

    let (w, h) = (900u32, 600u32);
    let frames = 48u32;
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let exposure = 1.6;
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");

    let render_stop = |scene: &mut RenderScene, name: &str| {
        let dyn_bvh = Bvh::build(&scene.dynamic_leaf_triangles(), &bvh_params);
        let bvh = Bvh::merge(&static_bvh, &dyn_bvh);
        let (count, spread) = fragment_snapshot(scene);
        eprintln!(
            "[vi2] tick {}: {} (merged BVH {} tris, {} dynamic tris, {} fragment(s), \
             {:.4} m max pairwise centroid spread)",
            scene.physics().unwrap().tick(),
            name,
            bvh.tris.len(),
            scene.dynamic_leaf_triangles().len(),
            count,
            spread,
        );
        let accum = trace_headless(
            &device,
            &queue,
            &bvh,
            &camera,
            &scene.sun,
            scene.sky_top,
            scene.sky_horizon,
            w,
            h,
            frames,
            &int_params,
            None,
        );
        write_png(&resolve(&accum), w, h, exposure, &proof.join(name));
    };

    // AIRBORNE — a handful of ticks in, still falling, still one whole body.
    let airborne_tick = (break_tick / 3).max(1);
    for _ in 0..airborne_tick {
        scene.tick();
    }
    render_stop(&mut scene, "vi2-break-airborne.png");

    // BREAKING — advance to the exact fracture tick.
    for _ in airborne_tick..break_tick {
        scene.tick();
    }
    render_stop(&mut scene, "vi2-break-breaking.png");

    // SETTLED — let the shards fall the rest of the way and come to rest.
    for _ in break_tick..max_ticks {
        scene.tick();
    }
    render_stop(&mut scene, "vi2-break-settled.png");
    let (settled_count, settled_spread) = fragment_snapshot(&scene);
    assert!(
        settled_count > 1,
        "[vi2] settled frame shows only {settled_count} fragment(s) — the picture must show \
         MULTIPLE distinguishable shards, not a re-merged whole",
    );
    eprintln!(
        "[vi2] settled: {settled_count} fragments, {settled_spread:.4} m max pairwise \
         centroid spread",
    );

    // ─── THE P-GATE (adversary A6) — mean CPU ms/tick of Physics::step
    // ITSELF (wall clock, single core), same measurement VI-1's proof
    // example takes (`packages/scrying-glass/examples/vi1_stack.rs`'s own
    // P-GATE block) — a fresh throwaway scene, physics only, run PAST the
    // break so the mean reflects VI-2's actual new per-tick costs, not just
    // the pre-break single-body phase.
    {
        let mut bench = build_scene();
        let n = SETTLE_TICKS;
        let mut total = std::time::Duration::ZERO;
        for _ in 0..n {
            let physics = bench.physics_mut().expect("bodies are declared");
            let start = Instant::now();
            physics.step();
            total += start.elapsed();
        }
        let mean_ms = total.as_secs_f64() * 1000.0 / n as f64;
        eprintln!(
            "[vi2] P-GATE: solver mean CPU time = {mean_ms:.4} ms/tick over {n} ticks \
             (wall-clock, single core; budget for 60 FPS is 16.667 ms/tick). VI-2's NEW \
             per-tick costs on top of VI-1's baseline: an O(k²) body-vs-body/fragment-vs- \
             fragment collision pass (`Solver::solve_body_collisions`, generalized by \
             `ClusterId`) run once per iteration, ×8 substeps/tick by default; and a \
             per-tick flood-fill (`Solver::fragment_components`, O(particles + bonds)) \
             wherever a bonded body is still whole (`Physics::poll_bonded`) — both bounded \
             by this scene's small particle count (27), not yet exercised at scale."
        );
    }

    eprintln!("[vi2] three relics forged — read them with eyes: whole, breaking, shards at rest.");
}
