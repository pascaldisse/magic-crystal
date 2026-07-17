//! RITE V · V0 render-side ordeals — the embodied body enters the traced
//! world. These are CPU-only (no GPU): they prove the `body` sigil composes
//! into world-space skinned triangles that ride the DYNAMIC partition (re-fed
//! to the traced BVH every tick, like the living layer), stand on the seawall,
//! and are deterministic.

use std::path::Path;

use crystal::{Core, load_world_dir};
use glam::Vec3;
use homunculus::{Pose, Skeleton};
use sama::{GaitParams, Locomotion, LocomotionParams};
use scrying_glass::scene::{
    BodyInstance, LeafTriangle, RenderScene, SceneParameters, SunDefaults, WalkerPose,
    contact_passing_ticks,
};
use vessel::{Body, Preset};

/// A scripted straight WALK for the walker along +x on the seawall band
/// (z=18, nari's authored strip), starting exactly at her authored xz so tick 0
/// is a zero-displacement idle. `step` metres per tick over `ticks` ticks; the
/// eye y deliberately CLIMBS (2.5 + 0.5·k) so a grounded body's y — which must
/// track the FLOOR, not the eye — provably decouples from it. The FINAL WELD
/// ordeals drive `command_bodies_walked` with this so nari TRACKS it.
fn walker_walk(step: f32, ticks: usize) -> Vec<WalkerPose> {
    (0..ticks)
        .map(|k| WalkerPose {
            position: Vec3::new(step * k as f32, 2.5 + 0.5 * k as f32, 18.0),
            yaw: 0.0,
        })
        .collect()
}

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

fn load_naruko() -> Core {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).expect("load naruko");
    core
}

/// The `body` sigil composes into a standing embodied vessel: nari, the nari
/// preset, with a full skinned triangle soup (matching the vessel's 2268-tri
/// idle mesh).
#[test]
fn v0_body_composes_from_the_sigil() {
    let core = load_naruko();
    let scene = RenderScene::from_ecs(core.world, &naruko_params()).expect("render scene");

    // RITE V·V2 added the pink cat, so TWO bodies now carry the sigil; they
    // sort by gaia id ("nari" < "naruko_cat"), so nari stays index 0.
    assert_eq!(
        scene.bodies.len(),
        2,
        "nari + the pink cat carry body sigils"
    );
    let nari = &scene.bodies[0];
    assert_eq!(nari.gaia_id, "nari");
    assert_eq!(nari.preset, "nari");
    // The compose skins the whole vessel (nari idle mesh = 2268 tris).
    assert_eq!(
        nari.world_tris.len(),
        2268,
        "nari body must be the full skinned vessel"
    );
    println!(
        "[v0-render] nari body composed: {} tris",
        nari.world_tris.len()
    );
}

/// The body's triangles ENTER THE DYNAMIC PARTITION — the exact soup the traced
/// BVH splices each tick, on top of the living layer — and they PERSIST across a
/// world tick (the per-tick splice cadence, kami's precedent). She also stands
/// on the seawall (feet at the seawall top y = 1.4, derived from her placement).
#[test]
fn v0_body_enters_dynamic_partition_each_tick() {
    let mut scene = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");
    // RITE V·V2: BOTH embodied bodies (nari + the cat) enter the partition.
    let body_tris: usize = scene.bodies.iter().map(|b| b.world_tris.len()).sum();
    let living = scene.dynamics.leaf_triangles().len();

    let dynamic = scene.dynamic_leaf_triangles();
    assert_eq!(
        dynamic.len(),
        living + body_tris,
        "dynamic partition = living layer + every embodied body"
    );

    // Feet on the seawall: the lowest body vertex is the seawall top (y=1.4).
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for t in &scene.bodies[0].world_tris {
        for p in &t.positions {
            y_min = y_min.min(p[1]);
            y_max = y_max.max(p[1]);
            // She stands within the seawall footprint in x/z.
            assert!(
                p[0].abs() < 1.0 && (p[2] - 18.0).abs() < 1.0,
                "on the seawall"
            );
        }
    }
    // Derived: feet = 2.505 + local_min_y(-1.104977) = 1.400023 (seawall top).
    assert!(
        (y_min - 1.4).abs() < 1e-2,
        "nari feet on the seawall top (y=1.4), got {y_min}"
    );
    assert!(
        y_max > 3.0,
        "nari stands ~2.1 m tall, head near y=3.5, got {y_max}"
    );

    // The splice is per-tick: after advancing the world clock the body is STILL
    // in the dynamic partition (V0 idle ⇒ same triangles; V1 drives the pose).
    scene.tick();
    let after = scene.dynamic_leaf_triangles();
    let body_tris_after: usize = scene.bodies.iter().map(|b| b.world_tris.len()).sum();
    assert_eq!(
        after.len(),
        scene.dynamics.leaf_triangles().len() + body_tris_after,
        "the bodies re-enter the dynamic partition every tick"
    );
    println!(
        "[v0-render] dynamic partition = {} living + {} body = {}; persists across tick",
        living,
        body_tris,
        dynamic.len()
    );
}

/// Determinism: two independent loads compose the SAME world-space body — the
/// skinned triangle positions are byte-identical (the ENTROPY law through the
/// whole render-side weld).
#[test]
fn v0_body_render_is_deterministic() {
    let a = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("a");
    let b = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("b");
    let bytes = |scene: &RenderScene| -> Vec<u8> {
        let mut out = Vec::new();
        for t in &scene.bodies[0].world_tris {
            for p in &t.positions {
                for c in p {
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
            for c in &t.albedo {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out
    };
    assert_eq!(
        bytes(&a),
        bytes(&b),
        "the embodied body must compose byte-identically"
    );
    println!("[v0-render] body render bytes identical across loads");
}

// ─────────────────────────────────────────────────────────────────────────
// RITE V · V1 — SHE WALKS. The walker's velocity drives sama; sama's pose
// drives the skin per tick. These CPU ordeals prove the seam (pose == skin
// input, determinism), the DERIVED foot-ground contact (Guardian finding 1 —
// no hover), and that two cycle ticks read as visibly distinct limb configs.
// ─────────────────────────────────────────────────────────────────────────

/// Canonical byte image of a pose — the local rotations sama emits.
fn pose_bytes(pose: &Pose) -> Vec<u8> {
    let mut out = Vec::new();
    for q in &pose.local_rotations {
        for c in q.to_array() {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    out
}

/// Canonical byte image of a skinned soup — positions + albedo.
fn tris_bytes(tris: &[LeafTriangle]) -> Vec<u8> {
    let mut out = Vec::new();
    for t in tris {
        for p in &t.positions {
            for c in p {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        for c in &t.albedo {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    out
}

/// A walk-then-idle command stream (walker velocity magnitudes, m·s⁻¹): idle,
/// accelerate into a walk, hold, then stop. Long enough to cross the state
/// machine's thresholds and a full blend.
fn command_stream() -> Vec<f32> {
    let walk = walk_speed();
    let mut s = vec![0.0; 6];
    s.extend(std::iter::repeat_n(walk, 60)); // walk speed (GAIA_PLAYER_WALK)
    s.extend(std::iter::repeat_n(0.0, 20)); // stop → blend back to idle
    s
}

/// The walker's walk speed (m·s⁻¹) — the SAME env-driven parameter the
/// embodiment uses (`GAIA_PLAYER_WALK`, default 6), never a bare literal (F10).
fn walk_speed() -> f32 {
    scrying_glass::player::PlayerParams::from_env()
        .expect("player params")
        .walk_speed
}

/// ORDEAL — sama's pose IS the skinning input, every tick (0e0 exact). An
/// INDEPENDENT sama state machine, fed the same command stream, reproduces the
/// pose the body skinned; and the body's `world_tris` are byte-identical to
/// that pose skinned through the vessel — the pose is never re-derived or
/// nudged between sama and the skin.
#[test]
fn v1_sama_pose_is_the_skinning_input_each_tick() {
    let mut scene = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");
    // The independent oracle: nari's skeleton + a fresh identical machine.
    let skeleton = Skeleton::humanoid();
    let mut oracle = Locomotion::new(LocomotionParams::default());

    for (i, &speed) in command_stream().iter().enumerate() {
        scene.command_bodies(speed);
        let body: &BodyInstance = &scene.bodies[0];
        let oracle_pose = oracle.step(&skeleton, speed);

        // (1) The pose the body used == sama's pose, exactly.
        assert_eq!(
            pose_bytes(body.pose()),
            pose_bytes(&oracle_pose),
            "tick {i}: body pose must be sama's pose (0e0)"
        );
        // (2) world_tris ARE that pose skinned — no divergence in the skin step.
        assert_eq!(
            tris_bytes(&body.world_tris),
            tris_bytes(&body.skin_current()),
            "tick {i}: world_tris must be the current pose skinned (0e0)"
        );
    }
    println!(
        "[v1] sama pose == skinning input for all {} ticks (0e0)",
        command_stream().len()
    );
}

/// ORDEAL — the gait is byte-identical across two independent runs (ENTROPY).
/// Two fresh scenes, driven by the same walker-velocity stream, skin the same
/// world-space triangles every tick.
#[test]
fn v1_gait_is_deterministic_byte_identical() {
    let mut a = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("a");
    let mut b = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("b");
    for &speed in &command_stream() {
        a.command_bodies(speed);
        b.command_bodies(speed);
        assert_eq!(
            tris_bytes(&a.bodies[0].world_tris),
            tris_bytes(&b.bodies[0].world_tris),
            "the walked body must be byte-identical across runs"
        );
    }
    println!("[v1] gait byte-identical across two runs (all ticks)");
}

/// ORDEAL — DERIVED foot-ground contact (Guardian finding 1: she must NOT
/// hover). The body's lowest world-space vertex at idle rests on the realm
/// floor under her feet (the seawall top, read straight from the realm as the
/// grounding source), within a tolerance DERIVED from the marching-cubes cell
/// size — never nudged by eye. The residual is orders of magnitude under the
/// cell (the placement is exact to float; the cell only bounds how well a
/// discretised sole approximates the true skin).
#[test]
fn v1_derived_foot_ground_contact_no_hover() {
    let scene = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");

    // Ground truth from the realm: the seawall top face, derived from geometry.
    let ground_y = scrying_glass::scene::top_flat_surface_y(&load_naruko().world, "naruko_seawall")
        .expect("seawall query")
        .expect("seawall is a flat slab");

    // The body's lowest world vertex (idle) — the contact point.
    let mut world_min = f32::INFINITY;
    for t in &scene.bodies[0].world_tris {
        for p in &t.positions {
            world_min = world_min.min(p[1]);
        }
    }

    // Tolerance DERIVED from mesh resolution: one marching-cubes cell. The cell
    // is the idle mesh's largest bounding extent divided by the meshing
    // resolution — the finest the discretised sole can resolve.
    let preset = Preset::nari();
    let body = Body::from_preset(&preset);
    let (lo, hi) = body.idle_local_bounds().expect("idle bounds");
    let extent = hi - lo;
    let cell = extent.max_element() / preset.vessel.resolution as f32;

    let residual = (world_min - ground_y).abs();
    println!(
        "[v1] contact: ground_y(seawall)={ground_y:.6} lowest_vertex_y={world_min:.6} \
         residual={residual:.2e} tol(cell={cell:.4})",
    );
    assert!(
        residual <= cell,
        "the body must stand ON the floor (no hover): residual {residual:.2e} > cell {cell:.4}",
    );
}

/// ORDEAL (F1 — ONE FLOOR) — the bodies ground on the SAME post-transmute leaf
/// floor the WALKER walks on. `RenderScene::leaf_positions()` is the exact `Vec`
/// `main.rs` feeds `Ground::from_positions` for the walker; a floor built from
/// it, queried under nari, holds her feet to the SAME 1.19e-7-class contact the
/// weld derived — NOT a second, forkable floor. If a future weld tolerance ever
/// forks the two floor sources, `walker_y` drifts from her feet past the cell
/// and this ordeal goes RED (the silent-fork the adversary named is now loud).
#[test]
fn v1_body_grounds_on_the_walker_floor_f1() {
    use scrying_glass::player::Ground;

    let scene = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");
    // The WALKER's floor: EXACTLY the positions `main.rs` grounds the walker on.
    let walker_floor = Ground::from_positions(&scene.leaf_positions());

    // nari's feet (lowest world vertex) and her body column (centroid x/z).
    let mut world_min = f32::INFINITY;
    let (mut cx, mut cz, mut n) = (0.0f64, 0.0f64, 0.0f64);
    for t in &scene.bodies[0].world_tris {
        for p in &t.positions {
            world_min = world_min.min(p[1]);
            cx += p[0] as f64;
            cz += p[2] as f64;
            n += 1.0;
        }
    }
    let (cx, cz) = ((cx / n) as f32, (cz / n) as f32);
    let walker_y = walker_floor
        .height_at(cx, cz, f32::INFINITY)
        .expect("the walker floor exists under nari");

    // Same discretization tolerance as the sibling contact ordeal (one cell).
    let preset = Preset::nari();
    let body = Body::from_preset(&preset);
    let (lo, hi) = body.idle_local_bounds().expect("idle bounds");
    let cell = (hi - lo).max_element() / preset.vessel.resolution as f32;

    let residual = (world_min - walker_y).abs();
    println!(
        "[v1-f1] walker_floor_y={walker_y:.6} nari_feet={world_min:.6} \
         residual={residual:.2e} tol(cell={cell:.4}) — ONE floor source",
    );
    assert!(
        residual <= cell,
        "nari must stand on the WALKER's floor (one source): residual {residual:.2e} > cell {cell:.4}",
    );
}

/// ORDEAL — two fixed cycle ticks read as CONTACT vs PASSING (visibly distinct
/// limb configuration). The two ticks are DERIVED from the walk gait (swing
/// foot lowest vs highest), and the leg pose between them differs beyond a
/// derived floor: the summed absolute thigh-angle change across both legs is a
/// large fraction of the stride swing, not noise.
#[test]
fn v1_contact_and_passing_are_distinct_poses() {
    let preset = Preset::nari();
    let body = Body::from_preset(&preset);
    let params = GaitParams::walk();
    let (contact, passing) = contact_passing_ticks(&body, &params);
    assert_ne!(
        contact, passing,
        "contact and passing must be different ticks"
    );

    // The leg configuration difference, measured on the thigh bones (the primary
    // stride signal). Sum the absolute angle change of every `.thigh` bone.
    let skeleton = &preset.skeleton;
    let pose_c = sama::gait_pose(skeleton, &params, contact);
    let pose_p = sama::gait_pose(skeleton, &params, passing);
    let mut leg_delta = 0.0f32;
    for (i, bone) in skeleton.bones.iter().enumerate() {
        if bone.name.ends_with(".thigh") {
            let a = pose_c.local_rotations[i];
            let b = pose_p.local_rotations[i];
            leg_delta += a.angle_between(b);
        }
    }
    // Derived floor: half the stride amplitude (radians) — a real stance change,
    // not float noise. `stride` is the gait's leg swing amplitude.
    let floor = 0.5 * params.stride;
    println!(
        "[v1] contact tick={contact} passing tick={passing} leg_delta={leg_delta:.4} rad \
         (floor {floor:.4})",
    );
    assert!(
        leg_delta > floor,
        "contact and passing must be visibly distinct: leg_delta {leg_delta:.4} <= {floor:.4}",
    );
}

/// ORDEAL — her traced occlusion REPAINTS the ground (Guardian finding 2). The
/// pleroma already traces occlusion; this proves it with a traced probe in the
/// lantern precedent's style: the direct-sun radiance on the seawall directly
/// UNDER her body is darker than a spot BESIDE her by more than a derived floor
/// (she blocks the sun there), while the SAME under-vs-beside probe taken FAR
/// from her shows ~0 difference (the null — the darkening is HER shadow, not a
/// global dip). No shadow code is written: the body is simply real to the light.
#[test]
fn v1_body_casts_a_traced_shadow_on_the_seawall() {
    use scrying_glass::bvh::{Bvh, BvhParams};

    let mut scene = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");
    // She is walking (past the blend) when the light traces her.
    for _ in 0..30 {
        scene.command_bodies(walk_speed());
    }
    let sun = scene.sun;

    // Every surface the light can hit — the realm plus her walking body.
    let mut tris = scene.leaf_triangles();
    tris.extend(scene.dynamic_leaf_triangles());
    let bvh = Bvh::build(&tris, &BvhParams::default());

    // Seawall top DERIVED from the realm geometry — the SAME source the sibling
    // contact ordeal grounds on (`top_flat_surface_y`), never a bare literal
    // (F10). The shadow probe and the drop projection both read this.
    let seawall_top =
        scrying_glass::scene::top_flat_surface_y(&load_naruko().world, "naruko_seawall")
            .expect("seawall query")
            .expect("seawall is a flat slab");
    let eps = 5e-3_f32;
    // Direct-sun luminance on a flat (normal +Y) ground point: sun colour ×
    // intensity × max(0, N·L) × visibility toward the sun (the traced shadow).
    let luminance = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let n_dot_l = sun.direction[1].max(0.0);
    let sun_lum = luminance(sun.color) * sun.intensity * n_dot_l;
    let radiance = |x: f32, z: f32| -> f32 {
        let origin = [x, seawall_top + eps, z];
        let occluded = bvh.occluded(origin, sun.direction, eps, 500.0);
        if occluded { 0.0 } else { sun_lum }
    };

    // Where her shadow FALLS is derived, not guessed: her body centroid,
    // projected onto the seawall top ALONG the sun direction (the sun sits up
    // and to +x/+z, so the shadow lands to -x/-z of her feet).
    let mut centroid = [0.0f64; 3];
    let mut n = 0.0f64;
    for t in &scene.bodies[0].world_tris {
        for p in &t.positions {
            centroid[0] += p[0] as f64;
            centroid[1] += p[1] as f64;
            centroid[2] += p[2] as f64;
            n += 1.0;
        }
    }
    let centroid = [
        (centroid[0] / n) as f32,
        (centroid[1] / n) as f32,
        (centroid[2] / n) as f32,
    ];
    let drop = (centroid[1] - seawall_top) / sun.direction[1];
    let shadow_x = centroid[0] - drop * sun.direction[0];
    let shadow_z = centroid[2] - drop * sun.direction[2];

    // In her shadow vs beside her (7 m along +x, clear of her body on the same
    // seawall band). THE MIRROR PROOF re-derived this offset: the panel
    // (x[2.96,3.04], y≤ 4.4) casts a sun shadow onto the wall top over
    // x∈[0.96,3.04] — the old +3 probe (x≈2.25) sat inside the GLASS's shadow;
    // +6 (x≈5.25) lands in the x=6 chain post's 0.28 m strip (x 5.30±0.14 at
    // this z). +7 (x≈6.25) is derived clear of panel, posts (next strip at
    // x≥13.3) and the stall's shadow (z≥22.5), still on the wall top x≤ 60.
    let under = radiance(shadow_x, shadow_z);
    let beside = radiance(shadow_x + 7.0, shadow_z);
    let shadow_diff = beside - under;

    // The null: the identical probe pair 40 m away, where she casts nothing.
    let far_under = radiance(shadow_x + 40.0, shadow_z);
    let far_beside = radiance(shadow_x + 47.0, shadow_z);
    let null_diff = (far_beside - far_under).abs();

    // Derived floor: half the full direct-sun luminance — a real occlusion, not
    // noise. (A full block gives the whole sun term.)
    let floor = 0.5 * sun_lum;
    println!(
        "[v1-shadow] shadow=({shadow_x:.3},{shadow_z:.3}) centroid=({:.3},{:.3},{:.3}) \
         under={under:.4} beside={beside:.4} shadow_diff={shadow_diff:.4} \
         null_diff={null_diff:.2e} floor={floor:.4} (sun_lum={sun_lum:.4})",
        centroid[0], centroid[1], centroid[2],
    );
    assert!(
        shadow_diff > floor,
        "her body must darken the ground under her: diff {shadow_diff:.4} <= floor {floor:.4}",
    );
    assert!(
        null_diff < floor,
        "far from her the ground is unshadowed (null): null_diff {null_diff:.2e} >= floor {floor:.4}",
    );
}

// ---------------------------------------------------------------------------
// RITE V · V2 — THE PINK CAT ANIMATES. The cat carries a `behavior`
// `{kind:"cat"}`, so its body is MINDED: it drives its own idle loop from the
// world clock (Sit → TailFlick → Walk circuit → Sit), independent of the
// walker. nari (no behavior) stays walker-driven. These ordeals prove the
// wiring: the minded cat moves on its circuit and returns home, nari does not
// move when the walker is still, and the whole animated stream is deterministic.
// ---------------------------------------------------------------------------

/// The cat body picks up its mind and nari stays mindless.
#[test]
fn v2_cat_body_is_minded_nari_is_not() {
    let scene = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");
    let cat = scene
        .bodies
        .iter()
        .find(|b| b.gaia_id == "naruko_cat")
        .expect("cat body");
    let nari = scene
        .bodies
        .iter()
        .find(|b| b.gaia_id == "nari")
        .expect("nari body");
    assert_eq!(cat.preset, "pink_cat");
    assert!(cat.is_minded(), "the cat carries a behavior spirit");
    assert!(!nari.is_minded(), "nari is walker-driven, not minded");
    println!(
        "[v2-wire] cat is minded ({} tris), nari is not",
        cat.world_tris.len()
    );
}

/// Ticking the world with the walker STILL (speed 0): the minded cat still
/// walks its circuit (position leaves home by ~radius, then returns), while
/// nari — walker-driven at speed 0 — never moves. The clock is what animates
/// the cat, not the walker.
#[test]
fn v2_cat_walks_its_circuit_while_nari_holds() {
    let mut scene = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");
    let home = [-5.0_f32, 23.0_f32]; // authored cat home xz
    let cat_idx = scene
        .bodies
        .iter()
        .position(|b| b.gaia_id == "naruko_cat")
        .unwrap();
    let nari_idx = scene
        .bodies
        .iter()
        .position(|b| b.gaia_id == "nari")
        .unwrap();
    let nari_start = scene.bodies[nari_idx].world_origin();

    // One full loop is ~17 s at 1/60 dt ≈ 1030 ticks; sweep 1100 to cover it.
    let mut max_from_home = 0.0_f32;
    let mut min_walk_speed_seen = f32::INFINITY;
    let mut walked = false;
    for _ in 0..1100 {
        scene.command_bodies(0.0); // walker STILL
        let o = scene.bodies[cat_idx].world_origin();
        let d = ((o[0] - home[0]).powi(2) + (o[2] - home[1]).powi(2)).sqrt();
        max_from_home = max_from_home.max(d);
        let sp = scene.bodies[cat_idx].commanded_speed();
        if sp > 0.0 {
            walked = true;
            min_walk_speed_seen = min_walk_speed_seen.min(sp);
        }
        scene.tick();
    }

    // The cat reached out to ~its 1.4 m circuit radius at some tick.
    assert!(
        max_from_home > 1.0,
        "cat must walk out on its circuit, max_from_home={max_from_home}"
    );
    assert!(
        walked,
        "cat must command a positive speed during the Walk phase"
    );
    assert!(
        min_walk_speed_seen > 0.0 && min_walk_speed_seen.is_finite(),
        "walk speed positive"
    );

    // nari never moved (walker still, and she is mindless).
    let nari_end = scene.bodies[nari_idx].world_origin();
    let nari_moved = ((nari_end[0] - nari_start[0]).powi(2)
        + (nari_end[1] - nari_start[1]).powi(2)
        + (nari_end[2] - nari_start[2]).powi(2))
    .sqrt();
    assert!(
        nari_moved < 1e-6,
        "mindless nari must not move when the walker is still, moved {nari_moved}"
    );
    println!(
        "[v2-wire] cat reached {max_from_home:.3} m from home on its circuit; nari held (moved {nari_moved:.2e})"
    );
}

/// DETERMINISM of the animated body: two scenes ticked the SAME number of times
/// with the walker still produce BYTE-IDENTICAL cat triangles (the clock-driven
/// idle loop is pure).
#[test]
fn v2_cat_animation_is_byte_identical() {
    let run = || -> Vec<u8> {
        let mut scene =
            RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");
        let idx = scene
            .bodies
            .iter()
            .position(|b| b.gaia_id == "naruko_cat")
            .unwrap();
        for _ in 0..400 {
            scene.command_bodies(0.0);
            scene.tick();
        }
        // sample deep in the Walk phase (tick 400 ≈ 6.7 s > sit+flick)
        tris_bytes(&scene.bodies[idx].world_tris)
    };
    assert_eq!(
        run(),
        run(),
        "animated cat body must be byte-identical across runs"
    );
    println!("[v2-wire] animated cat body byte-identical across two runs");
}

// ---------------------------------------------------------------------------
// RITE V FINAL WELD — THE BODY JOINS THE WALKER. nari's body declares
// `follows: "walker"` in worlds/naruko, so she is ATTACHED: her position/yaw
// track the walker each tick and her gait speed is DERIVED from the per-tick
// displacement (not the global broadcast). These ordeals prove the attachment
// (position == walker, 0e0 in xz), the derived velocity source (== displacement,
// != broadcast), the sigil wiring (nari attached, cat/mindless not), and the
// shadow following the walked-to body. The minded cat + presence paths are
// UNAFFECTED (their ordeals above stay green; nari falls back to the broadcast
// when no walker pose is given, so the V1/V2 broadcast ordeals are untouched).
// ---------------------------------------------------------------------------

/// The FINAL-WELD sigil is READ: nari declares `follows: "walker"` (attached),
/// the cat is minded (never attached), and the crate is a rigid physics body
/// (not a skinned vessel at all — absent from `bodies`).
#[test]
fn weld_attachment_sigil_is_read() {
    let scene = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");
    let nari = scene
        .bodies
        .iter()
        .find(|b| b.gaia_id == "nari")
        .expect("nari body");
    let cat = scene
        .bodies
        .iter()
        .find(|b| b.gaia_id == "naruko_cat")
        .expect("cat body");
    assert!(
        nari.follows_walker(),
        "nari's body declares follows: walker"
    );
    assert!(
        !cat.follows_walker(),
        "the minded cat is not walker-attached"
    );
    assert!(!nari.is_minded(), "an attached body is not minded");
    println!("[weld] sigil read: nari attached, cat minded (not attached)");
}

/// ORDEAL — an ATTACHED body's position TRACKS the walker every tick (0e0 in
/// xz). Over a scripted straight walk (broadcast fed a bogus 99 m·s⁻¹ to prove
/// it is IGNORED), nari's world xz equals the walker's xz to the bit; her y is
/// the DERIVED grounded height (feet on the seawall, not the walker's eye y).
#[test]
fn weld_attached_body_position_equals_walker_0e0() {
    let mut scene = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");
    let nari_idx = scene
        .bodies
        .iter()
        .position(|b| b.gaia_id == "nari")
        .unwrap();
    // 25 ticks → x ∈ [0, 2.4], the FLAT seawall band. The 21-vessel realm added
    // the mirror panel at [3, 2.9, 18] (a thin 0.08 m slab whose horizontal top
    // at y=4.4 the F1 floor treats as walkable ground, x-footprint [2.96, 3.04]),
    // and the chain-post at x=6 (footprint [5.86, 6.14]) still sits beyond it. So
    // the flat band clear of BOTH is x < 2.96; walking to x=2.4 keeps a 0.56 m
    // margin to the panel — the tracking claim stays proven on ground where the
    // grounded y provably holds constant (no panel/post top under the column).
    let walk = walker_walk(0.1, 25);
    // The grounded model-origin y at tick 0 (seawall is flat — it must not move).
    let grounded_y = {
        scene.command_bodies_walked(99.0, Some(walk[0]));
        scene.bodies[nari_idx].world_origin()[1]
    };
    for (k, pose) in walk.iter().enumerate() {
        // Broadcast a bogus speed: an attached body must NOT use it.
        scene.command_bodies_walked(99.0, Some(*pose));
        let o = scene.bodies[nari_idx].world_origin();
        assert_eq!(
            o[0], pose.position.x,
            "tick {k}: attached body x must equal walker x (0e0)"
        );
        assert_eq!(
            o[2], pose.position.z,
            "tick {k}: attached body z must equal walker z (0e0)"
        );
        // y is the DERIVED grounded height (feet on the flat seawall) — it holds
        // constant even as the walker's EYE y climbs far above it (decoupled).
        assert!(
            (o[1] - grounded_y).abs() < 1e-4,
            "tick {k}: attached body y stays grounded ({o1}), not the climbing eye {eye}",
            o1 = o[1],
            eye = pose.position.y,
        );
        assert!(
            pose.position.y - o[1] > 0.4 * k as f32 - 0.1,
            "tick {k}: the eye ({eye}) rises away from the grounded body ({o1})",
            eye = pose.position.y,
            o1 = o[1],
        );
    }
    let end = scene.bodies[nari_idx].world_origin();
    println!(
        "[weld] attached body tracked the walker to ({:.3},{:.3},{:.3}) — xz == walker (0e0)",
        end[0], end[1], end[2]
    );
}

/// ORDEAL — the attached body's gait speed is DERIVED from the walker's actual
/// horizontal displacement per tick, NOT the broadcast. Fed a bogus broadcast
/// (99), nari's commanded speed equals the exact displacement/dt (the same
/// value re-derived independently from the scripted poses), and is finite and
/// far from 99 while she walks.
#[test]
fn weld_gait_speed_is_derived_not_broadcast() {
    let mut scene = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");
    let nari_idx = scene
        .bodies
        .iter()
        .position(|b| b.gaia_id == "nari")
        .unwrap();
    let dt = naruko_params().tick_dt as f32;
    let walk = walker_walk(0.1, 40);
    let mut prev = {
        let o = scene.bodies[nari_idx].world_origin();
        [o[0], o[2]]
    };
    let mut walked = false;
    for (k, pose) in walk.iter().enumerate() {
        scene.command_bodies_walked(99.0, Some(*pose));
        let dx = pose.position.x - prev[0];
        let dz = pose.position.z - prev[1];
        let expected = (dx * dx + dz * dz).sqrt() / dt;
        let got = scene.bodies[nari_idx].commanded_speed();
        assert_eq!(
            got, expected,
            "tick {k}: gait speed must be the derived walker displacement, not the broadcast"
        );
        assert!(
            (got - 99.0).abs() > 1.0,
            "tick {k}: the broadcast (99) must be ignored, got {got}"
        );
        if got > 0.0 {
            walked = true;
        }
        let o = scene.bodies[nari_idx].world_origin();
        prev = [o[0], o[2]];
    }
    assert!(
        walked,
        "the attached body must derive a positive gait speed"
    );
    // At step 0.1 m/tick over dt=1/60, the derived speed is 6 m/s (walk).
    println!(
        "[weld] gait derived from walker displacement (== 0.1/dt = {:.3} m/s), broadcast 99 ignored",
        0.1 / dt
    );
}

/// ORDEAL — the traced SHADOW follows the WALKED-TO body (she casts where she
/// STANDS after the walk, not where she was authored). After a scripted walk
/// down the seawall the body's projected shadow darkens the ground beneath its
/// NEW position by more than a derived floor, while the SAME probe at her
/// AUTHORED spot (now vacated) shows ~no darkening (the null — she left it).
#[test]
fn weld_shadow_follows_the_walked_body() {
    use scrying_glass::bvh::{Bvh, BvhParams};

    let mut scene = RenderScene::from_ecs(load_naruko().world, &naruko_params()).expect("scene");
    let nari_idx = scene
        .bodies
        .iter()
        .position(|b| b.gaia_id == "nari")
        .unwrap();
    // Walk her +x along the seawall to a visibly different spot (x≈20), mid-stride.
    let walk = walker_walk(0.1, 200);
    for pose in &walk {
        scene.command_bodies_walked(0.0, Some(*pose));
        scene.tick();
    }
    let sun = scene.sun;

    let mut tris = scene.leaf_triangles();
    tris.extend(scene.dynamic_leaf_triangles());
    let bvh = Bvh::build(&tris, &BvhParams::default());

    let seawall_top =
        scrying_glass::scene::top_flat_surface_y(&load_naruko().world, "naruko_seawall")
            .expect("seawall query")
            .expect("seawall is a flat slab");
    let eps = 5e-3_f32;
    let luminance = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let n_dot_l = sun.direction[1].max(0.0);
    let sun_lum = luminance(sun.color) * sun.intensity * n_dot_l;
    let radiance = |x: f32, z: f32| -> f32 {
        let origin = [x, seawall_top + eps, z];
        if bvh.occluded(origin, sun.direction, eps, 500.0) {
            0.0
        } else {
            sun_lum
        }
    };

    // Where her shadow FALLS at her WALKED-TO position: her body centroid,
    // projected onto the seawall top along the sun direction.
    let mut centroid = [0.0f64; 3];
    let mut n = 0.0f64;
    for t in &scene.bodies[nari_idx].world_tris {
        for p in &t.positions {
            centroid[0] += p[0] as f64;
            centroid[1] += p[1] as f64;
            centroid[2] += p[2] as f64;
            n += 1.0;
        }
    }
    let centroid = [
        (centroid[0] / n) as f32,
        (centroid[1] / n) as f32,
        (centroid[2] / n) as f32,
    ];
    // Sanity: she actually walked away from x=0 (her authored spot).
    assert!(
        centroid[0] > 10.0,
        "the walked-to body must be far from its authored x=0, got x={}",
        centroid[0]
    );
    let drop = (centroid[1] - seawall_top) / sun.direction[1];
    let shadow_x = centroid[0] - drop * sun.direction[0];
    let shadow_z = centroid[2] - drop * sun.direction[2];

    let under = radiance(shadow_x, shadow_z);
    let beside = radiance(shadow_x + 3.0, shadow_z);
    let shadow_diff = beside - under;

    // The null: the probe pair back at her AUTHORED spot (x≈0), which she has
    // VACATED — no body there now, so no darkening. The 21-vessel realm added
    // the mirror panel at [3, 2.9, 18]; at this null z (shadow_z≈17.64) the
    // panel's OWN cast shadow darkens the seawall across x∈[1.0, 3.0], so the
    // old +3.0 beside-probe landed IN the panel shadow (a false darkening, not
    // the body's). Re-derive the beside offset to +7.0: at x=7.0 the ground is
    // lit (nearest shadow is the x=6 chain-post's at x≈5.5 — a 1.5 m margin,
    // and the panel shadow ends at x=3.0, a 4 m margin), while null_under at
    // x=0 is lit 1.0 m clear of the panel shadow's x=1.0 edge. Both probes now
    // read the same lit ground → null_diff≈0 (the true vacated null).
    const NULL_BESIDE_DX: f32 = 7.0;
    let null_under = radiance(0.0, shadow_z);
    let null_beside = radiance(NULL_BESIDE_DX, shadow_z);
    let null_diff = (null_beside - null_under).abs();

    let floor = 0.5 * sun_lum;
    println!(
        "[weld-shadow] walked-to x={:.2} under={under:.4} beside={beside:.4} \
         shadow_diff={shadow_diff:.4} vacated_null={null_diff:.2e} floor={floor:.4}",
        centroid[0]
    );
    assert!(
        shadow_diff > floor,
        "the walked-to body must darken the ground under its NEW position: diff {shadow_diff:.4} <= floor {floor:.4}"
    );
    assert!(
        null_diff < floor,
        "her authored spot is now vacated (null): null_diff {null_diff:.2e} >= floor {floor:.4}"
    );
}
