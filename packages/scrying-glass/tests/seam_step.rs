//! RITE VII · VII-1 — THE WALKER CROSSES ONTO GENERATED GROUND.
//!
//! VII-0b already fed a generated terrain patch through the ONE seal path
//! (`scene::append_terrain`) into BOTH collision worlds — the walker floor
//! (`Ground::from_positions(leaf_positions_of(&chains))`, RULING-6 gated) and
//! the solver collision world (`collider_triangles` -> `Solver.collider`).
//! These ordeals prove the SEAM is real, through the REAL weld and the REAL
//! player/solver code paths:
//!
//!  1. a walker crosses authored->generated AND generated->authored ground
//!     with no pose discontinuity (per-tick gated-floor delta within a DERIVED
//!     bound, grounded every crossing tick) — both directions, one guard;
//!  2. the guard is DISCRIMINATING — mis-author the seam 1 m off the field and
//!     the same guard fires (not a vacuous tail);
//!  3. a rigid `body` rests ON generated triangles in the SOLVER collision
//!     world at the DERIVED analytic height (not just the render floor);
//!  4. the same rest holds at PLANETARY tile magnitude with `render_origin`
//!     co-located — the collider triangles are placed exact (ruling 4), same
//!     as the render triangles;
//!  5. GUARDIAN RULING 6's contact-patch gate bites GENERATED geometry
//!     identically to authored (centre admitted, past-edge rejected).
//!
//! The per-tick bound derivation lives in
//! `docs/perf/2026-07-17-vii1-seam-step-epsilon-derivation.md`. Nothing here
//! is a plucked literal.

use crystal::{EcsWorld, load_world_dir};
use scrying_glass::player::{Ground, Key, Player, PlayerParams, Pose, contact_tolerance};
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults};
use seed::terrain::height_at_grid_index;
use seed::{Seed, TerrainSigil};

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

const TICK_DT: f32 = 1.0 / 60.0;

/// The DERIVED per-tick gated-floor continuity bound (see the derivation doc).
/// `Δfloor_max = v_walk·dt·g_max + slack`, every term a live constant — never
/// a frozen number. `g_max = tan(acos(WALL_NORMAL_Y_COS_CUTOFF))` is exactly
/// `contact_tolerance(1.0)` (the public function is `radius * that tangent`),
/// so this ties to the SAME cutoff `Ground::from_positions` drops walls at.
fn floor_delta_bound(walk_speed: f32, coord_magnitude: f32) -> f32 {
    let g_max = contact_tolerance(1.0); // = tan(acos(WALL_NORMAL_Y_COS_CUTOFF))
    let horizontal_step = walk_speed * TICK_DT;
    let slack = coord_magnitude.abs().max(1.0) * f32::EPSILON * 16.0;
    horizontal_step * g_max + slack
}

/// Build a scratch seam realm on disk: a flat authored plane (top at
/// `plane_top_y`) abutting the `z = tile_origin_z` edge of a generated tile,
/// spanning a narrow x band around the crossing column so the walker crosses
/// on the C0-matched column and nowhere else. Loaded fresh and welded through
/// the REAL `from_ecs`. Returns the built scene.
fn seam_realm(
    tag: &str,
    terrain_json: &str,
    plane_top_y: f32,
) -> (std::path::PathBuf, RenderScene) {
    let dir = std::env::temp_dir().join(format!("gaia_vii1_{tag}_{}", std::process::id()));
    let scenes = dir.join("scenes");
    std::fs::create_dir_all(&scenes).expect("create temp scenes dir");
    // The authored plane: 0.5 m thick, top at plane_top_y, x∈[30,34],
    // z∈[-30,0] (abutting the tile's z=0 edge). Center y = plane_top_y - 0.25.
    let plane_center_y = plane_top_y - 0.25;
    let scene = format!(
        r##"{{
          "authored_shore": {{
            "transform": {{ "position": [0, 0, 0] }},
            "mesh": {{ "parts": [
              {{ "shape": "box", "size": [4, 0.5, 30], "position": [32, {plane_center_y}, -15], "color": "#5a4a6c" }}
            ] }}
          }},
          "naruko_first_ground": {{ "terrain": {terrain_json} }}
        }}"##
    );
    std::fs::write(scenes.join("main.json"), scene).expect("write temp scene");
    let mut world = EcsWorld::default();
    load_world_dir(&dir, &mut world).expect("load the seam realm");
    let render = RenderScene::from_ecs(world, &params()).expect("weld the seam realm");
    (dir, render)
}

/// One tick's witness on the crossing: pose + the gated floor height under the
/// walker's CURRENT column (the same query the controller stands on).
struct Step {
    pose: Pose,
    floor: Option<f32>,
}

/// Settle then walk the crossing, sampling the gated floor under the walker
/// each tick. `keys` is held for the walk phase (settle phase holds nothing).
fn walk(
    ground: &Ground,
    params: PlayerParams,
    spawn_eye: glam::Vec3,
    yaw: f32,
    settle_ticks: u32,
    walk_ticks: u32,
    keys: &[Key],
) -> Vec<Step> {
    let mut player = Player::new(params, spawn_eye, yaw);
    for _ in 0..settle_ticks {
        player.step(TICK_DT, ground);
    }
    let mut trace = Vec::with_capacity(walk_ticks as usize);
    for k in keys {
        player.keys.insert(*k);
    }
    for _ in 0..walk_ticks {
        player.step(TICK_DT, ground);
        let p = player.pose();
        let floor = ground.height_at(p.position.x, p.position.z, f32::INFINITY);
        trace.push(Step { pose: p, floor });
    }
    trace
}

/// The field height at the crossing column's SEAM vertex — the value the
/// authored plane's top is set to, making the seam C0 along the walker's path.
/// The crossing column is the tile's x-centre grid vertex; the seam edge is
/// the tile's `local_j = 0` (`z = tile_origin_z`) row.
fn seam_setup(terrain_json: &str) -> (f32, f32, f32, f32) {
    let sigil: TerrainSigil = serde_json::from_str(terrain_json).expect("parse terrain sigil");
    let tile = sigil.tile();
    let tparams = sigil.params();
    let world_seed: Seed = sigil.world_seed();
    let (origin_x, origin_z) = seed::tile_origin_m(tile, &tparams);
    let n = tparams.grid_resolution as i64;
    let cell = tparams.cell_size_m();
    let local_i = n / 2; // x-centre grid vertex
    let world_x0 = origin_x as f32 + local_i as f32 * cell;
    let global_i = tile.tile_x * n + local_i;
    let global_j = tile.tile_y * n; // local_j = 0, the seam edge row
    let h_seam = height_at_grid_index(world_seed, &tparams, global_i, global_j);
    (world_x0, origin_z as f32, h_seam, cell)
}

/// ORDEAL 1 — THE SEAM STEP, both directions, pose-trace continuous. The
/// walker crosses authored->generated and generated->authored; every
/// consecutive gated-floor delta on the crossing is within the DERIVED bound,
/// and the walker stays grounded across the seam (a discontinuity would either
/// spike the delta or throw it airborne). Gentle tile so the whole path is
/// walkable; plane authored at the field's own seam value so the seam is C0.
#[test]
fn walker_crosses_seam_forward_and_back_is_pose_trace_continuous() {
    // Gentle, fully-walkable tile; a distinct diorama seed.
    let terrain_json = r##"{ "seed": 20260717, "tile_x": 0, "tile_y": 0, "height_amplitude": 1.5, "color": "#4a7c59" }"##;
    let (world_x0, origin_z, h_seam, _cell) = seam_setup(terrain_json);
    let (dir, scene) = seam_realm("cross", terrain_json, h_seam);
    let ground = Ground::from_positions(&scene.leaf_positions());
    let pparams = PlayerParams::from_env().expect("player params");
    let bound = floor_delta_bound(pparams.walk_speed, world_x0.abs().max(origin_z.abs()));

    // FORWARD: spawn on the authored plane (z<0), yaw=π (forward=+Z), walk
    // across z=0 onto the generated tile.
    let spawn_fwd = glam::Vec3::new(world_x0, h_seam + pparams.eye_stand + 0.5, origin_z - 6.0);
    let fwd = walk(
        &ground,
        pparams,
        spawn_fwd,
        std::f32::consts::PI,
        180,
        260,
        &[Key::Forward],
    );

    // REVERSE: spawn on the generated tile (z>0), yaw=0 (forward=−Z), walk
    // back across z=0 onto the authored plane.
    let spawn_rev = glam::Vec3::new(world_x0, h_seam + pparams.eye_stand + 3.0, origin_z + 8.0);
    let rev = walk(&ground, pparams, spawn_rev, 0.0, 180, 260, &[Key::Forward]);

    for (name, trace, crossed_sign) in [("forward", &fwd, 1.0f32), ("reverse", &rev, -1.0f32)] {
        // The walker actually CROSSED the seam (z spans clearly negative to
        // clearly positive relative to the seam edge, in the right direction).
        let zs: Vec<f32> = trace.iter().map(|s| s.pose.position.z - origin_z).collect();
        let min_z = zs.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_z = zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            min_z < -1.0 && max_z > 1.0,
            "{name}: walker must cross the seam edge (z-origin span [{min_z:.2},{max_z:.2}])"
        );
        assert!(
            crossed_sign * (zs.last().unwrap() - zs.first().unwrap()) > 2.0,
            "{name}: walker must travel across in the intended direction"
        );

        // Continuity: on every consecutive pair of GROUNDED ticks, the gated
        // floor under the walker changes by ≤ the derived bound, and the
        // walker never loses the floor while crossing (no NaN/None mid-cross).
        let mut max_delta = 0.0f32;
        let mut prev: Option<(f32, bool)> = None;
        let mut grounded_ticks = 0u32;
        for s in trace {
            let floor = match s.floor {
                Some(f) => f,
                None => {
                    // Losing the floor is only legitimate off the walkable
                    // region; on the crossing band it is a discontinuity.
                    let z0 = s.pose.position.z - origin_z;
                    assert!(
                        !(-1.0..=1.0).contains(&z0),
                        "{name}: floor vanished at the seam (z-origin {z0:.2})"
                    );
                    prev = None;
                    continue;
                }
            };
            if s.pose.grounded {
                grounded_ticks += 1;
            }
            if let Some((pf, pg)) = prev
                && pg
                && s.pose.grounded
            {
                let d = (floor - pf).abs();
                max_delta = max_delta.max(d);
                assert!(
                    d <= bound,
                    "{name}: gated-floor jumped {d:.5} m in one grounded tick (derived bound {bound:.5} m) — pose discontinuity at the seam"
                );
            }
            prev = Some((floor, s.pose.grounded));
        }
        assert!(
            grounded_ticks > 100,
            "{name}: the walker should be grounded for most of the crossing ({grounded_ticks} ticks)"
        );
        eprintln!(
            "[ordeal] seam {name}: max per-tick gated-floor delta {max_delta:.5} m ≤ derived bound {bound:.5} m; grounded {grounded_ticks} ticks; z-span [{min_z:.2},{max_z:.2}] about the seam edge"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// ORDEAL 2 — THE GUARD IS DISCRIMINATING (anti-vacuous). Author the plane 1 m
/// ABOVE the field's seam value: now the seam is a real 1 m step, and the same
/// forward crossing must produce a per-tick gated-floor delta that EXCEEDS the
/// derived bound (or throws the walker off the floor). Proves ordeal 1's guard
/// actually detects a discontinuity, not that it trivially passes.
#[test]
fn seam_height_mismatch_is_caught_by_the_guard() {
    let terrain_json = r#"{ "seed": 20260717, "tile_x": 0, "tile_y": 0, "height_amplitude": 1.5 }"#;
    let (world_x0, origin_z, h_seam, _cell) = seam_setup(terrain_json);
    // MIS-AUTHORED: plane 1 m too high (> the derived ~0.318 m step bound).
    let mismatch = 1.0f32;
    let (dir, scene) = seam_realm("mismatch", terrain_json, h_seam + mismatch);
    let ground = Ground::from_positions(&scene.leaf_positions());
    let pparams = PlayerParams::from_env().expect("player params");
    let bound = floor_delta_bound(pparams.walk_speed, world_x0.abs().max(origin_z.abs()));
    assert!(
        mismatch > bound,
        "the injected mismatch {mismatch} must exceed the bound {bound} to be a genuine discontinuity"
    );

    let spawn = glam::Vec3::new(
        world_x0,
        h_seam + mismatch + pparams.eye_stand + 0.5,
        origin_z - 6.0,
    );
    let trace = walk(
        &ground,
        pparams,
        spawn,
        std::f32::consts::PI,
        180,
        260,
        &[Key::Forward],
    );

    // The crossing must reveal the discontinuity: either a floor delta over the
    // bound, or the walker going airborne over the step (feet left the floor).
    let mut max_delta = 0.0f32;
    let mut lost_ground_mid_cross = false;
    let mut prev: Option<f32> = None;
    for s in &trace {
        let z0 = s.pose.position.z - origin_z;
        match s.floor {
            Some(f) => {
                if let Some(pf) = prev {
                    max_delta = max_delta.max((f - pf).abs());
                }
                prev = Some(f);
            }
            None => prev = None,
        }
        if (-2.0..=2.0).contains(&z0) && !s.pose.grounded {
            lost_ground_mid_cross = true;
        }
    }
    assert!(
        max_delta > bound || lost_ground_mid_cross,
        "a 1 m seam step must be CAUGHT: max floor delta {max_delta:.5} m (bound {bound:.5} m), airborne-mid-cross={lost_ground_mid_cross} — the guard is vacuous if neither triggers"
    );
    eprintln!(
        "[ordeal] mismatch caught: max per-tick floor delta {max_delta:.5} m vs bound {bound:.5} m (airborne-mid-cross={lost_ground_mid_cross})"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Build a scratch realm: a generated tile + one rigid `body` hung above its
/// centre, welded at `render_origin`. Returns the scene + the analytic rest
/// height derived from the field and the body's own dials.
fn body_on_terrain_realm(
    tag: &str,
    tile_x: i64,
    tile_y: i64,
    render_origin: [f64; 3],
) -> (std::path::PathBuf, RenderScene, f64) {
    let terrain_json = format!(
        r#"{{ "seed": 20260717, "tile_x": {tile_x}, "tile_y": {tile_y}, "height_amplitude": 1.5 }}"#
    );
    let sigil: TerrainSigil = serde_json::from_str(&terrain_json).unwrap();
    let tile = sigil.tile();
    let tparams = sigil.params();
    let world_seed = sigil.world_seed();
    let n = tparams.grid_resolution as i64;
    let cell = tparams.cell_size_m();
    let local = n / 2; // tile-centre grid vertex, in BOTH axes

    // The field height AT THE DROP COLUMN (the tile centre), via the exact
    // i64 lattice path — never through an f32 world coordinate (which would
    // itself lose precision at planetary magnitude; that trap is exactly
    // what ruling 4 pays off, so the analytic reference must not fall into
    // it either). global index = tile*n + local, in both axes.
    let field_center = height_at_grid_index(
        world_seed,
        &tparams,
        tile.tile_x * n + local,
        tile.tile_y * n + local,
    );

    // CAMERA-RELATIVE placement: with render_origin co-located with the
    // tile origin (the far case) OR at the world origin for the near tile,
    // the tile's local grid vertex (local*cell) IS its render-space position
    // — the terrain placement offset (tile_origin - render_origin, an exact
    // f64 subtract) is [0,0,0] in the co-located case and the tile's own
    // origin in the near case. Compute the drop column in render-LOCAL
    // coordinates directly (small numbers), never by casting a planetary
    // world coordinate down to f32.
    let offset_x = (seed::tile_origin_m(tile, &tparams).0 - render_origin[0]) as f32;
    let offset_z = (seed::tile_origin_m(tile, &tparams).1 - render_origin[2]) as f32;
    let body_x = offset_x + local as f32 * cell;
    let body_z = offset_z + local as f32 * cell;
    let field_local_y = field_center - render_origin[1] as f32;

    let half = 0.4f32; // body half-height (size.y/2)
    let contact_radius = 0.05f32;
    let drop_y = field_local_y + 6.0;
    let dir = std::env::temp_dir().join(format!("gaia_vii1body_{tag}_{}", std::process::id()));
    let scenes = dir.join("scenes");
    std::fs::create_dir_all(&scenes).expect("scenes dir");
    let scene = format!(
        r##"{{
          "ground": {{ "terrain": {terrain_json} }},
          "falling_crate": {{
            "transform": {{ "position": [{body_x}, {drop_y}, {body_z}] }},
            "mesh": {{ "parts": [ {{ "shape": "box", "size": [0.8, 0.8, 0.8], "position": [0,0,0], "color": "#6a4a2c" }} ] }},
            "body": {{ "shape": "box", "size": [0.8, 0.8, 0.8], "density": 500, "resolution": [3,3,3], "contact_radius": {contact_radius}, "rigidity": 1.0 }}
          }}
        }}"##
    );
    std::fs::write(scenes.join("main.json"), scene).expect("write scene");
    let mut world = EcsWorld::default();
    load_world_dir(&dir, &mut world).expect("load the body-on-terrain realm");
    let render =
        RenderScene::from_ecs_at(world, &params(), render_origin).expect("weld body-on-terrain");
    let analytic_rest = field_local_y as f64 + half as f64 + contact_radius as f64;
    (dir, render, analytic_rest)
}

/// ORDEAL 3 — THE BODY RESTS ON GENERATED TRIANGLES IN THE SOLVER. A rigid box
/// hung above the generated patch centre falls and settles at the DERIVED
/// analytic height `field(centre) + half_height + contact_radius`. If the
/// generated triangles were NOT in `Solver.collider`, the body would fall
/// through to the void — so a rest at the field proves the solver sees them.
#[test]
fn rigid_body_rests_on_generated_terrain_in_the_solver_collision_world() {
    let (dir, mut scene, analytic_rest) = body_on_terrain_realm("near", 0, 0, [0.0, 0.0, 0.0]);
    let start = scene.body_position("falling_crate").expect("body")[1];
    let mut last = start;
    for tick in 0..900u64 {
        scene.tick();
        let y = scene.body_position("falling_crate").unwrap()[1];
        if tick > 120 && (y - last).abs() < 1e-6 {
            break;
        }
        last = y;
    }
    let rest = scene.body_position("falling_crate").unwrap()[1];
    eprintln!(
        "[ordeal] body on generated terrain: start y={start:.4} rest y={rest:.4} analytic={analytic_rest:.4} Δ={:.5}",
        (rest - analytic_rest).abs()
    );
    assert!(
        start > analytic_rest + 4.0,
        "body started well above the field"
    );
    // Loose tol: the fBm patch under the footprint is not perfectly flat, so
    // the resting box tilts/settles a little off the single-vertex analytic
    // value — bounded by the field relief across the 0.8 m footprint
    // (gradient·halfwidth ≈ 0.3·0.4 ≈ 0.12 m) plus the contact margin.
    const REST_TOL: f64 = 0.2;
    assert!(
        (rest - analytic_rest).abs() < REST_TOL,
        "body rest y {rest} != derived analytic {analytic_rest} (tol {REST_TOL}) — did the generated triangles reach the solver collider?"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// ORDEAL 4 — THE COLLIDER RESPECTS THE COORDINATE LAW (ruling 4). The SAME
/// rest, with the tile at PLANETARY magnitude and `render_origin` co-located:
/// the collider triangles are placed `i64 -> f64 -> subtract -> f32` exact, so
/// the body still rests at the field value — the collider inherits the render's
/// camera-relative placement, no precision lost at planetary distance.
#[test]
fn rigid_body_rests_on_generated_terrain_at_planetary_render_origin() {
    // Ruling 4's own worked magnitude.
    let tile_x = 10_000_000i64;
    let tile_y = -10_000_000i64;
    // Co-locate render_origin with the tile origin (camera-relative guarantee).
    let sigil: TerrainSigil = serde_json::from_str(&format!(
        r#"{{ "seed": 20260717, "tile_x": {tile_x}, "tile_y": {tile_y}, "height_amplitude": 1.5 }}"#
    ))
    .unwrap();
    let (ox, oz) = seed::tile_origin_m(sigil.tile(), &sigil.params());
    let render_origin = [ox, 0.0, oz];

    let (dir, mut scene, analytic_rest) =
        body_on_terrain_realm("far", tile_x, tile_y, render_origin);
    let start = scene.body_position("falling_crate").expect("body")[1];
    let mut last = start;
    for tick in 0..900u64 {
        scene.tick();
        let y = scene.body_position("falling_crate").unwrap()[1];
        if tick > 120 && (y - last).abs() < 1e-6 {
            break;
        }
        last = y;
    }
    let rest = scene.body_position("falling_crate").unwrap()[1];
    eprintln!(
        "[ordeal] body on generated terrain @ planetary tile ({tile_x},{tile_y}): rest y={rest:.4} analytic={analytic_rest:.4} Δ={:.5}",
        (rest - analytic_rest).abs()
    );
    assert!(
        start > analytic_rest + 4.0,
        "body started well above the field"
    );
    const REST_TOL: f64 = 0.2;
    assert!(
        (rest - analytic_rest).abs() < REST_TOL,
        "at planetary render_origin the body rest y {rest} != derived analytic {analytic_rest} (tol {REST_TOL}) — the collider lost the coordinate law"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// ORDEAL 5 — RULING 6 BITES GENERATED GEOMETRY. On a terrain-ONLY realm the
/// contact-patch gate admits a column at the patch centre (all 8 probes land
/// on generated floor) and REJECTS a column past the patch edge (probes fall
/// off the generated surface into no-floor) — the gate runs on generated
/// triangles exactly as it does on authored ones (same `Ground::height_at`).
#[test]
fn ruling6_contact_patch_gate_bites_generated_geometry() {
    let terrain_json = r#"{ "seed": 20260717, "tile_x": 0, "tile_y": 0, "height_amplitude": 1.5 }"#;
    let sigil: TerrainSigil = serde_json::from_str(terrain_json).unwrap();
    let tile = sigil.tile();
    let tparams = sigil.params();
    let (ox, oz) = seed::tile_origin_m(tile, &tparams);
    let tile_size = tparams.tile_size_m;

    let (dir, scene) = seam_realm_terrain_only("gate", terrain_json);
    let ground = Ground::from_positions(&scene.leaf_positions());

    // Centre column — admitted (a big patch under the whole contact disc).
    let cx = ox as f32 + tile_size / 2.0;
    let cz = oz as f32 + tile_size / 2.0;
    assert!(
        ground.height_at(cx, cz, f32::INFINITY).is_some(),
        "the patch centre must be admitted by the ruling-6 gate on generated floor"
    );

    // Past-edge column — the contact disc straddles the patch's outer edge, so
    // some probes fall off the generated surface onto no floor: rejected.
    let radius = PlayerParams::from_env().unwrap().contact_radius;
    let edge_x = ox as f32 + tile_size - radius * 0.5; // disc pokes past +x edge
    let edge_z = cz;
    let admitted = ground.height_at(edge_x, edge_z, f32::INFINITY).is_some();
    assert!(
        !admitted,
        "a contact disc straddling the generated patch's outer edge must be REJECTED by ruling 6 (some probes have no floor) — the gate is not biting generated geometry"
    );
    eprintln!(
        "[ordeal] ruling 6 on generated geometry: centre ({cx:.1},{cz:.1}) admitted, past-edge ({edge_x:.1},{edge_z:.1}) rejected"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Terrain-only scratch realm (no authored plane), for the gate-edge ordeal.
fn seam_realm_terrain_only(tag: &str, terrain_json: &str) -> (std::path::PathBuf, RenderScene) {
    let dir = std::env::temp_dir().join(format!("gaia_vii1to_{tag}_{}", std::process::id()));
    let scenes = dir.join("scenes");
    std::fs::create_dir_all(&scenes).expect("scenes dir");
    let scene = format!(r#"{{ "ground": {{ "terrain": {terrain_json} }} }}"#);
    std::fs::write(scenes.join("main.json"), scene).expect("write scene");
    let mut world = EcsWorld::default();
    load_world_dir(&dir, &mut world).expect("load terrain-only realm");
    let render = RenderScene::from_ecs(world, &params()).expect("weld terrain-only realm");
    (dir, render)
}
