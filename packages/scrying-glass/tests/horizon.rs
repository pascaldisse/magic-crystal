//! RITE VII · VII-2 — THE HORIZON STREAMS (data-driven residency ordeals).
//!
//! The horizon around a moving walker is held in FINITE memory by DATA-DRIVEN
//! residency (`scrying_glass::horizon::HorizonRing`): tiles materialize from
//! `(seed, coords)` ahead of the walker and are evicted behind it, under a hard
//! byte budget — never authored streaming volumes (a forbidden concept).
//!
//!  1. a long straight walk materializes ahead / evicts behind, resident BYTES
//!     ≤ budget every tick (asserted INSIDE the loop), and an identical flight
//!     replays an identical load/evict sequence (determinism invariant);
//!  2. render_origin rebases as the walker advances — the WORLD pose trace is
//!     continuous across the rebase (per-tick world delta within a DERIVED
//!     bound, reusing VII-1's floor-delta pattern), and the camera-relative
//!     LOCAL coordinates stay bounded by the residency reach (the point of
//!     rebasing);
//!  3. the SAME walk + rebase at PLANETARY tile magnitude (±10,000,000) holds
//!     the same invariants — ruling 4 (i64-exact) + camera-relative rendering
//!     paid end to end;
//!  4. an evicted tile LEAVES the collision world — a tile behind the horizon
//!     no longer collides (sabotage-style negative), through the REAL
//!     `RenderScene::from_ecs_at` weld's `Ground`.
//!
//! Every bound is derived live; nothing here is a plucked literal.

use glam::Vec3;
use scrying_glass::horizon::{tile_byte_cost, HorizonRing};
use scrying_glass::player::{contact_tolerance, Ground, Key, Player, PlayerParams};
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults};
use seed::terrain::{tile_origin_m, TerrainParams, TerrainTile};

const TICK_DT: f32 = 1.0 / 60.0;

fn scene_params() -> SceneParameters {
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
        tick_dt: TICK_DT as f64,
        sun: SunDefaults {
            sun_color: "#ffe2b0".into(),
            sun_intensity: 1.1,
            sun_position: [60.0, 90.0, 30.0],
            ambient_intensity: 0.32,
        },
        emission_intensity: 2.5,
    }
}

/// A gentle, fully-walkable, COARSE terrain (cheap to transmute in tests):
/// 16 m tiles, an 8-cell grid, modest relief — the crossing stays walkable so
/// the pose-trace ordeals never fight a wall.
fn walkable_terrain() -> TerrainParams {
    let mut p = TerrainParams::derive(16.0);
    p.grid_resolution = 8;
    p.height_amplitude = 0.8;
    p
}

const SEED: u64 = 20260717;

/// The DERIVED per-tick WORLD-position continuity bound for a grounded walker:
/// horizontal ≤ `v·dt`, vertical (terrain follow) ≤ `v·dt·g_max` where
/// `g_max = contact_tolerance(1.0) = tan(acos(WALL_NORMAL_Y_COS_CUTOFF))` —
/// the SAME wall cutoff `Ground` drops non-floor triangles at (VII-1's derivation).
/// Total ≤ `v·dt·√(1+g_max²)` + fp slack (an f32 term on the small LOCAL
/// magnitude, an f64 term on the possibly-planetary render-origin magnitude,
/// since the world delta is accumulated in f64).
fn world_step_bound(walk_speed: f32, local_mag: f32, origin_mag: f64) -> f64 {
    let g_max = contact_tolerance(1.0) as f64;
    let horizontal = walk_speed as f64 * TICK_DT as f64;
    let slack = local_mag.abs().max(1.0) as f64 * f32::EPSILON as f64 * 32.0
        + origin_mag.abs().max(1.0) * f64::EPSILON * 32.0;
    horizontal * (1.0 + g_max * g_max).sqrt() + slack
}

// ---------------------------------------------------------------------------
// ORDEAL 1 — the horizon streams: materialize ahead, evict behind, budget held.
// ---------------------------------------------------------------------------

/// Advance a straight-line observer flight and return, per tick, the
/// `(loaded, evicted)` tile-key sequence — the determinism witness.
fn flight_sequence(
    budget: u64,
    steps: u64,
    v: f64,
) -> Vec<(Vec<(i64, i64)>, Vec<(i64, i64)>)> {
    let mut ring = HorizonRing::new(SEED, walkable_terrain(), Some("#4a7c59".into()), budget)
        .expect("ring");
    let ts = ring.params().tile_size_m as f64;
    let z = ts * 0.5;
    let mut x = 0.0f64;
    let mut seq = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        x += v * TICK_DT as f64;
        let tick = ring.update(x, z);
        seq.push((
            tick.loaded.iter().map(|t| (t.tile_x, t.tile_y)).collect(),
            tick.evicted.iter().map(|t| (t.tile_x, t.tile_y)).collect(),
        ));
    }
    seq
}

#[test]
fn ordeal1_horizon_materializes_ahead_evicts_behind_under_budget() {
    let tparams = walkable_terrain();
    let tb = tile_byte_cost(&tparams);
    // Budget sized for exactly a 7×7 = 49-tile residency square (radius 3).
    let budget = 49 * tb;
    let mut ring =
        HorizonRing::new(SEED, tparams, Some("#4a7c59".into()), budget).expect("ring");
    assert_eq!(ring.radius_tiles(), 3, "√49 = 7 = 2·3+1 ⇒ radius 3");

    let v = PlayerParams::from_env().expect("player params").walk_speed as f64;
    let ts = ring.params().tile_size_m as f64;
    let z = ts * 0.5;
    let start_tile = ring.tile_at(0.0, z);

    let tiles_to_cross = 20.0;
    let ticks = (tiles_to_cross * ts / (v * TICK_DT as f64)).ceil() as u64 + 120;
    let mut x = 0.0f64;
    let mut total_loads = 0usize;
    let mut total_evicts = 0usize;
    for _ in 0..ticks {
        x += v * TICK_DT as f64;
        let tick = ring.update(x, z);
        // INVARIANT 1 — budget never exceeded, asserted INSIDE the loop.
        assert!(
            tick.resident_bytes <= ring.budget_bytes(),
            "resident bytes {} exceeded budget {}",
            tick.resident_bytes,
            ring.budget_bytes()
        );
        // INVARIANT 2 — the walker's own tile (and its whole square) resident.
        assert!(
            ring.is_resident(tick.observer_tile),
            "observer tile {:?} must be resident",
            tick.observer_tile
        );
        // The budget is exactly the required square, so residency stays tight
        // at (2·3+1)² = 49 tiles — everything else is evicted immediately.
        assert_eq!(ring.resident_count(), 49, "residency square held tight");
        total_loads += tick.loaded.len();
        total_evicts += tick.evicted.len();
    }
    // Materialize-ahead AND evict-behind both actually happened.
    assert!(total_loads > 0, "tiles must materialize ahead over the walk");
    assert!(total_evicts > 0, "tiles must be evicted behind over the walk");

    // The tile the walk started in is now ~20 tiles behind the horizon: gone.
    assert!(
        !ring.is_resident(start_tile),
        "a tile far behind the horizon must be evicted, not resident"
    );
    // The ground under and ahead of the walker IS resident.
    let cur = ring.tile_at(x, z);
    assert!(ring.is_resident(cur), "the walker's current tile is resident");
    assert!(
        ring.is_resident(cur.neighbor(1, 0)),
        "the tile one ahead (load-ahead) is resident"
    );

    eprintln!(
        "[ordeal] horizon walk: {total_loads} materializations, {total_evicts} evictions over {ticks} ticks; \
         resident held at {} tiles = {} B ≤ budget {} B; start tile {:?} evicted",
        ring.resident_count(),
        ring.resident_bytes(),
        ring.budget_bytes(),
        start_tile
    );

    // INVARIANT 3 — an identical flight replays an identical load/evict sequence.
    let steps = 400u64;
    let a = flight_sequence(budget, steps, v);
    let b = flight_sequence(budget, steps, v);
    assert_eq!(a, b, "identical flight must replay an identical load/evict sequence");
    let seq_loads: usize = a.iter().map(|(l, _)| l.len()).sum();
    let seq_evicts: usize = a.iter().map(|(_, e)| e.len()).sum();
    assert!(seq_loads > 0 && seq_evicts > 0, "the replayed flight must load AND evict");
    eprintln!(
        "[ordeal] determinism: {steps}-tick flight replays byte-identical ({seq_loads} loads, {seq_evicts} evicts, both runs equal)"
    );
}

// ---------------------------------------------------------------------------
// ORDEALS 2 & 3 — render_origin rebases; the WORLD pose trace stays continuous.
// ---------------------------------------------------------------------------

/// Outcome of a rebasing walk: the worst per-tick world-position delta, the
/// derived bound, how many render_origin rebases fired, and the worst LOCAL
/// (camera-relative) coordinate magnitude the body ever reached.
struct RebaseWalk {
    max_world_delta: f64,
    bound: f64,
    rebases: u32,
    max_local_mag: f32,
    grounded_ticks: u32,
    budget_ok: bool,
}

/// Settle a walker on the resident ground at `start_world`, then walk +x for
/// `walk_ticks`, rebasing render_origin to the observer's tile whenever the
/// walker changes tile. The WORLD pose (`local + render_origin`, accumulated in
/// f64) must never step more than the derived bound between consecutive
/// grounded ticks — the rebase is invisible in world space by construction, so
/// a discontinuity there is a real bug.
fn rebasing_walk(start_world_x: f64, start_world_z: f64, walk_ticks: u32) -> RebaseWalk {
    rebasing_walk_mode(start_world_x, start_world_z, walk_ticks, false)
}

/// As [`rebasing_walk`], but `sabotage = true` DROPS the `player.rebase` frame
/// shift on every render_origin change (while still moving the origin and
/// rebuilding the ground) — the walker's world pose then jumps by ~a tile each
/// rebase. The anti-vacuous twin: the SAME continuity bound must be EXCEEDED,
/// proving the guard detects a real discontinuity rather than trivially
/// passing. In sabotage mode the inline continuity assert is disarmed and the
/// worst delta is returned for the twin to judge.
fn rebasing_walk_mode(
    start_world_x: f64,
    start_world_z: f64,
    walk_ticks: u32,
    sabotage: bool,
) -> RebaseWalk {
    let sp = scene_params();
    let mut ring =
        HorizonRing::new(SEED, walkable_terrain(), Some("#4a7c59".into()), {
            // Radius-2 square (25 tiles): reach = 2·16 = 32 m of local coords.
            25 * tile_byte_cost(&walkable_terrain())
        })
        .expect("ring");
    assert_eq!(ring.radius_tiles(), 2);
    let reach_mag = (ring.radius_tiles() as f32 + 1.0) * ring.params().tile_size_m;

    let pp = PlayerParams::from_env().expect("player params");

    // Build the resident ground around the start, camera-relative to the start
    // tile's origin.
    let mut origin = ring.render_origin_for(start_world_x, start_world_z);
    ring.update(start_world_x, start_world_z);
    let mut ground = build_ground(&ring, origin, &sp);

    // Spawn the eye in LOCAL coordinates above the ground, and settle.
    let local_spawn = Vec3::new(
        (start_world_x - origin[0]) as f32,
        (start_world_z as f32) * 0.0 + 8.0, // 8 m above, let it fall onto the field
        (start_world_z - origin[2]) as f32,
    );
    // Walk +x: forward = (sin yaw, 0, -cos yaw), so yaw = +π/2 faces +X (the
    // seam test's yaw=π faces +Z at the same convention).
    let mut player = Player::new(pp, local_spawn, std::f32::consts::FRAC_PI_2);
    for _ in 0..240 {
        player.step(TICK_DT, &ground);
    }

    let mut prev_world: Option<Vec3Wide> = None;
    let mut max_world_delta = 0.0f64;
    let mut max_local_mag = 0.0f32;
    let mut rebases = 0u32;
    let mut grounded_ticks = 0u32;
    let mut budget_ok = true;
    let v = pp.walk_speed;
    let bound = world_step_bound(v, reach_mag, origin_mag(origin));

    player.keys.insert(Key::Forward);
    for _ in 0..walk_ticks {
        player.step(TICK_DT, &ground);
        let p = player.pose();
        max_local_mag = max_local_mag
            .max(p.position.x.abs().max(p.position.z.abs()));
        let world = Vec3Wide {
            x: p.position.x as f64 + origin[0],
            y: p.position.y as f64 + origin[1],
            z: p.position.z as f64 + origin[2],
            grounded: p.grounded,
        };
        if p.grounded {
            grounded_ticks += 1;
        }
        if let Some(pw) = prev_world {
            // Only judge continuity across grounded→grounded pairs (a settle
            // fall or an intended jump is not a seam discontinuity).
            if pw.grounded && p.grounded {
                let d = world.dist(&pw);
                max_world_delta = max_world_delta.max(d);
                if !sabotage {
                    assert!(
                        d <= bound,
                        "world pose jumped {d:.6} m in one grounded tick (derived bound {bound:.6} m) — a rebase or seam discontinuity"
                    );
                }
            }
        }
        prev_world = Some(world);

        // Rebase render_origin to the walker's current tile if it changed.
        ring.update(world.x, world.z);
        if !budget_or_ok(&ring) {
            budget_ok = false;
        }
        let new_origin = ring.render_origin_for(world.x, world.z);
        if new_origin != origin {
            let delta = Vec3::new(
                (origin[0] - new_origin[0]) as f32,
                (origin[1] - new_origin[1]) as f32,
                (origin[2] - new_origin[2]) as f32,
            );
            if !sabotage {
                player.rebase(delta);
            }
            ground = build_ground(&ring, new_origin, &sp);
            origin = new_origin;
            rebases += 1;
        }
    }

    RebaseWalk {
        max_world_delta,
        bound,
        rebases,
        max_local_mag,
        grounded_ticks,
        budget_ok,
    }
}

fn budget_or_ok(ring: &HorizonRing) -> bool {
    ring.resident_bytes() <= ring.budget_bytes()
}

fn origin_mag(origin: [f64; 3]) -> f64 {
    origin[0].abs().max(origin[2].abs())
}

/// A world-space pose accumulated in f64 (exact at planetary magnitude), with
/// the grounded flag carried alongside for the continuity gate.
#[derive(Clone, Copy)]
struct Vec3Wide {
    x: f64,
    y: f64,
    z: f64,
    grounded: bool,
}
impl Vec3Wide {
    fn dist(&self, o: &Vec3Wide) -> f64 {
        let dx = self.x - o.x;
        let dy = self.y - o.y;
        let dz = self.z - o.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}
/// Build the walker floor from the ring's resident tiles through the REAL weld
/// (`RenderScene::from_ecs_at` — the SOLE geometry path), camera-relative to
/// `origin`.
fn build_ground(ring: &HorizonRing, origin: [f64; 3], sp: &SceneParameters) -> Ground {
    let scene: RenderScene = ring.scene_at(origin, sp).expect("weld resident tiles");
    Ground::from_positions(&scene.leaf_positions())
}

#[test]
fn ordeal2_render_origin_rebases_pose_trace_continuous() {
    // Start near the world origin; walk far enough to cross several tiles.
    let w = rebasing_walk(8.0, 8.0, 900);
    assert!(w.budget_ok, "budget held across the whole rebasing walk");
    assert!(
        w.rebases >= 2,
        "the walk must cross tiles and rebase render_origin at least twice (got {})",
        w.rebases
    );
    assert!(
        w.grounded_ticks > 300,
        "the walker should stay grounded for most of the walk ({} ticks)",
        w.grounded_ticks
    );
    // The camera-relative coordinates stayed bounded by the residency reach —
    // the whole point of rebasing (they never grow with world distance).
    let reach = w_reach();
    assert!(
        w.max_local_mag <= reach,
        "local coords {} must stay within the residency reach {reach} (rebasing keeps them small)",
        w.max_local_mag
    );
    eprintln!(
        "[ordeal] rebase near origin: {} rebases, max world-tick delta {:.6} m ≤ bound {:.6} m; max |local| {:.2} m ≤ reach {:.2} m",
        w.rebases, w.max_world_delta, w.bound, w.max_local_mag, reach
    );
}

#[test]
fn ordeal2b_sabotaged_rebase_trips_the_continuity_guard() {
    // Skip the player frame-shift on each rebase: the world pose must JUMP past
    // the derived bound — proving ordeal 2's guard is discriminating, not vacuous.
    let w = rebasing_walk_mode(8.0, 8.0, 900, true);
    assert!(
        w.rebases >= 2,
        "the sabotage walk must still cross tiles ({} rebases)",
        w.rebases
    );
    assert!(
        w.max_world_delta > w.bound,
        "a dropped render_origin rebase MUST spike the world pose past the bound \
         (max delta {:.4} m vs bound {:.4} m) — else the continuity guard is vacuous",
        w.max_world_delta,
        w.bound
    );
    eprintln!(
        "[ordeal] anti-vacuous: dropping the rebase shift spikes the world pose to {:.3} m ≫ bound {:.3} m — the guard bites",
        w.max_world_delta, w.bound
    );
}

#[test]
fn ordeal3_planetary_magnitude_walk_holds_the_invariants() {
    // Ruling 4's own worked magnitude: tile ±10,000,000. Start at that tile's
    // origin so render_origin is planetary from tick zero.
    let tparams = walkable_terrain();
    let far_tile = TerrainTile::new(10_000_000, -10_000_000);
    let (ox, oz) = tile_origin_m(far_tile, &tparams);
    // Start a little inside the tile so the walker crosses its far edge.
    let w = rebasing_walk(ox + 2.0, oz + 8.0, 900);
    assert!(w.budget_ok, "budget held across the planetary rebasing walk");
    assert!(
        w.rebases >= 2,
        "the planetary walk must rebase render_origin at least twice (got {})",
        w.rebases
    );
    assert!(
        w.grounded_ticks > 300,
        "grounded for most of the planetary walk ({} ticks)",
        w.grounded_ticks
    );
    let reach = w_reach();
    assert!(
        w.max_local_mag <= reach,
        "at planetary magnitude local coords {} must STILL stay within reach {reach} — camera-relative rebasing is what pays ruling 4",
        w.max_local_mag
    );
    eprintln!(
        "[ordeal] rebase @ planetary tile ({},{}): origin ~{:.3e} m; {} rebases, max world-tick delta {:.6} m ≤ bound {:.6} m; max |local| {:.2} m ≤ reach {:.2} m",
        far_tile.tile_x, far_tile.tile_y, ox, w.rebases, w.max_world_delta, w.bound, w.max_local_mag, reach
    );
}

/// The residency reach in meters used by [`rebasing_walk`] (radius-2 square of
/// 16 m tiles ⇒ (2+1)·16 = 48 m worst-case local magnitude).
fn w_reach() -> f32 {
    let t = walkable_terrain();
    (2.0 + 1.0) * t.tile_size_m
}

// ---------------------------------------------------------------------------
// ORDEAL 4 — an evicted tile leaves the collision world (sabotage negative).
// ---------------------------------------------------------------------------

#[test]
fn ordeal4_evicted_tile_leaves_the_collision_world() {
    let sp = scene_params();
    let tparams = walkable_terrain();
    let budget = 25 * tile_byte_cost(&tparams); // radius 2
    let mut ring = HorizonRing::new(SEED, tparams, Some("#4a7c59".into()), budget).expect("ring");
    let ts = ring.params().tile_size_m as f64;

    // Position A — establish residency; tile T = the observer's own tile.
    let ax = 0.0f64;
    let az = ts * 0.5;
    ring.update(ax, az);
    let t = ring.tile_at(ax, az);
    assert!(ring.is_resident(t), "T resident at A");

    let origin_a = ring.render_origin_for(ax, az);
    let ground_a = build_ground(&ring, origin_a, &sp);
    let (tcx, tcz) = tile_center(t, ring.params());
    let ta_local = (
        (tcx - origin_a[0]) as f32,
        (tcz - origin_a[2]) as f32,
    );
    assert!(
        ground_a
            .height_at(ta_local.0, ta_local.1, f32::INFINITY)
            .is_some(),
        "a resident tile's centre MUST collide (present in the collision world)"
    );

    // Walk far past the horizon so T is evicted behind us.
    let bx = ax + ts * (ring.radius_tiles() + 5) as f64;
    ring.update(bx, az);
    assert!(
        !ring.is_resident(t),
        "T must be evicted once the walker is well past the horizon"
    );

    let origin_b = ring.render_origin_for(bx, az);
    let ground_b = build_ground(&ring, origin_b, &sp);
    let tb_local = (
        (tcx - origin_b[0]) as f32,
        (tcz - origin_b[2]) as f32,
    );
    // THE NEGATIVE — the evicted tile no longer collides.
    assert!(
        ground_b
            .height_at(tb_local.0, tb_local.1, f32::INFINITY)
            .is_none(),
        "an EVICTED tile behind the horizon must NOT collide — but the collision world still answered at its centre"
    );
    // Positive control at B — the walker's current tile DOES collide.
    let cur = ring.tile_at(bx, az);
    let (ccx, ccz) = tile_center(cur, ring.params());
    let cur_local = ((ccx - origin_b[0]) as f32, (ccz - origin_b[2]) as f32);
    assert!(
        ground_b
            .height_at(cur_local.0, cur_local.1, f32::INFINITY)
            .is_some(),
        "the walker's current resident tile must still collide at B"
    );

    eprintln!(
        "[ordeal] eviction leaves collision: tile {:?} collided at A, GONE from the collision world at B (walker {} tiles ahead)",
        t,
        ring.radius_tiles() + 5
    );
}

fn tile_center(tile: TerrainTile, params: &TerrainParams) -> (f64, f64) {
    let (ox, oz) = tile_origin_m(tile, params);
    let half = params.tile_size_m as f64 / 2.0;
    (ox + half, oz + half)
}
