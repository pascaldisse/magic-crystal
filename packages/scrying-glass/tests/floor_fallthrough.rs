//! FLOOR FALL-THROUGH ORDEALS — the Architect's live-window "fall through the
//! ground while walking" report, pinned as CI ordeals.
//!
//! The hunt (see `examples/floor_fallthrough_hunt.rs`) drove the live
//! `Player::step` against the live naruko `Ground` over the whole reachable
//! realm (161×121 static grid at 0.5 m, 3500+ settle-and-walk sweeps in
//! walk/run/run+jump) and found NO interior fall-through: the terra plate is
//! watertight. Every below-plaza event is the plate's DESIGNED edge —
//!
//!   * NORTH rim (terra top y=0 ends at z=8): a 1.35 m step-DOWN onto the
//!     authored `naruko_sea` plate (top y≈-1.35), reachable by walking around
//!     the 120 m-wide `naruko_seawall` at z=18. The body stays GROUNDED.
//!   * SOUTH rim (terra ends at z=68, the sea plate does NOT extend past z=40):
//!     an OFF-WORLD drop to `void_y` → last-safe respawn. The realm boundary.
//!
//! These ordeals LOCK that finding: a genuine tunnel (body ending below a
//! GATED floor that still exists ABOVE it at its own column — walkable floor it
//! should be standing on) must NEVER occur on any walk. Work is a FIXED,
//! bounded walk list (no wall-clock bail needed — the predecessor sweep hung
//! because it re-ran O(triangles) gated queries over the whole realm; here the
//! walk count and tick budget are constants, so the ordeal always terminates
//! fast). FINDING for a future atom (NOT fixed here): `Ground` height queries
//! are O(triangles) linear scans (`raw_height_at` scans all ~10.4k tris,
//! `height_at_gated` ≈9× that) — an acceleration structure is a separate
//! exact-perf task, not a fall-through bug.

use std::path::Path;

use crystal::{Core, load_world_dir};
use glam::Vec3;
use scrying_glass::player::{Ground, Key, Player, PlayerParams, contact_tolerance};
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults};

fn naruko_params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: 60.0,
        near: 0.1,
        far: 4_000.0,
        sky_top: "#20152f".into(),
        sky_horizon: "#9a627d".into(),
        mesh_color: "#9aa0a6".into(),
        radial_segments: 24,
        camera_position: [0.0, 1.7, 24.0],
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

const DT: f32 = 1.0 / 60.0;
const PLAZA_Y: f32 = 0.0;
// A tick of walk-height smoothing lag, well above float noise and well below
// the plaza's 0.5 m slab thickness. Feet below PLAZA_Y - this is "below the
// plaza" — the cheap trigger that pays for a settle-classify.
const FALL_EPS: f32 = 0.25;

fn naruko_ground_and_params() -> (Ground, PlayerParams) {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene =
        RenderScene::from_ecs(std::mem::take(&mut core.world), &naruko_params()).expect("scene");
    let ground = Ground::from_positions(&scene.leaf_positions());
    let pp = PlayerParams::from_env().expect("player params");
    (ground, pp)
}

fn gated_floor(ground: &Ground, pp: &PlayerParams, x: f32, z: f32, ceiling: f32) -> Option<f32> {
    ground.height_at_gated(x, z, ceiling, pp.contact_radius, contact_tolerance(pp.contact_radius))
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Fall {
    /// Landed grounded on a real lower floor (the sea plate). Expected ledge.
    StepDown,
    /// Left the plate over its true authored edge → void_y. Realm boundary.
    OffWorld,
    /// Ended below a GATED floor that still exists ABOVE it at its own column:
    /// passed THROUGH walkable floor. THE bug. Carries the floor-above y.
    Tunnel(f32),
}

/// Release the keys and settle up to `SETTLE` ticks; the resting state names
/// the fall. Grounded → StepDown. Still airborne with a gated floor found
/// > ground_snap above the feet at this column → Tunnel. Else → OffWorld.
fn classify_fall(ground: &Ground, pp: &PlayerParams, player: &mut Player) -> Fall {
    const SETTLE: u32 = 120;
    player.keys.clear();
    for _ in 0..SETTLE {
        player.step(DT, ground);
        if player.grounded {
            return Fall::StepDown;
        }
    }
    let feet = player.position.y - player.eye_height;
    match gated_floor(ground, pp, player.position.x, player.position.z, 50.0) {
        Some(gy) if gy > feet + pp.ground_snap => Fall::Tunnel(gy),
        _ => Fall::OffWorld,
    }
}

/// Settle grounded at `start` (eye pose derived from a floor `y`), then hold
/// `keys` for `walk_ticks`, tapping jump periodically when `jump`. Returns the
/// first below-plaza event classified, or None if the body never dropped.
fn settle_and_walk(
    ground: &Ground,
    pp: &PlayerParams,
    start: (f32, f32),
    floor_y: f32,
    yaw: f32,
    keys: &[Key],
    run: bool,
    jump: bool,
    settle: u32,
    walk_ticks: u32,
) -> Option<(f32, f32, f32, Fall)> {
    let eye = Vec3::new(start.0, floor_y + pp.eye_stand + 0.2, start.1);
    let mut player = Player::new(*pp, eye, yaw);
    for _ in 0..settle {
        player.step(DT, ground);
    }
    if !player.grounded {
        return None; // never settled — not a valid walk start (sea/void)
    }
    for k in keys {
        player.keys.insert(*k);
    }
    if run {
        player.keys.insert(Key::Run);
    }
    for t in 0..walk_ticks {
        if jump && t % 30 == 15 {
            player.keys.insert(Key::Jump);
        } else {
            player.keys.remove(&Key::Jump);
        }
        player.step(DT, ground);
        let feet = player.position.y - player.eye_height;
        if feet < PLAZA_Y - FALL_EPS || player.vy < -8.0 {
            let (fx, fz) = (player.position.x, player.position.z);
            let fall = classify_fall(ground, pp, &mut player);
            return Some((fx, fz, feet, fall));
        }
    }
    None
}

/// ORDEAL 1 — WALK THE REALM: no interior fall-through. A FIXED, bounded set of
/// walks — the suspect features the Architect passes plus a coarse interior
/// grid — driven by the live `Player::step` against the live naruko `Ground`.
/// Every below-plaza event settle-classifies to a legitimate StepDown (sea
/// plate) or OffWorld (realm edge); a genuine Tunnel through supported interior
/// floor FAILS the ordeal. Bounded work ⇒ always terminates (anti-hang law).
#[test]
fn walk_the_realm_has_no_interior_fallthrough() {
    let (ground, pp) = naruko_ground_and_params();

    // SUSPECT ROUTES — over/around the named features (x,z, dir keys, run, jump).
    // Forward=-z, Back=+z, Right=+x, Left=-x at yaw 0.
    let suspects: &[(&str, (f32, f32), &[Key], bool, bool)] = &[
        ("over-chrome-pedestal", (4.5, 40.0), &[Key::Forward], false, false),
        ("thru-chrome-run", (4.5, 40.0), &[Key::Forward], true, true),
        ("over-mirror-slab", (-6.5, 40.0), &[Key::Forward], false, false),
        ("thru-mirror-run", (-6.5, 40.0), &[Key::Forward], true, true),
        ("mirror-slab-cross-x", (-12.0, 28.0), &[Key::Right], false, false),
        ("over-mirror-minor", (3.0, 40.0), &[Key::Forward], false, false),
        ("crate-cluster-z13", (0.0, 30.0), &[Key::Forward], false, true),
        ("crate-cluster-cross", (-10.0, 13.0), &[Key::Right], true, false),
        ("plaza-seam-EW", (-14.0, 25.0), &[Key::Right], true, false),
        ("plaza-seam-NS", (0.0, 34.0), &[Key::Forward], true, true),
    ];

    let mut worst: Vec<(&str, f32, f32, f32, Fall)> = Vec::new();
    let mut tunnels = 0usize;
    let mut walks = 0usize;

    for (name, start, keys, run, jump) in suspects {
        let f0 = match gated_floor(&ground, &pp, start.0, start.1, 50.0) {
            Some(y) => y,
            None => continue,
        };
        walks += 1;
        if let Some((fx, fz, feet, fall)) =
            settle_and_walk(&ground, &pp, *start, f0, 0.0, keys, *run, *jump, 45, 220)
        {
            if let Fall::Tunnel(_) = fall {
                tunnels += 1;
            }
            worst.push((name, fx, fz, feet, fall));
        }
    }

    // COARSE INTERIOR GRID — start on genuine plaza floor, walk each of two
    // cardinals (toward the north rim and along +x) at run speed with jump
    // taps. Bounded: 4×4 starts × 2 dirs = 32 walks max.
    let mut sx = -30.0f32;
    while sx <= 30.0 + 1e-4 {
        let mut sz = 16.0f32;
        while sz <= 64.0 + 1e-4 {
            if let Some(f0) = gated_floor(&ground, &pp, sx, sz, 50.0) {
                // Only start on the plaza plate itself (skip feature tops / sea).
                if (f0 - PLAZA_Y).abs() <= FALL_EPS {
                    for keys in [&[Key::Forward][..], &[Key::Right][..]] {
                        walks += 1;
                        if let Some((fx, fz, feet, fall)) = settle_and_walk(
                            &ground, &pp, (sx, sz), f0, 0.0, keys, true, true, 45, 90,
                        ) {
                            if let Fall::Tunnel(_) = fall {
                                tunnels += 1;
                                worst.push(("grid", fx, fz, feet, fall));
                            }
                        }
                    }
                }
            }
            sz += 16.0;
        }
        sx += 20.0;
    }

    println!("[walk-the-realm] {walks} walks, {tunnels} genuine tunnels");
    for (n, x, z, feet, fall) in worst.iter().take(20) {
        println!("  [{n}] ({x:.2},{z:.2}) feetY={feet:.3} {fall:?}");
    }
    assert_eq!(
        tunnels, 0,
        "a walk tunnelled THROUGH supported interior floor — the naruko terra \
         plate is no longer watertight; classified hits: {worst:?}"
    );
}

/// ORDEAL 2 — THE NORTH RIM STEP-DOWN, named. Walking north off the terra plate
/// where the 120 m seawall does NOT cover (|x| > 60) drops the body 1.35 m onto
/// the authored `naruko_sea` plate — it stays GROUNDED on real floor, it does
/// NOT fall to void. This is the EXACT mechanism behind every below-plaza event
/// the hunt flagged; pinning it means a future edit that deletes the sea plate
/// (turning the step-down into a true fall-through) surfaces here as a failure.
#[test]
fn terra_north_rim_steps_down_onto_sea_plate_grounded() {
    let (ground, pp) = naruko_ground_and_params();

    // Start on the terra plate north of the seawall gap (x=90 is well outside
    // the seawall's x∈[-60,60]) and walk north (Forward=-z) off the z=8 rim.
    let start = (90.0, 24.0);
    let f0 = gated_floor(&ground, &pp, start.0, start.1, 50.0)
        .expect("plaza floor under the north-rim start");
    assert!((f0 - PLAZA_Y).abs() <= FALL_EPS, "start must be on the plaza (y≈0), got {f0}");

    let eye = Vec3::new(start.0, f0 + pp.eye_stand + 0.2, start.1);
    let mut player = Player::new(pp, eye, 0.0);
    for _ in 0..45 {
        player.step(DT, &ground);
    }
    assert!(player.grounded, "did not settle grounded on the plaza");

    player.keys.insert(Key::Forward);
    // Walk north past the z=8 rim, then release and let the body land.
    let mut crossed = false;
    for _ in 0..260 {
        player.step(DT, &ground);
        if player.position.z < 8.0 - 0.5 {
            crossed = true;
            player.keys.clear();
        }
        if crossed && player.grounded {
            break;
        }
    }
    assert!(crossed, "never crossed the z=8 north rim while walking north");
    // Give it a few ticks to settle if it landed on the exact break tick.
    for _ in 0..120 {
        player.step(DT, &ground);
        if player.grounded {
            break;
        }
    }

    let feet = player.position.y - player.eye_height;
    println!(
        "[north-rim] rest ({:.2},{:.2}) feetY={:.3} grounded={} vy={:.3}",
        player.position.x, player.position.z, feet, player.grounded, player.vy
    );
    assert!(
        player.grounded,
        "north-rim step-down did NOT land grounded — the body fell through the \
         world instead of stepping onto the sea plate (feetY={feet:.3})"
    );
    assert!(
        player.position.y > pp.void_y + 1.0,
        "north-rim step-down fell to void_y — the sea plate is missing"
    );
    // The sea plate top is authored at y≈-1.35; the body should rest on it,
    // well below the plaza slab, NOT snapped back onto the plaza.
    assert!(
        feet < PLAZA_Y - FALL_EPS,
        "expected a step DOWN onto the lower sea plate, feet={feet:.3}"
    );
    // And the gated floor at the rest column is that sea plate, near -1.35.
    let g = gated_floor(&ground, &pp, player.position.x, player.position.z, 50.0)
        .expect("sea-plate floor under the rest column");
    assert!(
        (feet - g).abs() <= pp.ground_snap + 0.1,
        "body should rest ON the gated sea-plate floor: feet={feet:.3} gated={g:.3}"
    );
    assert!(
        (g - (-1.35)).abs() < 0.2,
        "the authored sea plate top should be ≈-1.35; gated floor here is {g:.3}"
    );
}
