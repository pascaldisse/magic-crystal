//! FLOOR FALL-THROUGH HUNT — offline repro of the Architect's live-window
//! "fall through the ground while walking" report.
//!
//! Drives the SAME `Player::step` against the SAME `Ground` the live window
//! builds (`Ground::from_positions(&render_scene.leaf_positions())`,
//! main.rs:1554) over the naruko realm, at live player defaults. Two organs:
//!
//!   1. STATIC PROBE MAP — a fine (x,z) grid over the plaza walk box compares
//!      the gated floor query (contact-patch, what the player actually stands
//!      on) against the raw query (radius 0 → patch disabled). A cell where
//!      the raw query finds the plaza but the gated query finds nothing (or
//!      something far lower) is a patch-gate hole: the player walking there
//!      loses the floor.
//!   2. WALK SWEEP — dense straight-line WASD walks across the box (both axes)
//!      plus lines threaded through the mirror base, chrome pedestal, stall
//!      and crate cluster. Each walk settles the body grounded, then holds
//!      forward; every tick logs pos/feetY/grounded/vy and flags any tick
//!      whose feet drop below the plaza floor (y=0) minus a derived epsilon.
//!
//! Run:  cargo run -p scrying-glass --release --example floor_fallthrough_hunt

use std::path::Path;
use std::time::Instant;

use crystal::{Core, load_world_dir};
use glam::Vec3;
use scrying_glass::player::{Ground, Key, Player, PlayerParams};
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

const PLAZA_Y: f32 = 0.0;
const DT: f32 = 1.0 / 60.0;
// A tick's worth of walk-height noise the ground_follow smoothing can lag by;
// well above float noise, well below the plaza's 0.5 m slab thickness. Feet
// below PLAZA_Y - this over solid plaza is a genuine fall-through, not lag.
const FALL_EPS: f32 = 0.25;

fn build() -> (Ground, PlayerParams) {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params)
        .expect("render scene");
    let ground = Ground::from_positions(&scene.leaf_positions());
    let pp = PlayerParams::from_env().expect("player params");
    (ground, pp)
}

/// Static probe: raw (patch-disabled) vs gated floor at a column.
fn probe(ground: &Ground, pp: &PlayerParams, x: f32, z: f32) -> (Option<f32>, Option<f32>) {
    let ceiling = 50.0;
    let raw = ground.height_at_gated(x, z, ceiling, 0.0, 0.0);
    let tol = scrying_glass::player::contact_tolerance(pp.contact_radius);
    let gated = ground.height_at_gated(x, z, ceiling, pp.contact_radius, tol);
    (raw, gated)
}

/// The GATED floor the player ACTUALLY stands on at a column (contact-patch),
/// looked for under `ceiling`.
fn gated_floor(ground: &Ground, pp: &PlayerParams, x: f32, z: f32, ceiling: f32) -> Option<f32> {
    let tol = scrying_glass::player::contact_tolerance(pp.contact_radius);
    ground.height_at_gated(x, z, ceiling, pp.contact_radius, tol)
}

/// What a below-plaza tick actually is, decided by RELEASING the keys and
/// letting the body settle — the only honest way to tell a legitimate
/// step-down from a tunnel, because the trigger fires mid-drop.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Fall {
    /// The body landed grounded on a real lower floor (naruko's sea plate at
    /// y≈-1.35 north of the terra rim). Expected ledge behaviour, not a bug.
    StepDown,
    /// The body left the terra plate over its TRUE authored edge (z>68 south /
    /// |x|>200 / any column with no floor under it) and fell to void_y. The
    /// realm boundary, not a floor-query fall-through.
    OffWorld,
    /// The body ended below a GATED floor that still exists ABOVE it at its own
    /// column — it passed THROUGH walkable floor it should be standing on. THE
    /// bug this hunt exists to catch. Carries the floor-above y.
    Tunnel(f32),
}

/// Classify a fall in progress: stop walking, settle up to `SETTLE` ticks, then
/// read the resting state. Grounded → StepDown. Still airborne with a gated
/// floor above the feet at this column → Tunnel. Otherwise → OffWorld.
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
    // Look for ANY gated floor above the fallen body at its column (ceiling well
    // above the plaza). A floor found > ground_snap above the feet is floor the
    // body should be resting on but is beneath — a genuine tunnel.
    match gated_floor(ground, pp, player.position.x, player.position.z, 50.0) {
        Some(gy) if gy > feet + pp.ground_snap => Fall::Tunnel(gy),
        _ => Fall::OffWorld,
    }
}

fn height_field(ground: &Ground, pp: &PlayerParams) {
    println!("\n== GATED FLOOR HEIGHT FIELD over walk box (each cell = gated floor y) ==");
    let tol = scrying_glass::player::contact_tolerance(pp.contact_radius);
    let mut none_cells = 0usize;
    let mut zi = 35.0f32;
    println!("       x: -10  -8  -6  -4  -2   0   2   4   6   8  10");
    while zi >= 15.0 - 1e-4 {
        let mut row = format!("z={zi:5.1} ");
        let mut xi = -10.0f32;
        while xi <= 10.0 + 1e-4 {
            let g = ground.height_at_gated(xi, zi, 50.0, pp.contact_radius, tol);
            match g {
                None => { row.push_str("  __"); none_cells += 1; }
                Some(y) if y.abs() < 0.05 => row.push_str("  .."), // plaza
                Some(y) => row.push_str(&format!("{y:4.0}")),
            }
            xi += 2.0;
        }
        println!("{row}");
        zi -= 2.0;
    }
    println!("(.. = plaza y~0, number = other floor y, __ = NO floor)");
    println!("none-floor cells in coarse grid: {none_cells}");
}

fn static_map(ground: &Ground, pp: &PlayerParams) {
    println!("\n== STATIC PROBE MAP (raw floor present but gated missing/lower) ==");
    // Widened to the whole terra footprint the player can reach on foot near
    // spawn (terra box top y=0 spans x∈[-200,200] z∈[8,68]); a gate hole here is
    // a spot where the RAW query finds solid floor but the patch-gate rejects
    // it — exactly the ruling-6 fall-through mechanism, if it exists.
    let mut holes = 0usize;
    let mut interior_holes = 0usize; // holes NOT on the terra rim (z in (8,68))
    let mut worst: Vec<(f32, f32, f32, String)> = Vec::new();
    let (mut x0, x1, mut z0, z1, step) = (-40.0f32, 40.0f32, 8.0f32, 68.0f32, 0.5f32);
    let _ = (&mut x0, &mut z0);
    let mut x = x0;
    while x <= x1 + 1e-4 {
        let mut z = z0;
        while z <= z1 + 1e-4 {
            let (raw, gated) = probe(ground, pp, x, z);
            // Is there plaza-level floor here at all (raw)?
            let raw_on_plaza = raw.map(|r| (r - PLAZA_Y).abs() <= FALL_EPS).unwrap_or(false);
            if raw_on_plaza {
                let g_ok = gated.map(|g| (g - PLAZA_Y).abs() <= FALL_EPS).unwrap_or(false);
                if !g_ok {
                    holes += 1;
                    // The terra plate top spans z in [8,68]; a hole strictly
                    // inside that band (not on the z=8 / z=68 rim line) would be
                    // a genuine interior patch-gate hole. Rim holes are the
                    // designed plate edge (step-down to the sea plate / world
                    // boundary), not fall-throughs.
                    if z > 8.0 + 1e-3 && z < 68.0 - 1e-3 {
                        interior_holes += 1;
                    }
                    let desc = match gated {
                        None => "gated=NONE".to_string(),
                        Some(g) => format!("gated={g:.3}"),
                    };
                    if worst.len() < 60 {
                        worst.push((x, z, raw.unwrap(), desc));
                    }
                }
            }
            z += step;
        }
        x += step;
    }
    println!(
        "grid {}x{} step {step}: {holes} patch-gate hole cells (raw plaza, gated not) — \
         {interior_holes} INTERIOR (z in (8,68)), rest on the terra rim (z=8 step-down / z=68 edge)",
        ((x1 - x0) / step) as i32 + 1,
        ((z1 - z0) / step) as i32 + 1
    );
    for (x, z, raw, desc) in worst.iter().take(60) {
        println!("  HOLE ({x:6.2},{z:6.2}) raw={raw:.3} {desc}");
    }
}

struct Walk {
    name: &'static str,
    start: (f32, f32),
    key: Key, // travel direction key (with yaw 0: Forward=-z, Right=+x)
    yaw: f32,
    ticks: u32,
}

fn run_walk(ground: &Ground, pp: &PlayerParams, w: &Walk) -> Option<(f32, f32, f32, Fall)> {
    // Spawn a little above the plaza and settle grounded.
    let eye = Vec3::new(w.start.0, PLAZA_Y + pp.eye_stand + 0.3, w.start.1);
    let mut player = Player::new(*pp, eye, w.yaw);
    for _ in 0..120 {
        player.step(DT, ground);
    }
    if !player.grounded {
        println!(
            "  [{}] WARN did not settle grounded at start ({:.2},{:.2}) feetY={:.3}",
            w.name,
            w.start.0,
            w.start.1,
            player.position.y - player.eye_height
        );
    }
    player.keys.insert(w.key);
    // Every 40 ticks, tap jump to exercise the run+jump landing snap band (d).
    for t in 0..w.ticks {
        if t % 40 == 20 { player.keys.insert(Key::Jump); } else { player.keys.remove(&Key::Jump); }
        player.step(DT, ground);
        let feet = player.position.y - player.eye_height;
        // Trigger on the FIRST tick below the plaza slab, then classify by
        // settling: step-down onto the sea plate vs true off-world edge vs a
        // genuine tunnel through supported floor.
        if feet < PLAZA_Y - FALL_EPS {
            let (fx, fz) = (player.position.x, player.position.z);
            let fall = classify_fall(ground, pp, &mut player);
            return Some((fx, fz, feet, fall));
        }
    }
    None
}

/// Grid-start coverage: from every start cell, settle grounded then walk each
/// of the four cardinal keys (walk, run, and run+jump-tap variants). A flagged
/// tick is classified TUNNEL when a raw floor sits >0.3 m ABOVE the body's feet
/// (solid ground the body should be resting on but is now beneath = the real
/// bug) versus EDGE (raw floor at/below the feet = the body walked off a ledge).
fn coverage_sweep(ground: &Ground, pp: &PlayerParams) {
    // Focused on the REPORTED region — the naruko plaza/spawn walk box with a
    // few metres of margin — not the whole realm. Dense grid starts, walk each
    // cardinal in walk/run/run+jump. Anti-hang: per-column progress, a derived
    // max-walk bail, and a wall-clock budget that dumps state and returns
    // rather than stalling silently (the failure mode that killed the last run).
    println!("\n== COVERAGE SWEEP (grid starts, walk/run/jump, classified) ==");
    // Whole reachable realm around spawn (terra top spans x∈[-200,200]
    // z∈[8,68]; the walk box is a subset). Start step 3 with 100-tick walks
    // (~10–23 m) overlaps coverage between starts. EDGE coords are captured so
    // a genuine interior hole is distinguishable from the terra rim.
    let (x0, x1, z0, z1, step) = (-30.0f32, 30.0f32, 8.0f32, 66.0f32, 3.0f32);
    let settle = 45u32; // flat plaza grounds a 0.2 m spawn well under this
    let walk_ticks = 100u32; // ~10–23 m of travel per walk
    let budget = std::time::Duration::from_secs(140);
    let start_clock = Instant::now();
    let dirs = [("W", Key::Forward), ("S", Key::Back), ("A", Key::Left), ("D", Key::Right)];
    let modes: [(&str, bool, bool); 3] =
        [("walk", false, false), ("run", true, false), ("runjump", true, true)];
    let mut tunnels = 0usize;
    let mut edges = 0usize;
    let mut walks_run = 0usize;
    let mut tunnel_hits: Vec<(f32, f32, f32, f32, &str)> = Vec::new();
    let mut edge_hits: Vec<(f32, f32, f32)> = Vec::new(); // sx,sz start + drop z
    let cols = ((x1 - x0) / step).round() as i32 + 1;
    let mut col = 0;
    let mut bailed = false;
    let mut sx = x0;
    'outer: while sx <= x1 + 1e-4 {
        col += 1;
        let mut sz = z0;
        while sz <= z1 + 1e-4 {
            // Only start where there is genuine floor (skip sea/void starts).
            let (raw0, _g) = probe(ground, pp, sx, sz);
            if raw0.is_none() { sz += step; continue; }
            let start_floor = raw0.unwrap();
            for (_dn, dk) in dirs {
                for (mn, run, jump) in modes {
                    if start_clock.elapsed() > budget {
                        bailed = true;
                        break 'outer;
                    }
                    let eye = Vec3::new(sx, start_floor + pp.eye_stand + 0.2, sz);
                    let mut player = Player::new(*pp, eye, 0.0);
                    for _ in 0..settle { player.step(DT, ground); }
                    if !player.grounded { continue; }
                    player.keys.insert(dk);
                    if run { player.keys.insert(Key::Run); }
                    walks_run += 1;
                    for t in 0..walk_ticks {
                        if jump && t % 30 == 15 { player.keys.insert(Key::Jump); }
                        else { player.keys.remove(&Key::Jump); }
                        player.step(DT, ground);
                        let feet = player.position.y - player.eye_height;
                        // CHEAP trigger (no query): feet under the plaza slab, or
                        // a runaway downward velocity not yet caught by a floor.
                        // Only THEN pay the settle-classify to name it.
                        if feet < PLAZA_Y - FALL_EPS || player.vy < -8.0 {
                            let (fx, fz) = (player.position.x, player.position.z);
                            match classify_fall(ground, pp, &mut player) {
                                Fall::Tunnel(gy) => {
                                    tunnels += 1;
                                    if tunnel_hits.len() < 30 {
                                        tunnel_hits.push((fx, fz, feet, gy, mn));
                                    }
                                }
                                Fall::StepDown | Fall::OffWorld => {
                                    edges += 1;
                                    if edge_hits.len() < 40 {
                                        edge_hits.push((sx, sz, fz));
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
            sz += step;
        }
        println!(
            "  ... column {col}/{cols} (x={sx:.0}) done  walks={walks_run} tunnels={tunnels} edges={edges}  t={:.1}s",
            start_clock.elapsed().as_secs_f64()
        );
        sx += step;
    }
    if bailed {
        println!(
            "  !! WALL-CLOCK BAIL after {:.1}s at column {col}/{cols} (x={sx:.0}) — partial coverage",
            start_clock.elapsed().as_secs_f64()
        );
    }
    println!("TUNNEL events (body below a solid floor): {tunnels}");
    for (x, z, feet, r, mn) in &tunnel_hits {
        println!("  TUNNEL ({x:6.2},{z:6.2}) feetY={feet:.3} floor_above={r:.3} mode={mn}");
    }
    println!("EDGE drops (walked off a ledge, expected): {edges}");
    // Show the spread of edge drops: start cell -> z where the feet left the
    // ground. If every drop z clusters at the terra rim (z≈8 north / z≈68
    // south / |x|>terra) it's the realm boundary; interior z means a hole.
    for (sx, sz, dz) in edge_hits.iter().take(40) {
        println!("  edge start({sx:6.1},{sz:6.1}) -> dropped at z={dz:6.2}");
    }
    println!("coverage: {walks_run} walks driven in {:.1}s", start_clock.elapsed().as_secs_f64());
}

fn walk_sweep(ground: &Ground, pp: &PlayerParams) {
    println!("\n== WALK SWEEP ==");
    let mut walks: Vec<Walk> = Vec::new();
    // West→East lanes (yaw so Right=+x). Right key = +x at yaw 0.
    let mut z = 15.0f32;
    while z <= 35.0 + 1e-4 {
        walks.push(Walk { name: "EW-lane", start: (-10.0, z), key: Key::Right, yaw: 0.0, ticks: 240 });
        z += 1.0;
    }
    // South→North lanes (Forward=-z at yaw 0 → travels toward smaller z; start high z).
    let mut x = -10.0f32;
    while x <= 10.0 + 1e-4 {
        walks.push(Walk { name: "NS-lane", start: (x, 35.0), key: Key::Forward, yaw: 0.0, ticks: 260 });
        x += 1.0;
    }
    // Threaded lines through named features (run through their bases):
    // mirror base @(-6.5,28) rotY-0.5; chrome pedestal @(4.5,29.5); stall; mirror_minor @(3,18).
    walks.push(Walk { name: "thru-mirror", start: (-6.5, 40.0), key: Key::Forward, yaw: 0.0, ticks: 400 });
    walks.push(Walk { name: "thru-chrome", start: (4.5, 40.0), key: Key::Forward, yaw: 0.0, ticks: 400 });
    walks.push(Walk { name: "thru-mirror-x", start: (-12.0, 28.0), key: Key::Right, yaw: 0.0, ticks: 400 });
    walks.push(Walk { name: "thru-chrome-x", start: (-2.0, 29.5), key: Key::Right, yaw: 0.0, ticks: 400 });
    walks.push(Walk { name: "thru-minor", start: (3.0, 40.0), key: Key::Forward, yaw: 0.0, ticks: 400 });
    // Run variant (high-speed) straight through the mirror + chrome to probe
    // the snap-band tunneling suspect (d).
    // (Run key held alongside travel.)

    let mut fell = 0usize;
    let mut tunnels = 0usize;
    for w in &walks {
        if let Some((x, z, feet, fall)) = run_walk(ground, pp, w) {
            fell += 1;
            if matches!(fall, Fall::Tunnel(_)) { tunnels += 1; }
            println!(
                "  {} [{}] start({:.1},{:.1}) -> ({:.2},{:.2}) feetY={:.3} {:?}",
                if matches!(fall, Fall::Tunnel(_)) { "TUNNEL" } else { "edge" },
                w.name, w.start.0, w.start.1, x, z, feet, fall
            );
        }
    }
    // Run-speed pass on the two feature lines.
    for w in &[
        Walk { name: "RUN-thru-mirror", start: (-6.5, 42.0), key: Key::Forward, yaw: 0.0, ticks: 400 },
        Walk { name: "RUN-thru-chrome", start: (4.5, 42.0), key: Key::Forward, yaw: 0.0, ticks: 400 },
        Walk { name: "RUN-EW-mid", start: (-12.0, 25.0), key: Key::Right, yaw: 0.0, ticks: 400 },
    ] {
        let eye = Vec3::new(w.start.0, PLAZA_Y + pp.eye_stand + 0.3, w.start.1);
        let mut player = Player::new(*pp, eye, w.yaw);
        for _ in 0..120 { player.step(DT, ground); }
        player.keys.insert(w.key);
        player.keys.insert(Key::Run);
        for _ in 0..w.ticks {
            player.step(DT, ground);
            let feet = player.position.y - player.eye_height;
            // Cheap trigger: feet below the plaza slab, then settle-classify.
            if feet < PLAZA_Y - FALL_EPS {
                let (fx, fz) = (player.position.x, player.position.z);
                let fall = classify_fall(ground, pp, &mut player);
                fell += 1;
                if matches!(fall, Fall::Tunnel(_)) { tunnels += 1; }
                println!(
                    "  {} [{}] -> ({fx:.2},{fz:.2}) feetY={feet:.3} {fall:?}",
                    if matches!(fall, Fall::Tunnel(_)) { "TUNNEL" } else { "edge" }, w.name
                );
                break;
            }
        }
    }
    println!(
        "walks: {} lanes, {fell} below-plaza events ({tunnels} genuine TUNNELS, rest edge/step-down)",
        walks.len() + 3
    );
}

fn main() {
    let t0 = Instant::now();
    let (ground, pp) = build();
    println!(
        "naruko ground: {} walkable triangles; contact_radius={:.4} ground_snap={:.3} void_y={:.1}  (build {:.1}s)",
        ground.triangle_count(),
        pp.contact_radius,
        pp.ground_snap,
        pp.void_y,
        t0.elapsed().as_secs_f64()
    );
    let t = Instant::now(); height_field(&ground, &pp); println!("[height_field {:.1}s]", t.elapsed().as_secs_f64());
    let t = Instant::now(); static_map(&ground, &pp);   println!("[static_map {:.1}s]", t.elapsed().as_secs_f64());
    let t = Instant::now(); coverage_sweep(&ground, &pp); println!("[coverage_sweep {:.1}s]", t.elapsed().as_secs_f64());
    let t = Instant::now(); walk_sweep(&ground, &pp);   println!("[walk_sweep {:.1}s]", t.elapsed().as_secs_f64());
    println!("[TOTAL {:.1}s]", t0.elapsed().as_secs_f64());
}
