//! RITE VII · VII-2 relic forge — THE HORIZON STREAMS.
//!
//! A walker, MID-JOURNEY across a world that was never stored, stands on
//! generated ground and looks out at a horizon held by DATA-DRIVEN residency
//! (`scrying_glass::horizon::HorizonRing`): the tiles under and ahead of the
//! feet materialized from `(seed, coords)` as the observer advanced; the tiles
//! behind were evicted; the whole resident set sits under a hard byte budget.
//!
//!   proof/vii2-horizon-near.png  — from tile ~20 of the journey, the walker
//!       settled by the REAL player physics on the resident ground, looking
//!       along the walk direction: the streamed horizon of generated hills.
//!   proof/vii2-horizon-far.png   — the SAME ring driven to PLANETARY tile
//!       magnitude (±10,000,000), render_origin rebased camera-relative: the
//!       identical horizon, proving the ground streams the same 10^8 m from
//!       the origin as it does beside it (ruling 4 paid end to end).
//!
//! Run:  cargo run -p scrying-glass --release --example vii2_horizon

use std::path::Path;

use glam::Vec3;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::horizon::{tile_byte_cost, HorizonRing};
use scrying_glass::integrator::{headless_device, resolve, trace_headless, IntegratorParams};
use scrying_glass::player::{Ground, Player, PlayerParams};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};
use seed::terrain::{tile_origin_m, TerrainParams, TerrainTile};

const SEED: u64 = 20260717;
const TICK_DT: f32 = 1.0 / 60.0;

fn scene_params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: 60.0,
        near: 0.1,
        far: 4_000.0,
        sky_top: "#243b6b".into(),
        sky_horizon: "#c9a26b".into(),
        mesh_color: "#6f8f5f".into(),
        radial_segments: 24,
        camera_position: [0.0, 2.0, 22.0],
        camera_yaw: 0.0,
        camera_pitch: 0.0,
        cluster_error_threshold: 1.0,
        tick_dt: TICK_DT as f64,
        sun: SunDefaults {
            sun_color: "#ffe6bd".into(),
            sun_intensity: 1.25,
            sun_position: [220.0, 120.0, 60.0],
            ambient_intensity: 0.34,
        },
        emission_intensity: 2.5,
    }
}

/// Rolling, walkable terrain with visible relief for the horizon shot.
fn horizon_terrain() -> TerrainParams {
    let mut p = TerrainParams::derive(48.0);
    p.grid_resolution = 16;
    p.height_amplitude = 4.0;
    p
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn write_png(img: &[Vec3], w: u32, h: u32, exposure: f32, path: &Path) {
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
    eprintln!("[vii2] wrote {}", path.display());
}

/// Settle a walker from a spawn eye onto the given floor (REAL `Player::step`).
fn settle(ground: &Ground, spawn_eye: Vec3, yaw: f32, ticks: u32) -> (Vec3, bool, f32) {
    let params = PlayerParams::from_env().expect("player params");
    let mut player = Player::new(params, spawn_eye, yaw);
    for _ in 0..ticks {
        player.step(TICK_DT, ground);
    }
    let p = player.pose();
    (p.position, p.grounded, p.position.y - p.eye_height)
}

/// Drive the ring from a start tile forward `journey_tiles` tiles along +x,
/// updating residency each step exactly as a walking observer would — then
/// return the resident scene + walker eye at the mid-journey position, placed
/// camera-relative to the observer's tile origin.
fn horizon_shot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    start_tile: TerrainTile,
    journey_tiles: i64,
    out: &Path,
) {
    let sp = scene_params();
    let tparams = horizon_terrain();
    let tb = tile_byte_cost(&tparams);
    // Budget sized for a 17×17 = 289-tile residency square (radius 8): the
    // horizon reaches 8·48 = 384 m in every direction.
    let budget = 289 * tb;
    let mut ring = HorizonRing::new(SEED, tparams, Some("#6f8f5f".into()), budget).expect("ring");
    eprintln!(
        "[vii2] ring: budget {} B, tile {} B, radius {} tiles ({} m reach)",
        ring.budget_bytes(),
        ring.tile_bytes(),
        ring.radius_tiles(),
        ring.radius_tiles() as f32 * ring.params().tile_size_m
    );

    let ts = ring.params().tile_size_m as f64;
    let (sx, _) = tile_origin_m(start_tile, ring.params());
    let (_, sz) = tile_origin_m(start_tile, ring.params());
    // Walk the observer forward along +x from the start tile's centre.
    let z = sz + ts * 0.5;
    let v = PlayerParams::from_env().expect("player params").walk_speed as f64;
    let target_x = sx + journey_tiles as f64 * ts + ts * 0.5;
    let mut x = sx + ts * 0.5;
    while x < target_x {
        x += v * TICK_DT as f64;
        ring.update(x, z);
    }
    let world_x = x;
    let world_z = z;
    eprintln!(
        "[vii2] mid-journey: observer world x={world_x:.1} z={world_z:.1}; resident {} tiles = {} B (peak {} B) ≤ budget {} B; {} materializations, {} evictions",
        ring.resident_count(),
        ring.resident_bytes(),
        ring.stats().peak_resident_bytes,
        ring.budget_bytes(),
        ring.stats().loads,
        ring.stats().evictions
    );

    // Camera-relative to the observer's current tile origin.
    let origin = ring.render_origin_for(world_x, world_z);
    let scene: RenderScene = ring.scene_at(origin, &sp).expect("weld resident tiles");
    let ground = Ground::from_positions(&scene.leaf_positions());

    // Settle the walker at the observer position, in LOCAL coordinates.
    let local_spawn = Vec3::new(
        (world_x - origin[0]) as f32,
        30.0,
        (world_z - origin[2]) as f32,
    );
    let yaw = std::f32::consts::FRAC_PI_2; // face +x, the walk direction
    let (eye, grounded, feet_y) = settle(&ground, local_spawn, yaw, 500);
    eprintln!(
        "[vii2] walker settled: local eye=[{:.2},{:.2},{:.2}] feet_y={feet_y:.2} grounded={grounded}",
        eye.x, eye.y, eye.z
    );
    assert!(grounded, "the walker must be grounded before the horizon shot");

    // Look along +x, tilted slightly down, so the streamed horizon of generated
    // hills fills the frame ahead of the feet.
    let camera = Camera {
        eye: eye + Vec3::new(0.0, 0.6, 0.0),
        yaw,
        pitch: -0.14,
        fov_y_radians: sp.fov_y_degrees.to_radians(),
        near: sp.near,
        far: sp.far,
    };

    let (w, h) = (960u32, 600u32);
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    let accum = trace_headless(
        device,
        queue,
        &bvh,
        &camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        w,
        h,
        48,
        &int_params,
        None,
    );
    write_png(&resolve(&accum), w, h, 1.5, out);
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[vii2] no GPU adapter on this host — cannot forge the relic");
    };
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");

    // NEAR: mid-journey a short way from the world origin.
    horizon_shot(
        &device,
        &queue,
        TerrainTile::new(0, 0),
        20,
        &proof.join("vii2-horizon-near.png"),
    );

    // FAR: the SAME journey begun at planetary tile magnitude (ruling 4).
    horizon_shot(
        &device,
        &queue,
        TerrainTile::new(10_000_000, -10_000_000),
        20,
        &proof.join("vii2-horizon-far.png"),
    );

    eprintln!("[vii2] the horizon streams around a walker on ground no hand stored — read the relics with eyes.");
}
