//! RITE VII · VII-1 relic forge — THE WALKER CROSSES ONTO GENERATED GROUND.
//!
//! Two relics, on the GPU, under the ONE light pass (sun + sky):
//!
//!   proof/vii1-stand-first-ground.png — a WALKER settled by the REAL player
//!       physics ON `naruko_first_ground` (the canon terrain SIGIL, tile 0,2,
//!       seed 20260717 — no stored geometry), rendered FIRST-PERSON from the
//!       walker's own settled eye pose: the generated ground is what the
//!       standing body sees under and ahead of its feet. The settle drives
//!       `Player::step` against `Ground::from_positions(scene.leaf_positions())`
//!       — the same floor VII-1's ordeals walk — so the camera pose is the
//!       body's real rest on the generated field, not a hand-placed eye.
//!
//!   proof/vii1-seam.png — the seam DIORAMA (a flat authored shore meeting a
//!       gentle generated tile at a C0-matched edge), framed side-on so the
//!       authored ground and the generated ground read as one continuous
//!       surface — the seam the pose-trace ordeal crosses. The walker's
//!       settled feet column is printed to the log (read it beside the pixels).
//!
//! Run:  cargo run -p scrying-glass --release --example vii1_seam

use std::path::Path;

use glam::Vec3 as GVec3;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::player::{Ground, Player, PlayerParams};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

use crystal::{EcsWorld, load_world_dir};

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
    eprintln!("[vii1] wrote {}", path.display());
}

fn camera_from_pose(eye: GVec3, yaw: f32, pitch: f32) -> Camera {
    Camera {
        eye,
        yaw,
        pitch,
        fov_y_radians: 55.0_f32.to_radians(),
        near: 0.1,
        far: 4_000.0,
    }
}

fn camera_look(eye: [f32; 3], look_at: [f32; 3]) -> Camera {
    let f = (GVec3::from_array(look_at) - GVec3::from_array(eye)).normalize();
    Camera {
        eye: GVec3::from_array(eye),
        yaw: (-f.x).atan2(-f.z),
        pitch: f.y.asin(),
        fov_y_radians: 55.0_f32.to_radians(),
        near: 0.1,
        far: 4_000.0,
    }
}

/// Settle a walker from a spawn eye onto the given floor, returning its rest
/// pose. Drives the REAL `Player::step` (gravity + gated ground-follow).
fn settle(ground: &Ground, spawn_eye: GVec3, yaw: f32, ticks: u32) -> (GVec3, f32, bool, f32) {
    let params = PlayerParams::from_env().expect("player params");
    let mut player = Player::new(params, spawn_eye, yaw);
    let dt = 1.0 / 60.0;
    for _ in 0..ticks {
        player.step(dt, ground);
    }
    let p = player.pose();
    (
        p.position,
        p.eye_height,
        p.grounded,
        p.position.y - p.eye_height,
    )
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[vii1] no GPU adapter on this host — cannot forge the relic");
    };
    let (w, h) = (900u32, 600u32);
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let frames = 48u32;
    let exposure = 1.6;
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");
    let params = naruko_params();

    // ---- Relic 1: the walker stands on naruko_first_ground (canon tile) ----
    {
        let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
        let mut world = EcsWorld::default();
        load_world_dir(&world_path, &mut world).expect("load naruko");
        let scene = RenderScene::from_ecs(world, &params).expect("render scene");
        let ground = Ground::from_positions(&scene.leaf_positions());

        // Tile 0,2 default 64 m ⇒ origin (0,128), centre (32,·,160). Spawn the
        // eye above the centre and let the body fall onto the generated field.
        let spawn = GVec3::new(32.0, 25.0, 158.0);
        // Face into the tile interior (+x, +z), a touch downward, so the frame
        // holds the generated ground under and ahead of the feet.
        let yaw = -std::f32::consts::FRAC_PI_4; // look toward +x/−z blend
        let (eye, eye_h, grounded, feet_y) = settle(&ground, spawn, yaw, 400);
        eprintln!(
            "[vii1] walker settled on naruko_first_ground: eye=[{:.2},{:.2},{:.2}] feet_y={feet_y:.3} grounded={grounded} eye_h={eye_h:.2}",
            eye.x, eye.y, eye.z
        );
        assert!(
            grounded,
            "the walker must be grounded on the first ground before the shot"
        );

        let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
        let camera = camera_from_pose(eye, yaw, -0.32);
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
        write_png(
            &resolve(&accum),
            w,
            h,
            exposure,
            &proof.join("vii1-stand-first-ground.png"),
        );
    }

    // ---- Relic 2: the seam diorama (authored shore meets generated tile) ----
    {
        // A gentle generated tile (matches the ordeal diorama) with a flat
        // authored shore whose top meets the field's seam value along the
        // crossing column — the C0 seam the pose-trace ordeal walks.
        use seed::TerrainSigil;
        use seed::terrain::height_at_grid_index;
        let terrain_json = r##"{ "seed": 20260717, "tile_x": 0, "tile_y": 0, "height_amplitude": 1.5, "color": "#4a7c59" }"##;
        let sigil: TerrainSigil = serde_json::from_str(terrain_json).unwrap();
        let tparams = sigil.params();
        let n = tparams.grid_resolution as i64;
        let cell = tparams.cell_size_m();
        let world_seed = sigil.world_seed();
        let local_i = n / 2;
        let h_seam = height_at_grid_index(world_seed, &tparams, local_i, 0);
        let x0 = local_i as f32 * cell;
        let plane_center_y = h_seam - 0.25;

        let dir = std::env::temp_dir().join(format!("gaia_vii1relic_{}", std::process::id()));
        let scenes = dir.join("scenes");
        std::fs::create_dir_all(&scenes).unwrap();
        let doc = format!(
            r##"{{
              "authored_shore": {{
                "transform": {{ "position": [0, 0, 0] }},
                "mesh": {{ "parts": [
                  {{ "shape": "box", "size": [24, 0.5, 30], "position": [{x0}, {plane_center_y}, -15], "color": "#5a4a6c" }}
                ] }}
              }},
              "first_ground": {{ "terrain": {terrain_json} }}
            }}"##
        );
        std::fs::write(scenes.join("main.json"), &doc).unwrap();
        let mut world = EcsWorld::default();
        load_world_dir(&dir, &mut world).expect("load seam diorama");
        let scene = RenderScene::from_ecs(world, &params).expect("render seam");
        let ground = Ground::from_positions(&scene.leaf_positions());

        // Settle the walker on the authored shore at the crossing column and
        // print its feet column beside the pixels.
        let spawn = GVec3::new(x0, h_seam + 6.0, -6.0);
        let (eye, _eh, grounded, feet_y) = settle(&ground, spawn, std::f32::consts::PI, 300);
        eprintln!(
            "[vii1] seam diorama: walker feet=[{:.2},{:.3},{:.2}] grounded={grounded}; seam edge z=0 at h={h_seam:.3}",
            eye.x, feet_y, eye.z
        );

        // Side-on framing: eye off to +x, above, looking at the seam edge so
        // the flat shore (z<0) and the generated ground (z>0) meet in frame.
        let camera = camera_look([x0 + 34.0, 12.0, 6.0], [x0, h_seam, 4.0]);
        let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
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
        write_png(
            &resolve(&accum),
            w,
            h,
            exposure,
            &proof.join("vii1-seam.png"),
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    eprintln!("[vii1] the walker stands on ground no hand placed — read the relics with eyes.");
}
