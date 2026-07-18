//! ELEMENTS P3 relic forge — THE FIRST THING FALLS. A wooden crate (a realm
//! `body`) is hung above the Naruko pier near the ramen stall; the world tick
//! runs the Elements' rigid solver, collides it against the STATIC realm mesh,
//! and writes its moving pose back — so its triangles ride the dynamic BVH
//! splice and the ONE traced light sees it fall, impact, and come to rest on
//! the planks. Three FIXED-TICK renders, on the real realm, on the GPU:
//!
//!   proof/p3-falling.png  — the crate mid-air above the pier
//!   proof/p3-impact.png   — the crate meeting the planks
//!   proof/p3-rest.png     — the crate settled on the pier
//!
//! Determinism: the tick index is the entropy coordinate; two runs render the
//! same frames. Run:  cargo run -p scrying-glass --release --example p3_crate

use std::path::Path;

use glam::Vec3 as GVec3;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

use crystal::{EcsWorld, load_world_dir};

/// Naruko authoring dials (mirror the window / a2 defaults).
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
    eprintln!("[p3] wrote {}", path.display());
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[p3] no GPU adapter on this host — cannot forge the relic");
    };

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut world = EcsWorld::default();
    load_world_dir(&world_path, &mut world).expect("load naruko");
    let params = naruko_params();
    let mut scene = RenderScene::from_ecs(world, &params).expect("render scene");
    eprintln!(
        "[p3] naruko: {} static leaf tris, {} declared bod(ies)",
        scene.leaf_triangles().len(),
        scene.physics().map(|p| p.bindings().len()).unwrap_or(0),
    );

    // The static BVH is built once (the pier, the realm); the crate rides the
    // DYNAMIC splice re-built each captured tick.
    let bvh_params = BvhParams::default();
    let static_bvh = Bvh::build(&scene.leaf_triangles(), &bvh_params);

    // The camera: a three-quarter view down the pier from the stall side, the
    // crate + chrome orb in the near foreground against the night sea.
    let camera = camera_at([-5.5, 4.2, 21.0], [-11.2, 1.8, 13.0], 55.0);

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

    // Fixed tick stops: mid-air, meeting the planks, settled (the crate settles
    // ≈ tick 91; see the physics ordeal). Rendered in sequence on ONE scene, so
    // the state carries (the entropy coordinate advances).
    let stops = [
        ("p3-falling.png", 25u64),
        ("p3-impact.png", 62u64),
        ("p3-rest.png", 150u64),
    ];
    let mut current = 0u64;
    for (name, target) in stops {
        while current < target {
            scene.tick();
            current += 1;
        }
        let crate_pos = scene.body_position("naruko_crate").expect("crate body");
        let dyn_bvh = Bvh::build(&scene.dynamic_leaf_triangles(), &bvh_params);
        let bvh = Bvh::merge(&static_bvh, &dyn_bvh);
        eprintln!(
            "[p3] tick {current}: crate at [{:.3}, {:.3}, {:.3}]  (merged BVH {} tris)",
            crate_pos[0],
            crate_pos[1],
            crate_pos[2],
            bvh.tris.len(),
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
    }
    eprintln!("[p3] three relics forged — read them with eyes.");
}
