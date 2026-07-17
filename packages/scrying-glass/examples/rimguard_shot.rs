//! RIM-GUARD SHOT — offline own-eyes render of the naruko south boundary
//! guard (realm-rimguard branch). From `world_spawn` looking south toward
//! the z=68 terra rim: confirms `naruko_seawall_south` is present, Carcosa-
//! palette consistent with the existing `naruko_seawall`, and correctly
//! placed against the plaza.
//!
//! Run: cargo run -p scrying-glass --release --example rimguard_shot

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

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
    if c <= 0.003_130_8 {
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
    eprintln!("[rimguard_shot] wrote {}", path.display());
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[rimguard_shot] no GPU adapter on this host");
    };

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();

    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let tris = scene.leaf_triangles();
    let bvh = Bvh::build(&tris, &BvhParams::default());
    eprintln!("[rimguard_shot] naruko: {} leaf triangles", tris.len());

    let (w, h) = (960u32, 640u32);
    let frames = 48u32;
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };

    // world_spawn: { position: [0, 7, 44], yaw: 0 } — eye pose the player
    // actually boots at. Looking south (+z, toward the z=68 rim / new
    // naruko_seawall_south) with a slight downward tilt to keep the wall
    // and plaza in frame.
    let cam = camera_at([0.0, 1.7, 50.0], [0.0, 1.0, 67.0], 55.0);

    let accum = trace_headless(
        &device,
        &queue,
        &bvh,
        &cam,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        w,
        h,
        frames,
        &int_params,
        None,
    );
    let img = resolve(&accum);
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");
    write_png(&img, w, h, 1.0, &proof.join("rimguard-south.png"));

    let raw = trace_headless_aov(&device, &queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, w, h);
    let (albedo, normal, depth) = split_aov(&raw);
    write_png(&albedo, w, h, 1.0, &proof.join("rimguard-south-albedo.png"));
    let norm_disp: Vec<GVec3> = normal.iter().map(|n| (*n + GVec3::ONE) * 0.5).collect();
    write_png(&norm_disp, w, h, 1.0, &proof.join("rimguard-south-normal.png"));
    let maxd = depth.iter().cloned().fold(0.0f32, f32::max).max(1.0);
    let depth_disp: Vec<GVec3> = depth.iter().map(|d| GVec3::splat(d / maxd)).collect();
    write_png(&depth_disp, w, h, 1.0, &proof.join("rimguard-south-depth.png"));
    eprintln!("[rimguard_shot] done. max depth={maxd:.2}");
}
