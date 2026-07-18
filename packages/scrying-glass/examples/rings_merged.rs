//! SIGNAL RINGS relic forge — re-rendered against the MERGED realm (main's
//! V2 cat + N2 presences + the three signal rings). Loads the real Naruko
//! realm and renders the lighthouse head-on so the three violet concentric
//! ring vessels (signal_ring_a/b/c) read around the beacon.
//!
//!   proof/rings-merged.png — the lighthouse broadcasts: three violet rings
//!                            (R = 6/10/14) about the beacon axis [0,56.5,-117],
//!                            the pulse expanded at t ≈ 2 s.
//!
//! Run: cargo run -p scrying-glass --release --example rings_merged

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

fn naruko_params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: 45.0,
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
            sun_intensity: 0.7,
            sun_position: [60.0, 90.0, 30.0],
            ambient_intensity: 0.22,
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
    eprintln!("[rings] wrote {}", path.display());
}

fn ticked(
    params: &SceneParameters,
    ticks: u64,
) -> (RenderScene, Vec<scrying_glass::scene::LeafTriangle>) {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let mut scene = RenderScene::from_ecs(std::mem::take(&mut core.world), params).expect("scene");
    for _ in 0..ticks {
        scene.command_bodies(0.0);
        scene.tick();
    }
    eprintln!(
        "[rings] dynamic entities = {} (lantern + beacon + ring a/b/c + crate)",
        scene.dynamics.entities().len()
    );
    let mut tris = scene.leaf_triangles();
    tris.extend(scene.dynamic_leaf_triangles());
    (scene, tris)
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[rings] no GPU adapter on this host — cannot forge the relic");
    };
    let params = naruko_params();
    let (w, h) = (960u32, 640u32);
    let frames = 64u32;
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 5,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");

    // Head-on at the lighthouse: eye on the beacon axis (x=0), below and ahead
    // of the tower, looking up the tower toward the beacon [0,56.5,-120] so the
    // three violet rings about [0,56.5,z] read as concentric circles.
    let cam = camera_at([0.0, 34.0, -12.0], [0.0, 52.0, -117.0], 45.0);
    // t ≈ 2 s → 120 ticks: the pulse has driven the ring radii outward.
    let (scene, tris) = ticked(&params, 120);
    let bvh = Bvh::build(&tris, &BvhParams::default());
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
    write_png(&resolve(&accum), w, h, 1.0, &proof.join("rings-merged.png"));
}
