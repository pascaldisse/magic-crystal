//! RITE V · V0 relic forge — THE BODY STANDS. nari, the embodied vessel, stands
//! on the Naruko seawall against the pink dawn, lit by the SAME traced light as
//! every surface. Her body is the `body` sigil composed by the engine: the nari
//! vessel skinned over the homunculus skeleton at sama's canonical idle pose
//! (tick 0), spliced into the traced BVH's dynamic partition. Loads the real
//! Naruko realm and renders two relics headlessly on the GPU:
//!
//!   proof/v0-nari.png        — front, framing her on the seawall vs the sky
//!   proof/v0-nari-orbit.png  — the same body from another shoulder
//!
//! Run:  cargo run -p scrying-glass --release --example v0_nari

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
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
    eprintln!("[v0] wrote {}", path.display());
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[v0] no GPU adapter on this host — cannot forge the relic");
    };

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();

    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    eprintln!(
        "[v0] embodied vessels: {}",
        scene
            .bodies
            .iter()
            .map(|b| format!("{}({} tris)", b.gaia_id, b.world_tris.len()))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Static + dynamic (living layer + the embodied body) geometry into one BVH,
    // exactly as the window builds it.
    let mut tris = scene.leaf_triangles();
    tris.extend(scene.dynamic_leaf_triangles());
    let bvh = Bvh::build(&tris, &BvhParams::default());
    eprintln!(
        "[v0] naruko: {} leaf triangles (incl. the body)",
        tris.len()
    );

    let (w, h) = (900u32, 600u32);
    let frames = 48u32;
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };

    // nari stands on the seawall, feet y=1.4, head ~y=3.5 (mid ~2.45). Frame her
    // whole body against the pink horizon, camera near her own height so the
    // sky/sea horizon line sits behind her silhouette.
    // Front: straight on from the sea side, camera above her head aiming down
    // toward the horizon so her whole silhouette (obsidian head, dark seifuku)
    // stands against the PINK dawn band, not the near-black upper sky.
    let cam_front = camera_at([0.0, 3.3, 24.0], [0.0, 2.2, 18.0], 46.0);
    // Orbit: from her right shoulder, over the open sea (the left side is
    // crowded by the stall massing).
    let cam_orbit = camera_at([4.6, 2.7, 23.5], [0.0, 2.45, 18.0], 46.0);

    for (cam, name) in [(cam_front, "v0-nari.png"), (cam_orbit, "v0-nari-orbit.png")] {
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
        write_png(&img, w, h, 1.0, &proof.join(name));
    }
    eprintln!("[v0] the body stands.");
}
