//! RITE V · V1 relic forge — SHE WALKS. The walker's velocity drives sama's
//! state machine; sama's pose drives the skin per tick. nari, mid-stride on the
//! Naruko seawall, lit by the SAME traced light as every surface — and her body
//! is real to that light, so it casts a traced shadow on the seawall (no shadow
//! code; the pleroma already traces occlusion). Loads the real Naruko realm and
//! renders three relics headlessly on the GPU:
//!
//!   proof/v1-contact.png  — the CONTACT tick (swing foot low, stance split)
//!   proof/v1-passing.png  — the PASSING tick (swing foot high, mid-swing)
//!   proof/v1-shadow.png   — an angle showing the ground darkening under her
//!
//! The two gait ticks are DERIVED from the walk cycle (`contact_passing_ticks`),
//! the SAME two the V1 ordeal proves distinct. Run:
//!   cargo run -p scrying-glass --release --example v1_nari

use std::path::Path;

use glam::Vec3 as GVec3;

use sama::GaitParams;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::scene::{
    Camera, RenderScene, SceneParameters, SunDefaults, contact_passing_ticks,
};
use vessel::{Body, Preset};

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
    eprintln!("[v1] wrote {}", path.display());
}

/// Drive a fresh scene's body to a pure walk pose whose limb configuration is
/// that of gait tick `cycle_tick` (adding one full cycle so the idle→walk blend
/// is long finished — the pose is then the exact procedural gait pose). Returns
/// the world-space triangle soup (realm + her posed body) ready to trace.
fn walked(
    params: &SceneParameters,
    cycle_tick: u64,
    cycle: u64,
) -> (RenderScene, Vec<scrying_glass::scene::LeafTriangle>) {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let mut scene = RenderScene::from_ecs(std::mem::take(&mut core.world), params).expect("scene");

    // Walk (default speed 6 m·s⁻¹) until the last emitted pose is at tick
    // `cycle_tick + cycle` — same phase as `cycle_tick`, blend complete.
    let target = cycle_tick + cycle;
    for _ in 0..=target {
        scene.command_bodies(6.0);
    }

    let mut tris = scene.leaf_triangles();
    tris.extend(scene.dynamic_leaf_triangles());
    (scene, tris)
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[v1] no GPU adapter on this host — cannot forge the relic");
    };

    let params = naruko_params();

    // The two representative gait ticks — derived from the walk cycle, the SAME
    // pair the V1 ordeal proves distinct.
    let body = Body::from_preset(&Preset::nari());
    let gait = GaitParams::walk();
    let cycle = (1.0 / (gait.cadence * gait.dt)).round() as u64;
    let (contact_tick, passing_tick) = contact_passing_ticks(&body, &gait);
    eprintln!("[v1] contact tick={contact_tick} passing tick={passing_tick} (cycle {cycle})");

    let (w, h) = (900u32, 600u32);
    let frames = 48u32;
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");

    // Front, camera above her head aiming down toward the pink horizon so her
    // whole mid-stride silhouette stands against the dawn band (V0's reframe).
    let cam_front = camera_at([0.0, 3.3, 24.0], [0.0, 2.2, 18.0], 46.0);
    // Shadow / contact: the sun sits up and to +x/+z, so her shadow falls to
    // -x/-z of her feet, onto the seawall TOP (y=1.4, the strip she stands on).
    // Look STEEPLY DOWN from the +x/+z (sun) side so the seawall-top plane fills
    // the frame — her boots planted on it, the dark cast-shadow streak lying on
    // the same lit strip beside her feet. This top-down framing is the anti-hover
    // proof: feet and shadow share one visible surface (no thin-ledge parallax).
    let cam_shadow = camera_at([2.6, 5.0, 20.6], [-0.9, 1.4, 17.4], 46.0);

    let shots = [
        (contact_tick, cam_front, "v1-contact.png"),
        (passing_tick, cam_front, "v1-passing.png"),
        (passing_tick, cam_shadow, "v1-shadow.png"),
    ];

    for (tick, cam, name) in shots {
        let (scene, tris) = walked(&params, tick, cycle);
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
        let img = resolve(&accum);
        write_png(&img, w, h, 1.0, &proof.join(name));
    }
    eprintln!("[v1] she walks.");
}
