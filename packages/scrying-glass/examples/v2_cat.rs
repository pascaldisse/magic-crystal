//! RITE V · V2 relic forge — THE PINK CAT, and the family finale.
//!
//! Loads the real Naruko realm and renders two relics headlessly on the GPU:
//!
//!   proof/v2-cat.png     — CLOSE: the pink cat sitting by the ramen stall,
//!                          the real lantern's pink light touching her.
//!   proof/v2-family.png  — THE RITE FINALE: one composed shot holding all
//!                          three embodied ones — nari mid-stride on the
//!                          seawall, the crate at rest on the pier, the pink
//!                          cat by the stall.
//!
//! The cat animates from the world clock (her kami idle loop); nari walks from
//! the walker velocity; the crate settles under the Elements' physics as the
//! world ticks. Run:
//!   cargo run -p scrying-glass --release --example v2_cat

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
    eprintln!("[v2] wrote {}", path.display());
}

/// A fresh Naruko scene, ticked `ticks` times with the walker driving nari at
/// `nari_speed` (the cat ignores it — she runs her own clock loop — and the
/// crate settles under physics as the clock advances). Returns the scene and
/// the full world-space triangle soup (realm + every posed body).
fn ticked(
    params: &SceneParameters,
    ticks: u64,
    nari_speed: f32,
) -> (RenderScene, Vec<scrying_glass::scene::LeafTriangle>) {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let mut scene = RenderScene::from_ecs(std::mem::take(&mut core.world), params).expect("scene");

    for _ in 0..ticks {
        scene.command_bodies(nari_speed);
        scene.tick();
    }
    // Report where the embodied ones ended up (honest placement log).
    for b in &scene.bodies {
        let o = b.world_origin();
        eprintln!(
            "[v2] body {:>10} preset={:<8} minded={} origin=[{:.2},{:.2},{:.2}] speed={:.2}",
            b.gaia_id,
            b.preset,
            b.is_minded(),
            o[0],
            o[1],
            o[2],
            b.commanded_speed()
        );
    }
    if let Some(p) = scene.physics() {
        for binding in p.bindings() {
            let pose = p.pose(binding);
            eprintln!(
                "[v2] crate {:>12} rest=[{:.2},{:.2},{:.2}]",
                binding.gaia_id, pose.position[0], pose.position[1], pose.position[2]
            );
        }
    }

    let mut tris = scene.leaf_triangles();
    tris.extend(scene.dynamic_leaf_triangles());
    (scene, tris)
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[v2] no GPU adapter on this host — cannot forge the relic");
    };
    let params = naruko_params();
    let (w, h) = (900u32, 600u32);
    let frames = 64u32;
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 5,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");

    // CLOSE cat shot — she is SITTING at home (tick 30 ≈ 0.5 s < her 3 s sit),
    // by the stall, the lantern (glow at [-7.5,3.5,20]) up and to her right.
    // Camera close, low, looking across her toward the lantern so its pink
    // spill lands on her lit flank.
    let cam_cat = camera_at([-2.7, 1.05, 25.6], [-5.6, 0.7, 21.4], 38.0);
    let (_scene_c, tris_c) = ticked(&params, 30, 0.0);
    let bvh_c = Bvh::build(&tris_c, &BvhParams::default());
    let accum_c = trace_headless(
        &device,
        &queue,
        &bvh_c,
        &cam_cat,
        &_scene_c.sun,
        _scene_c.sky_top,
        _scene_c.sky_horizon,
        w,
        h,
        frames,
        &int_params,
        None,
    );
    write_png(&resolve(&accum_c), w, h, 1.0, &proof.join("v2-cat.png"));

    // FAMILY FINALE — nari walking (speed 6, mid-stride), the crate settled on
    // the pier (physics over ~150 ticks), the cat still sitting by the stall.
    // A HIGH camera looking down over the stall roof so the stall never hides
    // nari (she sits on the seawall z=18, on the far side of the stall z=23–27
    // from a +z camera): the steep sightline clears the 2.9 m roof. Holds all
    // three — nari [0,~2.5,18], cat [-5,~0.4,23], crate [~-11.15,~1.5,13].
    let cam_family = camera_at([-5.4, 8.5, 31.0], [-5.4, 0.8, 17.0], 55.0);
    let (_scene_f, tris_f) = ticked(&params, 150, 6.0);
    let bvh_f = Bvh::build(&tris_f, &BvhParams::default());
    let accum_f = trace_headless(
        &device,
        &queue,
        &bvh_f,
        &cam_family,
        &_scene_f.sun,
        _scene_f.sky_top,
        _scene_f.sky_horizon,
        w,
        h,
        frames,
        &int_params,
        None,
    );
    write_png(&resolve(&accum_f), w, h, 1.0, &proof.join("v2-family.png"));

    eprintln!("[v2] the pink cat lives; the family stands.");
}
