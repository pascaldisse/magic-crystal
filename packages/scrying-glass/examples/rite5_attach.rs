//! RITE V FINAL WELD relic forge — THE BODY JOINS THE WALKER. nari's body
//! declares `follows: "walker"` in worlds/naruko, so she is ATTACHED: her
//! position/yaw track the walker each tick and her gait is DERIVED from the
//! per-tick displacement. Here the walker is SCRIPTED to walk down the seawall
//! to a visibly different spot, and her body is THERE, mid-stride, casting a
//! traced shadow on the ground beneath its NEW position (no shadow code; the
//! pleroma already traces occlusion). Loads the real Naruko realm and renders
//! two relics headlessly on the GPU:
//!
//!   proof/attach-walked.png  — the walker walked +x down the seawall; her body
//!                              THERE (x≈16), mid-stride, not at her authored x=0
//!   proof/attach-shadow.png  — a steep angle showing the ground darkening under
//!                              the walked-to body (her shadow follows her)
//!
//! Run: cargo run -p scrying-glass --release --example rite5_attach

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults, WalkerPose};

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
    eprintln!("[weld] wrote {}", path.display());
}

/// Load naruko and SCRIPT the walker walking +x down the seawall band (z=18) to
/// `target_x`, at a walk-pace 0.1 m/tick (≈6 m·s⁻¹, so the derived gait is a
/// real walk). nari's `follows: "walker"` body TRACKS it; the walk stops one
/// step short of a plant so she is mid-stride. Returns the scene and the traced
/// soup (realm + her posed body at the walked-to spot).
fn walked_to(
    params: &SceneParameters,
    target_x: f32,
) -> (RenderScene, Vec<scrying_glass::scene::LeafTriangle>) {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let mut scene = RenderScene::from_ecs(std::mem::take(&mut core.world), params).expect("scene");

    let step = 0.1_f32;
    let ticks = (target_x / step).round() as usize;
    for k in 0..=ticks {
        let pose = WalkerPose {
            position: GVec3::new(step * k as f32, 2.5, 18.0),
            yaw: 0.0,
        };
        // Broadcast 0: the attachment (not the broadcast) drives her gait.
        scene.command_bodies_walked(0.0, Some(pose));
        scene.tick();
    }
    let end = scene.bodies.iter().find(|b| b.gaia_id == "nari").unwrap();
    eprintln!(
        "[weld] walker scripted to x={target_x:.1}; nari's body at {:?} (speed {:.2} m/s)",
        end.world_origin(),
        end.commanded_speed(),
    );

    let mut tris = scene.leaf_triangles();
    tris.extend(scene.dynamic_leaf_triangles());
    (scene, tris)
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[weld] no GPU adapter on this host — cannot forge the relic");
    };

    let params = naruko_params();
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

    // She walks +x down the seawall to x≈16 — plainly displaced from her
    // authored x=0 (the empty seawall stretches behind her), and grounded on the
    // flat seawall band there (feet on y=1.4).
    let target_x = 16.0_f32;
    let (scene, tris) = walked_to(&params, target_x);
    let bvh = Bvh::build(&tris, &BvhParams::default());

    // WALKED shot: front-ish, framed on her walked-to position so the vacated
    // seawall to -x is visible behind her (she is here now, not there).
    let cam_walked = camera_at([target_x + 2.0, 3.3, 24.0], [target_x, 2.2, 18.0], 50.0);
    // SHADOW shot: steep top-down from the +x/+z (sun) side over her walked-to
    // feet, so the seawall-top plane fills the frame with her cast-shadow streak
    // lying on the same lit strip beside her boots (anti-hover framing).
    let cam_shadow = camera_at(
        [target_x + 2.6, 5.0, 20.6],
        [target_x - 0.9, 1.4, 17.4],
        46.0,
    );

    for (cam, name) in [
        (cam_walked, "attach-walked.png"),
        (cam_shadow, "attach-shadow.png"),
    ] {
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
    eprintln!("[weld] the body joins the walker — she walks where he walks.");
}
