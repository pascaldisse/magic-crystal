//! WELD REPAIR — OWN-BODY CULL relic forge. `follows: "walker"` is restored
//! (the interim 31bae8b detach reverted), so nari's body is welded exactly to
//! the walker's own pose again — which would render her mesh INSIDE the
//! first-person camera (the Architect-blinding bug her interim detach parked)
//! unless the render path culls it FROM THAT EXACT EYE. This forges two
//! headless GPU relics proving the cull at the render seam
//! (`RenderScene::dynamic_leaf_triangles_for_eye`, spliced in by both
//! `main.rs`'s `advance_world` and `capture_pose`):
//!
//!   proof/own-eye-default.png — camera AT the walker's own eye (the exact
//!                               pose fed to `command_bodies_walked`): her
//!                               body is ABSENT — the view is clear, matching
//!                               ordinary first-person play (no more blind).
//!   proof/own-eye-side.png    — a FOREIGN eye a few metres off (a moving
//!                               `/scry?pos=...` or diorama camera): her body
//!                               IS drawn, standing exactly at the walker's
//!                               position — the weld still holds, just not
//!                               drawn into the walker's own eye.
//!
//! Run: cargo run -p scrying-glass --release --example rite5_own_eye

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::scene::{
    Camera, OWN_EYE_EPSILON_M, RenderScene, SceneParameters, SunDefaults, WalkerPose,
};

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
    eprintln!("[own-eye] wrote {}", path.display());
}

fn load_naruko() -> (RenderScene, SceneParameters) {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();
    let scene =
        RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("render scene");
    (scene, params)
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[own-eye] no GPU adapter on this host — cannot forge the relic");
    };

    let (mut scene, _params) = load_naruko();

    // The walker stands at nari's authored spot (her welded body's own place)
    // and drives ONE tick — `last_walker_eye` now holds this EXACT pose, the
    // identity `dynamic_leaf_triangles_for_eye` culls against.
    let walker = WalkerPose {
        position: GVec3::new(0.0, 2.5, 18.0),
        yaw: 0.0,
    };
    scene.command_bodies_walked(0.0, Some(walker));
    let nari_tris = scene
        .bodies
        .iter()
        .find(|b| b.gaia_id == "nari")
        .expect("nari body")
        .world_tris
        .len();
    eprintln!("[own-eye] nari's welded body carries {nari_tris} tris at the walker's own pose");

    let statics = scene.leaf_triangles();
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

    // DEFAULT-EYE shot: the render camera IS the walker's own eye, looking
    // straight down the seawall (+x) the way ordinary first-person play does.
    let default_tris = scene.dynamic_leaf_triangles_for_eye(
        walker.position,
        OWN_EYE_EPSILON_M,
        /* force_draw */ false,
    );
    assert_eq!(
        default_tris.len(),
        scene.dynamic_leaf_triangles().len() - nari_tris,
        "own-eye inventory must be short by exactly nari's tris"
    );
    let mut tris_default = statics.clone();
    tris_default.extend(default_tris);
    let bvh_default = Bvh::build(&tris_default, &BvhParams::default());
    let cam_default = camera_at(
        walker.position.to_array(),
        [walker.position.x, walker.position.y - 0.3, walker.position.z - 10.0],
        55.0,
    );

    // SIDE-EYE shot: a FOREIGN eye 4 m off the walker's own pose (a moving
    // `/scry?pos=...`, or a diorama camera) — nari's body is drawn, standing
    // exactly where the walker stands.
    let side_eye = GVec3::new(walker.position.x + 4.0, 3.3, walker.position.z + 6.0);
    let side_tris = scene.dynamic_leaf_triangles_for_eye(
        side_eye,
        OWN_EYE_EPSILON_M,
        /* force_draw */ false,
    );
    assert_eq!(
        side_tris.len(),
        scene.dynamic_leaf_triangles().len(),
        "a foreign eye must see the full inventory, nari included"
    );
    let mut tris_side = statics;
    tris_side.extend(side_tris);
    let bvh_side = Bvh::build(&tris_side, &BvhParams::default());
    let cam_side = camera_at(
        side_eye.to_array(),
        [walker.position.x, 2.0, walker.position.z],
        50.0,
    );

    for (bvh, cam, name) in [
        (&bvh_default, cam_default, "own-eye-default.png"),
        (&bvh_side, cam_side, "own-eye-side.png"),
    ] {
        let accum = trace_headless(
            &device,
            &queue,
            bvh,
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
    eprintln!(
        "[own-eye] weld repair proven: own eye omits exactly nari's {nari_tris} tris; \
         a foreign eye 4m off still draws the full {}-tri dynamic inventory.",
        scene.dynamic_leaf_triangles().len(),
    );
}
