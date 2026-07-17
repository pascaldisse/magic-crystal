//! RITE VII · VII-0b relic forge — THE FIRST GROUND (the render weld). A
//! terrain patch (`naruko_first_ground`, `worlds/naruko/scenes/main.json`) is
//! authored ONLY as a sigil — `{seed:20260717, tile_x:0, tile_y:2}` — no
//! stored geometry anywhere. The real `RenderScene::from_ecs` weld generates
//! it through VII-0a's `seed::tile_mesh` at load and seals it through the
//! SAME Great Chain path every other static part in the canon Naruko realm
//! rides. Two relics, on the real realm, on the GPU, under the ONE light
//! pass (sun + sky — no new lights):
//!
//!   proof/vii0-ground.png       — the patch framed from the authored realm
//!                                 side, looking south past the seawall/terra
//!                                 edge toward the generated ground
//!   proof/vii0-ground-orbit.png — the SAME patch, camera yawed ~90° around
//!                                 its center (still on the ground plane,
//!                                 same distance) — proving the geometry
//!                                 reads as real, seamless terrain from
//!                                 another angle, not a screen-facing trick
//!
//! Run:  cargo run -p scrying-glass --release --example vii0_ground

use std::path::Path;

use glam::Vec3 as GVec3;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{headless_device, resolve, trace_headless, IntegratorParams};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

use crystal::{load_world_dir, EcsWorld};

/// Naruko authoring dials (mirror the window / p3_crate defaults).
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
    eprintln!("[vii0] wrote {}", path.display());
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[vii0] no GPU adapter on this host — cannot forge the relic");
    };

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut world = EcsWorld::default();
    load_world_dir(&world_path, &mut world).expect("load naruko");
    let params = naruko_params();
    let scene = RenderScene::from_ecs(world, &params).expect("render scene");
    eprintln!(
        "[vii0] naruko: {} static leaf tris (canon realm + the generated patch, one seal path)",
        scene.leaf_triangles().len(),
    );

    let bvh_params = BvhParams::default();
    let bvh = Bvh::build(&scene.leaf_triangles(), &bvh_params);

    // The patch: tile_x=0, tile_y=2, default tile_size_m=64 ⇒ tile_origin =
    // (0,128) (grid_resolution derives to 64, cell_size_m=1.0 — see
    // `seed::TerrainParams::derive`'s Nyquist derivation). Center = origin +
    // tile_size/2 = (32, 0, 160) (y taken at the field's zero baseline; the
    // fBm relief rides ±9.6 m around it, `height_amplitude`).
    let patch_center = [32.0_f32, 0.0, 160.0];

    // Eye 1 — framed from the authored realm side: standing above the terra
    // plane near its forward edge (z≈50, short of terra's own z=68 end),
    // looking south past the seawall/pier area toward the generated ground.
    let eye_a = [32.0_f32, 20.0, 50.0];
    let camera_a = camera_at(eye_a, patch_center, 55.0);

    // Eye 2 — the SAME distance from the patch center, yawed ~90° around it
    // (rotate the eye's (x,z) offset from the center by 90° in the xz
    // plane: (0,-110) -> (110,0), i.e. approach from due +x instead of -z).
    let eye_b = [
        patch_center[0] + (eye_a[2] - patch_center[2]).abs(),
        eye_a[1],
        patch_center[2],
    ];
    let camera_b = camera_at(eye_b, patch_center, 55.0);

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

    for (name, camera) in [
        ("vii0-ground.png", camera_a),
        ("vii0-ground-orbit.png", camera_b),
    ] {
        eprintln!(
            "[vii0] {name}: eye [{:.2}, {:.2}, {:.2}] -> patch center {:?}",
            camera.eye.x, camera.eye.y, camera.eye.z, patch_center
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
    eprintln!("[vii0] the first ground stands — read it with eyes.");
}
