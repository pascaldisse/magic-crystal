//! A1 relic forge — the ramen steam plume rises above the Naruko stall, lit by
//! the SAME traced light as every surface (Rite VI, the Aether entering the
//! Pleroma). Loads the real Naruko realm, splices a participating-medium steam
//! column above the stall, and renders three relics headlessly on the GPU:
//!
//!   proof/a1-steam.png        — the plume, steam ON
//!   proof/a1-steam-orbit.png  — the same plume from another angle
//!   proof/a1-steam-off.png    — steam OFF (for the localized-diff report)
//!
//! It prints the honest frame cost (ms/frame with the medium marching) and the
//! steam-on-vs-off difference split into the PLUME region and the far sky (the
//! discriminating claim: the plume changes, the far sky does not).
//!
//! Run:  cargo run -p scrying-glass --release --example a1_steam

use std::path::Path;
use std::time::Instant;

use aether::{DensityGrid, HomogeneousMedium, SteamColumn};
use crystal::{Core, load_world_dir};
use glam::Vec3 as GVec3;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, MediumGpu, headless_device, resolve};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

/// Naruko scene parameters (mirrors the window defaults in `main.rs` — the same
/// realm a player boots). Nothing here is hardcoded into the engine; these are
/// the world's authoring dials.
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

/// The steam plume above the ramen stall. The stall (`naruko_stall_massing`) is
/// at world [-1,0,25]; its counter sits ~y=0.9 and its roof ~y=2.9. The plume
/// rises from just above the counter. Every value is a dial (never hardcoded in
/// the medium law) with a documented choice.
fn steam_medium() -> MediumGpu {
    // The density source: a rising, turbulent column (Aether preset, param set).
    let column = SteamColumn {
        base: aether::vec3(-1.0, 1.1, 25.6),
        height: 4.6,
        radius: 0.55,
        peak: 1.0,
        turbulence: 0.65,
        ..SteamColumn::default()
    };
    // Rasterize into the grid the GPU uploads (the SAME artifact the CPU marches).
    let vsize = 0.12;
    let dims = [30usize, 46, 30];
    // Box centered on the column base in x/z, rising over its height.
    let origin = aether::vec3(-2.8, 1.0, 23.8);
    let grid = DensityGrid::rasterize(dims, vsize, origin, &column);

    // Optics: bright steam — nearly pure scattering, strongly forward (g=0.6) so
    // the warm BACKLIGHT behind the plume forward-scatters toward the camera and
    // the wisp GLOWS (the way real backlit steam does over a night stall).
    let optics = HomogeneousMedium::new(0.05, 2.2, 0.6);
    let d = grid.dims();
    let o = grid.world_origin();
    // The steam's OWN light: a warm source behind + above the plume (decoupled
    // from the sky sun). Direction TOWARD it ≈ the camera's view direction, so
    // forward scatter carries its light to the eye. A dial, not hardcoded law.
    let light_dir = {
        let v = GVec3::new(-0.4, 0.5, -0.8).normalize();
        [v.x, v.y, v.z]
    };
    MediumGpu {
        dims: [d[0] as u32, d[1] as u32, d[2] as u32],
        voxel_size: grid.voxel_size() as f32,
        world_origin: [o.x as f32, o.y as f32, o.z as f32],
        sigma_a: optics.sigma_a as f32,
        sigma_s: optics.sigma_s as f32,
        g: optics.g as f32,
        far: 60.0,
        march_steps: 128,
        shadow_steps: 32,
        shadow_dist: 7.0,
        light_dir,
        light_color: [1.0, 0.62, 0.34], // warm lantern glow (linear rgb)
        light_intensity: 9.0,
        density: grid.data().to_vec(),
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

/// Render one relic; returns the resolved linear-radiance image and the honest
/// mean ms/frame for the accumulation loop (medium marching included).
#[allow(clippy::too_many_arguments)]
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &Camera,
    scene: &RenderScene,
    w: u32,
    h: u32,
    frames: u32,
    params: &IntegratorParams,
    medium: Option<&MediumGpu>,
) -> (Vec<GVec3>, f64) {
    let start = Instant::now();
    let accum = scrying_glass::integrator::trace_headless(
        device,
        queue,
        bvh,
        camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        w,
        h,
        frames,
        params,
        medium,
    );
    let ms_per_frame = start.elapsed().as_secs_f64() * 1e3 / frames as f64;
    (resolve(&accum), ms_per_frame)
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
    eprintln!("[a1] wrote {}", path.display());
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[a1] no GPU adapter on this host — cannot forge the relic");
    };

    // Load the real Naruko realm.
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();
    let scene =
        RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("render scene");

    // Static + dynamic geometry into one BVH (as the window does).
    let mut tris = scene.leaf_triangles();
    tris.extend(scene.dynamic_leaf_triangles());
    let bvh = Bvh::build(&tris, &BvhParams::default());
    eprintln!("[a1] naruko: {} leaf triangles", tris.len());

    let medium = steam_medium();

    let (w, h) = (900u32, 600u32);
    let frames = 40u32;
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let exposure = 1.0;

    // Front three-quarter view: the stall with the plume rising above its roof
    // against the night sky and distant city.
    let cam_front = camera_at([3.5, 3.4, 33.0], [-1.0, 4.2, 25.6], 55.0);
    // Orbit: the other shoulder, same plume.
    let cam_orbit = camera_at([-6.5, 3.6, 32.0], [-1.0, 4.2, 25.6], 55.0);

    let (img_on, ms_on) = render(
        &device,
        &queue,
        &bvh,
        &cam_front,
        &scene,
        w,
        h,
        frames,
        &int_params,
        Some(&medium),
    );
    let (img_off, ms_off) = render(
        &device,
        &queue,
        &bvh,
        &cam_front,
        &scene,
        w,
        h,
        frames,
        &int_params,
        None,
    );
    let (img_orbit, ms_orbit) = render(
        &device,
        &queue,
        &bvh,
        &cam_orbit,
        &scene,
        w,
        h,
        frames,
        &int_params,
        Some(&medium),
    );

    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");
    write_png(&img_on, w, h, exposure, &proof.join("a1-steam.png"));
    write_png(
        &img_orbit,
        w,
        h,
        exposure,
        &proof.join("a1-steam-orbit.png"),
    );
    write_png(&img_off, w, h, exposure, &proof.join("a1-steam-off.png"));

    // Localized-diff report: split the ON-vs-OFF difference into the PLUME
    // region (a column of pixels over the stall) and the FAR SKY (top-left
    // corner, far from the plume). The plume must change; the far sky must not.
    let mut plume_sum = 0.0f64;
    let mut plume_n = 0.0f64;
    let mut sky_sum = 0.0f64;
    let mut sky_n = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let d = (img_on[i] - img_off[i]).abs();
            let dv = (d.x + d.y + d.z) as f64 / 3.0;
            // Plume region: central column, upper-middle of the frame.
            let in_plume =
                x > w * 40 / 100 && x < w * 62 / 100 && y > h * 8 / 100 && y < h * 70 / 100;
            // Far sky: top-left corner, away from the plume.
            let in_sky = x < w * 20 / 100 && y < h * 20 / 100;
            if in_plume {
                plume_sum += dv;
                plume_n += 1.0;
            }
            if in_sky {
                sky_sum += dv;
                sky_n += 1.0;
            }
        }
    }
    let plume_diff = plume_sum / plume_n.max(1.0);
    let sky_diff = sky_sum / sky_n.max(1.0);

    println!("[a1] ── FRAME COST (honest, {w}x{h}, {frames} accum frames) ──");
    println!("[a1]   steam ON  : {ms_on:.2} ms/frame");
    println!(
        "[a1]   steam OFF : {ms_off:.2} ms/frame  (medium overhead {:.2} ms)",
        ms_on - ms_off
    );
    println!("[a1]   orbit ON  : {ms_orbit:.2} ms/frame");
    println!("[a1] ── STEAM ON vs OFF (localized diff) ──");
    println!("[a1]   plume region mean |Δ| = {plume_diff:.5}");
    println!("[a1]   far-sky      mean |Δ| = {sky_diff:.6}");
    println!(
        "[a1]   plume/sky ratio = {:.1}x  (discriminating: the plume changes, the sky does not)",
        plume_diff / sky_diff.max(1e-9)
    );
}
