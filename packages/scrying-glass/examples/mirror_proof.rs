//! THE MIRROR PROOF — the Architect asked the raytracer to show its hand:
//! "put dynamic lights and reflective materials in the scene and maybe some
//! mirrors to prove it." Realm data only: a polished panel (`naruko_mirror`,
//! metallic 1.0, roughness 0.03) stands on the seawall; a cyan emissive orb
//! (`naruko_kami_orb`) rides a kami `orbit` between nari and the glass; a
//! second panel (`naruko_mirror_minor`) faces the first across the wall for
//! mirror-in-mirror. Four relics, headless on the GPU:
//!
//!   proof/mirror-nari.png      — nari's body, her reflection, the lantern's
//!                                pink bulb in the glass, the pink horizon band
//!   proof/mirror-dynamic-a.png — the orbiting orb at angle ≈ π/2 (tick 105)
//!   proof/mirror-dynamic-b.png — the orb at angle ≈ π (tick 209): the cyan
//!                                ground pool AND its mirror image both moved
//!   proof/mirror-in-mirror.png — the corridor between the facing panels
//!
//! PLACEMENT DERIVATION (mirror plane x = 3, reflective face toward −x; the
//! reflection across the plane maps a world point (x,y,z) → (6−x, y, z)):
//!   nari    N  = [0, 2.47, 18]    → N' = [6, 2.47, 18]
//!   lantern L  = [−7.5, 3.5, 20]  → L' = [13.5, 3.5, 20]
//!   orb A (tick 105, angle 1.575) ≈ [−2.607, 2.2, 19.70] → A' ≈ [8.607, 2.2, 19.70]
//!   orb B (tick 209, angle ≈ π) ≈ [−4.20, 2.2, 18.11] → B' = [10.2, 2.2, 18.11]
//!     (angle 3π/2 was rejected: its image crossing z = 16.15 exits the glass)
//! Camera E = [−7.5, 2.0, 15.5] (eye height BELOW the image line, so rays that
//! strike the panel above nari's image reflect UPWARD toward −x, clear the
//! terra slab and land in the pink horizon band — the sky in the glass).
//! Sight-line crossings on the panel plane (t = (3−Ex)/(Px'−Ex)):
//!   E→N'  crosses at (y 2.37, z 17.44) — inside the panel y[1.4,4.4] z[17,19]
//!   E→L'  crosses at (y 2.75, z 17.75) — inside
//!   E→A'  crosses at (y 2.13, z 18.24) — inside
//!   E→B'  crosses at (y 2.12, z 17.05) — inside (5 cm from the edge)
//! HERO shot re-derived (her reflection must read unambiguously): eye
//! [-3, 1.9, 16.2] look-at [3, 2.7, 17.4] fov 45 — see `cam_hero` below.
//! Occlusion: every image sight-line passes nari's AABB z[17.92,18.08] at
//! z ≤ 17.46 when x = 0 — her body never covers her own reflection. The
//! lantern itself sits ~73° off the view axis: only its REFLECTION is in frame.
//!
//! DIFF METHODOLOGY (the lantern-bob / a2_steam precedent): a-vs-b mean |Δ|
//! split into the GROUND band under the orbit (world rect x[−4.6,−0.6]
//! z[16.4,19.9] at y=1.4, projected), the PANEL face (x=2.96, y[1.4,4.4],
//! z[17,19], projected), and the FAR SKY null (top rows). The ground and the
//! glass must move; the sky must not.
//!
//! Run:  cargo run -p scrying-glass --release --example mirror_proof

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

/// Naruko scene parameters (mirrors the window defaults in `main.rs` — the
/// same realm a player boots). Authoring dials, not engine constants.
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
    eprintln!("[mirror] wrote {}", path.display());
}

/// Project a world point through the camera onto the pixel grid, using the SAME
/// basis the primary rays are generated from (`Camera::basis`): a pixel ray is
/// `forward + right·sx·tan(fov/2)·aspect + up·sy·tan(fov/2)`, so the inverse is
/// sx = (v·right)/((v·fwd)·tanH·aspect), sy = (v·up)/((v·fwd)·tanH). Row 0 is
/// the TOP of the image (sy = +1). Returns None behind the camera.
fn project(cam: &Camera, p: GVec3, w: u32, h: u32) -> Option<(f32, f32)> {
    let (right, up, forward) = cam.basis();
    let v = p - cam.eye;
    let zf = v.dot(forward);
    if zf <= 0.0 {
        return None;
    }
    let tan_h = (cam.fov_y_radians * 0.5).tan();
    let aspect = w as f32 / h as f32;
    let sx = v.dot(right) / (zf * tan_h * aspect);
    let sy = v.dot(up) / (zf * tan_h);
    Some(((sx + 1.0) * 0.5 * w as f32, (1.0 - sy) * 0.5 * h as f32))
}

/// Screen-space bounding box of a set of world points (clamped to the frame).
fn screen_bbox(cam: &Camera, pts: &[GVec3], w: u32, h: u32) -> (u32, u32, u32, u32) {
    let mut x0 = f32::MAX;
    let mut y0 = f32::MAX;
    let mut x1 = f32::MIN;
    let mut y1 = f32::MIN;
    for &p in pts {
        if let Some((px, py)) = project(cam, p, w, h) {
            x0 = x0.min(px);
            y0 = y0.min(py);
            x1 = x1.max(px);
            y1 = y1.max(py);
        }
    }
    (
        (x0.max(0.0)) as u32,
        (y0.max(0.0)) as u32,
        (x1.min(w as f32 - 1.0)) as u32,
        (y1.min(h as f32 - 1.0)) as u32,
    )
}

fn region_mean_abs_diff(a: &[GVec3], b: &[GVec3], w: u32, rect: (u32, u32, u32, u32)) -> f64 {
    let (x0, y0, x1, y1) = rect;
    let mut sum = 0.0f64;
    let mut n = 0.0f64;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = (y * w + x) as usize;
            let d = (a[i] - b[i]).abs();
            sum += (d.x + d.y + d.z) as f64 / 3.0;
            n += 1.0;
        }
    }
    if n > 0.0 { sum / n } else { 0.0 }
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[mirror] no GPU adapter on this host — cannot forge the relic");
    };

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();

    let mut scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");

    // Static realm once; the living layer (orb, lantern, beacon, bodies) is
    // re-spliced per captured tick — exactly as the window builds it.
    let bvh_params = BvhParams::default();
    let static_bvh = Bvh::build(&scene.leaf_triangles(), &bvh_params);

    let (w, h) = (960u32, 640u32);
    let frames = 96u32;
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let exposure = 1.0f32;
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");

    // The derived camera (see module note): below the image line, off the
    // seawall on the sea side, nari left of frame, the glass right.
    let cam_main = camera_at([-7.5, 2.0, 15.5], [1.5, 2.7, 18.0], 55.0);
    // The HERO pose, derived by mirror math (plane x=3, image map (x,y,z)→(6−x,y,z)):
    // eye E=[-3,1.9,16.2] BELOW nari's image line ⇒ her silhouette reads against the
    // reflected pink sky INSIDE the glass; E→N' ([6,2.47,18]) crosses the panel at
    // (y 2.28, z 17.4), her full image spans crossings y 1.63→3.03 — all inside
    // y[1.4,4.4] z[17,19]; the ray never reaches her AABB z-slab [17.92,18.08]
    // before the glass; her real body stands ~20° off-axis at frame left.
    let cam_hero = camera_at([-3.0, 1.9, 16.2], [3.0, 2.7, 17.4], 45.0);
    // The corridor shot: between the facing panels, nari in the frame, the
    // recursion behind her front image.
    let cam_corridor = camera_at([-5.5, 2.7, 17.5], [3.0, 2.7, 18.3], 50.0);

    // Diff regions, PROJECTED from the derived world geometry (never plucked
    // pixel boxes): the ground band swept by the orbit, the panel face, and a
    // far-sky null strip (top eighth of the frame).
    let ground_pts: Vec<GVec3> = [
        [-4.6f32, 1.4, 16.4],
        [-0.6, 1.4, 16.4],
        [-4.6, 1.4, 19.9],
        [-0.6, 1.4, 19.9],
    ]
    .iter()
    .map(|p| GVec3::from_array(*p))
    .collect();
    let panel_pts: Vec<GVec3> = [
        [2.96f32, 1.4, 17.0],
        [2.96, 4.4, 17.0],
        [2.96, 1.4, 19.0],
        [2.96, 4.4, 19.0],
    ]
    .iter()
    .map(|p| GVec3::from_array(*p))
    .collect();
    let ground_rect = screen_bbox(&cam_main, &ground_pts, w, h);
    let panel_rect = screen_bbox(&cam_main, &panel_pts, w, h);
    let sky_rect = (0u32, 0u32, w - 1, h / 8);
    eprintln!(
        "[mirror] regions — ground {:?}  panel {:?}  sky {:?}",
        ground_rect, panel_rect, sky_rect
    );

    let render = |scene: &RenderScene, cam: &Camera| -> Vec<GVec3> {
        let dyn_bvh = Bvh::build(&scene.dynamic_leaf_triangles(), &bvh_params);
        let bvh = Bvh::merge(&static_bvh, &dyn_bvh);
        let accum = trace_headless(
            &device,
            &queue,
            &bvh,
            cam,
            &scene.sun,
            scene.sky_top,
            scene.sky_horizon,
            w,
            h,
            frames,
            &int_params,
            None,
        );
        resolve(&accum)
    };

    // Orbit closed form (worlds/naruko main.json `naruko_kami_orb`): center
    // [-2.6, 2.2, 18.1], r 1.6, speed 0.9 — position printed per stop so the
    // relic states where the emitter stood.
    let orbit = |tick: u64| -> [f64; 3] {
        let t = tick as f64 / 60.0;
        let a = t * 0.9;
        [-2.6 + a.cos() * 1.6, 2.2, 18.1 + a.sin() * 1.6]
    };

    // Tick 0 — the hero relic: her body, her image, the lantern's bulb and the
    // pink band in the glass, the orb at angle 0.
    let img_hero = render(&scene, &cam_hero);
    write_png(&img_hero, w, h, exposure, &proof.join("mirror-nari.png"));

    // Tick 105 (angle ≈ π/2) — dynamic A.
    let mut current = 0u64;
    while current < 105 {
        scene.tick();
        current += 1;
    }
    let pa = orbit(current);
    eprintln!(
        "[mirror] tick {current}: orb at [{:.3}, {:.3}, {:.3}]",
        pa[0], pa[1], pa[2]
    );
    let img_a = render(&scene, &cam_main);
    write_png(&img_a, w, h, exposure, &proof.join("mirror-dynamic-a.png"));

    // Tick 209 (angle ≈ π) — dynamic B, half an orbit from A's side of the ring.
    while current < 209 {
        scene.tick();
        current += 1;
    }
    let pb = orbit(current);
    eprintln!(
        "[mirror] tick {current}: orb at [{:.3}, {:.3}, {:.3}]",
        pb[0], pb[1], pb[2]
    );
    let img_b = render(&scene, &cam_main);
    write_png(&img_b, w, h, exposure, &proof.join("mirror-dynamic-b.png"));

    // The a-vs-b localized diff: the pool and the glass move, the sky is null.
    let g = region_mean_abs_diff(&img_a, &img_b, w, ground_rect);
    let p = region_mean_abs_diff(&img_a, &img_b, w, panel_rect);
    let s = region_mean_abs_diff(&img_a, &img_b, w, sky_rect);
    eprintln!(
        "[mirror] a-vs-b mean|Δ| — ground {:.6}  panel {:.6}  far-sky {:.6}",
        g, p, s
    );

    // The corridor — mirror in mirror (max_bounces 4: the recursion resolves
    // as deep as the bounce budget lets a path reach a lit surface).
    let img_c = render(&scene, &cam_corridor);
    write_png(&img_c, w, h, exposure, &proof.join("mirror-in-mirror.png"));

    eprintln!("[mirror] four relics forged — read them with eyes.");
}
