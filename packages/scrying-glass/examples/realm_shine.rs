//! REALM SHINE — the Architect's complaint answered: he looks at the live
//! window and sees darkness, no evidence of ray tracing. The transport is all
//! real (mirror vessels, moving emitters, the chrome sphere = the Rite IV L2
//! close object) but it was staged for SIDE cameras (`mirror_proof`), never
//! for the spawn eye. This relic renders the realm from EXACTLY where the
//! player's eyes open — the settled spawn pose — so the show is in his
//! sightline.
//!
//! THE SPAWN EYE. The scene `spawn` is [0,7,44] yaw 0; the window drops a body
//! there and gravity settles it onto the terra top (y=0) at eye height 1.7, so
//! the gameplay eye is [0, 1.7, 44] yaw 0 looking down −Z (scene.rs: yaw 0 ⇒
//! forward (0,0,−1)). FOV 60 vertical (the window default GAIA_NATIVE_FOV), a
//! 16:9 frame. That is the pose these two frames use — no side camera, no cheat.
//!
//! THE SHOW (pure realm data, worlds/naruko/scenes/main.json — zero engine
//! code; the `orbit` kami and the metallic-mirror BRDF already exist):
//!   naruko_show_chrome   — a chrome sphere r 2.1 on a dark pedestal at
//!                          [4.5, 3.6, 29.5] (metallic 1.0, roughness 0.02 =
//!                          the perfect-mirror delta lobe). The Rite IV close
//!                          object: a sphere reflects the WHOLE hemisphere, so
//!                          the lighthouse, the sea traces, the sky band and
//!                          the moving orbs all warp across it.
//!   naruko_show_mirror   — a tall polished panel [0.18,6.5,4.2] at [-6.5,3.4,28]
//!                          rotated -0.5 rad about +y (normal turned back toward
//!                          the eye), angled to catch the realm — from the spawn
//!                          eye it visibly reflects the chrome sphere + the orbs
//!                          (metallic 1.0, roughness 0.03).
//!   naruko_show_light_a/b/c — three orbiting emitters (violet #b98aff, cyan
//!                          #37e0ff, pink #ff6bb0) on kami `orbit` rings
//!                          centered [-1.5, y, 29] between the mirror and the
//!                          sphere. As they orbit, their reflections in the
//!                          chrome + mirror MOVE and their colored pools crawl
//!                          across the terra — bounce light, traced.
//!
//! TWO FRAMES, one camera, ticks apart (the motion proof):
//!   proof/realm-shine-a.png  — tick 60  (t=1.0 s)
//!   proof/realm-shine-b.png  — tick 210 (t=3.5 s)
//! Between them the violet orb sweeps 0.5·2.5 = 1.25 rad (72°), the cyan
//! −0.55·2.5 = −1.375 rad, the pink 0.8·2.5 = 2.0 rad (115°) — so their glints
//! on the chrome and their pools on the ground are demonstrably in different
//! places. The localized a-vs-b mean|Δ| is printed for three PROJECTED regions
//! (the chrome disc, the ground band under the orbits, a far-sky null): the
//! chrome and the ground must move, the sky must not.
//!
//! Run:  cargo run -p scrying-glass --release --example realm_shine

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

/// Naruko authoring dials — the SAME window defaults a player boots (main.rs),
/// so this render is the realm the Architect actually sees.
fn naruko_params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: 60.0,
        near: 0.1,
        far: 4_000.0,
        sky_top: "#20152f".into(),
        sky_horizon: "#9a627d".into(),
        mesh_color: "#9aa0a6".into(),
        radial_segments: 24,
        camera_position: [0.0, 1.7, 44.0],
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

/// The settled gameplay eye: [0, 1.7, 44] yaw 0 (forward −Z), pitch 0, FOV 60.
fn spawn_eye() -> Camera {
    Camera {
        eye: GVec3::new(0.0, 1.7, 44.0),
        yaw: 0.0,
        pitch: 0.0,
        fov_y_radians: 60f32.to_radians(),
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
    eprintln!("[shine] wrote {}", path.display());
}

/// Project a world point onto the pixel grid (same basis the primary rays use).
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
        x0.max(0.0) as u32,
        y0.max(0.0) as u32,
        x1.min(w as f32 - 1.0) as u32,
        y1.min(h as f32 - 1.0) as u32,
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

/// Closed-form orbit position (worlds/naruko main.json) for a stated tick.
fn orbit_at(center: [f64; 3], radius: f64, speed: f64, phase: f64, tick: u64) -> [f64; 3] {
    let t = tick as f64 / 60.0;
    let a = t * speed + phase;
    [
        center[0] + a.cos() * radius,
        center[1],
        center[2] + a.sin() * radius,
    ]
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[shine] no GPU adapter on this host — cannot forge the relic");
    };

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();
    let mut scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");

    // Static realm once; the living layer (orbs, lantern, beacon, bodies) is
    // re-spliced per captured tick — exactly as the window builds it.
    let bvh_params = BvhParams::default();
    let static_bvh = Bvh::build(&scene.leaf_triangles(), &bvh_params);

    let (w, h) = (1280u32, 720u32);
    let frames = 128u32;
    let int_params = IntegratorParams {
        spp: 3,
        max_bounces: 5,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let exposure = 1.0f32;
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");
    let cam = spawn_eye();

    // Diff regions projected from the derived world geometry (never plucked):
    //   CHROME  — the sphere disc [4.5,3.6,29.5] r 2.1 (its glints move).
    //   GROUND  — the terra band under the orbit rings (their pools crawl).
    //   SKY     — the top eighth of the frame (must NOT move).
    let chrome_pts: Vec<GVec3> = [
        [4.5f32 - 2.1, 3.6, 29.5],
        [4.5 + 2.1, 3.6, 29.5],
        [4.5, 3.6 + 2.1, 29.5],
        [4.5, 3.6 - 2.1, 29.5],
    ]
    .iter()
    .map(|p| GVec3::from_array(*p))
    .collect();
    let ground_pts: Vec<GVec3> = [
        [-5.0f32, 0.01, 25.0],
        [3.0, 0.01, 25.0],
        [-5.0, 0.01, 33.0],
        [3.0, 0.01, 33.0],
    ]
    .iter()
    .map(|p| GVec3::from_array(*p))
    .collect();
    let chrome_rect = screen_bbox(&cam, &chrome_pts, w, h);
    let ground_rect = screen_bbox(&cam, &ground_pts, w, h);
    let sky_rect = (0u32, 0u32, w - 1, h / 8);
    eprintln!(
        "[shine] regions — chrome {:?}  ground {:?}  sky {:?}",
        chrome_rect, ground_rect, sky_rect
    );

    let render = |scene: &RenderScene, cam: &Camera| -> Vec<GVec3> {
        let dyn_bvh = Bvh::build(&scene.dynamic_leaf_triangles(), &bvh_params);
        let bvh = Bvh::merge(&static_bvh, &dyn_bvh);
        let accum = trace_headless(
            &device, &queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, w, h,
            frames, &int_params, None,
        );
        resolve(&accum)
    };

    // Print the three orbs' positions at each captured tick so the relic states
    // exactly where the emitters stood (the motion is DATA, not vibes).
    let report = |tick: u64| {
        let a = orbit_at([-1.5, 2.0, 29.0], 3.0, 0.5, 0.0, tick);
        let b = orbit_at([-1.5, 3.6, 29.0], 2.5, -0.55, std::f64::consts::PI, tick);
        let c = orbit_at([-1.5, 5.2, 29.0], 2.8, 0.8, std::f64::consts::FRAC_PI_2, tick);
        eprintln!(
            "[shine] tick {tick}: violet [{:.2},{:.2},{:.2}]  cyan [{:.2},{:.2},{:.2}]  pink [{:.2},{:.2},{:.2}]",
            a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]
        );
    };

    // Frame A — tick 60.
    let mut current = 0u64;
    while current < 60 {
        scene.tick();
        current += 1;
    }
    report(current);
    let img_a = render(&scene, &cam);
    write_png(&img_a, w, h, exposure, &proof.join("realm-shine-a.png"));

    // Frame B — tick 210 (2.5 s later).
    while current < 210 {
        scene.tick();
        current += 1;
    }
    report(current);
    let img_b = render(&scene, &cam);
    write_png(&img_b, w, h, exposure, &proof.join("realm-shine-b.png"));

    let chrome = region_mean_abs_diff(&img_a, &img_b, w, chrome_rect);
    let ground = region_mean_abs_diff(&img_a, &img_b, w, ground_rect);
    let sky = region_mean_abs_diff(&img_a, &img_b, w, sky_rect);
    eprintln!(
        "[shine] a-vs-b mean|Δ| — chrome {:.6}  ground {:.6}  far-sky {:.6}",
        chrome, ground, sky
    );
    eprintln!("[shine] two frames forged — read them with eyes.");
}
