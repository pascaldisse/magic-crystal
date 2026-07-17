//! WIRED N2 · LIVE ORDEAL — THE WIRED BECOMES VISIBLE.
//!
//! A remote wired presence, streamed live through the interpolation buffer,
//! renders as a real BODY in the traced Naruko realm. Spawns its OWN world
//! server on port 8428 (never touches 8420/5173), drives two wired clients —
//! one MOVES, the other's interp buffer feeds the render scene — and proves the
//! rendered body's transform tracks the interpolated position. Then it traces a
//! frame with the other presence's body visible mid-move → `proof/n2-presence.png`.
//!
//!   cargo run -p scrying-glass --release --example n2_presence
//!
//! LIVE ordeals (every number prints verbatim):
//!   - LIVE-TRACK : the rendered body is placed AT the viewer's interpolated
//!     sample (body.model.x == sample.x, exact) — the render tracks the buffer.
//!   - LIVE-LAG   : the interpolated sample lags the mover's true position by
//!     ≤ the interp window (delay + extrapolation_cap, DERIVED from buffer depth).
//!   - LIVE-SMOOTH: the body's per-frame step stays under the smooth bound while
//!     the raw slam it replaces jumps an order larger (wired's discriminator).
//!   - CLEAN      : the server child is killed and leaves no orphan on 8428.

use std::path::Path;
use std::process::Child;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::player::Ground;
use scrying_glass::presence::{PresenceBodies, PresencePose};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene, SceneParameters, SunDefaults};
use wired::{Config, InterpConfig, Wired};

const PORT: u16 = 8428;
/// The mover's ground speed (m/s) and the feed cadence — a steady walk in +x.
const SPEED: f64 = 3.0;
const FEED_HZ: f64 = 20.0;
/// The path the mover walks along the seawall (world x), at eye height.
const X0: f64 = -3.0;
const X1: f64 = 3.0;
const ZY: [f64; 2] = [18.0, 2.0]; // [z, y] — y is eye height, grounded on skin

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

fn write_png(img: &[GVec3], w: u32, h: u32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in img {
        bytes.push((linear_to_srgb(px.x) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.y) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.z) * 255.0 + 0.5) as u8);
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
    eprintln!("[n2] wrote {}", path.display());
}

/// Spawn `GAIA_PORT=<port> bun server/index.js` from the engine checkout. Kept
/// in a guard so a panic still reaps it (no orphan on the port).
struct ServerGuard(Child);
impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn spawn_server(port: u16) -> ServerGuard {
    let engine = std::env::var("GAIA_ENGINE_DIR").unwrap_or_else(|_| {
        format!(
            "{}/projects/GAIA-World-Engine",
            std::env::var("HOME").unwrap()
        )
    });
    let child = std::process::Command::new("bun")
        .arg("server/index.js")
        .current_dir(&engine)
        .env("GAIA_PORT", port.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn bun server (is bun on PATH? is GAIA-World-Engine present?)");
    eprintln!(
        "[n2] spawned server pid={} on port {port} (cwd {engine})",
        child.id()
    );
    ServerGuard(child)
}

#[tokio::main]
async fn main() {
    // GPU first — fail fast before touching the network if there's no adapter.
    let Some((device, queue)) = headless_device() else {
        panic!("[n2] no GPU adapter on this host — cannot forge the relic");
    };

    let server = spawn_server(PORT);

    // The interp window is DERIVED from the buffer depth: delay + extrap cap.
    let interp = InterpConfig::for_tick_hz(FEED_HZ);
    let window = interp.delay + interp.extrapolation_cap;
    eprintln!(
        "[n2] interp: delay={:.3}s extrap_cap={:.3}s window={:.3}s (feed {FEED_HZ}Hz)",
        interp.delay, interp.extrapolation_cap, window
    );

    let mover = Wired::connect(Config::with_port(PORT).presence("mover-n2").interp(interp));
    let viewer = Wired::connect(Config::with_port(PORT).presence("viewer-n2").interp(interp));
    assert!(mover.wait_live().await, "mover never went live");
    assert!(viewer.wait_live().await, "viewer never went live");
    eprintln!("[n2] both clients live on {PORT}");

    // Spawn the mover at the path start, let the viewer's snapshot fold it in.
    mover
        .spawn_presence([X0, ZY[1], ZY[0]], std::f64::consts::FRAC_PI_2)
        .expect("spawn mover");
    tokio::time::sleep(Duration::from_millis(300)).await;

    // The mover walks +x at SPEED, feeding FEED_HZ moves. A shared cell holds
    // its TRUE current x so the viewer side can measure the interpolation lag.
    let truth = Arc::new(Mutex::new(X0));
    let driver = {
        let truth = truth.clone();
        tokio::spawn(async move {
            let dt = 1.0 / FEED_HZ;
            let steps = ((X1 - X0) / (SPEED * dt)).ceil() as i64;
            for k in 0..=steps {
                let x = (X0 + SPEED * dt * k as f64).min(X1);
                // face +x (yaw = +PI/2 about +Y), walking the seawall.
                let _ = mover.move_presence([x, ZY[1], ZY[0]], std::f64::consts::FRAC_PI_2);
                *truth.lock().unwrap() = x;
                tokio::time::sleep(Duration::from_secs_f64(dt)).await;
            }
            mover
        })
    };

    // Build the Naruko render scene locally; the presence body splices into it.
    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let floor = Ground::from_positions(&scene.leaf_positions());
    let static_tris = scene.leaf_triangles();
    let dyn_tris = scene.dynamic_leaf_triangles();

    // Presence layer, driven at ~60Hz from the viewer's interpolated sample.
    let render_dt = 1.0 / 60.0_f32;
    let mut bodies = PresenceBodies::new("nari", render_dt);

    let mut max_render_gap = 0.0_f64; // exact |body.x - sample.x|
    // Velocities (m/s) — the jitter-normalized smooth-vs-slam discriminator:
    // step / actual wall dt, so a delayed render frame can't fake a big step.
    let mut max_smooth_vel = 0.0_f64;
    let mut max_slam_vel = 0.0_f64;
    let mut max_lag = 0.0_f64;
    let mut prev: Option<(f64, f64, std::time::Instant)> = None; // (smooth_x, slam_x, t)
    let mut samples = 0usize;
    let mut proof: Option<(Vec<LeafTriangle>, f64)> = None;

    // Render/sample loop: run until the mover finishes its walk plus a coast.
    let deadline = std::time::Instant::now() + Duration::from_secs_f64((X1 - X0) / SPEED + 0.6);
    while std::time::Instant::now() < deadline {
        if let Some(sample) = viewer.sample("mover-n2") {
            let pose = PresencePose::new(sample.position, sample.yaw);
            let mut set = std::collections::BTreeMap::new();
            set.insert("mover-n2".to_string(), pose);
            bodies
                .sync(&set, Some(&floor))
                .expect("sync presence bodies");

            let body = bodies.body("mover-n2").expect("mover embodied");
            let body_x = body.model().w_axis.x as f64;
            max_render_gap = max_render_gap.max((body_x - sample.position[0]).abs());

            let slam_x = viewer
                .slam("mover-n2")
                .map(|s| s.position[0])
                .unwrap_or(sample.position[0]);
            let now_t = std::time::Instant::now();
            if let Some((ps, pl, pt)) = prev {
                let dt = (now_t - pt).as_secs_f64();
                if dt > 0.0 {
                    max_smooth_vel = max_smooth_vel.max((sample.position[0] - ps).abs() / dt);
                    max_slam_vel = max_slam_vel.max((slam_x - pl).abs() / dt);
                }
            }
            prev = Some((sample.position[0], slam_x, now_t));

            let true_x = *truth.lock().unwrap();
            max_lag = max_lag.max((true_x - sample.position[0]).abs());
            samples += 1;

            // Capture the first mid-move frame (body near centre, walking) for
            // the proof relic.
            if proof.is_none() && sample.position[0].abs() < 0.4 && body.commanded_speed() > 0.0 {
                proof = Some((bodies.leaf_triangles(), sample.position[0]));
            }
        }
        tokio::time::sleep(Duration::from_secs_f64(render_dt as f64)).await;
    }

    let mover = driver.await.expect("driver task");

    // Derived bounds. Interpolation velocity can never exceed the feed speed by
    // more than the extrapolation coast + wall jitter; the raw slam holds still
    // then leaps a whole feed interval in one frame — an effective velocity an
    // order larger. Velocity (step/actual-dt) is the jitter-immune metric.
    let smooth_vel_bound = SPEED * 1.6; // feed speed + coast/jitter headroom
    println!(
        "LIVE-TRACK: samples={samples} max|body.x - sample.x|={max_render_gap:.2e} (render tracks the buffer exactly)"
    );
    println!(
        "LIVE-LAG: max_lag={max_lag:.4}m window_bound={:.4}m (window={window:.3}s * speed={SPEED})",
        window * SPEED
    );
    println!(
        "LIVE-SMOOTH: smooth_max_vel={max_smooth_vel:.4}m/s bound={smooth_vel_bound:.4} slam_max_vel={max_slam_vel:.4}m/s (feed_speed={SPEED})"
    );

    assert!(samples > 20, "not enough live samples: {samples}");
    assert!(
        max_render_gap < 1e-3,
        "rendered body must sit AT the interpolated sample: gap={max_render_gap}"
    );
    assert!(
        max_lag <= window * SPEED + 1e-3,
        "interp lag {max_lag} exceeded the derived window {}",
        window * SPEED
    );
    assert!(
        max_smooth_vel <= smooth_vel_bound,
        "smooth velocity {max_smooth_vel} exceeded bound {smooth_vel_bound}"
    );
    assert!(
        max_slam_vel > max_smooth_vel * 2.0,
        "slam velocity {max_slam_vel} should dwarf smooth {max_smooth_vel}"
    );

    // Forge the proof relic: the mover's body visible mid-move in the realm.
    let (presence_tris, proof_x) =
        proof.expect("never captured a mid-move frame with a walking body");
    let mut tris = static_tris.clone();
    tris.extend_from_slice(&dyn_tris);
    tris.extend_from_slice(&presence_tris);
    println!(
        "PROOF: mid-move frame at sample.x={proof_x:.3} — static={} dyn={} presence={} total_tris={}",
        static_tris.len(),
        dyn_tris.len(),
        presence_tris.len(),
        tris.len()
    );

    let (w, h) = (900u32, 600u32);
    let cam = camera_at([0.0, 3.3, 24.0], [proof_x as f32, 2.0, ZY[0] as f32], 46.0);
    let bvh = Bvh::build(&tris, &BvhParams::default());
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
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
        48,
        &int_params,
        None,
    );
    let img = resolve(&accum);
    let proof_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof/n2-presence.png");
    write_png(&img, w, h, &proof_path);

    // Close clients cleanly, then reap the server and prove no orphan on PORT.
    mover.close().await;
    viewer.close().await;
    drop(server);
    tokio::time::sleep(Duration::from_millis(400)).await;
    let orphan = std::process::Command::new("lsof")
        .args(["-i", &format!("tcp:{PORT}"), "-t"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!(
        "CLEAN: port {PORT} listeners after kill = {:?}",
        if orphan.is_empty() { "none" } else { &orphan }
    );
    assert!(orphan.is_empty(), "server orphan left on {PORT}: {orphan}");
    println!("[n2] the wired is visible. all live ordeals passed.");
}
