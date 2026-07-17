//! RITE VI · VI-1 relic forge — THE STACK TOPPLES. Three wooden crates
//! (realm `body`s), authored resting one atop the other on the Naruko pier;
//! an `Op::Impulse` (the incantation surface, `crates/crystal/src/protocol.rs`)
//! shoves the top crate; the world tick advances the Elements' rigid solver —
//! now with rigid-vs-rigid particle collision, so the crates actually feel
//! each other — and the ONE traced light watches the stack topple, tumble,
//! and settle without jitter. Three FIXED-TICK renders, on the real realm, on
//! the GPU:
//!
//!   proof/vi1-stack-before.png — the stack settled, pre-impulse
//!   proof/vi1-stack-mid.png    — mid-topple (the tick of max angular
//!                                displacement of the pushed crate — found
//!                                programmatically, never eyeballed)
//!   proof/vi1-stack-after.png  — the aftermath, settled again
//!
//! Determinism: the tick index is the entropy coordinate; two runs render the
//! same frames (proven in `packages/scrying-glass/tests/physics.rs`). Run:
//!   cargo run -p scrying-glass --release --example vi1_stack

use std::path::Path;
use std::time::Instant;

use glam::Vec3 as GVec3;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

use crystal::{EcsWorld, ImpulseOp, Op, load_world_dir};

const STACK_IDS: [&str; 3] = [
    "naruko_stack_crate_0",
    "naruko_stack_crate_1",
    "naruko_stack_crate_2",
];
/// Op data — the caller's choice, never an engine constant (never-hardcode
/// law: the ENGINE takes this as a parameter; this literal is the test/proof
/// harness picking its own scenario, same footing as every ordeal's impulse).
const PUSH_DELTA_VELOCITY: [f64; 3] = [3.0, 0.0, 0.0];
const SETTLE_TICKS: u64 = 120;
const TOPPLE_TICKS: u64 = 600;

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
    eprintln!("[vi1] wrote {}", path.display());
}

fn build_scene() -> RenderScene {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut world = EcsWorld::default();
    load_world_dir(&world_path, &mut world).expect("load naruko");
    RenderScene::from_ecs(world, &naruko_params()).expect("render scene")
}

fn impulse_op() -> Op {
    Op::Impulse(ImpulseOp {
        id: STACK_IDS[2].to_string(),
        delta_velocity: PUSH_DELTA_VELOCITY,
        ..Default::default()
    })
}

/// Angle (radians) between the pushed crate's rotated up-axis and world up —
/// zero at rest, largest mid-tumble. Read straight from the physics seam's
/// pose (the same rotation the ECS transform, and the rendered triangles,
/// carry that tick).
fn tilt_from_level(scene: &RenderScene) -> f64 {
    let physics = scene.physics().expect("bodies are declared");
    let binding = physics
        .bindings()
        .iter()
        .find(|b| b.gaia_id == STACK_IDS[2])
        .expect("pushed crate binding");
    let pose = physics.pose(binding);
    let up_y = pose.rotation_columns[1][1]; // world-up mapped through the rotation, y-component
    let mapped_len = pose.rotation_columns[1]
        .iter()
        .map(|c| c * c)
        .sum::<f64>()
        .sqrt();
    (up_y / mapped_len).clamp(-1.0, 1.0).acos()
}

/// The stack's authored centroid — the mean of the three authored crate
/// positions read straight from the loaded world (never a hardcoded camera
/// target unrelated to the data).
fn stack_authored_centroid() -> [f32; 3] {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut world = EcsWorld::default();
    load_world_dir(&world_path, &mut world).expect("load naruko");
    let scene = RenderScene::from_ecs(world, &naruko_params()).expect("render scene");
    let mut sum = [0.0f64; 3];
    for id in STACK_IDS {
        let p = scene.body_position(id).expect("stack crate body");
        for i in 0..3 {
            sum[i] += p[i];
        }
    }
    [
        (sum[0] / 3.0) as f32,
        (sum[1] / 3.0) as f32,
        (sum[2] / 3.0) as f32,
    ]
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[vi1] no GPU adapter on this host — cannot forge the relic");
    };

    // ─── PASS A — SILENT: find the mid-topple tick by max angular
    // displacement of the pushed crate, programmatically (never eyeballed).
    let mid_tick = {
        let mut scene = build_scene();
        for _ in 0..SETTLE_TICKS {
            scene.tick();
        }
        scene.tick_with_ops(&[impulse_op()]);
        let mut best_tick = 0u64;
        let mut best_tilt = tilt_from_level(&scene);
        for t in 1..=TOPPLE_TICKS {
            scene.tick();
            let tilt = tilt_from_level(&scene);
            if tilt > best_tilt {
                best_tilt = tilt;
                best_tick = t;
            }
        }
        eprintln!(
            "[vi1] max angular displacement {:.2}° at topple-tick {best_tick} (of {TOPPLE_TICKS})",
            best_tilt.to_degrees()
        );
        best_tick
    };

    // ─── PASS B — RENDER: replay the SAME deterministic episode, capturing
    // three fixed stops.
    let mut scene = build_scene();
    eprintln!(
        "[vi1] naruko: {} static leaf tris, {} declared bod(ies)",
        scene.leaf_triangles().len(),
        scene.physics().map(|p| p.bindings().len()).unwrap_or(0),
    );
    let bvh_params = BvhParams::default();
    let static_bvh = Bvh::build(&scene.leaf_triangles(), &bvh_params);

    let centroid = stack_authored_centroid();
    let camera = camera_at(
        [centroid[0] + 4.6, centroid[1] + 2.3, centroid[2] + 7.0],
        centroid,
        50.0,
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
    let exposure = 1.6;
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");

    let render_stop = |scene: &mut RenderScene, name: &str| {
        let dyn_bvh = Bvh::build(&scene.dynamic_leaf_triangles(), &bvh_params);
        let bvh = Bvh::merge(&static_bvh, &dyn_bvh);
        eprintln!(
            "[vi1] tick {}: {} (merged BVH {} tris)",
            scene.physics().unwrap().tick(),
            name,
            bvh.tris.len(),
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
    };

    for _ in 0..SETTLE_TICKS {
        scene.tick();
    }
    render_stop(&mut scene, "vi1-stack-before.png");

    scene.tick_with_ops(&[impulse_op()]);
    for t in 1..=TOPPLE_TICKS {
        scene.tick();
        if t == mid_tick {
            render_stop(&mut scene, "vi1-stack-mid.png");
        }
    }
    render_stop(&mut scene, "vi1-stack-after.png");

    // ─── THE P-GATE — mean CPU ms/tick of Solver::step ITSELF (wall clock,
    // not amortized against the render/kami overhead the tick() calls above
    // also pay). A fresh throwaway scene, physics only, dt matches the realm.
    {
        let mut bench = build_scene();
        let n = 300u64;
        let mut total = std::time::Duration::ZERO;
        for _ in 0..n {
            let physics = bench.physics_mut().expect("bodies are declared");
            let start = Instant::now();
            physics.step();
            total += start.elapsed();
        }
        let mean_ms = total.as_secs_f64() * 1000.0 / n as f64;
        eprintln!(
            "[vi1] P-GATE: solver mean CPU time = {mean_ms:.4} ms/tick over {n} ticks \
             (wall-clock, single core; budget for 60 FPS is 16.667 ms/tick)"
        );
    }

    eprintln!("[vi1] three relics forged — read them with eyes.");
}
