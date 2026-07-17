//! RITE VI · VI-2 relic forge — SOMETHING BREAKS. A soft bonded crate
//! (realm `body`, `bonded: true`) is authored above the Naruko seawall (the
//! hard stone-like massing — `naruko_seawall`, colour `#3a2d4d`, the
//! Guardian-named drop target), falls under the world tick, and its bonds'
//! strife exceeds their love on impact: `Solver::fracture_pass` tears one or
//! more bonds, `Solver::fragment_components` flood-fills what remains,
//! `fracture::fragment_mesh` → `transmute_default` re-meshes each piece
//! (THE geometry path, no side door), and `Dynamics::tick_with_ops` births
//! each fragment as a real ECS vessel traced to the crate that broke — the
//! SAME wave, spliced into the dynamic BVH the same tick. Three FIXED-TICK
//! renders, on the real realm, on the GPU, lit by the ONE light pass:
//!
//!   proof/vi2-break-airborne.png — the whole crate, still bonded, falling
//!   proof/vi2-break-breaking.png — the tick fracture first tears a bond
//!   proof/vi2-break-settled.png  — the fragments at rest on the seawall
//!
//! Determinism: the tick index is the entropy coordinate; two runs render
//! the same frames (see `packages/scrying-glass/tests/vi2_break.rs` for the
//! byte-determinism proof over this exact scenario). Run:
//!   cargo run -p scrying-glass --release --example vi2_break

use std::path::Path;

use glam::Vec3 as GVec3;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::scene::{Camera, RenderScene, SceneParameters, SunDefaults};

use crystal::{load_world_dir, EcsWorld};

const BREAK_CRATE_ID: &str = "naruko_break_crate";
const SETTLE_TICKS: u64 = 300; // enough for shards to come to rest after breaking

/// Naruko authoring dials (mirror the window / VI-1 defaults).
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
    eprintln!("[vi2] wrote {}", path.display());
}

fn build_scene() -> RenderScene {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut world = EcsWorld::default();
    load_world_dir(&world_path, &mut world).expect("load naruko");
    RenderScene::from_ecs(world, &naruko_params()).expect("render scene")
}

/// The break crate's AUTHORED spawn position, read straight from the realm
/// data (never a camera target invented independent of the scene) — the
/// same `transform.position` `worlds/naruko/scenes/main.json` declares for
/// `naruko_break_crate`.
fn break_crate_authored_position() -> [f32; 3] {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut world = EcsWorld::default();
    load_world_dir(&world_path, &mut world).expect("load naruko");
    let transform_id = world.component_id("transform").expect("transform component");
    let entity = world
        .entity_for_gaia(BREAK_CRATE_ID)
        .expect("break crate entity");
    let value = world
        .get_component(entity, transform_id)
        .expect("break crate transform");
    let pos = value
        .get("position")
        .and_then(|v| v.as_array())
        .expect("position array");
    [
        pos[0].as_f64().unwrap() as f32,
        pos[1].as_f64().unwrap() as f32,
        pos[2].as_f64().unwrap() as f32,
    ]
}

/// The tick fracture first tears a bond — found programmatically by running
/// the deterministic drop and reading `Solver::fractures` (never eyeballed).
fn find_break_tick(max_ticks: u64) -> Option<u64> {
    let mut scene = build_scene();
    for t in 1..=max_ticks {
        scene.tick();
        let physics = scene.physics().expect("bodies declared");
        if !physics.solver().fractures.is_empty() {
            return Some(t);
        }
    }
    None
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[vi2] no GPU adapter on this host — cannot forge the relic");
    };

    // ─── PASS A — SILENT: find the exact break tick.
    let max_ticks = SETTLE_TICKS;
    let break_tick = find_break_tick(max_ticks).expect(
        "[vi2] the authored crate never broke within the settle window — either the drop \
         height, essence density, or fracture_threshold need retuning",
    );
    eprintln!("[vi2] fracture first observed at tick {break_tick} (of {max_ticks})");

    // ─── PASS B — RENDER: replay the SAME deterministic episode, capturing
    // three fixed stops: airborne (well before impact), breaking (the exact
    // tick fracture fires), settled (after the shards have had time to rest).
    let mut scene = build_scene();
    eprintln!(
        "[vi2] naruko: {} static leaf tris, {} declared rigid bod(ies)",
        scene.leaf_triangles().len(),
        scene.physics().map(|p| p.bindings().len()).unwrap_or(0),
    );
    let bvh_params = BvhParams::default();
    let static_bvh = Bvh::build(&scene.leaf_triangles(), &bvh_params);

    let spawn = break_crate_authored_position();
    let camera = camera_at(
        [spawn[0] + 5.0, spawn[1] + 1.5, spawn[2] + 7.0],
        spawn,
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
            "[vi2] tick {}: {} (merged BVH {} tris, {} dynamic tris)",
            scene.physics().unwrap().tick(),
            name,
            bvh.tris.len(),
            scene.dynamic_leaf_triangles().len(),
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

    // AIRBORNE — a handful of ticks in, still falling, still one whole body.
    let airborne_tick = (break_tick / 3).max(1);
    for _ in 0..airborne_tick {
        scene.tick();
    }
    render_stop(&mut scene, "vi2-break-airborne.png");

    // BREAKING — advance to the exact fracture tick.
    for _ in airborne_tick..break_tick {
        scene.tick();
    }
    render_stop(&mut scene, "vi2-break-breaking.png");

    // SETTLED — let the shards fall the rest of the way and come to rest.
    for _ in break_tick..max_ticks {
        scene.tick();
    }
    render_stop(&mut scene, "vi2-break-settled.png");

    eprintln!("[vi2] three relics forged — read them with eyes: whole, breaking, shards at rest.");
}
