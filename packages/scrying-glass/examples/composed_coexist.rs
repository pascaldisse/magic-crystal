//! MERGE PROOF (elements-p3 → main) — COEXISTENCE. The re-derivation merge
//! unites two "13th vessel" bodies into one 14-vessel realm: `nari` (a SKINNED
//! vessel body, Rite V) standing on the seawall, and `naruko_crate` (a PHYSICS
//! body, Elements P3) falling onto the pier. Both dynamics paths feed the ONE
//! traced BVH splice — the skinned vessel triangles AND the rigid-solver crate.
//! This render frames BOTH in a single wide, elevated shot on the MERGED realm,
//! at a MID-STRIDE tick — nari WALKING (her SAMA gait at the passing pose, swing
//! foot lifted) while the crate has come to rest — so the reconciliation merge
//! (rite5-v1 → main) itself ends in pixels:
//!
//!   proof/composed-coexist.png — nari walking on the seawall + the crate at rest
//!
//! Run:  cargo run -p scrying-glass --release --example composed_coexist

use std::path::Path;

use glam::Vec3 as GVec3;
use sama::GaitParams;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::scene::{
    Camera, RenderScene, SceneParameters, SunDefaults, contact_passing_ticks,
};
use vessel::{Body, Preset};

use crystal::{EcsWorld, load_world_dir};

/// Naruko authoring dials (mirror the p3 relic / a2 defaults).
fn naruko_params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: 60.0,
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
    eprintln!("[coexist] wrote {}", path.display());
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[coexist] no GPU adapter on this host — cannot forge the relic");
    };

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut world = EcsWorld::default();
    load_world_dir(&world_path, &mut world).expect("load naruko");
    let params = naruko_params();
    let mut scene = RenderScene::from_ecs(world, &params).expect("render scene");
    eprintln!(
        "[coexist] naruko: {} static leaf tris, {} skinned body/ies, {} declared physics bod(ies)",
        scene.leaf_triangles().len(),
        scene.bodies.len(),
        scene.physics().map(|p| p.bindings().len()).unwrap_or(0),
    );

    // The static BVH is built once (the pier, the seawall, the realm). The crate
    // rides the DYNAMIC splice; nari's skinned triangles are appended to the
    // dynamic partition too (`dynamic_leaf_triangles`), so ONE merged BVH holds
    // BOTH new bodies.
    let bvh_params = BvhParams::default();
    let static_bvh = Bvh::build(&scene.leaf_triangles(), &bvh_params);

    // A wide, elevated three-quarter shot from the sea side: nari stands at
    // x=0 z=18 (seawall), the crate settles near x=-11.15 z=13 (pier) — ~12 m
    // apart. Look at their midpoint [-5.5, 2, 15.5] from up and back so both
    // fall inside the 60° cone.
    let camera = camera_at([-4.5, 8.5, 33.0], [-5.5, 2.0, 15.5], 60.0);

    let (w, h) = (1000u32, 640u32);
    let frames = 64u32;
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let exposure = 1.6;
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");

    // MID-STRIDE target: nari walks (her SAMA gait driven by `command_bodies`)
    // AND the crate settles (`tick` drives the rigid solver). The passing gait
    // tick is DERIVED from the walk cycle (swing foot at its highest — the same
    // pose the V1 ordeal proves distinct); the crate comes to rest ≈ tick 91.
    // Run past both: at least the crate-settle horizon (150) AND the idle→walk
    // blend (one full cycle beyond the passing phase), landing on the passing
    // phase so nari is unmistakably mid-stride, not idle.
    let body = Body::from_preset(&Preset::nari());
    let gait = GaitParams::walk();
    let cycle = (1.0 / (gait.cadence * gait.dt)).round().max(1.0) as u64;
    let (_contact_tick, passing_tick) = contact_passing_ticks(&body, &gait);
    let mut target = 150u64;
    while target < passing_tick + cycle || target % cycle != passing_tick % cycle {
        target += 1;
    }
    let mut current = 0u64;
    while current < target {
        // The player-path order (main.rs `advance_world`): command the body from
        // the walker's speed, then advance the world clock (physics + dynamics).
        scene.command_bodies(6.0);
        scene.tick();
        current += 1;
    }
    let crate_pos = scene.body_position("naruko_crate").expect("crate body");
    let dyn_bvh = Bvh::build(&scene.dynamic_leaf_triangles(), &bvh_params);
    let bvh = Bvh::merge(&static_bvh, &dyn_bvh);
    eprintln!(
        "[coexist] tick {current} (passing {passing_tick}, cycle {cycle}): nari mid-stride, crate at rest [{:.3}, {:.3}, {:.3}]  (merged BVH {} tris — static + nari-skinned + crate)",
        crate_pos[0],
        crate_pos[1],
        crate_pos[2],
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
    write_png(
        &resolve(&accum),
        w,
        h,
        exposure,
        &proof.join("composed-coexist.png"),
    );
    eprintln!("[coexist] the merged realm holds both — read it with eyes.");
}
