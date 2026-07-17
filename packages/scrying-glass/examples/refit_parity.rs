//! REFIT PARITY GATE (LEVER 1) — prove the per-tick refit renders identically
//! to the per-tick REBUILD it replaces. A BVH is acceleration-only: for any ray
//! whose nearest surface is unambiguous the hit is topology-independent, so the
//! image is bit-identical BY CONSTRUCTION; the only freedom is the arbitrary
//! winner of an EXACT-depth coplanar tie (the same freedom SAH-vs-median has —
//! see bvh.rs). This example measures that freedom honestly.
//!
//! Method: warm the merged Naruko realm to the composed mid-stride tick, drive a
//! persistent `DynamicSplice` (refit path) forward a real history of ticks, then
//! at the SAME tick build a from-scratch REBUILD merge over the identical
//! dynamic tris. Render both deterministically (fixed seed/frames) at the three
//! law poses (front, wide, a2_steam) and report the FNV hash of each accum plus
//! the exact per-pixel divergence (cells differing, max channel delta).
//!
//! Run:  cargo run -p scrying-glass --release --example refit_parity

use std::f32::consts::PI;
use std::path::Path;

use aether::{DensityGrid, HomogeneousMedium, SteamColumn};
use crystal::{Core, load_world_dir};
use glam::Vec3 as GVec3;
use sama::GaitParams;
use scrying_glass::bvh::{Bvh, BvhParams, DynamicSplice, RefitParams, SpliceKind};
use scrying_glass::integrator::{
    IntegratorParams, MediumGpu, MediumLightGpu, headless_device, trace_headless,
};
use scrying_glass::scene::{
    Camera, EmissiveSource, RenderScene, SceneParameters, SunDefaults, SunLight,
    contact_passing_ticks, emissive_sources, top_flat_surface_y,
};
use vessel::{Body, Preset};

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

struct BoundLight {
    light: MediumLightGpu,
    color: [f32; 3],
    intensity: f32,
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn select_medium_light(
    sources: &[EmissiveSource],
    sun: &SunLight,
    emission_intensity: f32,
    plume_center: [f32; 3],
    fallback_reach: f32,
) -> BoundLight {
    let nearest = sources
        .iter()
        .min_by(|a, b| dist2(a.position, plume_center).total_cmp(&dist2(b.position, plume_center)));
    match nearest {
        Some(s) if dist2(s.position, plume_center).sqrt() <= fallback_reach => BoundLight {
            light: MediumLightGpu::Point {
                position: s.position,
            },
            color: s.color,
            intensity: emission_intensity * PI * s.radius * s.radius,
        },
        _ => BoundLight {
            light: MediumLightGpu::Directional {
                to_light: sun.direction,
            },
            color: sun.color,
            intensity: sun.intensity,
        },
    }
}

fn steam_medium(bound: &BoundLight, counter_top_y: f32) -> MediumGpu {
    let column = SteamColumn {
        base: aether::vec3(-1.0, counter_top_y as f64, 25.6),
        height: 4.2,
        radius: 0.85,
        peak: 1.0,
        turbulence: 0.7,
        ..SteamColumn::default()
    };
    let dims = [26usize, 36, 26];
    let origin = aether::vec3(-2.5, counter_top_y as f64, 24.1);
    let grid = DensityGrid::rasterize(dims, 0.12, origin, &column);
    let optics = HomogeneousMedium::new(0.001, 1.5, 0.4);
    let d = grid.dims();
    let o = grid.world_origin();
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
        light: bound.light,
        light_color: bound.color,
        light_intensity: bound.intensity,
        density: grid.data().to_vec(),
    }
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Per-pixel divergence between two accum buffers (post-mean): differing cells
/// and the max absolute channel delta of the running MEAN (sum/w).
fn divergence(a: &[[f32; 4]], b: &[[f32; 4]]) -> (usize, f32) {
    let mut diff = 0usize;
    let mut max_delta = 0.0f32;
    for (ca, cb) in a.iter().zip(b.iter()) {
        let wa = ca[3].max(1.0);
        let wb = cb[3].max(1.0);
        let mut differs = false;
        for k in 0..3 {
            let d = (ca[k] / wa - cb[k] / wb).abs();
            if d > max_delta {
                max_delta = d;
            }
            if d > 0.0 {
                differs = true;
            }
        }
        if differs {
            diff += 1;
        }
    }
    (diff, max_delta)
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[refit-parity] no GPU adapter on this host");
    };
    let (w, h) = (900u32, 600u32);
    let frames = 4u32;
    let history: u64 = std::env::var("REFIT_HISTORY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();
    let sources = emissive_sources(&core.world).expect("emissive sources");
    let counter_top_y = top_flat_surface_y(&core.world, "naruko_stall_massing")
        .expect("stall surface")
        .expect("flat serving surface");
    let mut scene =
        RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("render scene");

    let plume_center = [-1.0, counter_top_y + 1.7, 25.6];
    let bound = select_medium_light(
        &sources,
        &scene.sun,
        params.emission_intensity,
        plume_center,
        1.47,
    );
    let medium = steam_medium(&bound, counter_top_y);

    let bvh_params = BvhParams::default();
    let static_bvh = Bvh::build(&scene.leaf_triangles(), &bvh_params);

    // Warm to the composed passing-phase tick (parity_check precedent), minus the
    // refit history so the refit path accrues a real multi-tick topology drift
    // before we compare at the measure tick.
    let body = Body::from_preset(&Preset::nari());
    let gait = GaitParams::walk();
    let cycle = (1.0 / (gait.cadence * gait.dt)).round().max(1.0) as u64;
    let (_c, passing_tick) = contact_passing_ticks(&body, &gait);
    let mut target = 150u64;
    while target < passing_tick + cycle || target % cycle != passing_tick % cycle {
        target += 1;
    }
    let warm_to = target - history;
    for _ in 0..warm_to {
        scene.command_bodies(6.0);
        scene.tick();
    }

    // Build the persistent splice HERE, then drive it (refit path) forward the
    // full history to `target` — exactly how the render loop feeds it.
    let mut splice = DynamicSplice::build(
        &static_bvh,
        &scene.dynamic_leaf_triangles(),
        &bvh_params.dynamic(),
        RefitParams::default(),
    );
    let mut refits = 0u32;
    let mut rebuilds = 0u32;
    for _ in 0..history {
        scene.command_bodies(6.0);
        scene.tick();
        match splice.update(&static_bvh, &scene.dynamic_leaf_triangles()) {
            SpliceKind::Refit => refits += 1,
            SpliceKind::Rebuilt => rebuilds += 1,
        }
    }
    eprintln!(
        "[refit-parity] warmed to tick {target} (history {history}: {refits} refits, {rebuilds} rebuilds); \
         last splice = {:?}",
        splice.last_kind
    );

    // The REBUILD ground truth at the identical tick over the identical dyn tris.
    let dyn_now = scene.dynamic_leaf_triangles();
    let rebuild_merged = Bvh::merge(&static_bvh, &Bvh::build(&dyn_now, &bvh_params.dynamic()));
    let refit_merged = &splice.merged;
    assert_eq!(
        refit_merged.tris.len(),
        rebuild_merged.tris.len(),
        "refit and rebuild must span the same triangle count"
    );

    let int_params = IntegratorParams::default();
    let poses = [
        (
            "front",
            Camera {
                eye: GVec3::from_array(params.camera_position),
                yaw: params.camera_yaw,
                pitch: params.camera_pitch,
                fov_y_radians: params.fov_y_degrees.to_radians(),
                near: params.near,
                far: params.far,
            },
        ),
        (
            "wide",
            camera_at([-4.5, 8.5, 33.0], [-5.5, 2.0, 15.5], 60.0),
        ),
        (
            "a2_steam",
            camera_at([3.5, 3.4, 33.0], [-1.0, 4.2, 25.6], 55.0),
        ),
    ];

    let total_px = (w * h) as usize;
    println!(
        "[refit-parity] {w}x{h}, spp {}, {frames} frames, {total_px} px",
        int_params.spp
    );
    println!("| pose | refit hash | rebuild hash | diff px | diff % | max Δ |");
    println!("|------|-----------|--------------|---------|--------|-------|");
    let mut worst_pct = 0.0f64;
    for (name, cam) in &poses {
        let refit_accum = trace_headless(
            &device,
            &queue,
            refit_merged,
            cam,
            &scene.sun,
            scene.sky_top,
            scene.sky_horizon,
            w,
            h,
            frames,
            &int_params,
            Some(&medium),
        );
        let rebuild_accum = trace_headless(
            &device,
            &queue,
            &rebuild_merged,
            cam,
            &scene.sun,
            scene.sky_top,
            scene.sky_horizon,
            w,
            h,
            frames,
            &int_params,
            Some(&medium),
        );
        let rh = fnv(bytemuck::cast_slice(&refit_accum));
        let bh = fnv(bytemuck::cast_slice(&rebuild_accum));
        let (diff, max_delta) = divergence(&refit_accum, &rebuild_accum);
        let pct = 100.0 * diff as f64 / total_px as f64;
        worst_pct = worst_pct.max(pct);
        println!("| {name:8} | {rh:016x} | {bh:016x} | {diff} | {pct:.4}% | {max_delta:.2e} |");
    }
    println!(
        "[refit-parity] worst divergence {worst_pct:.4}% of pixels — {} \
         (bit-identical where 0%; any nonzero is exact-depth coplanar tie churn, same class as SAH-vs-median)",
        if worst_pct == 0.0 {
            "BIT-EXACT"
        } else {
            "tie-band only"
        }
    );
}
