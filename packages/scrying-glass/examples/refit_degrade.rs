//! REFIT DEGRADE SWEEP — derive `RefitParams::degrade_ratio` honestly instead
//! of freezing a literal. LEVER 1 (refit-not-rebuild) trades a per-tick REBUILD
//! for a per-tick REFIT: same topology, fresh bounds. Bounds loosen the longer
//! a tree goes un-rebuilt (the dynamic-root half-area grows), and loose bounds
//! cost extra GPU traversal. `degrade_ratio` is the multiple of the last-rebuild
//! half-area at which we pay the rebuild back. House law: tolerances are
//! DERIVED — measure the noise floor, gate ~10x the floor, prove the gate
//! actually discriminates real drift from measurement noise.
//!
//! Method (see docs/perf/2026-07-17-refit-degrade-derivation.md for the run):
//!  1. Warm the merged Naruko realm to the composed mid-stride tick (same
//!     scaffolding as `refit_parity.rs`).
//!  2. Build a `DynamicSplice` with a refit gate that NEVER trips
//!     (`degrade_ratio: f32::INFINITY`) so every tick refits — the pure
//!     degradation signal, unmasked by any self-correcting rebuild.
//!  3. Drive N real ticks. Every tick, compute the area-ratio = current
//!     refit-tree dynamic-root half-area / half-area of a FRESH build over the
//!     identical dynamic tris that tick.
//!  4. Every K ticks, GPU-trace (perf_audit's trace_frame style, wide pose —
//!     the worst pose per the audit) the refit-N-ticks merged tree and a fresh
//!     rebuild merged tree over the identical tris: ~4 warmup + ~16 measured
//!     frames each, mean+std.
//!  5. DRIFT = refit trace mean − rebuild trace mean. Noise floor = std of the
//!     rebuild trace means across samples. Gate = 10x floor. The derived
//!     `degrade_ratio` is the area-ratio at which drift first exceeds the gate,
//!     or — if the sweep never degrades enough to bite (a periodic walk cycle
//!     may never accumulate that much drift) — the max benign ratio observed
//!     times the same 10x headroom, stated plainly.
//!
//! Run:  cargo run -p scrying-glass --release --example refit_degrade

use std::f32::consts::PI;
use std::path::Path;
use std::time::Instant;

use aether::{DensityGrid, HomogeneousMedium, SteamColumn};
use crystal::{Core, load_world_dir};
use glam::Vec3 as GVec3;
use sama::GaitParams;
use scrying_glass::bvh::{Bvh, BvhParams, DynamicSplice, RefitParams};
use scrying_glass::integrator::{
    Integrator, IntegratorParams, IntegratorUniform, MediumGpu, MediumLightGpu, headless_device,
};
use scrying_glass::scene::{
    Camera, EmissiveSource, RenderScene, SceneParameters, SunDefaults, SunLight,
    contact_passing_ticks, emissive_sources, top_flat_surface_y,
};
use vessel::{Body, Preset};

/// Sweep dials (never hardcode — env-parameterised with the ledger defaults).
fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Naruko authoring dials (mirror `refit_parity.rs` / `perf_audit.rs` verbatim).
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

/// Half the surface area of an AABB — mirrors `bvh.rs`'s private `half_area`
/// (same formula; duplicated here since the example only has public access).
fn half_area(mn: [f32; 3], mx: [f32; 3]) -> f32 {
    let dx = (mx[0] - mn[0]).max(0.0);
    let dy = (mx[1] - mn[1]).max(0.0);
    let dz = (mx[2] - mn[2]).max(0.0);
    dx * dy + dy * dz + dz * dx
}

/// The dynamic partition's root half-area inside a `Bvh::merge` result. `merge`
/// places the dynamic root at node index 2 whenever both sides are nonempty
/// (`[root, static_root, dynamic_root, ...]` — see `bvh.rs`), which the merged
/// Naruko realm always is (static stall geometry + dynamic nari/kami/crate).
fn dynamic_root_half_area(merged: &Bvh) -> f32 {
    let node = merged.nodes[2];
    half_area(node.min, node.max)
}

/// One trace frame (uniform + dispatch + GPU wait) — perf_audit's trace_frame,
/// verbatim structure. Player-path reset semantics: moving geometry resets
/// accumulation every frame, so `samples_before = 0` each frame.
#[allow(clippy::too_many_arguments)]
fn trace_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    integrator: &Integrator,
    compute_bg: &wgpu::BindGroup,
    camera: &Camera,
    sun: &SunLight,
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    w: u32,
    h: u32,
    int_params: &IntegratorParams,
    medium: Option<&MediumGpu>,
) {
    let uniform = IntegratorUniform::build(
        camera,
        sun,
        sky_top,
        sky_horizon,
        w,
        h,
        integrator.node_count,
        integrator.tri_count,
        0,
        int_params,
        medium,
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("degrade-sweep integrate"),
    });
    integrator.dispatch(queue, &mut encoder, &uniform, compute_bg, w, h);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len().max(1) as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Trace `bvh` at `camera` for `warmup + measured` frames, returning
/// (mean_ms, std_ms) over the measured frames only.
#[allow(clippy::too_many_arguments)]
fn measure_trace_ms(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &Camera,
    sun: &SunLight,
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    w: u32,
    h: u32,
    int_params: &IntegratorParams,
    medium: Option<&MediumGpu>,
    warmup: u32,
    measured: u32,
) -> (f64, f64) {
    let integrator = Integrator::new(device, wgpu::TextureFormat::Rgba8UnormSrgb, bvh, medium);
    let accum = integrator.make_accum(device, w, h);
    let compute_bg = integrator.compute_bind_group(device, &accum);
    let mut samples = Vec::with_capacity(measured as usize);
    for frame in 0..(warmup + measured) {
        let t = Instant::now();
        trace_frame(
            device, queue, &integrator, &compute_bg, camera, sun, sky_top, sky_horizon, w, h,
            int_params, medium,
        );
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if frame >= warmup {
            samples.push(ms);
        }
    }
    mean_std(&samples)
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[refit-degrade] no GPU adapter on this host");
    };
    let (w, h) = (900u32, 600u32);

    let ticks = env_u32("GAIA_DEGRADE_TICKS", 300) as u64;
    let stride = env_u32("GAIA_DEGRADE_STRIDE", 25) as u64;
    let trace_warmup = env_u32("GAIA_DEGRADE_TRACE_WARMUP", 4);
    let trace_measured = env_u32("GAIA_DEGRADE_TRACE_FRAMES", 16);

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
    let bound = select_medium_light(&sources, &scene.sun, params.emission_intensity, plume_center, 1.47);
    let medium = steam_medium(&bound, counter_top_y);

    let bvh_params = BvhParams::default();
    let static_bvh = Bvh::build(&scene.leaf_triangles(), &bvh_params);

    // Warm to the composed mid-stride tick (refit_parity precedent, verbatim
    // scaffolding) — the sweep then drives forward from a real walking stride.
    let body = Body::from_preset(&Preset::nari());
    let gait = GaitParams::walk();
    let cycle = (1.0 / (gait.cadence * gait.dt)).round().max(1.0) as u64;
    let (_c, passing_tick) = contact_passing_ticks(&body, &gait);
    let mut target = 150u64;
    while target < passing_tick + cycle || target % cycle != passing_tick % cycle {
        target += 1;
    }
    for _ in 0..target {
        scene.command_bodies(6.0);
        scene.tick();
    }
    eprintln!("[refit-degrade] realm warmed to tick {target}; sweeping {ticks} ticks, stride {stride}");

    // Refit gate that NEVER trips — the pure degradation signal, unmasked by a
    // self-correcting rebuild. `max_refits: 0` = unlimited consecutive refits.
    let never_trip = RefitParams {
        degrade_ratio: f32::INFINITY,
        max_refits: 0,
    };
    let mut splice = DynamicSplice::build(
        &static_bvh,
        &scene.dynamic_leaf_triangles(),
        &bvh_params.dynamic(),
        never_trip,
    );

    let wide = camera_at([-4.5, 8.5, 33.0], [-5.5, 2.0, 15.5], 60.0);
    let int_params = IntegratorParams::default();

    println!(
        "[refit-degrade] {w}x{h}, wide pose, {trace_warmup} warmup + {trace_measured} measured trace frames per sample"
    );
    println!("| tick | area_ratio | refit ms (mean) | refit std | rebuild ms (mean) | rebuild std | drift ms |");
    println!("|------|------------|------------------|-----------|--------------------|-------------|----------|");

    let mut rebuild_means: Vec<f64> = Vec::new();
    let mut samples: Vec<(u64, f32, f64, f64)> = Vec::new(); // tick, ratio, drift, rebuild_mean
    let mut max_benign_ratio = 0.0f32;

    for tick in 1..=ticks {
        scene.command_bodies(6.0);
        scene.tick();
        let dyn_tris = scene.dynamic_leaf_triangles();
        splice.update(&static_bvh, &dyn_tris);

        if tick % stride == 0 {
            let fresh_dyn = Bvh::build(&dyn_tris, &bvh_params.dynamic());
            let area_ratio = dynamic_root_half_area(&splice.merged) / fresh_dyn.root_half_area().max(1e-9);
            let rebuild_merged = Bvh::merge(&static_bvh, &fresh_dyn);

            let (refit_mean, refit_std) = measure_trace_ms(
                &device, &queue, &splice.merged, &wide, &scene.sun, scene.sky_top, scene.sky_horizon,
                w, h, &int_params, Some(&medium), trace_warmup, trace_measured,
            );
            let (rebuild_mean, rebuild_std) = measure_trace_ms(
                &device, &queue, &rebuild_merged, &wide, &scene.sun, scene.sky_top, scene.sky_horizon,
                w, h, &int_params, Some(&medium), trace_warmup, trace_measured,
            );
            let drift = refit_mean - rebuild_mean;
            println!(
                "| {tick:4} | {area_ratio:10.4} | {refit_mean:16.4} | {refit_std:9.4} | {rebuild_mean:18.4} | {rebuild_std:11.4} | {drift:8.4} |"
            );
            rebuild_means.push(rebuild_mean);
            samples.push((tick, area_ratio, drift, rebuild_mean));
            max_benign_ratio = max_benign_ratio.max(area_ratio);
        }
    }

    // DERIVE: floor = std of rebuild trace means across samples (the honest
    // measurement noise on the ground-truth arm); gate = 10x floor.
    let (rebuild_of_rebuilds_mean, floor) = mean_std(&rebuild_means);
    let gate = 10.0 * floor;
    println!("[derive] noise floor (std of rebuild trace means across {} samples) = {floor:.4} ms", rebuild_means.len());
    println!("[derive] rebuild trace grand mean = {rebuild_of_rebuilds_mean:.4} ms");
    println!("[derive] gate = 10x floor = {gate:.4} ms");

    let first_bite = samples.iter().find(|(_, _, drift, _)| *drift > gate);
    let derived = match first_bite {
        Some((tick, ratio, drift, _)) => {
            println!(
                "[derive] drift first exceeds gate at tick {tick}: area_ratio {ratio:.4}, drift {drift:.4} ms > gate {gate:.4} ms"
            );
            println!("[derive] observed max benign area_ratio over the sweep = {max_benign_ratio:.4}");
            println!("[derive] result: degrade_ratio = R = {ratio:.4} (drift-discriminating gate)");
            *ratio
        }
        None => {
            println!(
                "[derive] drift NEVER exceeded the gate across the {} ticks / {} samples swept (a periodic walk cycle may never degrade past a bite)",
                ticks,
                samples.len()
            );
            println!("[derive] observed max benign area_ratio over the sweep = {max_benign_ratio:.4}");
            let result = max_benign_ratio * 10.0;
            println!(
                "[derive] result: degrade_ratio = max observed area_ratio x 10 (tolerance-law headroom) = {result:.4}"
            );
            result
        }
    };
    println!("[refit-degrade] VERDICT: derived degrade_ratio = {derived:.4}");
}
