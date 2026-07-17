//! PERF AUDIT — MEASUREMENT ONLY (night-2 ledger). The 60 FPS law: one frame
//! of the GROWN realm must fit 16.67 ms. The realm under audit is the merged
//! Naruko: nari WALKING (skinned per tick), the crate physics body, the kami
//! decorative six, the steam medium, 15+ vessels — every dynamic feeding the
//! ONE traced BVH splice per tick.
//!
//! Extends the profile-split precedent (`a2_steam.rs` timing lines,
//! `composed_coexist.rs` full-dynamics scene) into a per-PHASE per-frame
//! breakdown, example-level instrumentation only (no library changes):
//!
//!   skin      — `command_bodies` (SAMA step + re-skin of every embodied body)
//!   tick      — `scene.tick()` (KAMI behaviors + rigid physics)
//!   splice    — gather dynamic leaf tris + `Bvh::build(dyn)` + `Bvh::merge`
//!   upload    — `Integrator::update_bvh` + compute bind-group rebuild
//!   trace     — one accumulation frame dispatched + GPU wait (medium march
//!               included when the steam is bound — labeled)
//!   readback  — accum copy → map → read (headless-audit cost; the player
//!               window blits instead)
//!
//! Segments per pose (three, so fused phases resolve by honest deltas):
//!   DYN-ON     full dynamics + steam        (the living world)
//!   STATIC+MED frozen splice, steam ON      (derivation aid)
//!   STATIC     frozen splice, steam OFF     (the statue world)
//!   medium march (derived) = STATIC+MED.trace − STATIC.trace
//!   living-world price     = DYN-ON.total − STATIC.total
//!
//! Two poses: (a) front spawn (the authored camera), (b) wide composed (the
//! coexist proof shot). 900×600, release, ≥5 warmup discarded, N measured.
//!
//! Run:  cargo run -p scrying-glass --release --example perf_audit

use std::f32::consts::PI;
use std::path::Path;
use std::time::Instant;

use aether::{DensityGrid, HomogeneousMedium, SteamColumn};
use crystal::{Core, load_world_dir};
use glam::Vec3 as GVec3;
use sama::GaitParams;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    Integrator, IntegratorParams, IntegratorUniform, MediumGpu, MediumLightGpu, headless_device,
};
use scrying_glass::scene::{
    Camera, EmissiveSource, RenderScene, SceneParameters, SunDefaults, SunLight,
    contact_passing_ticks, emissive_sources, top_flat_surface_y,
};
use vessel::{Body, Preset};

/// Bytes one accumulation cell occupies (vec4<f32>) — mirrors the integrator's
/// private `ACCUM_CELL`.
const ACCUM_CELL: u64 = 16;

/// Audit dials (never hardcode — env-parameterised with the ledger defaults).
fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Naruko authoring dials (mirror `composed_coexist.rs` — the merged realm).
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

/// A resolved medium light (mirrors `a2_steam.rs` — the A2 true binding).
struct BoundLight {
    light: MediumLightGpu,
    color: [f32; 3],
    intensity: f32,
    label: String,
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Bind the medium's in-scatter light to a REAL scene source (a2 precedent:
/// nearest emitter within `fallback_reach`, else the sun — all derived).
fn select_medium_light(
    sources: &[EmissiveSource],
    sun: &SunLight,
    emission_intensity: f32,
    plume_center: [f32; 3],
    fallback_reach: f32,
) -> BoundLight {
    let nearest = sources.iter().min_by(|a, b| {
        let da = dist2(a.position, plume_center);
        let db = dist2(b.position, plume_center);
        da.total_cmp(&db)
    });
    match nearest {
        Some(source) if dist2(source.position, plume_center).sqrt() <= fallback_reach => {
            BoundLight {
                light: MediumLightGpu::Point {
                    position: source.position,
                },
                color: source.color,
                intensity: emission_intensity * PI * source.radius * source.radius,
                label: source.id.clone(),
            }
        }
        _ => BoundLight {
            light: MediumLightGpu::Directional {
                to_light: sun.direction,
            },
            color: sun.color,
            intensity: sun.intensity,
            label: "sun".into(),
        },
    }
}

/// The ramen-stall steam plume (a2 dials verbatim — the SAME medium the last
/// honest ledger number measured).
fn steam_medium(bound: &BoundLight, counter_top_y: f32) -> MediumGpu {
    let plume_height = 4.2_f64;
    let column = SteamColumn {
        base: aether::vec3(-1.0, counter_top_y as f64, 25.6),
        height: plume_height,
        radius: 0.85,
        peak: 1.0,
        turbulence: 0.7,
        ..SteamColumn::default()
    };
    let vsize = 0.12;
    let dims = [26usize, 36, 26];
    let origin = aether::vec3(-2.5, counter_top_y as f64, 24.1);
    let grid = DensityGrid::rasterize(dims, vsize, origin, &column);
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

/// Per-phase sample sink: insertion-ordered, honest stats.
#[derive(Default)]
struct Recorder {
    names: Vec<&'static str>,
    samples: Vec<Vec<f64>>,
}

impl Recorder {
    fn push(&mut self, name: &'static str, ms: f64) {
        match self.names.iter().position(|n| *n == name) {
            Some(i) => self.samples[i].push(ms),
            None => {
                self.names.push(name);
                self.samples.push(vec![ms]);
            }
        }
    }

    /// (mean, std, min, max) for one phase.
    fn stats(&self, i: usize) -> (f64, f64, f64, f64) {
        let xs = &self.samples[i];
        let n = xs.len().max(1) as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mean, var.sqrt(), min, max)
    }

    /// Mean per-frame total across phases.
    fn total_mean(&self) -> f64 {
        (0..self.names.len()).map(|i| self.stats(i).0).sum()
    }
}

/// One trace frame (uniform + dispatch + GPU wait). Player-path reset
/// semantics: moving geometry resets accumulation every frame, so
/// `samples_before = 0` each frame (the honest 2spp-live tradeoff).
#[allow(clippy::too_many_arguments)]
fn trace_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    integrator: &Integrator,
    compute_bg: &wgpu::BindGroup,
    camera: &Camera,
    scene: &RenderScene,
    w: u32,
    h: u32,
    int_params: &IntegratorParams,
    medium: Option<&MediumGpu>,
) {
    let uniform = IntegratorUniform::build(
        camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        w,
        h,
        integrator.node_count,
        integrator.tri_count,
        0,
        int_params,
        medium,
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("audit integrate"),
    });
    integrator.dispatch(queue, &mut encoder, &uniform, compute_bg, w, h);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
}

/// One accum readback (copy → map → read → unmap) into a reusable buffer.
fn readback_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    accum: &wgpu::Buffer,
    readback: &wgpu::Buffer,
    bytes: u64,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("audit readback"),
    });
    encoder.copy_buffer_to_buffer(accum, 0, readback, 0, bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    encoder.map_buffer_on_submit(readback, wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r.map(|_| ()));
    });
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("readback channel").expect("map readback");
    let mapped = readback.get_mapped_range(..).expect("mapped readback");
    std::hint::black_box(&mapped[..16]);
    drop(mapped);
    readback.unmap();
}

fn print_table(pose: &str, config: &str, rec: &Recorder, budget_ms: f64) {
    for i in 0..rec.names.len() {
        let (mean, std, min, max) = rec.stats(i);
        println!(
            "| {pose} | {config} | {} | {mean:8.3} | {std:6.3} | {min:8.3} | {max:8.3} |",
            rec.names[i]
        );
    }
    let total = rec.total_mean();
    let verdict = if total <= budget_ms { "PASS" } else { "FAIL" };
    println!(
        "| {pose} | {config} | TOTAL | {total:8.3} |        |          |          |  {verdict} ({:.1} fps)",
        1000.0 / total.max(1e-9)
    );
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[audit] no GPU adapter on this host — cannot measure");
    };

    let w = env_u32("GAIA_AUDIT_W", 900);
    let h = env_u32("GAIA_AUDIT_H", 600);
    let warmup = env_u32("GAIA_AUDIT_WARMUP", 8);
    let frames = env_u32("GAIA_AUDIT_FRAMES", 48);
    let budget_ms = 1000.0 / 60.0;

    // Load the real merged Naruko realm; read the A2 bindings BEFORE the render
    // scene consumes the world (a2 precedent, verbatim).
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();
    let sources = emissive_sources(&core.world).expect("emissive sources");
    let counter_top_y = top_flat_surface_y(&core.world, "naruko_stall_massing")
        .expect("stall surface")
        .expect("the stall has a flat serving surface");
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

    // Warm the WORLD to the composed mid-stride steady state (coexist
    // precedent: past the crate-settle horizon, landing on the passing phase so
    // nari is unmistakably walking while we measure).
    let body = Body::from_preset(&Preset::nari());
    let gait = GaitParams::walk();
    let cycle = (1.0 / (gait.cadence * gait.dt)).round().max(1.0) as u64;
    let (_contact, passing_tick) = contact_passing_ticks(&body, &gait);
    let mut target = 150u64;
    while target < passing_tick + cycle || target % cycle != passing_tick % cycle {
        target += 1;
    }
    for _ in 0..target {
        scene.command_bodies(6.0);
        scene.tick();
    }
    eprintln!(
        "[audit] realm warmed to tick {target}: {} static tris, {} dynamic tris, {} bodies, {} physics binding(s), medium light = {}",
        scene.leaf_triangles().len(),
        scene.dynamic_leaf_triangles().len(),
        scene.bodies.len(),
        scene.physics().map(|p| p.bindings().len()).unwrap_or(0),
        bound.label,
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
    ];

    let bytes = (w as u64) * (h as u64) * ACCUM_CELL;
    let make_readback = || {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("audit readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    };

    println!(
        "[audit] {w}x{h}, spp {}, bounces {}, {warmup} warmup + {frames} measured frames, budget {budget_ms:.2} ms (60 FPS law)",
        int_params.spp, int_params.max_bounces
    );
    println!("| pose | config | phase | mean ms | std | min | max |");
    println!("|------|--------|-------|---------|-----|-----|-----|");

    let mut summaries: Vec<(String, f64, f64, f64)> = Vec::new();

    for (pose_name, camera) in &poses {
        // ── Segment A: DYN-ON — the living world, steam bound. ──────────────
        let dyn_bvh = Bvh::build(&scene.dynamic_leaf_triangles(), &bvh_params);
        let merged = Bvh::merge(&static_bvh, &dyn_bvh);
        let mut integrator = Integrator::new(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &merged,
            Some(&medium),
        );
        let accum = integrator.make_accum(&device, w, h);
        let readback = make_readback();
        let mut rec = Recorder::default();
        for frame in 0..(warmup + frames) {
            let measured = frame >= warmup;
            let t = Instant::now();
            scene.command_bodies(6.0);
            let skin_ms = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            scene.tick();
            let tick_ms = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            let dyn_bvh = Bvh::build(&scene.dynamic_leaf_triangles(), &bvh_params);
            let merged = Bvh::merge(&static_bvh, &dyn_bvh);
            let splice_ms = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            integrator.update_bvh(&device, &merged);
            let compute_bg = integrator.compute_bind_group(&device, &accum);
            let upload_ms = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            trace_frame(
                &device,
                &queue,
                &integrator,
                &compute_bg,
                camera,
                &scene,
                w,
                h,
                &int_params,
                Some(&medium),
            );
            let trace_ms = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            readback_frame(&device, &queue, &accum, &readback, bytes);
            let read_ms = t.elapsed().as_secs_f64() * 1e3;

            if measured {
                rec.push("skin (command_bodies)", skin_ms);
                rec.push("tick (physics+kami)", tick_ms);
                rec.push("splice (dyn build+merge)", splice_ms);
                rec.push("upload (update_bvh+bg)", upload_ms);
                rec.push("trace+medium (fused GPU)", trace_ms);
                rec.push("readback", read_ms);
            }
        }
        print_table(pose_name, "DYN-ON", &rec, budget_ms);
        let on_total = rec.total_mean();
        let on_trace = rec.stats(4).0;

        // Frozen splice for the static segments: the world as statue at the
        // last measured tick — same geometry, zero dynamics.
        let dyn_bvh = Bvh::build(&scene.dynamic_leaf_triangles(), &bvh_params);
        let frozen = Bvh::merge(&static_bvh, &dyn_bvh);

        // ── Segment B: STATIC+MED — frozen splice, steam still bound. ───────
        let integrator = Integrator::new(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &frozen,
            Some(&medium),
        );
        let accum = integrator.make_accum(&device, w, h);
        let compute_bg = integrator.compute_bind_group(&device, &accum);
        let readback = make_readback();
        let mut rec_med = Recorder::default();
        for frame in 0..(warmup + frames) {
            let t = Instant::now();
            trace_frame(
                &device,
                &queue,
                &integrator,
                &compute_bg,
                camera,
                &scene,
                w,
                h,
                &int_params,
                Some(&medium),
            );
            let trace_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            readback_frame(&device, &queue, &accum, &readback, bytes);
            let read_ms = t.elapsed().as_secs_f64() * 1e3;
            if frame >= warmup {
                rec_med.push("trace+medium (fused GPU)", trace_ms);
                rec_med.push("readback", read_ms);
            }
        }
        print_table(pose_name, "STATIC+MED", &rec_med, budget_ms);

        // ── Segment C: STATIC — frozen splice, steam OFF (the statue). ──────
        let integrator =
            Integrator::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, &frozen, None);
        let accum = integrator.make_accum(&device, w, h);
        let compute_bg = integrator.compute_bind_group(&device, &accum);
        let readback = make_readback();
        let mut rec_off = Recorder::default();
        for frame in 0..(warmup + frames) {
            let t = Instant::now();
            trace_frame(
                &device,
                &queue,
                &integrator,
                &compute_bg,
                camera,
                &scene,
                w,
                h,
                &int_params,
                None,
            );
            let trace_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            readback_frame(&device, &queue, &accum, &readback, bytes);
            let read_ms = t.elapsed().as_secs_f64() * 1e3;
            if frame >= warmup {
                rec_off.push("trace (no medium, GPU)", trace_ms);
                rec_off.push("readback", read_ms);
            }
        }
        print_table(pose_name, "STATIC", &rec_off, budget_ms);

        let off_total = rec_off.total_mean();
        let med_trace = rec_med.stats(0).0;
        let off_trace = rec_off.stats(0).0;
        println!(
            "| {pose_name} | derived | medium march (MED−STATIC trace) | {:8.3} |        |          |          |",
            med_trace - off_trace
        );
        println!(
            "| {pose_name} | derived | living-world price (ON−STATIC total) | {:8.3} |        |          |          |",
            on_total - off_total
        );
        summaries.push((
            pose_name.to_string(),
            on_total,
            off_total,
            med_trace - off_trace,
        ));
        std::hint::black_box(on_trace);
    }

    println!("[audit] ── VERDICT (budget {budget_ms:.2} ms) ──");
    for (pose, on, off, med) in &summaries {
        println!(
            "[audit]   {pose}: DYN-ON {on:.2} ms ({}) · STATIC {off:.2} ms ({}) · medium march ≈ {med:.2} ms · living-world price ≈ {:.2} ms",
            if *on <= budget_ms { "PASS" } else { "FAIL" },
            if *off <= budget_ms { "PASS" } else { "FAIL" },
            on - off,
        );
    }
}
