//! LIVE-LOOP AUDIT — MEASUREMENT ONLY (window-playable lane).
//!
//! The contradiction: the merged 60-FPS `perf_audit` PASSES (front 11.26 ms /
//! wide 13.23 ms at 900x600, judged on the CPU/GPU-OVERLAP pipelined wall) yet
//! the LIVE window's HUD reads 27.5-34.4 ms (~30 fps) at the vista pose. This
//! harness replicates `run_render_loop`'s EXACT per-frame sequence — NOT the
//! audit's overlap shape — at the live defaults (640x480, spp 2, NO medium in
//! the surface path, vista pose) and phase-splits it, then measures the same
//! frame under the proven overlap lever for the delta.
//!
//! Live serial per-frame (mirrors `Renderer::advance_world` + `Renderer::render`
//! minus the surface `present`, which a headless host cannot vsync):
//!   skin   — `command_bodies_walked(0, walker)`  (SAMA re-skin)
//!   tick   — `scene.tick()`                       (kami + rigid physics)
//!   splice — `DynamicSplice::update`              (refit/rebuild + merge)
//!   upload — `update_bvh` + compute bind-group    (buffer rebuild)
//!   trace  — dispatch (spp accum, NO medium) + `poll(wait)`  (GPU wait)
//!   blit   — offscreen blit + surface-shaped blit + `poll(wait)`
//!
//! The serial SUM is the true live CPU+GPU frame cost (the loop never overlaps
//! CPU and GPU — it submits then waits, present pass excepted). The OVERLAP
//! total is the pipelined wall the audit already proved byte-identical.
//!
//! Run:  cargo run -p scrying-glass --release --example live_loop_audit

use std::path::Path;
use std::time::Instant;

use crystal::{Core, load_world_dir};
use glam::Vec3 as GVec3;
use sama::GaitParams;
use scrying_glass::bvh::{Bvh, BvhParams, DynamicSplice, RefitParams, SpliceKind};
use scrying_glass::integrator::{
    Integrator, IntegratorParams, IntegratorUniform, headless_device,
};
use scrying_glass::scene::{
    Camera, RenderScene, SceneParameters, SunDefaults, WalkerPose, contact_passing_ticks,
};
use vessel::{Body, Preset};

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Naruko authoring dials — mirror `perf_audit::naruko_params` verbatim; only
/// the camera is overridden to the vista the Architect actually stands at.
fn naruko_params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: env_f32("GAIA_NATIVE_FOV", 60.0),
        near: 0.1,
        far: 4_000.0,
        sky_top: "#20152f".into(),
        sky_horizon: "#9a627d".into(),
        mesh_color: "#9aa0a6".into(),
        radial_segments: 24,
        camera_position: [
            env_f32("GAIA_NATIVE_SPAWN_X", 0.0),
            env_f32("GAIA_NATIVE_SPAWN_Y", 1.7),
            env_f32("GAIA_NATIVE_SPAWN_Z", 24.0),
        ],
        camera_yaw: env_f32("GAIA_NATIVE_CAMERA_YAW", 0.0),
        camera_pitch: env_f32("GAIA_NATIVE_CAMERA_PITCH", 0.0),
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
    fn median(&self, i: usize) -> f64 {
        let mut xs = self.samples[i].clone();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    }
    fn stats(&self, i: usize) -> (f64, f64, f64) {
        let xs = &self.samples[i];
        let n = xs.len().max(1) as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mean, min, max)
    }
    fn median_total(&self) -> f64 {
        (0..self.names.len()).map(|i| self.median(i)).sum()
    }
}

fn print_table(label: &str, rec: &Recorder, budget_ms: f64) {
    println!("| config | phase | median ms | mean | min | max |");
    println!("|--------|-------|-----------|------|-----|-----|");
    for i in 0..rec.names.len() {
        let med = rec.median(i);
        let (mean, min, max) = rec.stats(i);
        println!(
            "| {label} | {} | {med:8.3} | {mean:8.3} | {min:8.3} | {max:8.3} |",
            rec.names[i]
        );
    }
    let total = rec.median_total();
    let verdict = if total <= budget_ms { "PASS" } else { "FAIL" };
    println!(
        "| {label} | TOTAL (median sum) | {total:8.3} |          |          |          |  {verdict} ({:.1} fps)",
        1000.0 / total.max(1e-9)
    );
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[live-audit] no GPU adapter on this host — cannot measure");
    };

    let w = env_u32("GAIA_NATIVE_RENDER_W", 640);
    let h = env_u32("GAIA_NATIVE_RENDER_H", 480);
    let surf_w = env_u32("GAIA_NATIVE_WIDTH", 960);
    let surf_h = env_u32("GAIA_NATIVE_HEIGHT", 640);
    let warmup = env_u32("GAIA_AUDIT_WARMUP", 8);
    let frames = env_u32("GAIA_AUDIT_FRAMES", 80);
    let budget_ms = 1000.0 / 60.0;

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();
    let mut scene =
        RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("render scene");

    let mut int_params = IntegratorParams::default();
    int_params.spp = env_u32("GAIA_NATIVE_SPP", 2);
    int_params.max_bounces = env_u32("GAIA_NATIVE_MAX_BOUNCES", 4);

    let bvh_params = BvhParams::default();
    let static_bvh = Bvh::build(&scene.leaf_triangles(), &bvh_params);

    // Warm to the composed mid-stride steady state (coexist precedent).
    let body = Body::from_preset(&Preset::nari());
    let gait = GaitParams::walk();
    let cycle = (1.0 / (gait.cadence * gait.dt)).round().max(1.0) as u64;
    let (_c, passing_tick) = contact_passing_ticks(&body, &gait);
    let mut target = 150u64;
    while target < passing_tick + cycle || target % cycle != passing_tick % cycle {
        target += 1;
    }
    let vista = WalkerPose {
        position: GVec3::from_array(params.camera_position),
        yaw: params.camera_yaw,
    };
    for _ in 0..target {
        scene.command_bodies_walked(0.0, Some(vista));
        scene.tick();
    }

    let camera = Camera {
        eye: GVec3::from_array(params.camera_position),
        yaw: params.camera_yaw,
        pitch: params.camera_pitch,
        fov_y_radians: params.fov_y_degrees.to_radians(),
        near: params.near,
        far: params.far,
    };
    eprintln!(
        "[live-audit] {w}x{h} internal, surface {surf_w}x{surf_h}, spp {}, vista eye={:?} yaw={} — {} static / {} dynamic tris, {} bodies",
        int_params.spp,
        params.camera_position,
        params.camera_yaw,
        scene.leaf_triangles().len(),
        scene.dynamic_leaf_triangles().len(),
        scene.bodies.len(),
    );

    let refit_params = RefitParams::default();

    // Two surface-shaped blit targets to mirror render()'s two blits
    // (offscreen present + surface present).
    let make_target = |lbl: &str| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(lbl),
                size: wgpu::Extent3d {
                    width: surf_w,
                    height: surf_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    };
    let off_view = make_target("offscreen present");
    let surf_view = make_target("surface present");

    // Snapshot the invariant lighting so the per-frame uniform builder does not
    // hold a borrow on `scene` (which every frame mutably ticks).
    let sun = scene.sun.clone();
    let sky_top = scene.sky_top;
    let sky_horizon = scene.sky_horizon;
    let uniform_for = |integrator: &Integrator, samples_before: u32| {
        IntegratorUniform::build(
            &camera,
            &sun,
            sky_top,
            sky_horizon,
            w,
            h,
            integrator.node_count,
            integrator.tri_count,
            samples_before,
            &int_params,
            None, // the live surface path binds NO medium
        )
    };

    // ── SERIAL — the live loop shape exactly ────────────────────────────────
    {
        let mut splice = DynamicSplice::build(
            &static_bvh,
            &scene.dynamic_leaf_triangles(),
            &bvh_params.dynamic(),
            refit_params,
        );
        let mut integrator = Integrator::new(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &splice.merged,
            None,
        );
        let accum = integrator.make_accum(&device, w, h);
        let mut rec = Recorder::default();
        let mut refits = 0u32;
        let mut rebuilds = 0u32;
        // clone the warmed scene state is not needed — dynamics move on, but the
        // vista pose + steady stride keep frame cost representative.
        for frame in 0..(warmup + frames) {
            let measured = frame >= warmup;

            let t = Instant::now();
            let animating = scene.command_bodies_walked(0.0, Some(vista));
            let skin_ms = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            scene.tick();
            let tick_ms = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            let kind = splice.update(&static_bvh, &scene.dynamic_leaf_triangles());
            let splice_ms = t.elapsed().as_secs_f64() * 1e3;
            match kind {
                SpliceKind::Refit => refits += 1,
                SpliceKind::Rebuilt => rebuilds += 1,
            }

            let t = Instant::now();
            integrator.update_bvh(&device, &splice.merged);
            let compute_bg = integrator.compute_bind_group(&device, &accum);
            let blit_bg = integrator.blit_bind_group(&device, &accum);
            let upload_ms = t.elapsed().as_secs_f64() * 1e3;

            // Moving geometry resets accumulation every frame (2spp live).
            let samples_before = 0u32;
            let t = Instant::now();
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("live trace"),
            });
            integrator.dispatch(&queue, &mut enc, &uniform_for(&integrator, samples_before), &compute_bg, w, h);
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let trace_ms = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("live blit"),
            });
            integrator.blit(&mut enc, &off_view, &blit_bg, "offscreen present");
            integrator.blit(&mut enc, &surf_view, &blit_bg, "surface present");
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let blit_ms = t.elapsed().as_secs_f64() * 1e3;

            let _ = animating;
            if measured {
                rec.push("skin (command_bodies_walked)", skin_ms);
                rec.push("tick (physics+kami)", tick_ms);
                rec.push("splice (refit/rebuild+merge)", splice_ms);
                rec.push("upload (update_bvh+bg)", upload_ms);
                rec.push("trace (spp, no medium, GPU wait)", trace_ms);
                rec.push("blit (2x present, GPU wait)", blit_ms);
            }
        }
        eprintln!(
            "[live-audit] SERIAL splice: {refits} refits, {rebuilds} rebuilds over {} frames",
            warmup + frames
        );
        print_table("SERIAL (live loop)", &rec, budget_ms);
    }

    // ── OVERLAP — the proven pipelined lever wired into the live shape ──────
    {
        let mut splice = DynamicSplice::build(
            &static_bvh,
            &scene.dynamic_leaf_triangles(),
            &bvh_params.dynamic(),
            refit_params,
        );
        let mut integrator = Integrator::new(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &splice.merged,
            None,
        );
        let accum = integrator.make_accum(&device, w, h);
        let mut rec = Recorder::default();
        let mut pending: Option<wgpu::SubmissionIndex> = None;
        let mut frame_start = Instant::now();
        for frame in 0..(warmup + frames) {
            let measured = frame >= warmup;
            let this_frame_start = frame_start;

            scene.command_bodies_walked(0.0, Some(vista));
            scene.tick();
            splice.update(&static_bvh, &scene.dynamic_leaf_triangles());
            integrator.update_bvh(&device, &splice.merged);
            let compute_bg = integrator.compute_bind_group(&device, &accum);
            let blit_bg = integrator.blit_bind_group(&device, &accum);

            // Submit THIS frame's trace + blits WITHOUT waiting.
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("live overlap frame"),
            });
            integrator.dispatch(&queue, &mut enc, &uniform_for(&integrator, 0), &compute_bg, w, h);
            integrator.blit(&mut enc, &off_view, &blit_bg, "offscreen present");
            integrator.blit(&mut enc, &surf_view, &blit_bg, "surface present");
            let idx = queue.submit(Some(enc.finish()));

            // Complete the PREVIOUS frame (it ran on the GPU while this frame's
            // CPU stages executed above).
            if let Some(prev) = pending.take() {
                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: Some(prev),
                    timeout: None,
                });
            }
            pending = Some(idx);

            let next = Instant::now();
            if measured {
                rec.push("frame (wall, pipelined)", next.duration_since(this_frame_start).as_secs_f64() * 1e3);
            }
            frame_start = next;
        }
        if let Some(prev) = pending.take() {
            let _ = device.poll(wgpu::PollType::Wait { submission_index: Some(prev), timeout: None });
        }
        print_table("OVERLAP (live loop)", &rec, budget_ms);
    }

    println!("[live-audit] budget {budget_ms:.2} ms (60 FPS law). Serial > budget in (16.67, 33.3) ⇒ Fifo vsync halves to ~30 fps live.");
}
