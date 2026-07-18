//! TEACHER/BENCHMARK SURFACE (ITEM 16, de-chartered — was "THE ONE RENDER PATH") — production-resolution budget measurement. Times the
//! two neural compute passes of the chartered path at the REAL production
//! resolutions (trace/denoise 640×480 low → upscale ×UPSCALE_SCALE to the
//! surface) via GPU timestamp queries on THIS host's adapter, and prints an
//! honest phase table against the 16.67 ms / 60-fps wall.
//!
//! Not a gate — a measuring instrument (the numbers feed docs + the wall
//! verdict). Run: cargo run -p scrying-glass --release --example onepath_budget

use std::path::Path;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::deserialize_weights as denoiser_weights;
use scrying_glass::denoiser_gpu::{GpuDenoiser, headless_device_timed};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, RenderScene};
use scrying_glass::upscaler::{deserialize_weights as upscaler_weights, upscale_image};
use scrying_glass::upscaler_dataset::{
    PRODUCTION_LOW_HEIGHT, PRODUCTION_LOW_WIDTH, UPSCALE_SCALE, naruko_params,
};
use scrying_glass::upscaler_gpu::{GpuUpscaler, headless_device_f16_timed};

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let Some((device, queue)) = headless_device_timed() else {
        eprintln!("no GPU adapter on this host — cannot measure");
        return;
    };
    let timed = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    if !timed {
        eprintln!("adapter lacks TIMESTAMP_QUERY — cannot bracket compute passes; aborting");
        return;
    }

    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());

    let low_w = PRODUCTION_LOW_WIDTH;
    let low_h = PRODUCTION_LOW_HEIGHT;
    let target_w = low_w * UPSCALE_SCALE;
    let target_h = low_h * UPSCALE_SCALE;
    println!(
        "ONE RENDER PATH budget — trace/denoise {low_w}x{low_h} → upscale x{UPSCALE_SCALE} → {target_w}x{target_h}"
    );

    // A representative pose (naruko front).
    let camera = Camera {
        eye: glam::Vec3::new(0.0, 1.6, 6.0),
        yaw: 0.0,
        pitch: 0.0,
        fov_y_radians: 55f32.to_radians(),
        near: 0.05,
        far: 1000.0,
    };

    // Low-res noisy radiance (1 spp) — the denoiser input at 640×480.
    let noisy_params = IntegratorParams {
        spp: 1,
        ..IntegratorParams::default()
    };
    let low_noisy = resolve(&trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        low_w,
        low_h,
        1,
        &noisy_params,
        None,
    ));
    // Low-res AOVs for the denoise pass.
    let low_aov = trace_headless_aov(
        &device,
        &queue,
        &bvh,
        &camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        low_w,
        low_h,
    );
    let (low_albedo, low_normal, low_depth) = split_aov(&low_aov);
    // Target-res AOVs for the upscale pass (geometry-only, full res).
    let hi_aov = trace_headless_aov(
        &device,
        &queue,
        &bvh,
        &camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        target_w,
        target_h,
    );
    let (hi_albedo, hi_normal, hi_depth) = split_aov(&hi_aov);

    let denoiser = GpuDenoiser::new(
        &device,
        &denoiser_weights(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("data/denoiser-weights-v1.bin"),
            )
            .expect("denoiser weights"),
        )
        .expect("deserialize denoiser"),
    );
    let upscaler = GpuUpscaler::new(
        &device,
        &upscaler_weights(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("data/upscaler-weights-v1.bin"),
            )
            .expect("upscaler weights"),
        )
        .expect("deserialize upscaler"),
    );

    const REPEATS: u32 = 32;

    // The chartered path denoises the LOW frame (640×480), then upscales the
    // denoised radiance to the surface. Measure both passes at those exact
    // shapes.
    let mut denoise_ms = denoiser
        .time_dispatches_ms(
            &device,
            &queue,
            &low_noisy,
            &low_albedo,
            &low_normal,
            &low_depth,
            low_w,
            low_h,
            REPEATS,
        )
        .expect("timed denoise");
    // Upscale input is the low-res (denoised) radiance; here we time the pass
    // shape with the low frame (the pass cost is independent of the radiance
    // values, only the shapes/weights).
    let mut upscale_ms = upscaler
        .time_dispatches_ms(
            &device, &queue, &low_noisy, low_w, low_h, &hi_albedo, &hi_normal, &hi_depth, target_w,
            target_h, REPEATS,
        )
        .expect("timed upscale");

    let d_min = denoise_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let d_med = median(&mut denoise_ms);
    let u_min = upscale_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let u_med = median(&mut upscale_ms);

    // FAST upscaler (fp16 threadgroup-cached, MODE A) on an f16 device.
    let mut uf_med = f64::NAN;
    let mut uf_min = f64::NAN;
    let mut fast_parity = f64::NAN;
    if let Some((fdev, fq)) = headless_device_f16_timed() {
        let fast_mlp = upscaler_weights(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("data/upscaler-weights-v1.bin"),
            )
            .unwrap(),
        )
        .unwrap();
        let fast = GpuUpscaler::new_fast(&fdev, &fast_mlp);
        // parity vs CPU reference (a small correctness witness alongside timing).
        let cpu = upscale_image(
            &fast_mlp, &low_noisy, low_w, low_h, &hi_albedo, &hi_normal, &hi_depth, target_w,
            target_h,
        );
        let gpu = fast.upscale(
            &fdev, &fq, &low_noisy, low_w, low_h, &hi_albedo, &hi_normal, &hi_depth, target_w,
            target_h,
        );
        let mag = rmse(&cpu, &vec![glam::Vec3::ZERO; cpu.len()]).max(1e-12);
        fast_parity = (rmse(&gpu, &cpu) / mag) as f64;
        if let Some(mut ms) = fast.time_dispatches_ms(
            &fdev, &fq, &low_noisy, low_w, low_h, &hi_albedo, &hi_normal, &hi_depth, target_w,
            target_h, REPEATS,
        ) {
            uf_min = ms.iter().cloned().fold(f64::INFINITY, f64::min);
            uf_med = median(&mut ms);
        }
    } else {
        eprintln!("[fast] host lacks SHADER_F16 — fast upscaler not measured");
    }

    println!("\n=== PHASE TABLE (GPU compute pass, timestamp-bracketed, median of {REPEATS}) ===");
    println!("  denoise  {low_w}x{low_h}       min {d_min:8.3} ms  median {d_med:8.3} ms");
    println!(
        "  upscale (naive fp32)  {target_w}x{target_h}  min {u_min:8.3} ms  median {u_med:8.3} ms"
    );
    println!(
        "  upscale (FAST f16-tg) {target_w}x{target_h}  min {uf_min:8.3} ms  median {uf_med:8.3} ms  parity_rel(vs CPU)={fast_parity:.3e}"
    );
    let combined = d_med + if uf_med.is_nan() { u_med } else { uf_med };
    println!("  (combined below uses FAST upscale when available)");
    println!("  denoise+upscale combined median: {combined:7.3} ms");
    println!("  60-fps wall: 16.667 ms;  neural headroom budget target: ~8.6 ms");
    if combined <= 8.6 {
        println!("  VERDICT: neural stages fit the ~8.6 ms headroom.");
    } else if combined <= 16.667 {
        println!(
            "  VERDICT: neural stages exceed the ~8.6 ms overlap headroom but fit under 16.67 ms raw."
        );
    } else {
        println!(
            "  VERDICT: neural stages EXCEED the 16.67 ms wall — over budget by {:.3} ms.",
            combined - 16.667
        );
    }
}
