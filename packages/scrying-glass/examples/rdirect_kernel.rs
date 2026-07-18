//! R-DIRECT GPU KERNEL — MEASURE. The fused single-dispatch kernel timed on
//! THIS M1 at the real shapes, with a GPU-vs-CPU parity gate.
//!
//! For two shapes — the SPIKE (low 48×32 → native 96×64, where the net was
//! trained/validated) and NATIVE window (low 640×480 → native 960×640, the
//! canonical present size from src/main.rs GAIA_NATIVE_* defaults) — it:
//!   - traces a LOW 1-spp radiance frame + the NATIVE-res G-buffer (motion=0,
//!     static-pose honest gap);
//!   - runs the CPU reference (`direct_render_image`) and BOTH GPU kernels
//!     (f32 `rdirect.wgsl`, fp16 MODE A `rdirect_fast.wgsl`);
//!   - asserts GPU-vs-CPU parity within a DERIVED bound (fp32 for the f32
//!     kernel; fp32 + f16-storage term for the fast kernel);
//!   - times each kernel's compute pass via GPU TIMESTAMP_QUERY (warm-up +
//!     median/min of many dispatches) and prints ms vs the 16.67ms 60fps
//!     budget.
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_kernel

use std::path::Path;

use glam::{Vec2, Vec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::denoiser_dataset::{law_poses, naruko_params};
use scrying_glass::rdirect::{Mlp, RdirectConfig, deserialize_weights, direct_render_image};
use scrying_glass::rdirect_gpu::{GpuRdirect, headless_device_f16_timed, headless_device_timed};
use scrying_glass::scene::RenderScene;

/// One measured shape's traced buffers.
struct Frame {
    label: &'static str,
    low_radiance: Vec<Vec3>,
    low_w: u32,
    low_h: u32,
    hi_albedo: Vec<Vec3>,
    hi_normal: Vec<Vec3>,
    hi_depth: Vec<f32>,
    hi_motion: Vec<Vec2>,
    target_w: u32,
    target_h: u32,
}

#[allow(clippy::too_many_arguments)]
fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    scene: &RenderScene,
    camera: &scrying_glass::scene::Camera,
    label: &'static str,
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
) -> Frame {
    let noisy_params = IntegratorParams { spp: 1, ..IntegratorParams::default() };
    let low_radiance = resolve(&trace_headless(
        device, queue, bvh, camera, &scene.sun, scene.sky_top, scene.sky_horizon,
        low_w, low_h, 1, &noisy_params, None,
    ));
    let (hi_albedo, hi_normal, hi_depth) = split_aov(&trace_headless_aov(
        device, queue, bvh, camera, &scene.sun, scene.sky_top, scene.sky_horizon,
        target_w, target_h,
    ));
    let hi_motion = vec![Vec2::ZERO; (target_w * target_h) as usize];
    eprintln!("[rdirect-kernel] traced '{label}' low={low_w}x{low_h} native={target_w}x{target_h}");
    Frame {
        label, low_radiance, low_w, low_h, hi_albedo, hi_normal, hi_depth, hi_motion,
        target_w, target_h,
    }
}

fn cpu_render(mlp: &Mlp, f: &Frame) -> Vec<Vec3> {
    direct_render_image(
        mlp, &f.low_radiance, f.low_w, f.low_h, &f.hi_albedo, &f.hi_normal, &f.hi_depth,
        &f.hi_motion, f.target_w, f.target_h,
    )
}

/// Derived fp32 GPU-vs-CPU relative parity bound (Higham dot-product analysis,
/// same method as viii3b): (macs + transcendental ULP budget) · u32.
fn f32_bound(mlp: &Mlp) -> f64 {
    let macs: u64 = mlp.layer_dims().iter().map(|&(i, o)| i as u64 * o as u64).sum();
    // 23-feature path: 4 tap log-demods (12 ln) + 1 depth ln + 3 output exp = 16
    // transcendentals, 4-ULP budget each (Metal fast-math). MAC term dominates.
    const TRANSCENDENTAL_ULP_BUDGET: u64 = 16 * 4;
    let u32_ = f32::EPSILON as f64;
    (macs + TRANSCENDENTAL_ULP_BUDGET) as f64 * u32_
}

/// Derived fp16-storage (MODE A) relative parity bound: f32 accumulate does not
/// compound, so error is dominated by rounding each weight+activation to f16
/// once (~2·u16 rel) plus the fp32 dot-product term (macs·u32). (fp16 verdict.)
fn f16_bound(mlp: &Mlp) -> f64 {
    let macs: u64 = mlp.layer_dims().iter().map(|&(i, o)| i as u64 * o as u64).sum();
    let u16 = 2f64.powi(-11);
    let u32_ = f32::EPSILON as f64;
    2.0 * u16 + (macs as f64) * u32_
}

fn median_min(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = v[0];
    let med = v[v.len() / 2];
    (med, min)
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[rdirect-kernel] no GPU adapter — cannot trace/measure");
    };

    let bytes = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rdirect-weights-v1.bin"),
    )
    .expect("read committed rdirect-weights-v1.bin");
    let mlp = deserialize_weights(&bytes).expect("deserialize committed rdirect weights");
    let macs: u64 = mlp.macs();
    println!("[rdirect-kernel] net: {} layers, macs/pixel={macs}", mlp.layer_dims().len());

    // Build naruko scene once.
    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());

    let poses = law_poses(&params);
    let front = poses.iter().find(|(n, _)| *n == "front").expect("front pose").1.clone();

    // Two shapes: the spike (trained/validated) + native window present.
    let frames = vec![
        render_frame(&device, &queue, &bvh, &scene, &front, "spike 96x64", 48, 32, 96, 64),
        render_frame(&device, &queue, &bvh, &scene, &front, "native 960x640", 640, 480, 960, 640),
    ];

    // GPU devices: f32 timing (TIMESTAMP_QUERY) + fp16 fast (SHADER_F16).
    let f32_gpu = GpuRdirect::new(&device, &mlp);
    // A dedicated TIMESTAMP_QUERY device for the f32 kernel (the default
    // headless device does not request the feature).
    let f32_timed = headless_device_timed().map(|(d, q)| {
        let g = GpuRdirect::new(&d, &mlp);
        (d, q, g)
    });
    let f16_dev = headless_device_f16_timed();

    let bound_f32 = f32_bound(&mlp);
    let bound_f16 = f16_bound(&mlp);
    println!("[rdirect-kernel] derived parity rel bounds: f32={bound_f32:.3e}  fp16-MODE-A={bound_f16:.3e}");
    println!("[rdirect-kernel] 60fps budget = 16.67 ms/frame; target kernel ≤ ~10ms leaves room for the trace\n");

    const WARMUP: u32 = 3;
    const TIMED: u32 = 10;

    let mut all_parity_ok = true;
    for f in &frames {
        println!("── {} (low {}x{} → native {}x{}, {} target px) ──",
            f.label, f.low_w, f.low_h, f.target_w, f.target_h, f.target_w * f.target_h);
        let cpu = cpu_render(&mlp, f);
        let mag = rmse(&cpu, &vec![Vec3::ZERO; cpu.len()]).max(1e-12);

        // ── f32 kernel: parity + timing ──
        let gpu_f32 = f32_gpu.render(
            &device, &queue, &f.low_radiance, f.low_w, f.low_h, &f.hi_albedo, &f.hi_normal,
            &f.hi_depth, &f.hi_motion, f.target_w, f.target_h,
        );
        let parity_f32 = rmse(&gpu_f32, &cpu) / mag;
        let ok_f32 = parity_f32 <= bound_f32;
        all_parity_ok &= ok_f32;
        print!("  f32 kernel  parity_rel={parity_f32:.3e} (bound {bound_f32:.3e}) {}",
            if ok_f32 { "OK" } else { "FAIL" });
        let f32_ms = f32_timed.as_ref().and_then(|(d, q, g)| g.time_dispatches_ms(
            d, q, &f.low_radiance, f.low_w, f.low_h, &f.hi_albedo, &f.hi_normal,
            &f.hi_depth, &f.hi_motion, f.target_w, f.target_h, WARMUP + TIMED,
        ));
        match f32_ms {
            Some(mut ms) => {
                ms.drain(0..WARMUP as usize);
                let (med, min) = median_min(ms);
                println!("  |  GPU pass: median={med:.3}ms min={min:.3}ms  ({:.1}% of 16.67ms budget)",
                    med / 16.67 * 100.0);
            }
            None => println!("  |  (no TIMESTAMP_QUERY device — timing unavailable)"),
        }

        // ── fp16 MODE A fast kernel: parity + timing (if SHADER_F16) ──
        match &f16_dev {
            Some((fdev, fq)) => {
                let fast = GpuRdirect::new_fast(fdev, &mlp);
                let gpu_f16 = fast.render(
                    fdev, fq, &f.low_radiance, f.low_w, f.low_h, &f.hi_albedo, &f.hi_normal,
                    &f.hi_depth, &f.hi_motion, f.target_w, f.target_h,
                );
                let parity_f16 = rmse(&gpu_f16, &cpu) / mag;
                let ok_f16 = parity_f16 <= bound_f16;
                all_parity_ok &= ok_f16;
                print!("  fp16 MODE-A parity_rel={parity_f16:.3e} (bound {bound_f16:.3e}) {}",
                    if ok_f16 { "OK" } else { "FAIL" });
                // Skip the very slow fp16-native timing (~2s/dispatch, occupancy-
                // bound); time only the spike shape. Native fp16 measured
                // separately at ~2056ms (WARMUP8/TIMED40 run).
                let time_this = f.target_w * f.target_h <= 100_000;
                let fast_ms = if time_this {
                    fast.time_dispatches_ms(
                        fdev, fq, &f.low_radiance, f.low_w, f.low_h, &f.hi_albedo, &f.hi_normal,
                        &f.hi_depth, &f.hi_motion, f.target_w, f.target_h, WARMUP + TIMED,
                    )
                } else {
                    None
                };
                match fast_ms {
                    Some(mut ms) => {
                        ms.drain(0..WARMUP as usize);
                        let (med, min) = median_min(ms);
                        println!("  |  GPU pass: median={med:.3}ms min={min:.3}ms  ({:.1}% of budget)",
                            med / 16.67 * 100.0);
                    }
                    None => println!("  |  (native fp16 timing skipped — ~2056ms measured separately)"),
                }
            }
            None => println!("  fp16 MODE-A: host lacks SHADER_F16 — fast kernel unmeasurable here"),
        }
        println!();
    }

    println!("[rdirect-kernel] ALL PARITY GATES {}", if all_parity_ok { "PASSED" } else { "FAILED" });
    assert!(all_parity_ok, "a GPU kernel drifted from the CPU reference beyond its derived bound");

    // ── CONSTRUCTIVE SWEEP: what net size hits the 60fps budget at native? ──
    // Timing-only (random weights — ms depends on MAC/memory shape, not the
    // trained values), f32 no-cache kernel at native 960×640. Answers "what
    // remains": the per-pixel MLP is memory-bound in its weight fetch, so ms
    // scales with MACs/pixel; this finds the params ceiling for ≤10ms.
    if let Some((d, q, _)) = &f32_timed {
        let native = &frames[1];
        println!("── NET-SIZE SWEEP @ native 960x640 (f32 no-cache kernel, timing-only) ──");
        println!("  {:>8} {:>7} {:>10}  {:>10} {:>8}", "layers×w", "macs", "f16 KB", "median ms", "%budget");
        let configs = [(2usize,32usize),(3,32),(3,48),(4,48),(4,64),(5,64)];
        for (hl, hw) in configs {
            let cfg = RdirectConfig { hidden_layers: hl, hidden_width: hw };
            let m = Mlp::new_random(cfg, 1);
            let params: u64 = m.macs() + m.layer_dims().iter().map(|&(_, o)| o as u64).sum::<u64>();
            let g = GpuRdirect::new(d, &m);
            if let Some(mut ms) = g.time_dispatches_ms(
                d, q, &native.low_radiance, native.low_w, native.low_h, &native.hi_albedo,
                &native.hi_normal, &native.hi_depth, &native.hi_motion, native.target_w,
                native.target_h, WARMUP + TIMED,
            ) {
                ms.drain(0..WARMUP as usize);
                let (med, _min) = median_min(ms);
                println!("  {:>6}×{:<2} {:>7} {:>9.1}K  {:>9.2} {:>7.0}%",
                    hl, hw, m.macs(), params as f64 * 2.0 / 1024.0, med, med / 16.67 * 100.0);
            }
        }
    }
}
