//! R-DIRECT GPU ORDEALS — the fused single-dispatch kernel port of the CPU
//! reference direct renderer (src/rdirect.rs → src/rdirect.wgsl / rdirect_fast.wgsl
//! + src/rdirect_gpu.rs). The exact house pattern the VIII-3b upscaler port set:
//!
//!   (a) GPU self-determinism: the same current frame rendered twice on the
//!       same device is byte-identical.
//!   (b) GPU-vs-CPU parity: the f32 GPU output matches the CPU reference
//!       within a tolerance DERIVED from the net's op count + fp32 unit
//!       roundoff (never a frozen literal). The fp16 MODE-A fast kernel matches
//!       within the derived f16-storage bound.
//!   (c) THE BAN: both shaders carry `// BAN-SCOPED` and no cross-frame
//!       vocabulary; `rdirect_gpu.rs` is re-witnessed here.
//!
//! Runs cheap: the SPIKE shape (low 48×32 → native 96×64, where the net was
//! trained/validated). All GPU ordeals print + return early on a host without
//! an adapter (never a false green), matching viii3b_ordeals.rs.

use std::fs;
use std::path::Path;

use glam::{Vec2, Vec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser_dataset::{law_poses, naruko_params};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::rdirect::{Mlp, deserialize_weights, direct_render_image};
use scrying_glass::rdirect_gpu::{GpuRdirect, headless_device_f16_timed};
use scrying_glass::scene::RenderScene;

const LOW_W: u32 = 48;
const LOW_H: u32 = 32;
const TARGET_W: u32 = 96;
const TARGET_H: u32 = 64;

fn load_committed_weights() -> Mlp {
    let bytes = fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rdirect-weights-v1.bin"))
        .expect("read committed rdirect-weights-v1.bin");
    deserialize_weights(&bytes).expect("deserialize committed rdirect weights artifact")
}

fn build_naruko_scene() -> (Bvh, RenderScene) {
    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    (bvh, scene)
}

/// The spike frame: low 1-spp radiance + native G-buffer (motion=0, static).
type Frame = (Vec<Vec3>, Vec<Vec3>, Vec<Vec3>, Vec<f32>, Vec<Vec2>);

fn render_spike_frame(device: &wgpu::Device, queue: &wgpu::Queue) -> Frame {
    let (bvh, scene) = build_naruko_scene();
    let params = naruko_params();
    let camera = law_poses(&params)
        .into_iter()
        .find(|(n, _)| *n == "front")
        .expect("front pose")
        .1;
    let noisy_params = IntegratorParams { spp: 1, ..IntegratorParams::default() };
    let low = resolve(&trace_headless(
        device, queue, &bvh, &camera, &scene.sun, scene.sky_top, scene.sky_horizon,
        LOW_W, LOW_H, 1, &noisy_params, None,
    ));
    let (alb, nrm, dep) = split_aov(&trace_headless_aov(
        device, queue, &bvh, &camera, &scene.sun, scene.sky_top, scene.sky_horizon,
        TARGET_W, TARGET_H,
    ));
    let mot = vec![Vec2::ZERO; (TARGET_W * TARGET_H) as usize];
    (low, alb, nrm, dep, mot)
}

/// Derived fp32 GPU-vs-CPU relative bound (Higham dot-product analysis).
fn f32_bound(mlp: &Mlp) -> f64 {
    let macs: u64 = mlp.layer_dims().iter().map(|&(i, o)| i as u64 * o as u64).sum();
    const TRANSCENDENTAL_ULP_BUDGET: u64 = 16 * 4; // 12 tap ln + 1 depth ln + 3 exp
    (macs + TRANSCENDENTAL_ULP_BUDGET) as f64 * f32::EPSILON as f64
}

/// Derived fp16-storage (MODE A) relative bound: 2·u16 + macs·u32.
fn f16_bound(mlp: &Mlp) -> f64 {
    let macs: u64 = mlp.layer_dims().iter().map(|&(i, o)| i as u64 * o as u64).sum();
    2.0 * 2f64.powi(-11) + (macs as f64) * f32::EPSILON as f64
}

/// (a) The GPU port renders the same current frame byte-identically twice.
#[test]
fn a_gpu_inference_is_byte_identical_same_frame_twice() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[rdirect gpu determinism] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let (low, alb, nrm, dep, mot) = render_spike_frame(&device, &queue);
    let gpu = GpuRdirect::new(&device, &load_committed_weights());
    let a = gpu.render(&device, &queue, &low, LOW_W, LOW_H, &alb, &nrm, &dep, &mot, TARGET_W, TARGET_H);
    let b = gpu.render(&device, &queue, &low, LOW_W, LOW_H, &alb, &nrm, &dep, &mot, TARGET_W, TARGET_H);
    assert_eq!(a.len(), (TARGET_W * TARGET_H) as usize);
    assert_eq!(a, b, "GPU render is not byte-identical across two runs on the same device+frame");
}

/// (b) f32 GPU output matches the CPU reference within the derived fp32 bound.
#[test]
fn b_f32_gpu_matches_cpu_within_derived_bound() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[rdirect gpu parity] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let mlp = load_committed_weights();
    let bound = f32_bound(&mlp);
    let (low, alb, nrm, dep, mot) = render_spike_frame(&device, &queue);
    let cpu = direct_render_image(&mlp, &low, LOW_W, LOW_H, &alb, &nrm, &dep, &mot, TARGET_W, TARGET_H);
    let gpu = GpuRdirect::new(&device, &mlp);
    let gpu_out = gpu.render(&device, &queue, &low, LOW_W, LOW_H, &alb, &nrm, &dep, &mot, TARGET_W, TARGET_H);
    let mag = rmse(&cpu, &vec![Vec3::ZERO; cpu.len()]).max(1e-12);
    let parity = rmse(&gpu_out, &cpu) / mag;
    println!("[rdirect parity] f32 parity_rel={parity:.3e} bound={bound:.3e}");
    assert!(parity <= bound, "f32 GPU-vs-CPU spread {parity:.3e} exceeds derived bound {bound:.3e}");
}

/// (b') fp16 MODE-A fast kernel matches the CPU reference within the derived
/// f16-storage bound — proves the fast kernel is the SAME net, not a drift.
#[test]
fn b2_fp16_fast_kernel_matches_cpu_within_derived_bound() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[rdirect fp16 parity] no GPU adapter — ordeal could not run");
        return;
    };
    let Some((fdev, fq)) = headless_device_f16_timed() else {
        eprintln!("[rdirect fp16 parity] host lacks SHADER_F16 — fast kernel unmeasurable");
        return;
    };
    let mlp = load_committed_weights();
    let bound = f16_bound(&mlp);
    // Render the frame on the default device, run the CPU + fast-GPU nets.
    let (low, alb, nrm, dep, mot) = render_spike_frame(&device, &queue);
    let cpu = direct_render_image(&mlp, &low, LOW_W, LOW_H, &alb, &nrm, &dep, &mot, TARGET_W, TARGET_H);
    let fast = GpuRdirect::new_fast(&fdev, &mlp);
    let gpu = fast.render(&fdev, &fq, &low, LOW_W, LOW_H, &alb, &nrm, &dep, &mot, TARGET_W, TARGET_H);
    let mag = rmse(&cpu, &vec![Vec3::ZERO; cpu.len()]).max(1e-12);
    let parity = rmse(&gpu, &cpu) / mag;
    println!("[rdirect parity] fp16 MODE-A parity_rel={parity:.3e} bound={bound:.3e}");
    assert!(parity <= bound, "fp16 GPU-vs-CPU spread {parity:.3e} exceeds derived bound {bound:.3e}");
}

/// (c) THE BAN: both shaders + the driver carry `// BAN-SCOPED` and no
/// cross-frame vocabulary (the fixed list VIII-0/1 enforce).
#[test]
fn c_ban_no_temporal_vocabulary_in_the_gpu_kernel() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let forbidden = [
        "previous_frame", "history", "motion_vector", "temporal", "reproject",
        "warp", "feedback", "recurrent", "accum_prev", "prev_", "last_frame",
        "frame_history", "velocity",
    ];
    for rel in ["src/rdirect.wgsl", "src/rdirect_fast.wgsl", "src/rdirect_gpu.rs"] {
        let text = fs::read_to_string(root.join(rel)).unwrap_or_else(|_| panic!("read {rel}"));
        assert!(text.contains("// BAN-SCOPED"), "{rel} must carry the // BAN-SCOPED marker");
        for word in forbidden {
            assert!(
                !text.to_lowercase().contains(word),
                "forbidden temporal vocabulary '{word}' found in {rel}"
            );
        }
    }
}
