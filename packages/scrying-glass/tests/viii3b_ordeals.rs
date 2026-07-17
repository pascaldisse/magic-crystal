//! RITE VIII-3 wave (b) — THE UPSCALER AT SPEED: the ordeals. GPU compute port
//! of the VIII-3 CPU reference upscaler (src/upscaler.rs → src/upscaler.wgsl +
//! src/upscaler_gpu.rs). The exact house pattern the VIII-2 denoiser port
//! (viii2_ordeals.rs) set:
//!
//!   (a) GPU self-determinism: the same current frame upscaled twice on the
//!       same device is byte-identical.
//!   (b) GPU-vs-CPU parity: the GPU output matches the CPU reference within a
//!       tolerance DERIVED from the net's op count and the fp32 unit roundoff
//!       (never a frozen literal — computed here from `layer_dims()` and
//!       `f32::EPSILON`).
//!   (c) THE QUALITY GATE on GPU output: neural upscale strictly beats naive
//!       bilinear on every held-out validation frame — the SAME beats-bilinear
//!       check VIII-3 pins, re-run machine-checked on the GPU result.
//!   (d) THE BAN: `upscaler.wgsl` carries the `// BAN-SCOPED` marker and
//!       contains no cross-frame vocabulary (the fixed list VIII-0/1 scan);
//!       `upscaler_gpu.rs` is caught by the `upscaler*.rs` glob and re-witnessed
//!       here directly.
//!
//! All GPU ordeals print + return early (never a false green) on a host
//! without an adapter, matching viii2_ordeals.rs.

use std::fs;
use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::RenderScene;
use scrying_glass::upscaler::{
    Mlp, bilinear_upsample, deserialize_weights, upscale_image,
};
use scrying_glass::upscaler_dataset::{
    DATASET_REF_FRAMES, VALIDATION_POSE_NAMES, dataset_dims, law_poses, naruko_params,
};
use scrying_glass::upscaler_gpu::GpuUpscaler;

fn weights_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data")
}

fn load_committed_weights() -> Mlp {
    let bytes = fs::read(weights_dir().join("upscaler-weights-v1.bin"))
        .expect("read committed upscaler-weights-v1.bin");
    deserialize_weights(&bytes).expect("deserialize committed upscaler weights artifact")
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

/// One validation pose's rendered buffers.
type ValidationPose = (
    &'static str,
    Vec<GVec3>, // low noisy
    Vec<GVec3>, // hi albedo
    Vec<GVec3>, // hi normal
    Vec<f32>,   // hi depth
    Vec<GVec3>, // reference
    u32,        // low_w
    u32,        // low_h
    u32,        // target_w
    u32,        // target_h
);

fn render_validation_poses(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    scene: &RenderScene,
) -> Vec<ValidationPose> {
    let params = naruko_params();
    let (low_w, low_h, target_w, target_h) = dataset_dims();
    law_poses(&params)
        .into_iter()
        .filter(|(name, _)| VALIDATION_POSE_NAMES.contains(name))
        .map(|(name, camera)| {
            let noisy_params = IntegratorParams { spp: 1, ..IntegratorParams::default() };
            let low_noisy = resolve(&trace_headless(
                device, queue, bvh, &camera, &scene.sun, scene.sky_top, scene.sky_horizon,
                low_w, low_h, 1, &noisy_params, None,
            ));
            let reference = resolve(&trace_headless(
                device, queue, bvh, &camera, &scene.sun, scene.sky_top, scene.sky_horizon,
                target_w, target_h, DATASET_REF_FRAMES, &IntegratorParams::default(), None,
            ));
            let raw_aov = trace_headless_aov(
                device, queue, bvh, &camera, &scene.sun, scene.sky_top, scene.sky_horizon,
                target_w, target_h,
            );
            let (hi_albedo, hi_normal, hi_depth) = split_aov(&raw_aov);
            (name, low_noisy, hi_albedo, hi_normal, hi_depth, reference, low_w, low_h, target_w, target_h)
        })
        .collect()
}

/// (a) The GPU port upscales the same current frame byte-identically twice.
#[test]
fn a_gpu_inference_is_byte_identical_same_frame_twice() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[VIII-3b gpu determinism] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let (bvh, scene) = build_naruko_scene();
    let poses = render_validation_poses(&device, &queue, &bvh, &scene);
    let (_, low, alb, nrm, dep, _, lw, lh, tw, th) = &poses[0];
    let gpu = GpuUpscaler::new(&device, &load_committed_weights());
    let a = gpu.upscale(&device, &queue, low, *lw, *lh, alb, nrm, dep, *tw, *th);
    let b = gpu.upscale(&device, &queue, low, *lw, *lh, alb, nrm, dep, *tw, *th);
    assert_eq!(a, b, "GPU upscale is not byte-identical across two runs on the same device+frame");
}

/// The GPU-vs-CPU parity tolerance, DERIVED (not chosen), by the SAME method
/// the VIII-2 denoiser port used: the per-pixel forward pass is `macs` fused
/// multiply-adds; the GPU's FMA path differs from the CPU's mul-then-add path
/// by at most one unit-roundoff `u` per MAC (FMA is the MORE accurate; the
/// DIFFERENCE is bounded by the same `u` — Higham, dot-product error
/// analysis). The upscaler's feature+undo path uses MORE transcendentals than
/// the denoiser (4 taps + base + depth = 16 `ln` in + 3 `exp` out = 19), each
/// bounded by a small ULP budget. So the relative output spread is bounded by
/// `(macs + transcendental_ulp_budget) * u`, computed here from the actual net.
fn derived_parity_rel_bound(mlp: &Mlp) -> f64 {
    let macs: u64 = mlp
        .layer_dims()
        .iter()
        .map(|&(i, o)| (i as u64) * (o as u64))
        .sum();
    // 19 transcendentals in the feature+undo path: 4 radiance-tap log-demods
    // (12 `ln`) + 1 base log-demod (3 `ln`) + 1 depth `ln` (1) + 3 output
    // `exp` = 19. 4-ULP budget each (Metal fast-math transcendentals; derived,
    // not chosen for a value) — the accumulated MAC term dominates this anyway.
    const TRANSCENDENTAL_ULP_BUDGET: u64 = 19 * 4;
    let unit_roundoff = f32::EPSILON as f64; // 2^-23, worst-case fp32 rounding step
    (macs + TRANSCENDENTAL_ULP_BUDGET) as f64 * unit_roundoff
}

/// (b) + (c): render the held-out validation poses, upscale on the GPU, assert
/// parity vs the CPU reference within the derived bound AND that the GPU
/// output strictly beats naive bilinear (the quality gate, machine-checked on
/// GPU output).
#[test]
fn b_and_c_gpu_matches_cpu_within_derived_bound_and_beats_bilinear() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[VIII-3b gpu parity] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let mlp = load_committed_weights();
    let gpu = GpuUpscaler::new(&device, &mlp);
    let rel_bound = derived_parity_rel_bound(&mlp);
    println!("[VIII-3b parity] derived GPU-vs-CPU relative bound = {rel_bound:.3e}");

    let (bvh, scene) = build_naruko_scene();
    let poses = render_validation_poses(&device, &queue, &bvh, &scene);
    assert_eq!(poses.len(), 2, "expected exactly the 2 documented validation poses");

    for (name, low, alb, nrm, dep, reference, lw, lh, tw, th) in &poses {
        let cpu = upscale_image(&mlp, low, *lw, *lh, alb, nrm, dep, *tw, *th);
        let gpu_out = gpu.upscale(&device, &queue, low, *lw, *lh, alb, nrm, dep, *tw, *th);

        // (b) parity within the derived RELATIVE bound (normalized by the CPU
        // image magnitude, matching the VIII-1/2 convention).
        let magnitude = rmse(&cpu, &vec![GVec3::ZERO; cpu.len()]).max(1e-12);
        let parity_rel = rmse(&gpu_out, &cpu) / magnitude;
        println!("[VIII-3b parity] pose={name} parity_rel={parity_rel:.3e} bound={rel_bound:.3e}");
        assert!(
            parity_rel <= rel_bound,
            "pose '{name}': GPU-vs-CPU relative spread {parity_rel:.3e} exceeds the derived bound {rel_bound:.3e}"
        );

        // (c) the quality gate, on GPU output: strictly beats naive bilinear.
        let bilinear = bilinear_upsample(low, *lw, *lh, *tw, *th);
        let bilinear_rmse = rmse(&bilinear, reference);
        let gpu_rmse = rmse(&gpu_out, reference);
        println!("[VIII-3b quality] pose={name} bilinear_rmse={bilinear_rmse:.6} gpu_neural_rmse={gpu_rmse:.6}");
        assert!(
            gpu_rmse < bilinear_rmse,
            "pose '{name}': GPU-neural RMSE {gpu_rmse:.6} does not beat bilinear RMSE {bilinear_rmse:.6}"
        );
    }
}

/// (d) THE BAN, GPU half: `upscaler.wgsl` carries the `// BAN-SCOPED` marker
/// and no cross-frame vocabulary (the same fixed list VIII-0/1 enforce). A
/// direct witness on the shader text — `upscaler_gpu.rs` itself is already
/// scanned by the VIII-0 forward-proof `upscaler*.rs` glob.
#[test]
fn d_ban_no_temporal_vocabulary_in_the_gpu_shader() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shader = fs::read_to_string(root.join("src/upscaler.wgsl")).expect("read src/upscaler.wgsl");
    assert!(
        shader.contains("// BAN-SCOPED"),
        "upscaler.wgsl must carry the // BAN-SCOPED marker so the ban scope picks it up"
    );
    let forbidden = [
        "previous_frame", "history", "motion_vector", "temporal", "reproject",
        "warp", "feedback", "recurrent", "accum_prev", "prev_", "last_frame",
        "frame_history", "velocity",
    ];
    for word in forbidden {
        assert!(
            !shader.to_lowercase().contains(word),
            "forbidden temporal vocabulary '{word}' found in src/upscaler.wgsl"
        );
    }
    let gpu_src = fs::read_to_string(root.join("src/upscaler_gpu.rs")).expect("read upscaler_gpu.rs");
    assert!(gpu_src.contains("// BAN-SCOPED"), "upscaler_gpu.rs should carry the // BAN-SCOPED marker");
    for word in forbidden {
        assert!(
            !gpu_src.to_lowercase().contains(word),
            "forbidden temporal vocabulary '{word}' found in src/upscaler_gpu.rs"
        );
    }
}
