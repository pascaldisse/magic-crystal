//! RITE VIII-2 — THE DREAM AT SPEED: the ordeals. GPU compute port of the
//! VIII-1 CPU reference denoiser. See
//! docs/proposals/RITE-VIII-THE-DREAM-DENOISER.md §VIII-2.
//!
//!   (a) GPU self-determinism: the same current frame denoised twice on the
//!       same device is byte-identical.
//!   (b) GPU-vs-CPU parity: the GPU output matches the CPU reference within a
//!       tolerance DERIVED from the net's op count and the fp32 unit roundoff
//!       (never a frozen literal — computed here from `layer_dims()` and
//!       `f32::EPSILON`). The measured spread sits far under the bound.
//!   (c) THE QUALITY GATE on GPU output: denoised strictly beats noisy on
//!       every held-out validation frame — the SAME beats-noisy check VIII-1
//!       pins, re-run machine-checked on the GPU result.
//!   (d) THE BAN: `denoiser.wgsl` carries the `// BAN-SCOPED` marker and
//!       contains no temporal vocabulary (the same fixed list VIII-0/1 scan);
//!       `denoiser_gpu.rs` is caught by the `denoiser*.rs` glob and already
//!       scanned by the VIII-0 gate — here we re-witness the shader directly.
//!
//! All GPU ordeals print + return early (never a false green) on a host
//! without an adapter, matching `viii0_ordeals.rs` / `viii1_ordeals.rs`.

use std::fs;
use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::{Mlp, denoise_image, deserialize_weights};
use scrying_glass::denoiser_dataset::{
    DATASET_HEIGHT, DATASET_REF_FRAMES, DATASET_WIDTH, VALIDATION_POSE_NAMES, law_poses,
    naruko_params,
};
use scrying_glass::denoiser_gpu::GpuDenoiser;
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene, SunLight};

fn weights_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data")
}

fn load_committed_weights() -> Mlp {
    let bytes = fs::read(weights_dir().join("denoiser-weights-v1.bin"))
        .expect("read committed denoiser-weights-v1.bin");
    deserialize_weights(&bytes).expect("deserialize committed weights artifact")
}

/// The same small fixed pose the VIII-0/1 ordeals use — fast, non-vacuous.
fn fixed_pose() -> (Bvh, Camera, SunLight, [f32; 4], [f32; 4], u32, u32) {
    let tri = LeafTriangle {
        positions: [[-4.0, -2.0, -8.0], [4.0, -2.0, -8.0], [0.0, 4.0, -8.0]],
        albedo: [0.7, 0.4, 0.2],
        emission: [0.0, 0.0, 0.0],
        metallic: 0.0,
        roughness: 0.6,
    };
    let bvh = Bvh::build(&[tri], &BvhParams::default());
    let camera = Camera {
        eye: GVec3::new(0.0, 0.0, 2.0),
        yaw: 0.0,
        pitch: 0.0,
        fov_y_radians: 50f32.to_radians(),
        near: 0.05,
        far: 1000.0,
    };
    let sun = SunLight {
        direction: GVec3::new(0.3, 1.0, 0.4).normalize().to_array(),
        color: [1.0, 0.95, 0.85],
        intensity: 2.0,
        ambient_intensity: 0.15,
    };
    (bvh, camera, sun, [0.1, 0.12, 0.22, 1.0], [0.55, 0.42, 0.5, 1.0], 64, 48)
}

fn render_frame_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &Camera,
    sun: &SunLight,
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    w: u32,
    h: u32,
) -> (Vec<GVec3>, Vec<GVec3>, Vec<GVec3>, Vec<f32>) {
    let noisy_params = IntegratorParams { spp: 1, ..IntegratorParams::default() };
    let noisy = resolve(&trace_headless(
        device, queue, bvh, camera, sun, sky_top, sky_horizon, w, h, 1, &noisy_params, None,
    ));
    let raw_aov = trace_headless_aov(device, queue, bvh, camera, sun, sky_top, sky_horizon, w, h);
    let (albedo, normal, depth) = split_aov(&raw_aov);
    (noisy, albedo, normal, depth)
}

/// (a) The GPU port denoises the same current frame byte-identically twice.
#[test]
fn a_gpu_inference_is_byte_identical_same_frame_twice() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[VIII-2 gpu determinism] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let (bvh, camera, sun, sky_top, sky_horizon, w, h) = fixed_pose();
    let (noisy, albedo, normal, depth) =
        render_frame_buffers(&device, &queue, &bvh, &camera, &sun, sky_top, sky_horizon, w, h);
    let gpu = GpuDenoiser::new(&device, &load_committed_weights());
    let a = gpu.denoise(&device, &queue, &noisy, &albedo, &normal, &depth, w, h);
    let b = gpu.denoise(&device, &queue, &noisy, &albedo, &normal, &depth, w, h);
    assert_eq!(a, b, "GPU denoise is not byte-identical across two runs on the same device+frame");
}

/// The GPU-vs-CPU parity tolerance, DERIVED (not chosen): the per-pixel
/// forward pass is `macs` fused multiply-adds; the GPU's FMA path differs
/// from the CPU's mul-then-add path by at most one unit-roundoff `u` per MAC
/// (FMA is the MORE accurate of the two, so the DIFFERENCE is bounded by the
/// same `u` per op — Higham, *Accuracy and Stability of Numerical
/// Algorithms*, dot-product error analysis). The feature/undo transforms add
/// a handful of transcendentals (3 `ln` in, 1 `ln` depth, 3 `exp` out), each
/// bounded by a small ULP budget on either backend. So the relative output
/// spread is bounded by `(macs + transcendental_ulp_budget) * u`, computed
/// here from `layer_dims()` and `f32::EPSILON` — a frozen literal would be a
/// hardcode in costume; this recomputes from the actual net every run.
fn derived_parity_rel_bound(mlp: &Mlp) -> f64 {
    let macs: u64 = mlp
        .layer_dims()
        .iter()
        .map(|&(i, o)| (i as u64) * (o as u64))
        .sum();
    // 7 transcendentals in the feature+undo path; a generous 4-ULP budget
    // each (Metal fast-math transcendentals; still derived, not chosen for a
    // value) — the accumulated MAC term dominates this anyway.
    const TRANSCENDENTAL_ULP_BUDGET: u64 = 7 * 4;
    let unit_roundoff = f32::EPSILON as f64; // 2^-23, worst-case fp32 rounding step
    (macs + TRANSCENDENTAL_ULP_BUDGET) as f64 * unit_roundoff
}

/// (b) + (c): render the two held-out validation poses, denoise on the GPU,
/// assert parity vs the CPU reference within the derived bound AND that the
/// GPU output strictly beats noisy (the quality gate, machine-checked on GPU
/// output).
#[test]
fn b_and_c_gpu_matches_cpu_within_derived_bound_and_beats_noisy() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[VIII-2 gpu parity] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let mlp = load_committed_weights();
    let gpu = GpuDenoiser::new(&device, &mlp);
    let rel_bound = derived_parity_rel_bound(&mlp);
    println!("[VIII-2 parity] derived GPU-vs-CPU relative bound = {rel_bound:.3e}");

    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());

    let (w, h) = (DATASET_WIDTH, DATASET_HEIGHT);
    let val_poses: Vec<(&'static str, Camera)> = law_poses(&params)
        .into_iter()
        .filter(|(name, _)| VALIDATION_POSE_NAMES.contains(name))
        .collect();
    assert_eq!(val_poses.len(), 2, "expected exactly the 2 documented validation poses");

    for (name, camera) in val_poses {
        let (noisy, albedo, normal, depth) =
            render_frame_buffers(&device, &queue, &bvh, &camera, &scene.sun, scene.sky_top, scene.sky_horizon, w, h);
        let reference = resolve(&trace_headless(
            &device, &queue, &bvh, &camera, &scene.sun, scene.sky_top, scene.sky_horizon, w, h,
            DATASET_REF_FRAMES, &IntegratorParams::default(), None,
        ));

        let cpu = denoise_image(&mlp, &noisy, &albedo, &normal, &depth);
        let gpu_out = gpu.denoise(&device, &queue, &noisy, &albedo, &normal, &depth, w, h);

        // (b) parity within the derived RELATIVE bound (normalized by the
        // CPU image magnitude, matching the VIII-1 nondeterminism-margin
        // convention).
        let magnitude = rmse(&cpu, &vec![GVec3::ZERO; cpu.len()]).max(1e-12);
        let parity_rel = rmse(&gpu_out, &cpu) / magnitude;
        println!("[VIII-2 parity] pose={name} parity_rel={parity_rel:.3e} bound={rel_bound:.3e}");
        assert!(
            parity_rel <= rel_bound,
            "pose '{name}': GPU-vs-CPU relative spread {parity_rel:.3e} exceeds the derived bound {rel_bound:.3e}"
        );

        // (c) the quality gate, on GPU output: strictly beats noisy.
        let noisy_rmse = rmse(&noisy, &reference);
        let gpu_rmse = rmse(&gpu_out, &reference);
        println!("[VIII-2 quality] pose={name} noisy_rmse={noisy_rmse:.6} gpu_denoised_rmse={gpu_rmse:.6}");
        assert!(
            gpu_rmse < noisy_rmse,
            "pose '{name}': GPU-denoised RMSE {gpu_rmse:.6} does not beat noisy RMSE {noisy_rmse:.6}"
        );
    }
}

/// (d) THE BAN, GPU half: `denoiser.wgsl` carries the `// BAN-SCOPED` marker
/// and no temporal vocabulary (the same fixed list VIII-0/1 enforce). A
/// direct witness on the shader text — `denoiser_gpu.rs` itself is already
/// scanned by the VIII-0 forward-proof `denoiser*.rs` glob.
#[test]
fn d_ban_no_temporal_vocabulary_in_the_gpu_shader() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shader = fs::read_to_string(root.join("src/denoiser.wgsl")).expect("read src/denoiser.wgsl");
    assert!(
        shader.contains("// BAN-SCOPED"),
        "denoiser.wgsl must carry the // BAN-SCOPED marker so the ban scope picks it up"
    );
    let forbidden = [
        "previous_frame", "history", "motion_vector", "temporal", "reproject",
        "warp", "feedback", "recurrent", "accum_prev", "prev_", "last_frame",
        "frame_history", "velocity",
    ];
    for word in forbidden {
        assert!(
            !shader.to_lowercase().contains(word),
            "forbidden temporal vocabulary '{word}' found in src/denoiser.wgsl"
        );
    }

    // The gpu driver module is caught by the VIII-0 `denoiser*.rs` glob;
    // re-witness it here so this file documents the whole GPU seam is clean.
    let gpu_src = fs::read_to_string(root.join("src/denoiser_gpu.rs")).expect("read denoiser_gpu.rs");
    assert!(gpu_src.contains("// BAN-SCOPED"), "denoiser_gpu.rs should carry the // BAN-SCOPED marker");
    for word in forbidden {
        assert!(
            !gpu_src.to_lowercase().contains(word),
            "forbidden temporal vocabulary '{word}' found in src/denoiser_gpu.rs"
        );
    }
}
