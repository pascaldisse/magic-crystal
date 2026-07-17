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
//!   (e) weights hash-pin, GPU-scoped: the committed bytes `GpuDenoiser::new`
//!       consumes hash to the pinned provenance sha256 (viii1's `a4` already
//!       pins this for the CPU path; re-witnessed here because it is what
//!       the GPU upload actually reads), AND the flat payload the upload
//!       derives from those bytes (`Mlp::flat_weights()`) is a stable,
//!       deterministic transcription — two independent deserializations of
//!       the SAME committed bytes produce a bit-identical flat payload, and
//!       its length matches the layer geometry independently re-derived
//!       from `layer_dims()` (not trusting `GpuDenoiser::new`'s own internal
//!       assert).
//!   (f) THE BAN, signature half: every `pub fn` in `denoiser_gpu.rs` takes
//!       current-frame inputs only — no frame-index/history parameter
//!       anywhere (adversary A1 precedent from VIII-1/VIII-3: a second entry
//!       point cannot slip past unexamined).
//!
//! (a) cold×2, (e), and (f) port patterns from the unmerged reference branch
//! `rite8-viii2-ari` @ b97a9a0 (`packages/scrying-glass/tests/viii2_ordeals.rs`
//! there), adapted to this port's actual API (`GpuDenoiser`, `layer_dims()` —
//! no `layer_views()`/free `denoise_image_gpu`/`flatten_weights` here).
//!
//! All GPU ordeals print + return early (never a false green) on a host
//! without an adapter, matching `viii0_ordeals.rs` / `viii1_ordeals.rs`.

use std::fs;
use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::{Mlp, denoise_image, deserialize_weights, sha256_hex};
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

fn read_committed_weights_bytes() -> Vec<u8> {
    fs::read(weights_dir().join("denoiser-weights-v1.bin")).expect("read committed denoiser-weights-v1.bin")
}

fn load_committed_weights() -> Mlp {
    deserialize_weights(&read_committed_weights_bytes()).expect("deserialize committed weights artifact")
}

fn load_provenance() -> serde_json::Value {
    let text = fs::read_to_string(weights_dir().join("denoiser-weights-v1.provenance.json"))
        .expect("read committed provenance sidecar");
    serde_json::from_str(&text).expect("parse provenance JSON")
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
    // 7 transcendentals in the feature+undo path (3 `ln` demodulated-radiance
    // channels in + 1 `ln` depth in + 3 `exp` radiance channels out —
    // denoiser.rs `extract_features`/`undo_transform`); RE-VERIFY this count
    // if the feature transform ever changes shape. 4-ULP budget each (Metal
    // fast-math transcendentals; still derived, not chosen for a value) —
    // the accumulated MAC term dominates this anyway. This ULP-budget slack
    // must be revisited before any fp16 port (see docs/perf/2026-07-17-viii2-gpu-denoise.md).
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

/// (e) weights hash-pin, GPU-scoped (ported from `rite8-viii2-ari` @ b97a9a0
/// `d_gpu_uses_hash_pinned_committed_weights`, adapted — this port has no
/// free `flatten_weights` fn; `GpuDenoiser::new` derives its upload payload
/// from `Mlp::flat_weights()` inline, so the witness re-derives the same
/// invariant from public API instead of a helper that does not exist here).
/// The bytes `GpuDenoiser::new` loads are the pinned artifact (viii1's `a4`
/// pins this for the CPU path already; re-witnessed here because it is what
/// the GPU upload actually reads), and the flat payload derived from those
/// bytes is a stable, deterministic transcription whose length matches the
/// layer geometry — never re-derived, never silently reshaped.
#[test]
fn e_gpu_uploads_weights_matching_the_hash_pinned_committed_artifact() {
    let bytes = read_committed_weights_bytes();
    let provenance = load_provenance();
    let expected_hash = provenance["weights_sha256"]
        .as_str()
        .expect("weights_sha256 string in provenance JSON");
    let actual_hash = sha256_hex(&bytes);
    assert_eq!(
        actual_hash, expected_hash,
        "committed denoiser-weights-v1.bin does not hash to the pinned provenance sha256 — \
         GpuDenoiser::new would upload weights that drifted from their provenance"
    );

    // Two independent deserializations of the SAME committed bytes produce a
    // bit-identical flat weight payload (exact bit-pattern compare —
    // stricter than `==`, distinguishes +0/-0).
    let mlp_a = deserialize_weights(&bytes).expect("deserialize committed weights (a)");
    let mlp_b = deserialize_weights(&bytes).expect("deserialize committed weights (b)");
    let flat_a = mlp_a.flat_weights();
    let flat_b = mlp_b.flat_weights();
    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(
        bits(&flat_a),
        bits(&flat_b),
        "flat_weights() is not a deterministic transcription of the committed bytes — GpuDenoiser \
         could upload a different payload across two loads of the same artifact"
    );

    // The flat payload length matches the layer geometry, independently
    // re-derived here from `layer_dims()` (per layer: in*out weights + out
    // biases) — not trusting `GpuDenoiser::new`'s own internal assert.
    let dims = mlp_a.layer_dims();
    let expected_len: u32 = dims.iter().map(|&(i, o)| i * o + o).sum();
    assert_eq!(
        flat_a.len() as u32,
        expected_len,
        "flat_weights() length disagrees with the layer geometry GpuDenoiser::new uses to compute \
         per-layer upload offsets"
    );
}

/// Every `pub fn NAME(...) -> ... {` signature found in `text`, as
/// (name, full signature text INCLUDING the return-type arrow, EXCLUDING the
/// opening `{`). Plain textual scan (same as VIII-1/VIII-3's) — sufficient
/// for `denoiser_gpu.rs`'s simple signatures.
fn public_fn_signatures(text: &str) -> Vec<(String, String)> {
    let mut sigs = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find("pub fn ") {
        let start = search_from + rel;
        let name_start = start + "pub fn ".len();
        let name_end = text[name_start..].find('(').map(|e| name_start + e).unwrap_or(name_start);
        let name = text[name_start..name_end].trim().to_string();
        let body_start = text[start..].find('{').map(|e| start + e).unwrap_or(text.len());
        sigs.push((name, text[start..body_start].to_string()));
        search_from = (body_start + 1).max(start + 1);
    }
    sigs
}

/// (f) THE BAN, signature half (ported from `rite8-viii2-ari` @ b97a9a0
/// `e_ban_every_public_fn_signature_in_gpu_module_is_current_frame_only`,
/// matching the convention `viii1_ordeals.rs`'s `d_ban_every_public_fn_...`
/// and `viii3_ordeals.rs`'s `f_ban_every_public_fn_...` already use — 
/// adversary finding A1: extended from the primary entry point to EVERY
/// `pub fn`, so a second entry point (e.g. the timing helper) cannot slip
/// past unexamined). Every public function's signature (grepped directly)
/// takes current-frame buffers only — no frame-index/history parameter
/// anywhere, checked against the same fixed forbidden-substring list.
#[test]
fn f_ban_every_public_fn_signature_in_denoiser_gpu_takes_current_frame_inputs_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let text = fs::read_to_string(root.join("src/denoiser_gpu.rs")).expect("read denoiser_gpu.rs");
    let sigs = public_fn_signatures(&text);
    assert!(
        sigs.len() >= 3,
        "expected at least 3 `pub fn`s in src/denoiser_gpu.rs (found {}) — the textual scan may be broken",
        sigs.len()
    );
    let forbidden_params = ["frames", "frame_index", "prev", "history", "samples_before", "last_"];
    for (name, signature) in &sigs {
        for forbidden in forbidden_params {
            assert!(
                !signature.to_lowercase().contains(forbidden),
                "denoiser_gpu.rs `pub fn {name}` signature contains '{forbidden}' — every public entry \
                 point must take current-frame inputs only"
            );
        }
    }
}
