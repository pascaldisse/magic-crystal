//! TEACHER/BENCHMARK SURFACE (ITEM 16, de-chartered — was "THE ONE RENDER PATH") — fp16 denoiser verdict (bound re-derivation + BAN
//! re-proof by simulation). The budget question: can the denoiser run in fp16
//! without losing the RAZOR-THIN beats-noisy margin (orbit_-20: noisy 0.052073
//! vs denoised 0.042999 = 0.009074 headroom at 96×64)?
//!
//! Two honest fp16 modes are simulated in the CPU reference — a legitimate
//! measuring instrument, because the numerical question ("does fp16 precision
//! destroy the margin?") is answered by the SAME rounding arithmetic a GPU
//! f16 shader would perform; this is a test oracle, never a runtime path:
//!
//!   MODE A (storage/read fp16, MAC accumulate in f32): weights and per-layer
//!     activations are rounded to f16; the dot-product accumulator stays f32.
//!     This is the mode that fits the 13.8 KB (denoiser) / 55 KB (upscaler)
//!     weights into threadgroup memory in half the bytes — the actual budget
//!     lever (RENDER.md §8). Sane fp16 inference.
//!
//!   MODE B (FULL fp16, accumulate in f16 too): every accumulation step is
//!     rounded to f16. The fastest but the most fragile.
//!
//! The forbidden literal-margin trap is avoided: we RE-DERIVE the fp16 parity
//! bound from the net op-count and f16 unit-roundoff (u16 = 2^-11), then
//! MEASURE the actual beats-noisy RMSE on the two TRUE held-out orbits, and
//! print the verdict. Death certificate or survival — the numbers decide.
//!
//! ADVISORY NOTE: this is a bound-derivation + measurement example, not an
//! asserting ordeal (it prints, it does not `assert!`/gate CI) — correct for
//! fp16's current status as a test oracle only, never a runtime path. If
//! fp16 (MODE A) is ever wired as an actual runtime path, this must be
//! promoted to an asserting ordeal (panics/fails on regression), not left as
//! a printing example.
//!
//! Run: cargo run -p scrying-glass --release --example onepath_fp16_verdict

use std::path::Path;

use glam::Vec3;
use half::f16;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::{Mlp, denoise_image, deserialize_weights, pixel_features};
use scrying_glass::denoiser_dataset::{
    DATASET_HEIGHT, DATASET_REF_FRAMES, DATASET_WIDTH, VALIDATION_POSE_NAMES, law_poses,
    naruko_params,
};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::RenderScene;

const ALBEDO_DEMOD_EPS: f32 = 1e-3;
const NO_HIT_ALBEDO_THRESHOLD_SQ: f32 = 1e-8;

fn demod_divisor(a: Vec3) -> Vec3 {
    if a.length_squared() > NO_HIT_ALBEDO_THRESHOLD_SQ {
        a + Vec3::splat(ALBEDO_DEMOD_EPS)
    } else {
        Vec3::ONE
    }
}

fn undo_output(raw: [f32; 3], albedo: Vec3) -> Vec3 {
    let expm1 = Vec3::new(raw[0].exp() - 1.0, raw[1].exp() - 1.0, raw[2].exp() - 1.0);
    let demod = Vec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0));
    demod * demod_divisor(albedo)
}

fn h(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// fp16-simulated forward of the SAME net (same layer geometry, same fixed
/// accumulation order as `Mlp::forward`). `full_fp16` selects MODE B (round
/// the accumulator each step) vs MODE A (f32 accumulate). Weights and input
/// activations are always rounded to f16 (fp16 storage).
fn forward_fp16(dims: &[(u32, u32)], flat: &[f32], input: &[f32], full_fp16: bool) -> [f32; 3] {
    // fp16-rounded activation vector.
    let mut act: Vec<f32> = input.iter().map(|&v| h(v)).collect();
    let mut off = 0usize;
    for (li, &(in_dim, out_dim)) in dims.iter().enumerate() {
        let in_dim = in_dim as usize;
        let out_dim = out_dim as usize;
        let w_off = off;
        let b_off = off + in_dim * out_dim;
        off = b_off + out_dim;
        let is_last = li + 1 == dims.len();
        let mut next = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut sum = h(flat[b_off + o]);
            let row = w_off + o * in_dim;
            for i in 0..in_dim {
                let prod = h(flat[row + i]) * act[i];
                sum += prod;
                if full_fp16 {
                    sum = h(sum);
                }
            }
            next[o] = if is_last { sum } else { sum.max(0.0) };
        }
        // fp16 storage of activations between layers.
        act = next.iter().map(|&v| h(v)).collect();
    }
    [act[0], act[1], act[2]]
}

fn denoise_fp16(
    mlp: &Mlp,
    noisy: &[Vec3],
    albedo: &[Vec3],
    normal: &[Vec3],
    depth: &[f32],
    full_fp16: bool,
) -> Vec<Vec3> {
    let dims = mlp.layer_dims();
    let flat = mlp.flat_weights();
    (0..noisy.len())
        .map(|i| {
            let features = pixel_features(noisy[i], albedo[i], normal[i], depth[i]);
            let raw = forward_fp16(&dims, &flat, &features, full_fp16);
            undo_output(raw, albedo[i])
        })
        .collect()
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("no GPU adapter — cannot render frames for the verdict");
        return;
    };
    let bytes =
        std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("data/denoiser-weights-v1.bin"))
            .expect("read denoiser weights");
    let mlp = deserialize_weights(&bytes).expect("deserialize");

    // ── fp16 parity bound re-derivation (documented, machine-computed) ──
    let macs: u64 = mlp
        .layer_dims()
        .iter()
        .map(|&(i, o)| i as u64 * o as u64)
        .sum();
    let u16 = 2f64.powi(-11); // f16 unit roundoff (10-bit mantissa + implicit 1)
    let u32_ = f32::EPSILON as f64; // 2^-23
    // MODE A bound (f32 accumulate, f16 inputs): the accumulator does not
    // compound; the error is dominated by rounding each weight & activation to
    // f16 once — ~2·u16 relative per product, and the sum in f32 adds only the
    // fp32 dot-product term (macs·u32). So rel ≈ 2·u16 + macs·u32.
    let bound_mode_a = 2.0 * u16 + (macs as f64) * u32_;
    // MODE B bound (f16 accumulate): the dot-product error compounds in f16 —
    // classic (n)·u16 for an n-term sum (Higham). rel ≈ macs·u16 — expected
    // catastrophic for a 3488-MAC net.
    let bound_mode_b = (macs as f64) * u16;
    println!("[fp16 verdict] net macs = {macs}");
    println!(
        "[fp16 verdict] DERIVED MODE A (f16 storage, f32 acc) parity rel bound ≈ {bound_mode_a:.3e}"
    );
    println!(
        "[fp16 verdict] DERIVED MODE B (full f16 acc)         parity rel bound ≈ {bound_mode_b:.3e}"
    );

    // ── render the two TRUE held-out orbits at the pinned dataset res ──
    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    let (w, h_) = (DATASET_WIDTH, DATASET_HEIGHT);

    let mut all_pass_a = true;
    let mut all_pass_b = true;
    for (name, camera) in law_poses(&params)
        .into_iter()
        .filter(|(n, _)| VALIDATION_POSE_NAMES.contains(n))
    {
        let noisy_params = IntegratorParams {
            spp: 1,
            ..IntegratorParams::default()
        };
        let noisy = resolve(&trace_headless(
            &device,
            &queue,
            &bvh,
            &camera,
            &scene.sun,
            scene.sky_top,
            scene.sky_horizon,
            w,
            h_,
            1,
            &noisy_params,
            None,
        ));
        let reference = resolve(&trace_headless(
            &device,
            &queue,
            &bvh,
            &camera,
            &scene.sun,
            scene.sky_top,
            scene.sky_horizon,
            w,
            h_,
            DATASET_REF_FRAMES,
            &IntegratorParams::default(),
            None,
        ));
        let raw_aov = trace_headless_aov(
            &device,
            &queue,
            &bvh,
            &camera,
            &scene.sun,
            scene.sky_top,
            scene.sky_horizon,
            w,
            h_,
        );
        let (albedo, normal, depth) = split_aov(&raw_aov);

        let fp32 = denoise_image(&mlp, &noisy, &albedo, &normal, &depth);
        let a = denoise_fp16(&mlp, &noisy, &albedo, &normal, &depth, false);
        let b = denoise_fp16(&mlp, &noisy, &albedo, &normal, &depth, true);

        let noisy_rmse = rmse(&noisy, &reference);
        let fp32_rmse = rmse(&fp32, &reference);
        let a_rmse = rmse(&a, &reference);
        let b_rmse = rmse(&b, &reference);
        let mag = rmse(&fp32, &vec![Vec3::ZERO; fp32.len()]).max(1e-12);
        let parity_a = rmse(&a, &fp32) / mag;
        let parity_b = rmse(&b, &fp32) / mag;

        let beats_a = a_rmse < noisy_rmse;
        let beats_b = b_rmse < noisy_rmse;
        all_pass_a &= beats_a;
        all_pass_b &= beats_b;
        println!("\n── pose {name} (96×64) ──");
        println!("  noisy_rmse         {noisy_rmse:.6}");
        println!(
            "  fp32 denoised      {fp32_rmse:.6}  (margin over noisy {:.6})",
            noisy_rmse - fp32_rmse
        );
        println!(
            "  MODE A fp16        {a_rmse:.6}  (margin over noisy {:.6}, beats={beats_a})  parity_rel {parity_a:.3e} vs bound {bound_mode_a:.3e}",
            noisy_rmse - a_rmse
        );
        println!(
            "  MODE B full-fp16   {b_rmse:.6}  (margin over noisy {:.6}, beats={beats_b})  parity_rel {parity_b:.3e} vs bound {bound_mode_b:.3e}",
            noisy_rmse - b_rmse
        );
    }

    println!("\n=== fp16 VERDICT ===");
    println!("  MODE A (f16 storage, f32 accumulate): beats-noisy on ALL held-out = {all_pass_a}");
    println!("  MODE B (full f16 accumulate):         beats-noisy on ALL held-out = {all_pass_b}");
}
