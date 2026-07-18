//! R-DIRECT export — dump the REAL per-pixel feature buffer + the CPU-reference
//! net output for ONE pose at the spike shape (low 48×32 → native 96×64,
//! 6144 target px), for the native-Metal tensor harness to check PARITY against.
//!
//! Not timing, not GPU: this is the ground-truth bridge. It traces the same
//! naruko "front" pose the WGSL kernel measure uses, featurizes every target
//! pixel with `pixel_features` (23 f32 each), runs the committed trained net
//! `Mlp::forward` (the demod-log OUTPUT of the matmul chain — BEFORE
//! undo_log_demod, because the harness computes only the net forward), and
//! writes three little-endian f32 blobs next to the world data:
//!   - features.f32       : [N=6144][23]  row-major, the harness input X
//!   - expected.f32       : [N=6144][3]   the net forward output the harness must match
//!   - meta.json          : shapes + which weights + sha, so the harness is self-describing
//!
//! The weights the harness loads are the SAME committed `rdirect-weights-v1.bin`
//! (GAIARDR1) — the harness parses that format directly, so parity is exact-net.
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_export_features

use std::path::Path;

use glam::Vec2;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser_dataset::{law_poses, naruko_params};
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::rdirect::{
    INPUT_FEATURES, OUTPUT_CHANNELS, Mlp, deserialize_weights, pixel_features, weights_sha256,
};
use scrying_glass::scene::RenderScene;

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[rdirect-export] no GPU adapter — cannot trace");
    };

    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(manifest.join("data/rdirect-weights-v1.bin"))
        .expect("read committed rdirect-weights-v1.bin");
    let mlp: Mlp = deserialize_weights(&bytes).expect("deserialize committed rdirect weights");
    let sha = weights_sha256(&mlp);
    println!("[rdirect-export] net: {} layers, sha256={sha}", mlp.layer_dims().len());

    // Build naruko scene, trace the SAME "front" pose the kernel measure uses.
    let params = naruko_params();
    let world_path = manifest.join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    let poses = law_poses(&params);
    let front = poses.iter().find(|(n, _)| *n == "front").expect("front pose").1.clone();

    // Spike shape: low 48×32 → native 96×64 (6144 target px).
    let (low_w, low_h, target_w, target_h) = (48u32, 32u32, 96u32, 64u32);
    let noisy = IntegratorParams { spp: 1, ..IntegratorParams::default() };
    let low_radiance = resolve(&trace_headless(
        &device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon,
        low_w, low_h, 1, &noisy, None,
    ));
    let (hi_albedo, hi_normal, hi_depth) = split_aov(&trace_headless_aov(
        &device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon,
        target_w, target_h,
    ));
    let hi_motion = vec![Vec2::ZERO; (target_w * target_h) as usize];

    let n = (target_w * target_h) as usize;
    let mut features = Vec::<f32>::with_capacity(n * INPUT_FEATURES);
    let mut expected = Vec::<f32>::with_capacity(n * OUTPUT_CHANNELS);
    for ty in 0..target_h {
        for tx in 0..target_w {
            let i = (ty * target_w + tx) as usize;
            let f = pixel_features(
                &low_radiance, low_w, low_h, target_w, target_h, tx, ty, hi_albedo[i],
                hi_normal[i], hi_depth[i], hi_motion[i],
            );
            features.extend_from_slice(&f);
            // The matmul-chain output the harness reproduces: the net forward in
            // demod-log space, BEFORE undo_log_demod (that per-pixel expm1 is not
            // part of the batched matmul the harness times).
            let out = mlp.forward(&f);
            expected.extend_from_slice(&out);
        }
    }

    // Write next to the harness so it is self-locating.
    let out_dir = manifest.join("../../tools/metal4-probe/data");
    std::fs::create_dir_all(&out_dir).expect("mkdir tools/metal4-probe/data");

    let write_f32 = |name: &str, v: &[f32]| {
        let mut buf = Vec::<u8>::with_capacity(v.len() * 4);
        for &x in v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(out_dir.join(name), &buf).unwrap_or_else(|e| panic!("write {name}: {e}"));
    };
    write_f32("features.f32", &features);
    write_f32("expected.f32", &expected);
    // Copy the weights blob beside the harness too (single source: the committed bin).
    std::fs::write(out_dir.join("rdirect-weights-v1.bin"), &bytes).expect("copy weights");

    let dims: Vec<(u32, u32)> = mlp.layer_dims();
    let meta = format!(
        "{{\n  \"pose\": \"front\",\n  \"low\": [{low_w}, {low_h}],\n  \"native\": [{target_w}, {target_h}],\n  \"n_pixels\": {n},\n  \"input_features\": {INPUT_FEATURES},\n  \"output_channels\": {OUTPUT_CHANNELS},\n  \"weights_sha256\": \"{sha}\",\n  \"layer_dims\": {dims:?}\n}}\n"
    );
    std::fs::write(out_dir.join("meta.json"), meta).expect("write meta.json");

    println!(
        "[rdirect-export] wrote {} px × {} feat + {} out to {}",
        n, INPUT_FEATURES, OUTPUT_CHANNELS, out_dir.display()
    );
}
