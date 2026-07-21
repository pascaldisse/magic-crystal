//! NEURAL-LIVE N0.b GATE — the GPU FEATURE GATHER + zero-copy shared forward.
//!
//! Two isolated parities, on the SAME naruko "front" pose the N0.a export uses
//! (low 48×32 → native 96×64):
//!   GATE A (gather): the GPU `rdirect_gather.wgsl` `[N,23]` tensor, built from
//!     the traced `accum`/`aov` buffers, matches the CPU `pixel_features` for
//!     every pixel/feature.
//!   GATE B (forward): `RdirectLive::forward_shared` — run zero-copy over the
//!     SAME pooled MTLBuffer the gather wrote — matches the CPU `Mlp::forward`
//!     on the GPU-gathered features.
//! Together: GPU (gather → shared MTLBuffer → GEMM) == CPU direct render in
//! demod-log space, with no per-frame allocation.
//!
//! Also prints a live GATHER-ms median (the N0.b budget line). Skips (does not
//! fail) when no Metal/GPU device is present.

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::time::Instant;

use glam::Vec2;
use wgpu::util::DeviceExt;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser_dataset::{law_poses, naruko_params};
use scrying_glass::integrator::{
    headless_device, resolve, split_aov, trace_headless, trace_headless_aov, IntegratorParams,
};
use scrying_glass::rdirect::{deserialize_weights, pixel_features, Mlp, INPUT_FEATURES};
use scrying_glass::rdirect_gather::FeatureGather;
use scrying_glass::rdirect_live::RdirectLive;
use scrying_glass::scene::RenderScene;

fn weights_bytes() -> Vec<u8> {
    // STAGE D: parity of the SHIPPED default weights (v2) — GPU forward vs the
    // live CPU reference (deserialize_weights + Mlp, same weights, same math).
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::read(root.join("data/rdirect-weights-v2.bin")).expect("committed rdirect weights")
}

#[test]
fn n0b_gather_and_shared_forward_match_cpu() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[n0b] SKIP — no GPU adapter");
        return;
    };

    // Same scene + pose + shapes as the N0.a export.
    let params = naruko_params();
    let world_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    let front = law_poses(&params)
        .into_iter()
        .find(|(n, _)| *n == "front")
        .expect("front pose")
        .1;

    let (low_w, low_h, target_w, target_h) = (48u32, 32u32, 96u32, 64u32);
    let n = (target_w * target_h) as usize;

    let noisy = IntegratorParams { spp: 1, ..IntegratorParams::default() };
    // RAW accum (sum + count) at low res, and RAW aov (2 cells/px) at native.
    let accum_raw = trace_headless(
        &device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h,
        1, &noisy, None,
    );
    let aov_raw = trace_headless_aov(
        &device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, target_w,
        target_h,
    );

    // CPU reference feature vectors (motion = 0, as the export).
    let low_radiance = resolve(&accum_raw);
    let (hi_albedo, hi_normal, hi_depth) = split_aov(&aov_raw);
    let mut cpu_feats = Vec::<f32>::with_capacity(n * INPUT_FEATURES);
    for ty in 0..target_h {
        for tx in 0..target_w {
            let i = (ty * target_w + tx) as usize;
            let f = pixel_features(
                &low_radiance, low_w, low_h, target_w, target_h, tx, ty, hi_albedo[i],
                hi_normal[i], hi_depth[i], Vec2::ZERO,
            );
            cpu_feats.extend_from_slice(&f);
        }
    }

    // Upload the traced buffers as STORAGE (they stand in for the live path's
    // surface_accum / aov buffers — identical bytes).
    let accum_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("accum(low)"),
        contents: bytemuck::cast_slice(&accum_raw),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let aov_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("aov(native)"),
        contents: bytemuck::cast_slice(&aov_raw),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Build the live net + pooled shared buffers on the SAME wgpu device.
    let weights = weights_bytes();
    let live = match RdirectLive::from_wgpu_queue(&device, &queue, &weights, n) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[n0b] SKIP — no live Metal path: {e}");
            return;
        }
    };
    let feats_buf = live.feature_buffer().expect("pooled feature buffer");
    let gather = FeatureGather::new(&device);

    // One gather → the shared feature MTLBuffer.
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gather"),
    });
    gather.encode(
        &device, &queue, &mut enc, &accum_buf, &aov_buf, feats_buf, low_w, low_h, target_w,
        target_h,
    );
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    // GATE A — read the gather output back and compare to the CPU features.
    let feat_bytes = FeatureGather::feature_bytes(n);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("feat readback"),
        size: feat_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("feat copy"),
    });
    enc.copy_buffer_to_buffer(feats_buf, 0, &readback, 0, feat_bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    enc.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r.map(|_| ()));
    });
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("readback chan").expect("map feat readback");
    let mapped = readback.get_mapped_range(..).expect("mapped feats");
    let gpu_feats: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();

    assert_eq!(gpu_feats.len(), cpu_feats.len(), "feature tensor length");
    let mut max_feat_abs = 0f32;
    for i in 0..cpu_feats.len() {
        max_feat_abs = max_feat_abs.max((gpu_feats[i] - cpu_feats[i]).abs());
    }
    eprintln!("[n0b] GATE A gather: N={n} px × {INPUT_FEATURES} feat · max abs {max_feat_abs:.3e}");
    // GPU vs CPU f32 div + ln: last-ULP class. 1e-4 is orders above float drift,
    // a breach = a real wiring error (wrong tap / packing / offset).
    assert!(max_feat_abs < 1.0e-4, "gather vs CPU pixel_features abs {max_feat_abs:.3e} ≥ 1e-4");

    // GATE B — zero-copy shared forward over the SAME pooled MTLBuffer vs the CPU
    // net on the GPU-gathered features (isolates the GEMM from gather drift).
    let got = live.forward_shared(n).expect("shared forward ran");
    let mlp: Mlp = deserialize_weights(&weights).expect("cpu net");
    let out_c = live.out_channels();
    assert_eq!(got.len(), n * out_c, "forward output length");
    let mut max_fwd_abs = 0f32;
    for p in 0..n {
        let f = &gpu_feats[p * INPUT_FEATURES..(p + 1) * INPUT_FEATURES];
        let cpu = mlp.forward(f);
        for c in 0..out_c {
            max_fwd_abs = max_fwd_abs.max((got[p * out_c + c] - cpu[c]).abs());
        }
    }
    eprintln!("[n0b] GATE B shared forward: max abs {max_fwd_abs:.3e}");
    assert!(max_fwd_abs < 1.0e-3, "shared forward vs CPU net abs {max_fwd_abs:.3e} ≥ 1e-3");

    // Determinism of the shared forward.
    let got2 = live.forward_shared(n).expect("second shared forward");
    assert_eq!(got, got2, "shared forward not deterministic");

    // S8 A/B PARITY — `got` above is now the MPSGraph executable (S8 flipped
    // the default). Run the raw MPSMatrixMultiplication chain (the kept lab
    // A/B) over the SAME pooled buffers and confirm it still matches. Same
    // weights, same math; the chain is the honest slower measurement path.
    live.set_use_mpsgraph(false);
    let got_chain = live.forward_shared(n).expect("chain forward");
    live.set_use_mpsgraph(true);
    let mut max_ab = 0f32;
    for k in 0..got.len() {
        max_ab = max_ab.max((got[k] - got_chain[k]).abs());
    }
    eprintln!("[n0g] S8 MPSGraph(default) vs chain: max abs {max_ab:.3e}");
    assert!(max_ab < 1.0e-3, "S8 MPSGraph vs chain abs {max_ab:.3e} ≥ 1e-3");

    // GATHER-ms budget line: median wall of encode+submit+wait over 200 frames.
    let iters = 200;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gather-timed"),
        });
        gather.encode(
            &device, &queue, &mut enc, &accum_buf, &aov_buf, feats_buf, low_w, low_h, target_w,
            target_h,
        );
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[iters / 2];
    eprintln!(
        "[n0b] GATHER median {median:.3} ms over {iters} frames @ {target_w}×{target_h} ({n} px)"
    );
}
