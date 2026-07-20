//! V7-LIVE LANE STAGE 1 — E/D evidence probe.
//!
//! Dumps ONE frame's GPU-gathered E/D evidence (the new `gather_split.wgsl`
//! entry, reading the integrator's raw `accum_ed` split buffer exactly as the
//! live path's `evidence_split` branch does in `main.rs::NetPresent`) and
//! cross-checks it against the CPU reference: `rdirect::pixel_features_split`
//! fed by `trace_headless_split` — the SAME function + 1-spp seed convention
//! (`0x7abc + f*131 + 5`, f=0 here) the v7 trainer's `render_pose` uses to
//! acquire its E/D evidence (see examples/rdirect_train_v7e.rs). Same pose/
//! shapes the n0b gather gate uses: naruko "front", low 48×32 → native 96×64.
//!
//! Prints max-abs-diff over the 24 radiance-tap features (idx 0..24 of the
//! 35-feature `INPUT_FEATURES_SPLIT` row — E taps then D taps; history is
//! Stage 2, not written by `gather_split` yet).

use glam::Vec2;
use wgpu::util::DeviceExt;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser_dataset::{law_poses, naruko_params};
use scrying_glass::integrator::{
    headless_device, split_aov, trace_headless_aov, trace_headless_split, Integrator,
    IntegratorParams, IntegratorUniform,
};
use scrying_glass::rdirect::{pixel_features_split, INPUT_FEATURES_SPLIT};
use scrying_glass::rdirect_gather::FeatureGatherSplit;
use scrying_glass::scene::RenderScene;

const ACCUM_CELL: u64 = 16; // vec4<f32>

fn main() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[v7-ed-probe] SKIP — no GPU adapter");
        return;
    };

    let params = naruko_params();
    let world_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
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
    // Trainer's `render_pose` seed convention for frame f=0.
    let np = IntegratorParams { spp: 1, seed: 0x7abc + 5, ..IntegratorParams::default() };

    // ── GPU: ONE frame's raw accum_ed via the SAME dispatch_split the live
    // path's evidence_split branch drives, then the NEW gather_split kernel
    // over it — the exact wiring under test. ──
    let integrator = Integrator::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, &bvh, None);
    let accum = integrator.make_accum(&device, low_w, low_h); // unused binding, present-but-unread
    let accum_ed = integrator.make_split_buffer(&device, low_w, low_h);
    let compute_bg = integrator.compute_bind_group(&device, &accum);
    let split_bg = integrator.split_bind_group(&device, &accum_ed);
    let uniform = IntegratorUniform::build(
        &front, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h,
        integrator.node_count, integrator.tri_count, 0, &np, None,
    );
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("v7-ed-probe trace split"),
    });
    integrator.dispatch_split(&queue, &mut enc, &uniform, &compute_bg, &split_bg, low_w, low_h);
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let aov_raw = trace_headless_aov(
        &device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, target_w,
        target_h,
    );
    let aov_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("aov(native)"),
        contents: bytemuck::cast_slice(&aov_raw),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let gather_split = FeatureGatherSplit::new(&device);
    let feat_bytes = FeatureGatherSplit::feature_bytes(n);
    let feats_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("feats35"),
        size: feat_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("v7-ed-probe gather split"),
    });
    gather_split.encode(
        &device, &queue, &mut enc, &accum_ed, &aov_buf, &feats_buf, low_w, low_h, target_w,
        target_h,
    );
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("feats35 readback"),
        size: feat_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("feat copy"),
    });
    enc.copy_buffer_to_buffer(&feats_buf, 0, &readback, 0, feat_bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    enc.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r.map(|_| ()));
    });
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("readback chan").expect("map feats");
    let mapped = readback.get_mapped_range(..).expect("mapped feats");
    let gpu_feats: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    let _ = ACCUM_CELL; // documents the cell layout above; buffer sizing goes through make_split_buffer

    // ── CPU reference: trace_headless_split (== render_pose's evidence
    // acquisition) + pixel_features_split, SAME pose/seed. ──
    let (low_e, low_d) = trace_headless_split(
        &device, &queue, &bvh, &front, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h,
        1, &np,
    );
    let (hi_albedo, hi_normal, hi_depth) = split_aov(&aov_raw);
    let mut cpu_feats = Vec::<f32>::with_capacity(n * INPUT_FEATURES_SPLIT);
    for ty in 0..target_h {
        for tx in 0..target_w {
            let i = (ty * target_w + tx) as usize;
            let f = pixel_features_split(
                &low_e, &low_d, low_w, low_h, target_w, target_h, tx, ty, hi_albedo[i],
                hi_normal[i], hi_depth[i], Vec2::ZERO,
            );
            cpu_feats.extend_from_slice(&f);
        }
    }

    assert_eq!(gpu_feats.len(), cpu_feats.len(), "feature tensor length");
    let mut max_all = 0f32;
    let mut max_ed = 0f32; // radiance taps only, idx 0..24 (E then D)
    let mut max_tail = 0f32; // subpixel/albedo/normal/depth/motion, idx 24..35
    for p in 0..n {
        for k in 0..INPUT_FEATURES_SPLIT {
            let d = (gpu_feats[p * INPUT_FEATURES_SPLIT + k] - cpu_feats[p * INPUT_FEATURES_SPLIT + k]).abs();
            max_all = max_all.max(d);
            if k < 24 {
                max_ed = max_ed.max(d);
            } else {
                max_tail = max_tail.max(d);
            }
        }
    }
    println!(
        "[v7-ed-probe] N={n} px x {INPUT_FEATURES_SPLIT} feat (35, no history) \
         max-abs-diff: E/D taps {max_ed:.3e} · tail {max_tail:.3e} · overall {max_all:.3e}"
    );
}
