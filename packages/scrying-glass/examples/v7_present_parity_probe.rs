//! V7-LIVE LANE STAGE 3 — full-frame present parity: the live GPU pipeline
//! (Stage 2 `gather_hist_split` + the REAL v7 MPSGraph forward + Stage 3's
//! new `rdirect_evidence` GPU clamp kernels + `DemodPass`) vs the CPU
//! reference `rdirect::direct_render_sequence_hist_split` (the function that
//! is load-bearing for the stamped 55720b45 real-image ordeal — NOT called
//! from the GPU side, only as the independent truth to diff against).
//!
//! Two short sequences, per the room's ask:
//!   STILL  — the same camera pose 3 times (reprojection is a no-op / pure
//!            temporal accumulation; exercises the has_prev=true steady state).
//!   PAN    — a 3-frame -3/0/+3 deg orbit (Stage 2's own probe pose), so
//!            reprojection/disocclusion is genuinely exercised too.
//!
//! GPU side per frame: split trace (low_e/low_d) -> native AOV trace ->
//! `FeatureGatherHistSplit::encode` (39-in, GPU) -> readback -> REAL net
//! forward via `RdirectLive::forward_cpu_roundtrip` (same MPSGraph the live
//! app's async pipeline commits, just driven synchronously here) -> upload
//! -> `DemodPass::encode` (undo log-demod) -> `EvidenceClamp::encode_accumulate`
//! + `encode_clamp` (Stage 3's new kernels) -> readback the PRESENTED image
//! -> `EvidenceClamp::encode_pack` + `HistoryBuffers::swap` for next frame.
//!
//! CPU side: one `direct_render_sequence_hist_split` call over the whole
//! sequence (its own internal recurrence + evidence accum, untouched).
//!
//! Run: cargo run --release -j2 --example v7_present_parity_probe

use wgpu::util::DeviceExt;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser_dataset::{naruko_params, orbit_camera};
use scrying_glass::integrator::{
    headless_device, split_aov, trace_headless_aov, Integrator, IntegratorParams, IntegratorUniform,
};
use scrying_glass::rdirect::{
    direct_render_sequence_hist_split, evidence_clamp_gamma, verify_stamp, CamPose, HistFrameSplit,
};
use scrying_glass::rdirect_demod::DemodPass;
use scrying_glass::rdirect_evidence::EvidenceClamp;
use scrying_glass::rdirect_gather::{FeatureGatherHistSplit, HistoryBuffers};
use scrying_glass::rdirect_live::RdirectLive;
use scrying_glass::scene::{Camera, RenderScene};

const DEPTH_TOL: f32 = 0.05;
const NORMAL_THRESH: f32 = 0.85;

fn cam_pose(cam: &Camera, w: u32, h: u32) -> CamPose {
    let (right, up, forward) = cam.basis();
    CamPose {
        eye: cam.eye,
        right,
        up,
        forward,
        half_tan: (cam.fov_y_radians * 0.5).tan(),
        aspect: w as f32 / h as f32,
    }
}

fn read_f32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, bytes: u64) -> Vec<f32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy") });
    enc.copy_buffer_to_buffer(buf, 0, &readback, 0, bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    enc.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r.map(|_| ()));
    });
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("readback chan").expect("map");
    let mapped = readback.get_mapped_range(..).expect("mapped");
    let v: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    v
}

/// Run one (label, camera sequence) through the live GPU pipeline (Stage
/// 2+3 kernels + the real net) and the CPU reference, printing per-frame
/// max-abs-diff of the PRESENTED (post-clamp) image.
#[allow(clippy::too_many_arguments)]
fn run_sequence(
    label: &str,
    cams: &[Camera],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    integrator: &Integrator,
    bvh: &Bvh,
    scene: &RenderScene,
    live: &RdirectLive,
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
) -> f32 {
    let n = (target_w * target_h) as usize;
    let gamma = evidence_clamp_gamma();

    let hist_gather = FeatureGatherHistSplit::new(device);
    let mut hist_buf = HistoryBuffers::new(device, target_w, target_h);
    let demod = DemodPass::new(device);
    let evidence = EvidenceClamp::new(device);

    let evidence_sum = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity evidence sum"),
        size: EvidenceClamp::sum_bytes(n).max(1),
        // COPY_SRC added for this diagnostic probe's boundary-flip readback
        // only (main.rs's live evidence_sum stays COPY_DST-only, untouched).
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_dl_padded = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity out_dl padded"),
        size: FeatureGatherHistSplit::out_dl_bytes(n).max(1),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let present_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity present accum"),
        size: (n as u64).max(1) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut cpu_frames: Vec<HistFrameSplit> = Vec::new();
    let mut gpu_presented: Vec<Vec<f32>> = Vec::new(); // per-frame [n,4] xyz used
    let mut count: u32 = 0;
    // BOUNDARY-FLIP PROBE: per-frame (pre-clamp net_linear, clamp ceiling),
    // both [n] Vec3, so a flipped pixel can be checked against the ceiling.
    let mut boundary_probe: Vec<(Vec<f32>, Vec<glam::Vec3>)> = Vec::new();

    // Own the per-frame CPU-side low_e/low_d/albedo/normal/depth so the
    // HistFrameSplit borrows stay alive for the single direct_render call
    // after this loop.
    let mut owned: Vec<(Vec<glam::Vec3>, Vec<glam::Vec3>, Vec<glam::Vec3>, Vec<glam::Vec3>, Vec<f32>)> =
        Vec::new();

    for cam in cams {
        let np = IntegratorParams { spp: 1, seed: 0x51de + (count * 131), ..IntegratorParams::default() };
        let accum_ed = integrator.make_split_buffer(device, low_w, low_h);
        let compute_bg = integrator.compute_bind_group(device, &integrator.make_accum(device, low_w, low_h));
        let split_bg = integrator.split_bind_group(device, &accum_ed);
        let uniform = IntegratorUniform::build(
            cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, integrator.node_count,
            integrator.tri_count, 0, &np, None,
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parity trace split") });
        integrator.dispatch_split(queue, &mut enc, &uniform, &compute_bg, &split_bg, low_w, low_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let aov_raw = trace_headless_aov(
            device, queue, bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
        );
        let aov_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("parity aov(native)"),
            contents: bytemuck::cast_slice(&aov_raw),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let cur_cam = cam_pose(cam, target_w, target_h);
        let prev_cam = hist_buf.prev_cam.unwrap_or(cur_cam);

        // ── GPU: gather_hist_split (39-in) ──
        let feat_bytes = FeatureGatherHistSplit::feature_bytes(n);
        let feats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("parity feats39"),
            size: feat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parity gather") });
        hist_gather.encode(
            device, queue, &mut enc, &accum_ed, &aov_buf, &feats_buf, &hist_buf.prev_out_dl,
            &hist_buf.prev_aov, cur_cam, prev_cam, hist_buf.has_prev, hist_buf.w, hist_buf.h,
            DEPTH_TOL, NORMAL_THRESH, low_w, low_h, target_w, target_h,
        );
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let gpu_feats = read_f32(device, queue, &feats_buf, feat_bytes);

        // ── REAL net forward (the shape-generic MPSGraph, synchronous) ──
        let out_dl = live.forward_cpu_roundtrip(&gpu_feats).expect("v7 forward");
        assert_eq!(out_dl.len(), n * 3);
        let net_out_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("parity net out_dl (tight)"),
            contents: bytemuck::cast_slice(&out_dl),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // ── demod (undo log-demod) -> present_buf = net_lin (unclamped) ──
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parity demod") });
        demod.encode(device, queue, &mut enc, &net_out_buf, &aov_buf, &present_buf, n as u32, false);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        // BOUNDARY-FLIP PROBE (diagnostic only, this example): capture the
        // PRE-clamp net_linear so a later flipped pixel can be checked
        // against the clamp ceiling. Extra readback, not part of the timed
        // live path this probe cross-checks.
        let unclamped_lin = read_f32(device, queue, &present_buf, (n as u64) * 16);

        // ── Stage 3 evidence accumulate (its own submit, so the boundary
        // probe can read the updated sum before the clamp consumes it) ──
        count += 1;
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parity evidence accumulate") });
        evidence.encode_accumulate(queue, device, &mut enc, &accum_ed, &evidence_sum, low_w, low_h, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let sum_raw = read_f32(device, queue, &evidence_sum, (n as u64) * 16);
        let sum_vec3: Vec<glam::Vec3> = (0..n)
            .map(|p| glam::Vec3::new(sum_raw[p * 4], sum_raw[p * 4 + 1], sum_raw[p * 4 + 2]))
            .collect();
        let mean: Vec<glam::Vec3> = sum_vec3.iter().map(|&s| s / (count as f32)).collect();
        let ceiling_lin: Vec<glam::Vec3> = scrying_glass::rdirect::local_max_3x3(&mean, target_w, target_h)
            .iter()
            .map(|&m| gamma * m.max(glam::Vec3::ZERO))
            .collect();

        // ── clamp + pack (unchanged) ──
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parity evidence clamp+pack") });
        evidence.encode_clamp(queue, device, &mut enc, &evidence_sum, &present_buf, target_w, target_h, count, gamma);
        evidence.encode_pack(queue, device, &mut enc, &net_out_buf, &out_dl_padded, n as u32);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let presented = read_f32(device, queue, &present_buf, (n as u64) * 16);
        gpu_presented.push(presented);
        boundary_probe.push((unclamped_lin, ceiling_lin));

        // ── history swap (this frame's own RAW out_dl feeds next frame) ──
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parity swap") });
        hist_buf.swap(&mut enc, &out_dl_padded, &aov_buf, cur_cam, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        // ── CPU-side frame data (same trace/AOV, re-derived on CPU for the
        // reference call — same seeds via the SAME uniform the GPU used is
        // not re-derivable bit-exact from spp=1 RNG on CPU vs GPU taps, so
        // instead we feed the CPU reference the SAME accum_ed/aov data by
        // reading them back — the honest apples-to-apples comparison is
        // "same input radiance/AOV, does the REST of the pipeline agree",
        // not an independent RNG cross-check (that is Stage 1/2's own job,
        // already proven bit-exact separately). ──
        let low_e_d = read_f32(device, queue, &accum_ed, (low_w as u64) * (low_h as u64) * 32);
        let (low_e, low_d) = split_from_cells(&low_e_d, (low_w * low_h) as usize);
        let (hi_albedo, hi_normal, hi_depth) = split_aov(&aov_raw);
        owned.push((low_e, low_d, hi_albedo, hi_normal, hi_depth));
    }

    for (i, cam) in cams.iter().enumerate() {
        let (low_e, low_d, hi_albedo, hi_normal, hi_depth) = &owned[i];
        cpu_frames.push(HistFrameSplit {
            low_e,
            low_d,
            low_w,
            low_h,
            hi_albedo,
            hi_normal,
            hi_depth,
            target_w,
            target_h,
            cam: cam_pose(cam, target_w, target_h),
        });
    }

    let cpu_out = direct_render_sequence_hist_split(live.cpu_ref(), &cpu_frames, DEPTH_TOL, NORMAL_THRESH);

    let mut max_all = 0f32;
    for (f, (gpu, cpu)) in gpu_presented.iter().zip(cpu_out.iter()).enumerate() {
        let mut max_d = 0f32;
        let mut sum_d = 0f64;
        let mut over = 0u32;
        for p in 0..n {
            let gx = gpu[p * 4];
            let gy = gpu[p * 4 + 1];
            let gz = gpu[p * 4 + 2];
            let c = cpu[p];
            let d = (gx - c.x).abs().max((gy - c.y).abs()).max((gz - c.z).abs());
            max_d = max_d.max(d);
            sum_d += d as f64;
            if d > 1.0e-3 {
                over += 1;
            }
        }
        max_all = max_all.max(max_d);
        let mean_d = sum_d / n as f64;
        println!(
            "[v7-present-parity] {label} frame {f} N={n} max-abs-diff {max_d:.4e} mean-abs-diff {mean_d:.4e} px>1e-3={over}"
        );

        // BOUNDARY-FLIP PROBE: dump the 5 worst-diff pixels' pre-clamp
        // net_linear vs this frame's clamp ceiling (gamma*local_max_evidence)
        // — a flip is "benign fp min() boundary" if unclamped sits within
        // ~1e-3 of the ceiling on either side; anything wider is a real gap.
        if over > 0 {
            let (unclamped, ceiling) = &boundary_probe[f];
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| {
                let da = (gpu[a * 4] - cpu[a].x).abs().max((gpu[a * 4 + 1] - cpu[a].y).abs()).max((gpu[a * 4 + 2] - cpu[a].z).abs());
                let db = (gpu[b * 4] - cpu[b].x).abs().max((gpu[b * 4 + 1] - cpu[b].y).abs()).max((gpu[b * 4 + 2] - cpu[b].z).abs());
                db.partial_cmp(&da).unwrap()
            });
            for &p in idx.iter().take(5) {
                let u = glam::Vec3::new(unclamped[p * 4], unclamped[p * 4 + 1], unclamped[p * 4 + 2]);
                let c = ceiling[p];
                let dist = (u - c).abs();
                println!(
                    "[v7-present-parity]   {label} f{f} flip px={p} gpu=({:.5},{:.5},{:.5}) cpu=({:.5},{:.5},{:.5}) unclamped=({:.5},{:.5},{:.5}) ceiling=({:.5},{:.5},{:.5}) |unclamped-ceiling|=({:.2e},{:.2e},{:.2e})",
                    gpu[p * 4], gpu[p * 4 + 1], gpu[p * 4 + 2],
                    cpu[p].x, cpu[p].y, cpu[p].z,
                    u.x, u.y, u.z, c.x, c.y, c.z, dist.x, dist.y, dist.z
                );
            }
        }
    }
    println!("[v7-present-parity] {label} OVERALL max-abs-diff {max_all:.4e}");
    max_all
}

fn split_from_cells(cells: &[f32], n: usize) -> (Vec<glam::Vec3>, Vec<glam::Vec3>) {
    // Mirrors accum_ed layout: 2 vec4 cells/px = 8 f32/px (E sum+count, D
    // sum+count). Normalize sum/count -> radiance, same as the WGSL kernels.
    let mut e = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    for i in 0..n {
        let ec = &cells[i * 8..i * 8 + 4];
        let dc = &cells[i * 8 + 4..i * 8 + 8];
        let ecount = ec[3].max(1.0);
        let dcount = dc[3].max(1.0);
        e.push(glam::Vec3::new(ec[0] / ecount, ec[1] / ecount, ec[2] / ecount));
        d.push(glam::Vec3::new(dc[0] / dcount, dc[1] / dcount, dc[2] / dcount));
    }
    (e, d)
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[v7-present-parity] SKIP — no GPU adapter");
        return;
    };

    let weights_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rdirect-weights-v7.bin");
    let weights = std::fs::read(&weights_path).expect("read v7 weights");
    let stamp_path = scrying_glass::rdirect::stamp_path_for(&weights_path);
    let stamped = verify_stamp(&weights, &stamp_path);
    println!("[v7-present-parity] weights stamp PASS: {stamped}");
    let live = RdirectLive::from_system(&weights).expect("v7 net load");
    println!(
        "[v7-present-parity] live net loaded in_features={} out_channels={}",
        live.in_features(),
        live.out_channels()
    );

    let params = naruko_params();
    let world_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    let integrator = Integrator::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, &bvh, None);

    let (low_w, low_h, target_w, target_h) = (48u32, 32u32, 96u32, 64u32);
    let pivot = [0.0, 2.0, 0.0];

    let still_cam = orbit_camera(params.camera_position, pivot, 0.0, params.fov_y_degrees);
    let still: Vec<Camera> = vec![still_cam; 3];
    let pan: Vec<Camera> = [-3.0f32, 0.0, 3.0]
        .iter()
        .map(|yaw| orbit_camera(params.camera_position, pivot, *yaw, params.fov_y_degrees))
        .collect();

    let max_still = run_sequence(
        "still", &still, &device, &queue, &integrator, &bvh, &scene, &live, low_w, low_h, target_w, target_h,
    );
    let max_pan = run_sequence(
        "pan", &pan, &device, &queue, &integrator, &bvh, &scene, &live, low_w, low_h, target_w, target_h,
    );

    println!(
        "[v7-present-parity] SUMMARY still_max={max_still:.4e} pan_max={max_pan:.4e}"
    );
}
