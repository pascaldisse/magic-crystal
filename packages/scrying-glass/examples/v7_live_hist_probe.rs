//! V7-LIVE LANE STAGE 2 — recurrent history probe (39-in, GPU vs CPU).
//!
//! Drives a 3-FRAME small-pan pose sequence (orbit yaw -3/0/+3 deg around the
//! naruko "front" pivot — continuous camera motion, so reprojection is
//! genuinely exercised: most pixels reproject validly frame-to-frame, edges
//! disocclude) through:
//!
//!   GPU: Stage-1's `gather_split` (35-in, unmodified) feeding the NEW
//!   `gather_hist_split` entry (`FeatureGatherHistSplit`, rdirect_gather.rs)
//!   — the SAME wiring `NetPresent`'s `evidence_split` branch would use once
//!   Stage 3 lands a 39-in net. History ping-pong (`HistoryBuffers::swap`)
//!   runs between frames exactly as the live path would run it.
//!
//!   CPU: `pixel_features_split` (base 35) + a hand-transcribed copy of
//!   `rdirect::direct_render_sequence_hist_split`'s reprojection block
//!   (world = cur_cam.ray_dir*depth, `CamPose::reproject` into the prev
//!   camera, nearest-pixel depth/normal reject test, bilinear resample of
//!   `prev_dl` on accept) + `hist_features_split` — bit-for-bit the same
//!   spec the WGSL kernel implements. NOT calling
//!   `direct_render_sequence_hist_split` itself (that function also runs the
//!   net forward + evidence clamp, is load-bearing for the stamped real-image
//!   ordeal, and this lane must not touch/risk it) — this file re-derives the
//!   reprojection-only slice as an independent cross-check.
//!
//! "Previous net output" doesn't exist yet this stage (no 39-in net loaded —
//! that's Stage 3). Both CPU and GPU instead feed forward a SYNTHETIC but
//! IDENTICAL stand-in per frame: `out_dl(frame) = base_feat[0..3]` (that
//! frame's own E-tap0 demod-log value, itself Stage-1 parity-proven
//! bit-exact GPU vs CPU). This exercises the real plumbing — the ping-pong
//! buffer, the reprojection math, the depth/normal guard, the bilinear
//! resample — without depending on unbuilt Stage-3 net weights. Only history
//! features (35-38) are meaningful checks on frames 1-2; frame 0 checks the
//! zero/invalid rule.
//!
//! Run: cargo run --release -j2 --example v7_live_hist_probe

use glam::{Vec2, Vec3 as GVec3};
use wgpu::util::DeviceExt;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser_dataset::{naruko_params, orbit_camera};
use scrying_glass::integrator::{
    headless_device, split_aov, trace_headless_aov, trace_headless_split, Integrator,
    IntegratorParams, IntegratorUniform,
};
use scrying_glass::rdirect::{pixel_features_split, CamPose, INPUT_FEATURES_SPLIT};
use scrying_glass::rdirect_gather::{FeatureGatherHistSplit, HistoryBuffers};
use scrying_glass::scene::{Camera, RenderScene};

const DEPTH_TOL: f32 = 0.05; // matches examples/real_image_ordeal.rs DEPTH_TOL
const NORMAL_THRESH: f32 = 0.85; // matches examples/real_image_ordeal.rs NORMAL_THRESH

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

/// CPU cross-check reprojection — a bit-for-bit transcription of
/// `rdirect::direct_render_sequence_hist_split`'s per-pixel history block
/// (rdirect.rs), stopping short of the net forward / evidence clamp (out of
/// scope this stage; see module doc). `prev` is `None` on frame 0.
#[allow(clippy::too_many_arguments)]
fn cpu_history(
    cur_depth: f32,
    cur_normal: GVec3,
    cur_cam: &CamPose,
    tx: u32,
    ty: u32,
    tw: u32,
    th: u32,
    prev: Option<(&[[f32; 3]], &[f32], &[GVec3], &CamPose, u32, u32)>,
) -> ([f32; 3], f32) {
    match prev {
        None => ([0.0; 3], 0.0),
        Some((p_dl, p_depth, p_norm, p_cam, pw, ph)) => {
            let is_miss = cur_depth <= 0.0;
            let dir = cur_cam.ray_dir(tx, ty, tw, th);
            let dist = if is_miss { 1.0e5 } else { cur_depth };
            let world = cur_cam.eye + dir * dist;
            match p_cam.reproject(world, pw, ph) {
                None => ([0.0; 3], 0.0),
                Some((fx, fy)) => {
                    let ipx = fx.round().clamp(0.0, (pw - 1) as f32) as usize;
                    let ipy = fy.round().clamp(0.0, (ph - 1) as f32) as usize;
                    let pj = ipy * pw as usize + ipx;
                    let prev_depth = p_depth[pj];
                    let prev_miss = prev_depth <= 0.0;
                    let ok = if is_miss {
                        prev_miss
                    } else if prev_miss {
                        false
                    } else {
                        let dist_prev = (world - p_cam.eye).length();
                        let depth_ok = (dist_prev - prev_depth).abs() <= DEPTH_TOL * dist_prev.max(1e-4);
                        let normal_ok = cur_normal.dot(p_norm[pj]) >= NORMAL_THRESH;
                        depth_ok && normal_ok
                    };
                    if ok {
                        // Bilinear resample of the prev demod-log output (SAME
                        // `bilinear_vec3` shape as rdirect.rs, transcribed).
                        let x0 = fx.floor() as i32;
                        let y0 = fy.floor() as i32;
                        let txf = fx - x0 as f32;
                        let tyf = fy - y0 as f32;
                        let cl = |v: i32, hi: u32| v.clamp(0, hi as i32 - 1) as usize;
                        let x0c = cl(x0, pw);
                        let x1c = cl(x0 + 1, pw);
                        let y0c = cl(y0, ph);
                        let y1c = cl(y0 + 1, ph);
                        let idx = |x: usize, y: usize| y * pw as usize + x;
                        let g = |x: usize, y: usize| {
                            let d = p_dl[idx(x, y)];
                            GVec3::new(d[0], d[1], d[2])
                        };
                        let a = g(x0c, y0c);
                        let b = g(x1c, y0c);
                        let c = g(x0c, y1c);
                        let d = g(x1c, y1c);
                        let top = a * (1.0 - txf) + b * txf;
                        let bot = c * (1.0 - txf) + d * txf;
                        let s = top * (1.0 - tyf) + bot * tyf;
                        ([s.x, s.y, s.z], 1.0)
                    } else {
                        ([0.0; 3], 0.0)
                    }
                }
            }
        }
    }
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[v7-hist-probe] SKIP — no GPU adapter");
        return;
    };

    let params = naruko_params();
    let world_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());

    let pivot = [0.0, 2.0, 0.0];
    let cams: Vec<Camera> = [-3.0f32, 0.0, 3.0]
        .iter()
        .map(|yaw| orbit_camera(params.camera_position, pivot, *yaw, params.fov_y_degrees))
        .collect();

    let (low_w, low_h, target_w, target_h) = (48u32, 32u32, 96u32, 64u32);
    let n = (target_w * target_h) as usize;

    let integrator = Integrator::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, &bvh, None);
    let gather_hist = FeatureGatherHistSplit::new(&device);
    let mut hist_buf = HistoryBuffers::new(&device, target_w, target_h);

    // CPU-side rolling "previous frame" state (mirrors HistoryBuffers, in
    // system RAM): out_dl image, depth, normal, cam.
    let mut cpu_prev: Option<(Vec<[f32; 3]>, Vec<f32>, Vec<GVec3>, CamPose, u32, u32)> = None;

    let mut max_hist_all = 0f32;
    let mut max_base_all = 0f32;
    for (f, cam) in cams.iter().enumerate() {
        let np = IntegratorParams { spp: 1, seed: 0x7abc + (f as u32) * 131 + 5, ..IntegratorParams::default() };
        let accum_ed = integrator.make_split_buffer(&device, low_w, low_h);
        let compute_bg = integrator.compute_bind_group(&device, &integrator.make_accum(&device, low_w, low_h));
        let split_bg = integrator.split_bind_group(&device, &accum_ed);
        let uniform = IntegratorUniform::build(
            cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, integrator.node_count,
            integrator.tri_count, 0, &np, None,
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hist-probe trace split"),
        });
        integrator.dispatch_split(&queue, &mut enc, &uniform, &compute_bg, &split_bg, low_w, low_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let aov_raw = trace_headless_aov(
            &device, &queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
        );
        let aov_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aov(native)"),
            contents: bytemuck::cast_slice(&aov_raw),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let cur_cam = cam_pose(cam, target_w, target_h);

        // ── GPU: gather_hist_split into a 39-wide feats buffer ──
        let feat_bytes = FeatureGatherHistSplit::feature_bytes(n);
        let feats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("feats39"),
            size: feat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hist-probe gather"),
        });
        gather_hist.encode(
            &device, &queue, &mut enc, &accum_ed, &aov_buf, &feats_buf, &hist_buf.prev_out_dl,
            &hist_buf.prev_aov, cur_cam, hist_buf.prev_cam.unwrap_or(cur_cam), hist_buf.has_prev,
            hist_buf.w, hist_buf.h, DEPTH_TOL, NORMAL_THRESH, low_w, low_h, target_w, target_h,
        );
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let gpu_feats = read_f32(&device, &queue, &feats_buf, feat_bytes);

        // ── CPU reference: base(35) + history(4) ──
        let (low_e, low_d) = trace_headless_split(
            &device, &queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np,
        );
        let (hi_albedo, hi_normal, hi_depth) = split_aov(&aov_raw);

        let mut cpu_feats39 = Vec::<f32>::with_capacity(n * 39);
        let mut cur_out_dl: Vec<[f32; 3]> = Vec::with_capacity(n);
        for ty in 0..target_h {
            for tx in 0..target_w {
                let i = (ty * target_w + tx) as usize;
                let base = pixel_features_split(
                    &low_e, &low_d, low_w, low_h, target_w, target_h, tx, ty, hi_albedo[i],
                    hi_normal[i], hi_depth[i], Vec2::ZERO,
                );
                let prev_ref = cpu_prev
                    .as_ref()
                    .map(|(dl, d, nrm, c, pw, ph)| (dl.as_slice(), d.as_slice(), nrm.as_slice(), c, *pw, *ph));
                let (prev_dl, valid) =
                    cpu_history(hi_depth[i], hi_normal[i], &cur_cam, tx, ty, target_w, target_h, prev_ref);
                cpu_feats39.extend_from_slice(&base);
                cpu_feats39.push(prev_dl[0]);
                cpu_feats39.push(prev_dl[1]);
                cpu_feats39.push(prev_dl[2]);
                cpu_feats39.push(valid);
                // Synthetic "net output" stand-in for next frame's history —
                // SAME deterministic value both sides use (see module doc).
                cur_out_dl.push([base[0], base[1], base[2]]);
            }
        }

        // ── compare ──
        let mut max_base = 0f32;
        let mut max_hist = 0f32;
        for p in 0..n {
            for k in 0..INPUT_FEATURES_SPLIT {
                let d = (gpu_feats[p * 39 + k] - cpu_feats39[p * 39 + k]).abs();
                max_base = max_base.max(d);
            }
            for k in INPUT_FEATURES_SPLIT..39 {
                let d = (gpu_feats[p * 39 + k] - cpu_feats39[p * 39 + k]).abs();
                max_hist = max_hist.max(d);
            }
        }
        max_base_all = max_base_all.max(max_base);
        max_hist_all = max_hist_all.max(max_hist);
        println!(
            "[v7-hist-probe] frame {f} N={n} px x 39 feat — base(0-34) max-abs-diff {max_base:.3e} · \
             history(35-38) max-abs-diff {max_hist:.3e} (has_prev={})",
            hist_buf.has_prev
        );

        // ── GPU ping-pong swap: this frame's own out_dl → prev_out_dl, this
        // frame's aov → prev_aov, remember cur_cam. Feed the SAME synthetic
        // stand-in (E-tap0 of the 39-wide feats buffer, idx 0..3) as
        // "out_dl", written via the gather kernel itself (feats[0..3] ==
        // cur_out_dl by construction), so the GPU buffer we copy from is the
        // GPU's own computed value (not a CPU upload) — a true GPU-resident
        // ping-pong. ──
        let cur_out_dl_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cur out_dl (39-in E-tap0 stand-in)"),
            size: FeatureGatherHistSplit::out_dl_bytes(n),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Extract E-tap0 (feats[0..3]) into a padded vec4/px buffer via CPU
        // upload of the GPU-read-back feats (already read back above) — the
        // VALUES are GPU-computed (Stage-1 gather_split proven bit-exact),
        // only the repack (35/39-stride → vec4-stride) touches the CPU.
        let mut padded: Vec<f32> = Vec::with_capacity(n * 4);
        for p in 0..n {
            padded.push(gpu_feats[p * 39]);
            padded.push(gpu_feats[p * 39 + 1]);
            padded.push(gpu_feats[p * 39 + 2]);
            padded.push(0.0);
        }
        queue.write_buffer(&cur_out_dl_buf, 0, bytemuck::cast_slice(&padded));
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hist-probe swap"),
        });
        hist_buf.swap(&mut enc, &cur_out_dl_buf, &aov_buf, cur_cam, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        // CPU-side prev advance (mirrors the GPU swap exactly).
        cpu_prev = Some((cur_out_dl, hi_depth.clone(), hi_normal.clone(), cur_cam, target_w, target_h));
    }

    println!(
        "[v7-hist-probe] OVERALL base max-abs-diff {max_base_all:.3e} · history max-abs-diff {max_hist_all:.3e}"
    );
    // Same float-ULP-class bound as the Stage-1 n0b/v7-ed-probe gates.
    assert!(max_base_all < 1.0e-4, "base feature parity regressed vs Stage 1");
    assert!(max_hist_all < 1.0e-4, "history feature parity out of bound");
    println!("[v7-hist-probe] PASS — GPU 39-in gather (base+history) matches CPU reference");
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

// (aov buffers are built inline above via `device.create_buffer_init`.)
