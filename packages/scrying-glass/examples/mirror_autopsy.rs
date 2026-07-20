//! MIRROR AUTOPSY — one-off diagnostic, not a lane stage. The Architect:
//! "the mirror still looks weird" on the live window, looking at
//! `naruko_show_chrome` (the large chrome sphere r=2.1 at [4.5,3.6,29.5],
//! metallic 1.0 roughness 0.02 — pure specular). Camera = `spawn_eye`
//! (realm_shine.rs's [0,1.7,44] yaw0 pitch0 fov60), the SAME pose
//! `proof/realm-shine-a.png` already proves frames the sphere.
//!
//! Reuses the exact GPU pipeline `v7_present_parity_probe.rs` proved wired
//! (trace split -> AOV -> gather_hist_split(39) -> real v7 net forward ->
//! demod -> evidence accumulate -> clamp+pack -> swap), run for 3 identical
//! still frames (steady-state history), and on the LAST frame dumps:
//!   (1) presented   — the post-clamp image the live window would show.
//!   (2) evidence    — bilinear-upsampled low-res E+D composite (linear,
//!                     no net), i.e. exactly `evidence_composite_frame`'s
//!                     output — the clamp's own raw ceiling INPUT signal
//!                     before the temporal-mean/3x3-max/gamma pipeline.
//!   (3) preclamp    — the net's own linear output BEFORE the evidence
//!                     clamp (post-demod, pre-clamp `present_buf`).
//!
//! Run: cargo run --release -j2 --example mirror_autopsy

use wgpu::util::DeviceExt;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser_dataset::naruko_params;
use scrying_glass::integrator::{
    headless_device, split_aov, trace_headless_aov, Integrator, IntegratorParams, IntegratorUniform,
};
use scrying_glass::rdirect::{evidence_clamp_gamma, evidence_composite_frame, local_max_3x3, verify_stamp};
use scrying_glass::rdirect_demod::DemodPass;
use scrying_glass::rdirect_evidence::EvidenceClamp;
use scrying_glass::rdirect_gather::FeatureGatherHistSplit;
use scrying_glass::rdirect_gather::HistoryBuffers;
use scrying_glass::rdirect_live::RdirectLive;
use scrying_glass::scene::{Camera, RenderScene};

fn cam_pose(cam: &Camera, w: u32, h: u32) -> scrying_glass::rdirect::CamPose {
    let (right, up, forward) = cam.basis();
    scrying_glass::rdirect::CamPose {
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

fn split_from_cells(cells: &[f32], n: usize) -> (Vec<glam::Vec3>, Vec<glam::Vec3>) {
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

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn write_png(img: &[glam::Vec3], w: u32, h: u32, exposure: f32, path: &std::path::Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in img {
        bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
    }
    let file = std::fs::File::create(path).unwrap();
    let writer = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(writer, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("[mirror-autopsy] wrote {}", path.display());
}

/// Region stats (mean/max luminance + mean|Δ| of channel spread, a coarse
/// "how noisy/blown-out is this patch" reading) over a screen rect.
fn region_stats(img: &[glam::Vec3], w: u32, rect: (u32, u32, u32, u32), label: &str) {
    let (x0, y0, x1, y1) = rect;
    let mut sum = glam::Vec3::ZERO;
    let mut max_lum = 0f32;
    let mut n = 0f64;
    let mut lums: Vec<f32> = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let p = img[(y * w + x) as usize];
            let lum = 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
            sum += p;
            max_lum = max_lum.max(lum);
            lums.push(lum);
            n += 1.0;
        }
    }
    let mean = sum / n as f32;
    let mean_lum = 0.2126 * mean.x + 0.7152 * mean.y + 0.0722 * mean.z;
    // local variance as a sparkle/noise proxy: mean |lum - mean_lum|
    let mad: f64 = lums.iter().map(|&l| (l - mean_lum).abs() as f64).sum::<f64>() / n;
    println!(
        "[mirror-autopsy]   {label} rect={rect:?} n={n} mean=({:.4},{:.4},{:.4}) mean_lum={mean_lum:.4} max_lum={max_lum:.4} mad_lum={mad:.5}",
        mean.x, mean.y, mean.z
    );
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[mirror-autopsy] SKIP — no GPU adapter");
        return;
    };

    let weights_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rdirect-weights-v7.bin");
    let weights = std::fs::read(&weights_path).expect("read v7 weights");
    let stamp_path = scrying_glass::rdirect::stamp_path_for(&weights_path);
    let stamped = verify_stamp(&weights, &stamp_path);
    println!("[mirror-autopsy] weights stamp PASS: {stamped}");
    let live = RdirectLive::from_system(&weights).expect("v7 net load");
    println!(
        "[mirror-autopsy] live net loaded in_features={} out_channels={}",
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

    // spawn_eye (realm_shine.rs): the settled gameplay eye the player
    // actually sees, [0,1.7,44] yaw0 pitch0 fov60 — proven by
    // proof/realm-shine-a.png to frame naruko_show_chrome (the large
    // mirror sphere) squarely in view.
    let cam = Camera {
        eye: glam::Vec3::new(0.0, 1.7, 44.0),
        yaw: 0.0,
        pitch: 0.0,
        fov_y_radians: 60f32.to_radians(),
        near: 0.1,
        far: 4000.0,
    };

    // 16:9 to match the proof shot's own framing; 2x low/native ratio same
    // as the lane's own gather convention.
    let (low_w, low_h, target_w, target_h) = (240u32, 135u32, 480u32, 270u32);
    let n = (target_w * target_h) as usize;
    let gamma = evidence_clamp_gamma();

    let hist_gather = FeatureGatherHistSplit::new(&device);
    let mut hist_buf = HistoryBuffers::new(&device, target_w, target_h);
    let demod = DemodPass::new(&device);
    let evidence = EvidenceClamp::new(&device);

    let evidence_sum = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mirror-autopsy evidence sum"),
        size: EvidenceClamp::sum_bytes(n).max(1),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_dl_padded = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mirror-autopsy out_dl padded"),
        size: FeatureGatherHistSplit::out_dl_bytes(n).max(1),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let present_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mirror-autopsy present accum"),
        size: (n as u64).max(1) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let frames = 3u32; // steady-state history (same recurrence depth the
                        // live window reaches within a couple frames of a
                        // still eye)
    let mut count: u32 = 0;
    let mut last_presented: Vec<f32> = Vec::new();
    let mut last_unclamped: Vec<f32> = Vec::new();
    let mut last_evidence_raw: Vec<glam::Vec3> = Vec::new();

    for f in 0..frames {
        let np = IntegratorParams { spp: 1, seed: 0x51de + (f * 131), ..IntegratorParams::default() };
        let accum_ed = integrator.make_split_buffer(&device, low_w, low_h);
        let compute_bg = integrator.compute_bind_group(&device, &integrator.make_accum(&device, low_w, low_h));
        let split_bg = integrator.split_bind_group(&device, &accum_ed);
        let uniform = IntegratorUniform::build(
            &cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, integrator.node_count,
            integrator.tri_count, 0, &np, None,
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("trace split") });
        integrator.dispatch_split(&queue, &mut enc, &uniform, &compute_bg, &split_bg, low_w, low_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let aov_raw = trace_headless_aov(
            &device, &queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
        );
        let aov_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mirror-autopsy aov(native)"),
            contents: bytemuck::cast_slice(&aov_raw),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let cur_cam = cam_pose(&cam, target_w, target_h);
        let prev_cam = hist_buf.prev_cam.unwrap_or(cur_cam);

        let feat_bytes = FeatureGatherHistSplit::feature_bytes(n);
        let feats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mirror-autopsy feats39"),
            size: feat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gather") });
        hist_gather.encode(
            &device, &queue, &mut enc, &accum_ed, &aov_buf, &feats_buf, &hist_buf.prev_out_dl,
            &hist_buf.prev_aov, cur_cam, prev_cam, hist_buf.has_prev, hist_buf.w, hist_buf.h,
            0.05, 0.85, low_w, low_h, target_w, target_h,
        );
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let gpu_feats = read_f32(&device, &queue, &feats_buf, feat_bytes);

        let out_dl = live.forward_cpu_roundtrip(&gpu_feats).expect("v7 forward");
        assert_eq!(out_dl.len(), n * 3);
        let net_out_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mirror-autopsy net out_dl (tight)"),
            contents: bytemuck::cast_slice(&out_dl),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("demod") });
        demod.encode(&device, &queue, &mut enc, &net_out_buf, &aov_buf, &present_buf, n as u32, false);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let unclamped_lin = read_f32(&device, &queue, &present_buf, (n as u64) * 16);

        count += 1;
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("evidence accumulate") });
        evidence.encode_accumulate(&queue, &device, &mut enc, &accum_ed, &evidence_sum, low_w, low_h, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clamp+pack") });
        evidence.encode_clamp(&queue, &device, &mut enc, &evidence_sum, &present_buf, target_w, target_h, count, gamma);
        evidence.encode_pack(&queue, &device, &mut enc, &net_out_buf, &out_dl_padded, n as u32);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let presented = read_f32(&device, &queue, &present_buf, (n as u64) * 16);

        // Independent CPU-side evidence composite (task's item (2)): bilinear
        // upsample of THIS frame's low_e/low_d, exactly `evidence_composite_frame`.
        let low_e_d = read_f32(&device, &queue, &accum_ed, (low_w as u64) * (low_h as u64) * 32);
        let (low_e, low_d) = split_from_cells(&low_e_d, (low_w * low_h) as usize);
        let evidence_raw = evidence_composite_frame(&low_e, &low_d, low_w, low_h, target_w, target_h);

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("swap") });
        hist_buf.swap(&mut enc, &out_dl_padded, &aov_buf, cur_cam, target_w, target_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        println!("[mirror-autopsy] frame {f} done (has_prev={})", hist_buf.has_prev);
        last_presented = presented;
        last_unclamped = unclamped_lin;
        last_evidence_raw = evidence_raw;
    }

    // ---- write the 3 PNGs (last/steady-state frame) ----
    let to_vec3 = |flat: &[f32]| -> Vec<glam::Vec3> {
        (0..n).map(|p| glam::Vec3::new(flat[p * 4], flat[p * 4 + 1], flat[p * 4 + 2])).collect()
    };
    let presented_img = to_vec3(&last_presented);
    let preclamp_img = to_vec3(&last_unclamped);
    let evidence_img = last_evidence_raw;

    let proof_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof/neural-live");
    write_png(&presented_img, target_w, target_h, 1.0, &proof_dir.join("mirror-presented.png"));
    write_png(&evidence_img, target_w, target_h, 1.0, &proof_dir.join("mirror-evidence.png"));
    write_png(&preclamp_img, target_w, target_h, 1.0, &proof_dir.join("mirror-preclamp.png"));

    // Sphere screen-region: chrome sphere world [4.5,3.6,29.5] r 2.1, camera
    // as above. Projected bbox worked by hand from the same basis() the
    // realm_shine.rs `project` helper uses; reported here as a coarse rect
    // (this cam frames it left-of-center per realm-shine-a.png; use pixel
    // margins around the analytically nearest column/rows).
    let (right, up, forward) = cam.basis();
    let project = |p: glam::Vec3| -> Option<(f32, f32)> {
        let v = p - cam.eye;
        let zf = v.dot(forward);
        if zf <= 0.0 {
            return None;
        }
        let tan_h = (cam.fov_y_radians * 0.5).tan();
        let aspect = target_w as f32 / target_h as f32;
        let sx = v.dot(right) / (zf * tan_h * aspect);
        let sy = v.dot(up) / (zf * tan_h);
        Some(((sx + 1.0) * 0.5 * target_w as f32, (1.0 - sy) * 0.5 * target_h as f32))
    };
    let sphere_pts = [
        glam::Vec3::new(4.5 - 2.1, 3.6, 29.5),
        glam::Vec3::new(4.5 + 2.1, 3.6, 29.5),
        glam::Vec3::new(4.5, 3.6 + 2.1, 29.5),
        glam::Vec3::new(4.5, 3.6 - 2.1, 29.5),
    ];
    let mut x0 = f32::MAX;
    let mut y0 = f32::MAX;
    let mut x1 = f32::MIN;
    let mut y1 = f32::MIN;
    for &p in &sphere_pts {
        if let Some((px, py)) = project(p) {
            x0 = x0.min(px);
            y0 = y0.min(py);
            x1 = x1.max(px);
            y1 = y1.max(py);
        }
    }
    let rect = (
        x0.max(0.0) as u32,
        y0.max(0.0) as u32,
        x1.min(target_w as f32 - 1.0) as u32,
        y1.min(target_h as f32 - 1.0) as u32,
    );
    println!("[mirror-autopsy] sphere screen rect (projected): {rect:?}");
    println!("[mirror-autopsy] === REGION STATS (sphere) ===");
    region_stats(&presented_img, target_w, rect, "presented");
    region_stats(&evidence_img, target_w, rect, "evidence(raw E+D upsample)");
    region_stats(&preclamp_img, target_w, rect, "preclamp(net, pre-clamp)");

    // A full-frame sky-band control (top eighth) — should be flat/quiet in
    // all 3 (a sanity check the sphere numbers above aren't just generic
    // frame noise).
    let sky_rect = (0u32, 0u32, target_w - 1, target_h / 8);
    println!("[mirror-autopsy] === REGION STATS (sky control) ===");
    region_stats(&presented_img, target_w, sky_rect, "presented");
    region_stats(&evidence_img, target_w, sky_rect, "evidence(raw E+D upsample)");
    region_stats(&preclamp_img, target_w, sky_rect, "preclamp(net, pre-clamp)");

    println!("[mirror-autopsy] done");
}
