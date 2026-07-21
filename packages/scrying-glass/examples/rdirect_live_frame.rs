//! NEURAL-LIVE N0.c (evidence) — THE NET'S FRAME through the REAL live GPU path,
//! written to a PNG to READ. This runs exactly the N0.b live embodiment:
//!   trace low-res radiance (accum) + native AOV G-buffer
//!     → GPU gather (rdirect_gather.wgsl) into the pooled shared MTLBuffer
//!     → MPSGraph batched-GEMM forward_shared (zero-copy)
//!     → undo_log_demod by the native albedo
//!     → sRGB tonemap → PNG.
//! It is the net output the live WINDOW present will show (the window-blit +
//! GAIA_NATIVE_NET_PRESENT wiring in main.rs is the remaining N0.c step); this
//! is the same pixels, off the same code, so the frame is READABLE now.
//!
//! Weights were trained at the 96×64 static shape — the bigger shape here is
//! the honest "native quality may be rough" look (quality is N1's job).
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_live_frame

use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::{Vec2, Vec3};
use wgpu::util::DeviceExt;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser_dataset::{law_poses, naruko_params};
use scrying_glass::integrator::{
    headless_device, resolve, split_aov, trace_headless, trace_headless_aov, IntegratorParams,
};
use scrying_glass::rdirect::{bilinear_upsample, ALBEDO_DEMOD_EPS};
use scrying_glass::rdirect_gather::FeatureGather;
use scrying_glass::rdirect_live::RdirectLive;
use scrying_glass::scene::RenderScene;

const NO_HIT_SQ: f32 = 1e-8;

fn demod_divisor(albedo: Vec3) -> Vec3 {
    if albedo.length_squared() > NO_HIT_SQ {
        albedo + Vec3::splat(ALBEDO_DEMOD_EPS)
    } else {
        Vec3::ONE
    }
}

fn undo_log_demod(dl: Vec3, divisor: Vec3) -> Vec3 {
    let e = Vec3::new(dl.x.exp() - 1.0, dl.y.exp() - 1.0, dl.z.exp() - 1.0);
    Vec3::new(e.x.max(0.0), e.y.max(0.0), e.z.max(0.0)) * divisor
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn write_png(img: &[Vec3], w: u32, h: u32, exposure: f32, path: &Path) {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).unwrap();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in img {
        bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
    }
    let f = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("[n0c] wrote {}", path.display());
}

/// Exposure that maps the image's mean luminance to ~0.18 mid-gray (honest
/// auto-exposure; printed so the look is reproducible).
fn auto_exposure(img: &[Vec3]) -> f32 {
    let mut sum = 0f64;
    let mut n = 0u64;
    for p in img {
        let l = 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
        if l.is_finite() && l > 0.0 {
            sum += l as f64;
            n += 1;
        }
    }
    if n == 0 {
        return 1.0;
    }
    let mean = (sum / n as f64) as f32;
    (0.18 / mean.max(1e-6)).clamp(0.05, 50.0)
}

fn net_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    weights: &[u8],
    bvh: &Bvh,
    scene: &RenderScene,
    front: &scrying_glass::scene::Camera,
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
    tag: &str,
) {
    let n = (target_w * target_h) as usize;
    let noisy = IntegratorParams { spp: 1, ..IntegratorParams::default() };
    let accum_raw = trace_headless(
        device, queue, bvh, front, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1,
        &noisy, None,
    );
    let aov_raw = trace_headless_aov(
        device, queue, bvh, front, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
    );
    let (hi_albedo, _hi_normal, _hi_depth) = split_aov(&aov_raw);
    let low_radiance = resolve(&accum_raw);

    // Upload the live buffers as STORAGE, build the net + pool, gather + forward.
    let accum_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("accum"),
        contents: bytemuck::cast_slice(&accum_raw),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let aov_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("aov"),
        contents: bytemuck::cast_slice(&aov_raw),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let live = RdirectLive::from_wgpu_queue(device, queue, weights, n)
        .expect("live net on the wgpu Metal device");
    let feats = live.feature_buffer().expect("pooled feature buffer");
    let gather = FeatureGather::new(device);

    let t0 = Instant::now();
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    gather.encode(
        device, queue, &mut enc, &accum_buf, &aov_buf, feats, low_w, low_h, target_w, target_h,
    );
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let gather_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let out = live.forward_shared(n).expect("shared forward");
    let fwd_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Net demod-log → final radiance by the native albedo.
    let mut net_img = Vec::with_capacity(n);
    for i in 0..n {
        let div = demod_divisor(hi_albedo[i]);
        net_img.push(undo_log_demod(
            Vec3::new(out[3 * i], out[3 * i + 1], out[3 * i + 2]),
            div,
        ));
    }
    // Naive bilinear upscale of the SAME low radiance — the honest "what the net
    // replaces" reference panel.
    let bilinear = bilinear_upsample(&low_radiance, low_w, low_h, target_w, target_h);

    let exposure = auto_exposure(&net_img);
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../proof/neural-live");
    write_png(&net_img, target_w, target_h, exposure, &dir.join(format!("net_{tag}.png")));
    write_png(&bilinear, target_w, target_h, exposure, &dir.join(format!("bilinear_{tag}.png")));

    // Honest stats.
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut nan = 0u64;
    for p in &net_img {
        for c in [p.x, p.y, p.z] {
            if !c.is_finite() {
                nan += 1;
            } else {
                mn = mn.min(c);
                mx = mx.max(c);
            }
        }
    }
    eprintln!(
        "[n0c] {tag}: low {low_w}×{low_h} → net {target_w}×{target_h} ({n} px) · gather {gather_ms:.3} ms · forward {fwd_ms:.3} ms · radiance[min {mn:.3}, max {mx:.3}] non-finite {nan} · exposure {exposure:.3}"
    );
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[n0c] no GPU adapter");
    };
    let weights = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/rdirect-weights-v1.bin"),
    )
    .expect("committed rdirect weights");

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

    // Trained shape (best-case) and a 4× larger near-native shape (rough).
    net_frame(&device, &queue, &weights, &bvh, &scene, &front, 48, 32, 96, 64, "trained_96x64");
    net_frame(&device, &queue, &weights, &bvh, &scene, &front, 240, 160, 480, 320, "native_480x320");
    let _ = Vec2::ZERO;
}
