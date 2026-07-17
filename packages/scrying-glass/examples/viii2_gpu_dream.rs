//! RITE VIII-2 — THE DREAM AT SPEED: the proof. Loads the COMMITTED
//! hash-pinned weights (never retrains), renders the naruko `front` (train)
//! and `orbit_-20` (held-out validation) poses at the dataset resolution the
//! pinned bound was measured at, denoises each noisy 1-spp frame on the GPU
//! (`denoiser.wgsl` via `GpuDenoiser`), and:
//!
//!   - writes a before/after pair per pose: proof/viii2-gpu-front.png and
//!     proof/viii2-gpu-heldout.png, each = noisy | GPU-denoised side by side.
//!   - prints, honestly, RMSE(noisy,ref) / RMSE(gpu-denoised,ref) /
//!     RMSE(cpu-ref-denoised,ref) and the GPU-vs-CPU relative parity, plus
//!     the measured GPU denoise-pass cost in ms (timestamp queries) against
//!     the 60-fps frame budget headroom.
//!
//! Run:  cargo run -p scrying-glass --release --example viii2_gpu_dream

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::{Mlp, denoise_image, deserialize_weights};
use scrying_glass::denoiser_dataset::{
    DATASET_HEIGHT, DATASET_REF_FRAMES, DATASET_WIDTH, law_poses, naruko_params,
};
use scrying_glass::denoiser_gpu::{GpuDenoiser, headless_device_timed};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, RenderScene};

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn write_pair(a: &[GVec3], b: &[GVec3], w: u32, h: u32, exposure: f32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((2 * w * h * 3) as usize);
    for y in 0..h {
        for panel in [a, b] {
            for x in 0..w {
                let px = panel[(y * w + x) as usize];
                bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
                bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
                bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
            }
        }
    }
    let file = std::fs::File::create(path).unwrap();
    let writer = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(writer, 2 * w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("[viii2-gpu] wrote {}", path.display());
}

#[allow(clippy::too_many_arguments)]
fn dream_one_pose(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &Camera,
    scene: &RenderScene,
    cpu_mlp: &Mlp,
    gpu: &GpuDenoiser,
    role: &str,
    pose_name: &str,
    w: u32,
    h: u32,
    exposure: f32,
    out_path: &Path,
) {
    let noisy_params = IntegratorParams { spp: 1, ..IntegratorParams::default() };
    let noisy = resolve(&trace_headless(
        device, queue, bvh, camera, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, 1,
        &noisy_params, None,
    ));
    let reference = resolve(&trace_headless(
        device, queue, bvh, camera, &scene.sun, scene.sky_top, scene.sky_horizon, w, h,
        DATASET_REF_FRAMES, &IntegratorParams::default(), None,
    ));
    let raw_aov =
        trace_headless_aov(device, queue, bvh, camera, &scene.sun, scene.sky_top, scene.sky_horizon, w, h);
    let (albedo, normal, depth) = split_aov(&raw_aov);

    let cpu_denoised = denoise_image(cpu_mlp, &noisy, &albedo, &normal, &depth);
    let gpu_denoised = gpu.denoise(device, queue, &noisy, &albedo, &normal, &depth, w, h);

    let noisy_rmse = rmse(&noisy, &reference);
    let gpu_rmse = rmse(&gpu_denoised, &reference);
    let cpu_rmse = rmse(&cpu_denoised, &reference);
    let parity_abs = rmse(&gpu_denoised, &cpu_denoised);
    let magnitude = rmse(&cpu_denoised, &vec![GVec3::ZERO; cpu_denoised.len()]).max(1e-12);
    let parity_rel = parity_abs / magnitude;

    println!("[viii2-gpu] pose='{pose_name}' ({role})");
    println!("[viii2-gpu]   RMSE(noisy,    ref) = {noisy_rmse:.6}");
    println!("[viii2-gpu]   RMSE(gpu-den,  ref) = {gpu_rmse:.6}   beats noisy: {}",
        if gpu_rmse < noisy_rmse { "YES" } else { "NO" });
    println!("[viii2-gpu]   RMSE(cpu-den,  ref) = {cpu_rmse:.6}");
    println!("[viii2-gpu]   parity GPU-vs-CPU: abs={parity_abs:.3e} rel={parity_rel:.3e}");

    // GPU pass cost (timestamp queries; falls back to a note if unsupported).
    match gpu.time_dispatches_ms(device, queue, &noisy, &albedo, &normal, &depth, w, h, 32) {
        Some(mut ms) => {
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = ms[ms.len() / 2];
            let min = ms[0];
            println!("[viii2-gpu]   GPU denoise pass: min={min:.4} ms median={median:.4} ms (32 dispatches, {w}x{h})");
        }
        None => println!("[viii2-gpu]   GPU denoise pass: TIMESTAMP_QUERY unsupported on this device — ms UNMEASURED"),
    }

    write_pair(&noisy, &gpu_denoised, w, h, exposure, out_path);
}

fn main() {
    let Some((device, queue)) = headless_device_timed() else {
        panic!("[viii2-gpu] no GPU adapter on this host — cannot forge the dream at speed");
    };
    let has_ts = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    eprintln!("[viii2-gpu] device TIMESTAMP_QUERY = {has_ts}");

    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());

    let poses = law_poses(&params);
    let front = &poses.iter().find(|(n, _)| *n == "front").expect("front").1;
    let heldout = &poses.iter().find(|(n, _)| *n == "orbit_-20").expect("orbit_-20").1;

    let (w, h) = (DATASET_WIDTH, DATASET_HEIGHT);
    let exposure = 1.6;
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");

    let weights_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/denoiser-weights-v1.bin");
    let bytes = std::fs::read(&weights_path)
        .unwrap_or_else(|e| panic!("[viii2-gpu] read {} failed: {e}", weights_path.display()));
    let cpu_mlp = deserialize_weights(&bytes).expect("deserialize committed weights");
    let gpu = GpuDenoiser::new(&device, &cpu_mlp);

    println!("[viii2-gpu] 60fps budget headroom (measured, HANDOFF): front 11.26 ms / wide 13.23 ms");
    dream_one_pose(&device, &queue, &bvh, front, &scene, &cpu_mlp, &gpu,
        "TRAIN-split pose - reconstruction", "front", w, h, exposure,
        &proof.join("viii2-gpu-front.png"));
    dream_one_pose(&device, &queue, &bvh, heldout, &scene, &cpu_mlp, &gpu,
        "VALIDATION-split pose, held out - generalization", "orbit_-20", w, h, exposure,
        &proof.join("viii2-gpu-heldout.png"));

    // Present-resolution pass cost: the perf audit's frame budget (front
    // 11.26 ms) is measured at 900x600 (perf_audit GAIA_AUDIT_W/H defaults),
    // so time the denoise pass at that SAME resolution for an honest
    // budget comparison. The denoiser is per-pixel independent, so this is
    // the real added cost a 900x600 present frame would pay.
    {
        let (pw, ph) = (900u32, 600u32);
        let noisy_params = IntegratorParams { spp: 1, ..IntegratorParams::default() };
        let noisy = resolve(&trace_headless(
            &device, &queue, &bvh, front, &scene.sun, scene.sky_top, scene.sky_horizon, pw, ph, 1,
            &noisy_params, None,
        ));
        let raw_aov = trace_headless_aov(
            &device, &queue, &bvh, front, &scene.sun, scene.sky_top, scene.sky_horizon, pw, ph);
        let (albedo, normal, depth) = split_aov(&raw_aov);
        match gpu.time_dispatches_ms(&device, &queue, &noisy, &albedo, &normal, &depth, pw, ph, 32) {
            Some(mut ms) => {
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = ms[ms.len() / 2];
                let min = ms[0];
                let budget = 1000.0 / 60.0;
                println!("[viii2-gpu] PRESENT-RES denoise pass @ {pw}x{ph}: min={min:.4} ms median={median:.4} ms");
                println!("[viii2-gpu]   vs 60fps frame budget 16.67 ms (front headroom 11.26 ms measured): pass is {:.1}% of the 16.67 ms budget, {:.1}% of the 11.26 ms front headroom",
                    100.0 * median / budget, 100.0 * median / 11.26);
            }
            None => println!("[viii2-gpu] PRESENT-RES @ {pw}x{ph}: TIMESTAMP_QUERY unsupported — ms UNMEASURED"),
        }
    }

    eprintln!("[viii2-gpu] noisy | GPU-denoised pairs written.");
}
