//! THE ONE RENDER PATH — own-eyes proof. Renders one naruko vista and writes a
//! side-by-side: LEFT = the OLD runtime resolve (bilinear upscale of the noisy
//! 1-spp 640×480 trace); RIGHT = the chartered neural path (GPU denoise at
//! 640×480 → GPU neural upscale → present). Same trace, same AOVs — only the
//! resolve differs. Grain should be visibly reduced on the right, structure
//! intact. Written to proof/onepath-before-after.png (read by own eyes).
//!
//! Run: cargo run -p scrying-glass --release --example onepath_proof

use std::path::Path;

use glam::Vec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::deserialize_weights as denoiser_weights;
use scrying_glass::denoiser_gpu::{GpuDenoiser, headless_device_timed};
use scrying_glass::integrator::{
    IntegratorParams, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, RenderScene};
use scrying_glass::upscaler::{bilinear_upsample, deserialize_weights as upscaler_weights};
use scrying_glass::upscaler_dataset::naruko_params;
use scrying_glass::upscaler_gpu::GpuUpscaler;

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn write_pair(a: &[Vec3], b: &[Vec3], w: u32, h: u32, exposure: f32, path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let gap = 8u32;
    let out_w = 2 * w + gap;
    let mut bytes = vec![0u8; (out_w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let pa = a[(y * w + x) as usize];
            let pb = b[(y * w + x) as usize];
            for (panel_x, px) in [(x, pa), (x + w + gap, pb)] {
                let o = ((y * out_w + panel_x) * 3) as usize;
                bytes[o] = (linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8;
                bytes[o + 1] = (linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8;
                bytes[o + 2] = (linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8;
            }
        }
    }
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), out_w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .unwrap()
        .write_image_data(&bytes)
        .unwrap();
}

fn main() {
    let Some((device, queue)) = headless_device_timed() else {
        eprintln!("no GPU adapter");
        return;
    };
    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());

    // Viewable size (fast): 480×360 low → ×2 → 960×720 target.
    let (low_w, low_h) = (480u32, 360u32);
    let (tw, th) = (low_w * 2, low_h * 2);
    let camera = Camera {
        eye: Vec3::new(0.0, 1.6, 6.0),
        yaw: 0.0,
        pitch: 0.0,
        fov_y_radians: 55f32.to_radians(),
        near: 0.05,
        far: 1000.0,
    };

    let noisy_params = IntegratorParams {
        spp: 1,
        ..IntegratorParams::default()
    };
    let low_noisy = resolve(&trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        low_w,
        low_h,
        1,
        &noisy_params,
        None,
    ));
    let low_aov = trace_headless_aov(
        &device,
        &queue,
        &bvh,
        &camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        low_w,
        low_h,
    );
    let (low_alb, low_nrm, low_dep) = split_aov(&low_aov);
    let hi_aov = trace_headless_aov(
        &device,
        &queue,
        &bvh,
        &camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        tw,
        th,
    );
    let (hi_alb, hi_nrm, hi_dep) = split_aov(&hi_aov);

    // BEFORE: old runtime resolve = bilinear of the noisy low trace.
    let before = bilinear_upsample(&low_noisy, low_w, low_h, tw, th);

    // AFTER: chartered path = GPU denoise(low) → GPU neural upscale.
    let denoiser = GpuDenoiser::new(
        &device,
        &denoiser_weights(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("data/denoiser-weights-v1.bin"),
            )
            .unwrap(),
        )
        .unwrap(),
    );
    let upscaler = GpuUpscaler::new(
        &device,
        &upscaler_weights(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("data/upscaler-weights-v1.bin"),
            )
            .unwrap(),
        )
        .unwrap(),
    );
    let denoised_low = denoiser.denoise(
        &device, &queue, &low_noisy, &low_alb, &low_nrm, &low_dep, low_w, low_h,
    );
    let after = upscaler.upscale(
        &device,
        &queue,
        &denoised_low,
        low_w,
        low_h,
        &hi_alb,
        &hi_nrm,
        &hi_dep,
        tw,
        th,
    );

    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof/onepath-before-after.png");
    write_pair(&before, &after, tw, th, 1.6, &out);
    println!("wrote {} ({}x{} each panel)", out.display(), tw, th);
}
