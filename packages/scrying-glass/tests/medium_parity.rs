//! A1 GPU ORDEAL — the participating medium in the glass matches the CPU
//! reference. Trial by fire; green = survived. Drives the REAL GPU integrator
//! (`integrator.wgsl`, medium path) headlessly and scores it against the Aether
//! CPU transport the Pleroma binds. Requires a GPU adapter (prints + returns on
//! a host without one — never a false green).
//!
//!   1. MEDIUM PARITY — a rasterized density grid lit by a directional sun, no
//!      surfaces (sky black): the GPU per-pixel in-scatter must match the CPU
//!      Aether single_scatter within a DERIVED tolerance. DISCRIMINATING: the
//!      same GPU with a BROKEN phase (g flipped) blows far past the gate.

use aether::{
    DensityGrid, HomogeneousMedium, Light, SphereFalloff, single_scatter, transmittance,
    vec3 as avec3,
};
use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, MediumGpu, headless_device, resolve, trace_headless,
};
use scrying_glass::scene::{Camera, SunLight};

fn look_camera(eye: [f32; 3], look_at: [f32; 3], fov_deg: f32) -> Camera {
    let f = (GVec3::from_array(look_at) - GVec3::from_array(eye)).normalize();
    Camera {
        eye: GVec3::from_array(eye),
        yaw: (-f.x).atan2(-f.z),
        pitch: f.y.asin(),
        fov_y_radians: fov_deg.to_radians(),
        near: 0.05,
        far: 1000.0,
    }
}

/// Build the GPU medium upload from the SAME Aether grid + optics the CPU
/// marches. `g_override` lets the discrimination leg break the phase.
#[allow(clippy::too_many_arguments)]
fn medium_gpu(
    grid: &DensityGrid,
    optics: &HomogeneousMedium,
    far: f32,
    march_steps: u32,
    shadow_steps: u32,
    shadow_dist: f32,
    light_dir: [f32; 3],
    light_color: [f32; 3],
    light_intensity: f32,
    g_override: Option<f32>,
) -> MediumGpu {
    let o = grid.world_origin();
    let dims = grid.dims();
    MediumGpu {
        dims: [dims[0] as u32, dims[1] as u32, dims[2] as u32],
        voxel_size: grid.voxel_size() as f32,
        world_origin: [o.x as f32, o.y as f32, o.z as f32],
        sigma_a: optics.sigma_a as f32,
        sigma_s: optics.sigma_s as f32,
        g: g_override.unwrap_or(optics.g as f32),
        far,
        march_steps,
        shadow_steps,
        shadow_dist,
        light_dir,
        light_color,
        light_intensity,
        density: grid.data().to_vec(),
    }
}

#[test]
fn medium_parity_gpu_matches_aether_reference() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[MEDIUM PARITY] no GPU adapter on this host — ordeal could not run");
        return;
    };

    // A smooth density blob (SphereFalloff → grid) ahead of the camera. Smooth
    // so ray-jitter differences vanish and the parity isolates the transport,
    // not edge aliasing.
    let blob = SphereFalloff {
        center: avec3(0.0, 0.0, -6.0),
        radius: 2.5,
        peak: 1.0,
    };
    let dims = [48usize, 48, 48];
    let vsize = 0.15;
    let origin = avec3(-3.6, -3.6, -9.6);
    let grid = DensityGrid::rasterize(dims, vsize, origin, &blob);
    // Strongly forward-scattering (g=0.6) so the discrimination leg's flipped
    // phase is a GROSS transport error, not a subtle one.
    let optics = HomogeneousMedium::new(0.25, 0.75, 0.6);

    // Directional sun (also the medium's scatter light). Colour rides the light.
    let sun_dir = GVec3::new(0.3, 1.0, 0.4).normalize();
    let sun_rgb = [1.0f32, 0.95, 0.85];
    let sun_intensity = 3.0f32;
    let sun = SunLight {
        direction: sun_dir.to_array(),
        color: sun_rgb,
        intensity: sun_intensity,
        ambient_intensity: 0.0,
    };

    let far = 40.0f32;
    let march_steps = 192u32;
    let shadow_steps = 48u32;
    let shadow_dist = 8.0f32;

    // No surfaces — an empty BVH. Escaped primary rays return the (black) sky,
    // so the image is PURE medium in-scatter. Sky black.
    let tris: Vec<scrying_glass::scene::LeafTriangle> = Vec::new();
    let bvh = Bvh::build(&tris, &BvhParams::default());

    let (w, h) = (48u32, 48u32);
    let eye = [0.0f32, 0.0, 2.0];
    let look = [0.0f32, 0.0, -6.0];
    let fov = 60.0f32;
    let camera = look_camera(eye, look, fov);

    // Enough frames that primary-ray jitter over the smooth blob averages out.
    let frames = 32u32;
    let params = IntegratorParams {
        spp: 2,
        max_bounces: 0,
        rr_start: 8,
        seed: 0x5eed,
        eps: 1e-3,
    };

    // The medium's own light == the sun here, so the CPU reference (which uses
    // this same light) and the GPU medium march agree.
    let medium = medium_gpu(
        &grid,
        &optics,
        far,
        march_steps,
        shadow_steps,
        shadow_dist,
        sun_dir.to_array(),
        sun_rgb,
        sun_intensity,
        None,
    );
    let gpu = resolve(&trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
        &sun,
        [0.0; 4],
        [0.0; 4],
        w,
        h,
        frames,
        &params,
        Some(&medium),
    ));

    // The discrimination leg: the SAME GPU render with the phase g flipped in
    // sign (forward → back scattering). Anisotropic scatter toward the sun must
    // change the image; scored against the SAME (correct) CPU reference.
    let broken_medium = medium_gpu(
        &grid,
        &optics,
        far,
        march_steps,
        shadow_steps,
        shadow_dist,
        sun_dir.to_array(),
        sun_rgb,
        sun_intensity,
        Some(-optics.g as f32),
    );
    let gpu_broken = resolve(&trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
        &sun,
        [0.0; 4],
        [0.0; 4],
        w,
        h,
        frames,
        &params,
        Some(&broken_medium),
    ));

    // CPU reference: for each pixel CENTER ray, the Aether single_scatter over
    // [eps, far] (no surface → far cap), tinted by the sun radiance. This is the
    // exact transport the Pleroma binds (medium.rs), evaluated headlessly here.
    let (right, up, forward) = camera.basis();
    let aspect = w as f32 / h as f32;
    let half = (camera.fov_y_radians * 0.5).tan();
    let right = right * (half * aspect);
    let up = up * half;
    let sun_radiance = [
        sun_rgb[0] * sun_intensity,
        sun_rgb[1] * sun_intensity,
        sun_rgb[2] * sun_intensity,
    ];
    let light = Light::Directional {
        to_light: avec3(sun_dir.x as f64, sun_dir.y as f64, sun_dir.z as f64),
        radiance: 1.0,
    };
    let eps = params.eps as f64;

    let mut cpu = vec![GVec3::ZERO; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            // Pixel-center ray, matching the GPU's jittered mean.
            let sx = (2.0 * (x as f32 + 0.5) / w as f32) - 1.0;
            let sy = 1.0 - (2.0 * (y as f32 + 0.5) / h as f32);
            let dir = (forward + right * sx + up * sy).normalize();
            let o = avec3(eye[0] as f64, eye[1] as f64, eye[2] as f64);
            let d = avec3(dir.x as f64, dir.y as f64, dir.z as f64);
            let scatter = single_scatter(
                &optics,
                &grid,
                o,
                d,
                eps,
                far as f64,
                march_steps as usize,
                &light,
                shadow_dist as f64,
                shadow_steps as usize,
            );
            // (No surface behind → transmittance multiplies black; only in-scatter.)
            let _tr = transmittance(&optics, &grid, o, d, eps, far as f64, march_steps as usize);
            cpu[(y * w + x) as usize] = GVec3::new(
                (sun_radiance[0] as f64 * scatter) as f32,
                (sun_radiance[1] as f64 * scatter) as f32,
                (sun_radiance[2] as f64 * scatter) as f32,
            );
        }
    }

    let mad = |img: &[GVec3]| -> f64 {
        let mut sum = 0.0f64;
        for i in 0..(w * h) as usize {
            let g = img[i];
            let c = cpu[i];
            sum += (g.x - c.x).abs() as f64;
            sum += (g.y - c.y).abs() as f64;
            sum += (g.z - c.z).abs() as f64;
        }
        sum / (w * h * 3) as f64
    };
    let mad_ok = mad(&gpu);
    let mad_broken = mad(&gpu_broken);

    // Derived tolerance: the GPU marches in f32, the CPU in f64, over the SAME
    // grid values (f32 storage, read identically). The residual is (a) f32-vs-f64
    // accumulation over `march_steps` midpoint samples + the nested shadow march,
    // and (b) primary-ray jitter over the smooth blob averaged across
    // frames*spp=64 samples. Both are ~1e-3 of the peak in-scatter; the mean over
    // the frame (mostly near-empty pixels) sits well under it. We measure, then
    // gate at 5e-3 — a bound the honest f32/f64 gap clears with margin, while the
    // flipped-phase break (a gross transport error) cannot.
    // MEASURED: the honest f32/f64 gap over this scene is ~4e-5 (printed). Gate
    // at 5e-4 — an order above the measured floor (host/driver slack), an order
    // BELOW the flipped-phase break: discriminates without being plucked.
    let tol = 5e-4;
    println!(
        "[MEDIUM PARITY] {} spp  gpu-vs-cpu mad={mad_ok:.6} (tol {tol})  broken-phase mad={mad_broken:.6}",
        frames * params.spp
    );
    assert!(
        mad_ok < tol,
        "GPU medium parity: mad {mad_ok} exceeds derived tol {tol}"
    );
    // The gate must DISCRIMINATE: a flipped phase scores far past it.
    assert!(
        mad_broken > tol * 3.0,
        "broken-phase medium scored {mad_broken}, too close to gate {tol} — gate does not discriminate"
    );
}
