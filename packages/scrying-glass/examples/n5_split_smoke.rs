//! N5 smoke: prove the split shader validates on GPU and E+D == the ordinary
//! total radiance, per pixel (the load-bearing invariant — the teacher target
//! stays the converged total). Renders one naruko pose small at K frames both
//! ways and reports max |E+D - total| and the E/D energy share.

use std::path::Path;

use glam::Vec3 as GVec3;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, trace_headless, trace_headless_split,
};
use scrying_glass::scene::RenderScene;

fn lum(c: GVec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[n5-smoke] no GPU");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();
    let bvh = Bvh::build(&base_tris, &BvhParams::default());

    let all = scrying_glass::denoiser_dataset::law_poses(&params);
    let cam = all.iter().find(|(n, _)| *n == "front").unwrap().1.clone();
    let (w, h) = (240u32, 180u32);
    let frames = 8u32;
    let p = IntegratorParams { spp: 1, seed: 0x5eed, ..IntegratorParams::default() };

    let total = resolve(&trace_headless(
        &device, &queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, frames, &p, None,
    ));
    let (e, d) = trace_headless_split(
        &device, &queue, &bvh, &cam, &scene.sun, scene.sky_top, scene.sky_horizon, w, h, frames, &p,
    );

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_e = 0.0f64;
    let mut sum_d = 0.0f64;
    for i in 0..total.len() {
        let s = e[i] + d[i];
        let dv = (s - total[i]).abs();
        let a = dv.x.max(dv.y).max(dv.z);
        max_abs = max_abs.max(a);
        sum_abs += a as f64;
        sum_e += lum(e[i]) as f64;
        sum_d += lum(d[i]) as f64;
    }
    let mean_abs = sum_abs / total.len() as f64;
    let share_d = sum_d / (sum_e + sum_d).max(1e-9);
    println!(
        "[n5-smoke] {}x{} K={frames}: max|E+D-total|={max_abs:.3e} mean|.|={mean_abs:.3e}  E_lum={sum_e:.1} D_lum={sum_d:.1}  D_share={:.2}%",
        w, h, 100.0 * share_d
    );
    // E+D reconstructs the total to a per-sample FP reassociation of the same
    // terms (bright HDR firefly pixels carry the largest abs slack). The teacher
    // target is rendered INDEPENDENTLY (trace_headless, 96 frames) so this is a
    // sanity check, not a training dependency.
    assert!(mean_abs < 1e-3, "E+D drifts from total in the mean: {mean_abs}");
    assert!(max_abs < 0.2, "E+D max drift too large: {max_abs}");
    println!("[n5-smoke] SPLIT OK (E dominates=direct neon, D=sparse indirect firefly source).");
}
