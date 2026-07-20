//! CLAMP GAMMA SWEEP PROBE (ghoul, 2026-07-20). Smallest possible probe that
//! reuses the real-image ordeal's OWN sparkle/resid code path
//! (`sparkle_resid_per_mpx`, duplicated verbatim — the ordeal binary itself
//! is never touched, never run against the sealed weights, never allowed to
//! write/delete `data/rdirect-weights-v7.bin.stamp`) to sweep
//! `GAIA_V7_CLAMP_GAMMA` OFFSCREEN, own port, zero windows, without
//! re-tracing the scene per gamma value: the expensive GPU renders (still
//! sequence taps + converged teacher) happen ONCE per pose; only the cheap
//! CPU-side recurrent MLP + clamp + metrics are re-run per gamma, reusing
//! the traced buffers. `evidence_clamp_gamma()` is read fresh by
//! `direct_render_sequence_hist_split` on every call, so setting
//! `GAIA_V7_CLAMP_GAMMA` before each `sequence_render` call is sufficient
//! and correct.
//!
//! Protocol matches the sealed v7 stamp exactly: GAIA_ORDEAL_WEIGHTS=v7,
//! 640x480, still_len=10, ref_still=96, poses orbit_-20 + orbit_+40
//! (averaged, same as the ordeal). This writes NO stamp file — it never
//! calls `stamp_path_for`/`stamp_pass_text`, so the live sealed stamp for
//! v7 is untouched regardless of what any non-default gamma scores here.
//!
//! Run: cargo run -p scrying-glass --release --example gamma_sweep_probe

use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
    trace_headless_split,
};
use scrying_glass::rdirect::{
    CamPose, HistFrameSplit, deserialize_weights, direct_render_sequence_hist_split,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

const TARGET_W: u32 = 640;
const TARGET_H: u32 = 480;
const STILL_LEN: u32 = 10;
const REF_STILL: u32 = 96;
const SPARK_DELTA: f32 = 0.15;
const EXPOSURE: f32 = 1.6;
const GAMMAS: [f32; 4] = [1.5, 1.25, 1.0, 0.85];
const BRIGHT_PCTL: f32 = 0.99; // top 1% brightest teacher px = "known bright spots"

fn lum(c: GVec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

fn cam_pose(cam: &Camera, w: u32, h: u32) -> CamPose {
    let (right, up, forward) = cam.basis();
    CamPose { eye: cam.eye, right, up, forward, half_tan: (cam.fov_y_radians * 0.5).tan(), aspect: w as f32 / h as f32 }
}

struct FrameBufs {
    low_e: Vec<GVec3>,
    low_d: Vec<GVec3>,
    albedo: Vec<GVec3>,
    normal: Vec<GVec3>,
    depth: Vec<f32>,
    cam: CamPose,
}

fn render_frame(
    device: &wgpu::Device, queue: &wgpu::Queue, base_tris: &[LeafTriangle], scene: &RenderScene,
    cam: &Camera, seed: u32, low_w: u32, low_h: u32, tw: u32, th: u32,
) -> FrameBufs {
    let bvh = Bvh::build(base_tris, &BvhParams::default());
    let np = IntegratorParams { spp: 1, seed, ..IntegratorParams::default() };
    let (low_e, low_d) = trace_headless_split(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np,
    );
    let (albedo, normal, depth) = split_aov(&trace_headless_aov(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th,
    ));
    FrameBufs { low_e, low_d, albedo, normal, depth, cam: cam_pose(cam, tw, th) }
}

fn render_teacher(
    device: &wgpu::Device, queue: &wgpu::Queue, base_tris: &[LeafTriangle], scene: &RenderScene,
    cam: &Camera, ref_frames: u32, tw: u32, th: u32,
) -> Vec<GVec3> {
    let bvh = Bvh::build(base_tris, &BvhParams::default());
    resolve(&trace_headless(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, ref_frames,
        &IntegratorParams::default(), None,
    ))
}

/// Verbatim copy of `examples/real_image_ordeal.rs::sparkle_resid_per_mpx`
/// (that file is never edited/executed against sealed weights by this
/// probe) — same definition the ordeal's SPARKLE_BAR judges by.
fn sparkle_resid_per_mpx(net: &[GVec3], teacher: &[GVec3], w: u32, h: u32) -> f64 {
    let idx = |x: i32, y: i32| (y as usize) * w as usize + x as usize;
    let err = |x: i32, y: i32| lum(net[idx(x, y)]) - lum(teacher[idx(x, y)]);
    let mut count = 0u64;
    for y in 1..h as i32 - 1 {
        for x in 1..w as i32 - 1 {
            let e = err(x, y);
            if e <= SPARK_DELTA {
                continue;
            }
            let mut is_peak = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    if err(x + dx, y + dy) >= e {
                        is_peak = false;
                    }
                }
            }
            if is_peak {
                count += 1;
            }
        }
    }
    (count as f64) * 1.0e6 / (w as f64 * h as f64)
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn write_panel(panel: &[GVec3], w: u32, h: u32, exposure: f32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in panel {
        bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
    }
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("[gamma-sweep] wrote {}", path.display());
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[gamma-sweep] no GPU adapter");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();

    let low_w = TARGET_W / 2;
    let low_h = TARGET_H / 2;

    let wrel = "data/rdirect-weights-v7.bin";
    let wpath = Path::new(env!("CARGO_MANIFEST_DIR")).join(wrel);
    let mlp = deserialize_weights(&std::fs::read(&wpath).expect("read v7 weights")).expect("parse weights");

    let val_poses = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| val_poses.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();
    let poses = [("orbit_-20", find("orbit_-20")), ("orbit_+40", find("orbit_+40"))];

    let t0 = std::time::Instant::now();

    // ── ONE trace pass per pose (gamma-independent): still taps + teacher ──
    struct PoseData { bufs: Vec<FrameBufs>, teacher: Vec<GVec3> }
    let mut pose_data: Vec<PoseData> = Vec::new();
    for (pname, cam) in &poses {
        let bufs: Vec<FrameBufs> = (0..STILL_LEN)
            .map(|k| render_frame(&device, &queue, &base_tris, &scene, cam, 0x5eed + k * 101 + 7, low_w, low_h, TARGET_W, TARGET_H))
            .collect();
        let teacher = render_teacher(&device, &queue, &base_tris, &scene, cam, REF_STILL, TARGET_W, TARGET_H);
        eprintln!("[gamma-sweep] traced pose {pname} ({:.1}s elapsed)", t0.elapsed().as_secs_f64());
        pose_data.push(PoseData { bufs, teacher });
    }
    eprintln!("[gamma-sweep] all GPU tracing done at {:.1}s, sweeping {} gammas over cached buffers", t0.elapsed().as_secs_f64(), GAMMAS.len());

    // bright mask from pose 0's teacher (top 1% luminance = "known bright
    // spots": lighthouse lamp / lit windows in this scene's still framing).
    let t0_teacher = &pose_data[0].teacher;
    let mut lums: Vec<f32> = t0_teacher.iter().map(|&c| lum(c)).collect();
    let mut sorted = lums.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let thresh = sorted[((sorted.len() - 1) as f32 * BRIGHT_PCTL) as usize];
    let bright_mask: Vec<bool> = lums.iter().map(|&l| l >= thresh).collect();
    let n_bright = bright_mask.iter().filter(|&&b| b).count();
    lums.clear();
    let teacher_bright_mean: f32 = {
        let s: f64 = (0..t0_teacher.len()).filter(|&i| bright_mask[i]).map(|i| lum(t0_teacher[i]) as f64).sum();
        (s / n_bright as f64) as f32
    };
    eprintln!("[gamma-sweep] bright mask: {n_bright} px (top {:.0}% by teacher lum, thresh={thresh:.3}), teacher_bright_mean={teacher_bright_mean:.4}", (1.0 - BRIGHT_PCTL) * 100.0);

    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("proof/neural-live");
    println!("\ngamma,resid_still,sparkle_still,highlight_patch_mean,highlight_delta_vs_teacher_pct");

    let mut rows: Vec<(f32, f64, f64, f32, f32)> = Vec::new();
    for &gamma in &GAMMAS {
        // SAFETY: single-threaded example, no other thread reads/writes env.
        unsafe { std::env::set_var("GAIA_V7_CLAMP_GAMMA", format!("{gamma}")) };

        let mut resid_still = 0.0f64;
        let mut sparkle_still = 0.0f64;
        let mut shot: Option<Vec<GVec3>> = None;
        let mut highlight_mean_acc = 0.0f64;
        let mut highlight_pose_n = 0usize;

        for (pi, pd) in pose_data.iter().enumerate() {
            let hf: Vec<HistFrameSplit> = pd.bufs.iter().map(|b| HistFrameSplit {
                low_e: &b.low_e, low_d: &b.low_d, low_w, low_h,
                hi_albedo: &b.albedo, hi_normal: &b.normal, hi_depth: &b.depth,
                target_w: TARGET_W, target_h: TARGET_H, cam: b.cam,
            }).collect();
            let outs = direct_render_sequence_hist_split(&mlp, &hf, 0.05, 0.85);
            let settled = outs.last().unwrap();
            let r = rmse(settled, &pd.teacher) as f64;
            let sp = sparkle_resid_per_mpx(settled, &pd.teacher, TARGET_W, TARGET_H);
            resid_still += r;
            sparkle_still += sp;
            if pi == 0 {
                shot = Some(settled.clone());
                let s: f64 = (0..settled.len()).filter(|&i| bright_mask[i]).map(|i| lum(settled[i]) as f64).sum();
                highlight_mean_acc += s / n_bright as f64;
                highlight_pose_n += 1;
            }
        }
        let np = pose_data.len() as f64;
        resid_still /= np;
        sparkle_still /= np;
        let highlight_mean = (highlight_mean_acc / highlight_pose_n as f64) as f32;
        let highlight_delta_pct = (highlight_mean - teacher_bright_mean) / teacher_bright_mean * 100.0;

        if let Some(net) = &shot {
            write_panel(net, TARGET_W, TARGET_H, EXPOSURE, &proof.join(format!("gamma-sweep-{gamma}.png")));
        }

        println!("{gamma},{resid_still:.5},{sparkle_still:.2},{highlight_mean:.4},{highlight_delta_pct:+.2}");
        eprintln!("[gamma-sweep] gamma={gamma} resid_still={resid_still:.5} sparkle_still={sparkle_still:.2}/Mpx highlight_patch_mean={highlight_mean:.4} (teacher {teacher_bright_mean:.4}, delta {highlight_delta_pct:+.2}%) [{:.1}s elapsed]", t0.elapsed().as_secs_f64());
        rows.push((gamma, resid_still, sparkle_still, highlight_mean, highlight_delta_pct));
    }

    println!("\n[gamma-sweep] done in {:.1}s. rows:", t0.elapsed().as_secs_f64());
    for (g, r, s, hm, hd) in &rows {
        println!("  gamma={g:<5} resid={r:.5} sparkle={s:>6.2}/Mpx highlight_mean={hm:.4} highlight_delta={hd:+.2}%");
    }
}
