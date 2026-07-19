//! THE REAL-IMAGE ORDEAL (Architect, 2026-07-18: "REAL OR BLACK").
//!
//! THE REAL IMAGE BAR: the app presents a neural frame ONLY when the shipped
//! weights pass this ordeal. The bar models HIS eye — zero visible sparkle at
//! stillness, no ghost trails under motion, and an image that is genuinely the
//! converged teacher (not a smeared guess). It runs the RECURRENT N2 net over
//! f(seed) validation poses, STILL and PAN, and measures three IRON quantities:
//!
//!   1. resid_still  — RMSE(linear) of the settled still frame vs the converged
//!                     teacher. "Is it the real image?"  ≤ RESID_BAR.
//!   2. sparkle_still — isolated-bright-pixel count per megapixel on the settled
//!                     still frame (a pixel whose luminance jumps past its 3×3
//!                     neighbourhood median by SPARK_DELTA). "The dots." ≤ SPARKLE_BAR.
//!   3. tvar_still   — mean per-pixel temporal variance of luminance across the
//!                     settled tail frames. A converged running mean → ~0.
//!                     "Does it stop shimmering when he holds still?" ≤ TVAR_BAR.
//!   And under motion:
//!   4. resid_move   — RMSE of a mid-pan frame vs ITS OWN teacher. ≤ RESID_MOVE_BAR.
//!   5. ghost_excess — resid_move(history ON) − resid_move(history OFF). History
//!                     must not smear motion. ≤ GHOST_BAR.
//!
//! PASS ⇔ all five under bar. On PASS it writes the sidecar stamp beside the
//! weights (`<weights>.stamp`) that `main.rs`'s rig-build gate verifies; on FAIL
//! it writes NO stamp (or removes a stale one) and prints the exact residual
//! distance to each bar. The thresholds below are IRON — do NOT soften them to
//! make a net pass; a failing net earns honest BLACK.
//!
//! Run: cargo run -p scrying-glass --release --example real_image_ordeal
//!   GAIA_ORDEAL_WEIGHTS=v2|v3|<path>   (default v3)

use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::{Vec2, Vec3 as GVec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::rdirect::{
    CamPose, HistFrame, INPUT_FEATURES, Mlp, deserialize_weights, direct_render_sequence_hist,
    hist_features, pixel_features, stamp_pass_text, stamp_path_for,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

// ── IRON BARS (do not soften — the bar models HIS eye) ──────────────────────
const RESID_BAR: f64 = 0.035; // settled still RMSE vs teacher (v2 held-out ~.031-.035)
const SPARKLE_BAR: f64 = 40.0; // isolated bright px / megapixel at stillness
const SPARK_DELTA: f32 = 0.15; // linear luminance jump over 3×3 median = a "dot"
const TVAR_BAR: f64 = 5.0e-4; // mean temporal luminance variance over settled tail
const RESID_MOVE_BAR: f64 = 0.060; // mid-pan RMSE vs per-frame teacher
const GHOST_BAR: f64 = 0.012; // history must not raise motion RMSE beyond this

const DEPTH_TOL: f32 = 0.05; // reprojection depth guard (light-fix temporal.y band)
const NORMAL_THRESH: f32 = 0.85; // reprojection normal guard (light-fix temporal.z)

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

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

struct FrameBufs {
    low: Vec<GVec3>,
    albedo: Vec<GVec3>,
    normal: Vec<GVec3>,
    depth: Vec<f32>,
    cam: CamPose,
}

#[allow(clippy::too_many_arguments)]
fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_tris: &[LeafTriangle],
    scene: &RenderScene,
    cam: &Camera,
    seed: u32,
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
) -> FrameBufs {
    let bvh = Bvh::build(base_tris, &BvhParams::default());
    let np = IntegratorParams { spp: 1, seed, ..IntegratorParams::default() };
    let low = resolve(&trace_headless(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1,
        &np, None,
    ));
    let (albedo, normal, depth) = split_aov(&trace_headless_aov(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
    ));
    FrameBufs { low, albedo, normal, depth, cam: cam_pose(cam, target_w, target_h) }
}

fn render_teacher(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_tris: &[LeafTriangle],
    scene: &RenderScene,
    cam: &Camera,
    ref_frames: u32,
    target_w: u32,
    target_h: u32,
) -> Vec<GVec3> {
    let bvh = Bvh::build(base_tris, &BvhParams::default());
    resolve(&trace_headless(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
        ref_frames, &IntegratorParams::default(), None,
    ))
}

fn lum(c: GVec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// Isolated INVENTED bright dots per megapixel: pixels where the net's
/// luminance exceeds the CONVERGED TEACHER's by more than SPARK_DELTA AND that
/// excess is a strict local maximum of the signed error over the 3×3
/// neighbourhood — i.e. a firefly the net hallucinated that is NOT in the real
/// image. Measured against the teacher (not the image's own texture), so a
/// converged surface with real high-frequency detail scores ZERO and
/// teacher-vs-teacher is exactly 0. This is the dot the Architect sees.
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

/// Mean per-pixel temporal variance of luminance across a set of frames.
fn temporal_variance(frames: &[Vec<GVec3>]) -> f64 {
    if frames.len() < 2 {
        return 0.0;
    }
    let n = frames[0].len();
    let m = frames.len() as f64;
    let mut acc = 0.0f64;
    for i in 0..n {
        let mut s = 0.0f64;
        let mut s2 = 0.0f64;
        for f in frames {
            let l = lum(f[i]) as f64;
            s += l;
            s2 += l * l;
        }
        let mean = s / m;
        acc += (s2 / m - mean * mean).max(0.0);
    }
    acc / n as f64
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
    eprintln!("[ordeal] wrote {}", path.display());
}

/// Single-frame (history OFF) render — the no-memory baseline for ghost_excess.
fn render_single(mlp: &Mlp, f: &FrameBufs, tw: u32, th: u32, low_w: u32, low_h: u32) -> Vec<GVec3> {
    let motion = vec![Vec2::ZERO; (tw * th) as usize];
    let n = (tw * th) as usize;
    let mut out = vec![GVec3::ZERO; n];
    let uses_hist = mlp.layer_dims()[0].0 as usize == INPUT_FEATURES + 4;
    for ty in 0..th {
        for tx in 0..tw {
            let i = (ty * tw + tx) as usize;
            let albedo = f.albedo[i];
            let base = pixel_features(
                &f.low, low_w, low_h, tw, th, tx, ty, albedo, f.normal[i], f.depth[i], motion[i],
            );
            let dl = if uses_hist {
                mlp.forward(&hist_features(&base, [0.0; 3], 0.0))
            } else {
                mlp.forward(&base)
            };
            // undo log-demod
            let div = if albedo.length_squared() > 1e-8 { albedo + GVec3::splat(1e-3) } else { GVec3::ONE };
            let expm1 = GVec3::new(dl[0].exp() - 1.0, dl[1].exp() - 1.0, dl[2].exp() - 1.0);
            out[i] = GVec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0)) * div;
        }
    }
    out
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[ordeal] no GPU adapter");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();

    let target_w = env_u32("GAIA_ORDEAL_W", 480);
    let target_h = env_u32("GAIA_ORDEAL_H", 360);
    let low_w = target_w / 2;
    let low_h = target_h / 2;
    let still_len = env_u32("GAIA_ORDEAL_STILL", 10);
    let tail = env_u32("GAIA_ORDEAL_TAIL", 4).min(still_len);
    let pan_len = env_u32("GAIA_ORDEAL_PAN", 6);
    let pan_step = env_f32("GAIA_ORDEAL_PANSTEP", 0.004); // rad/frame ≈ 0.23°, a slow pan
    let ref_still = env_u32("GAIA_ORDEAL_REF_STILL", 96);
    let ref_move = env_u32("GAIA_ORDEAL_REF_MOVE", 48);
    let exposure = env_f32("GAIA_RDIRECT_EXPOSURE", 1.6);

    // weights selection + path
    let sel = std::env::var("GAIA_ORDEAL_WEIGHTS").unwrap_or_else(|_| "v3".to_string());
    let wrel = match sel.as_str() {
        "v1" => "data/rdirect-weights-v1.bin".to_string(),
        "v2" => "data/rdirect-weights-v2.bin".to_string(),
        "v3" => "data/rdirect-weights-v3.bin".to_string(),
        "v4" => "data/rdirect-weights-v4.bin".to_string(),
        "v5" => "data/rdirect-weights-v5.bin".to_string(),
        other => other.to_string(),
    };
    let wpath: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join(&wrel);
    let wbytes = match std::fs::read(&wpath) {
        Ok(b) => b,
        Err(e) => {
            println!("[ordeal] FAIL: cannot read weights {wrel}: {e} — no stamp, present BLACK.");
            std::process::exit(2);
        }
    };
    let mlp = deserialize_weights(&wbytes).expect("weights parse");
    let in_dim = mlp.layer_dims()[0].0 as usize;
    println!(
        "[ordeal] weights={wrel} in_dim={in_dim} ({}) res {target_w}x{target_h} still={still_len} pan={pan_len}",
        if in_dim == INPUT_FEATURES + 4 { "N2 recurrent (27)" } else { "v2 current-frame (23) — no memory" }
    );

    let val_poses = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| val_poses.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();
    let poses = [("orbit_-20", find("orbit_-20")), ("orbit_+40", find("orbit_+40"))];

    let t0 = Instant::now();
    let mut resid_still = 0.0f64;
    let mut sparkle_still = 0.0f64;
    let mut sparkle_teacher = 0.0f64;
    let mut tvar_still = 0.0f64;
    let mut resid_move = 0.0f64;
    let mut ghost_excess = 0.0f64;
    let mut shot_still: Option<(Vec<GVec3>, Vec<GVec3>)> = None; // (net, teacher)
    let mut shot_move: Option<(Vec<GVec3>, Vec<GVec3>)> = None;

    for (pname, cam) in &poses {
        // ── STILL: same camera, fresh seed each frame ──────────────────────
        let still_bufs: Vec<FrameBufs> = (0..still_len)
            .map(|k| render_frame(&device, &queue, &base_tris, &scene, cam, 0x5eed + k * 101 + 7, low_w, low_h, target_w, target_h))
            .collect();
        let teacher_still = render_teacher(&device, &queue, &base_tris, &scene, cam, ref_still, target_w, target_h);
        let hist_frames: Vec<HistFrame> = still_bufs
            .iter()
            .map(|b| HistFrame {
                low_radiance: &b.low, low_w, low_h,
                hi_albedo: &b.albedo, hi_normal: &b.normal, hi_depth: &b.depth,
                target_w, target_h, cam: b.cam,
            })
            .collect();
        let outs = direct_render_sequence_hist(&mlp, &hist_frames, DEPTH_TOL, NORMAL_THRESH);
        let settled = outs.last().unwrap();
        let r = rmse(settled, &teacher_still) as f64;
        let sp = sparkle_resid_per_mpx(settled, &teacher_still, target_w, target_h);
        let spt = sparkle_resid_per_mpx(&teacher_still, &teacher_still, target_w, target_h);
        let tv = temporal_variance(&outs[(still_len - tail) as usize..]);
        println!("[ordeal] {pname} STILL: resid={r:.5} sparkle={sp:.1}/Mpx (teacher {spt:.1}) tvar={tv:.3e}");
        resid_still += r;
        sparkle_still += sp;
        sparkle_teacher += spt;
        tvar_still += tv;
        if shot_still.is_none() {
            shot_still = Some((settled.clone(), teacher_still.clone()));
        }

        // ── PAN: yaw drifts each frame; per-frame teacher ─────────────────
        let mut pan_cams = Vec::new();
        for k in 0..pan_len {
            let mut c = cam.clone();
            c.yaw += pan_step * k as f32;
            pan_cams.push(c);
        }
        let pan_bufs: Vec<FrameBufs> = pan_cams
            .iter()
            .enumerate()
            .map(|(k, c)| render_frame(&device, &queue, &base_tris, &scene, c, 0x1234 + k as u32 * 97 + 3, low_w, low_h, target_w, target_h))
            .collect();
        let pan_hist: Vec<HistFrame> = pan_bufs
            .iter()
            .map(|b| HistFrame {
                low_radiance: &b.low, low_w, low_h,
                hi_albedo: &b.albedo, hi_normal: &b.normal, hi_depth: &b.depth,
                target_w, target_h, cam: b.cam,
            })
            .collect();
        let pan_outs = direct_render_sequence_hist(&mlp, &pan_hist, DEPTH_TOL, NORMAL_THRESH);
        // mid-pan frame (history has built up but camera is moving = ghost test)
        let mid = (pan_len - 1) as usize;
        let teacher_mid = render_teacher(&device, &queue, &base_tris, &scene, &pan_cams[mid], ref_move, target_w, target_h);
        let rm_hist = rmse(&pan_outs[mid], &teacher_mid) as f64;
        let single = render_single(&mlp, &pan_bufs[mid], target_w, target_h, low_w, low_h);
        let rm_single = rmse(&single, &teacher_mid) as f64;
        let ge = rm_hist - rm_single;
        println!("[ordeal] {pname} PAN mid: resid_hist={rm_hist:.5} resid_single={rm_single:.5} ghost_excess={ge:+.5}");
        resid_move += rm_hist;
        ghost_excess = ghost_excess.max(ge); // WORST ghost across poses
        if shot_move.is_none() {
            shot_move = Some((pan_outs[mid].clone(), teacher_mid.clone()));
        }
    }

    let np = poses.len() as f64;
    resid_still /= np;
    sparkle_still /= np;
    sparkle_teacher /= np;
    tvar_still /= np;
    resid_move /= np;

    // ── proof PNGs ──────────────────────────────────────────────────────────
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("proof/neural-live");
    if let Some((net, teacher)) = &shot_still {
        write_panel(net, target_w, target_h, exposure, &proof.join("s21-still.png"));
        write_panel(teacher, target_w, target_h, exposure, &proof.join("s21-still-teacher.png"));
    }
    if let Some((net, teacher)) = &shot_move {
        write_panel(net, target_w, target_h, exposure, &proof.join("s21-moving.png"));
        write_panel(teacher, target_w, target_h, exposure, &proof.join("s21-moving-teacher.png"));
    }

    // ── VERDICT ──────────────────────────────────────────────────────────────
    let p_resid = resid_still <= RESID_BAR;
    let p_spark = sparkle_still <= SPARKLE_BAR;
    let p_tvar = tvar_still <= TVAR_BAR;
    let p_rmove = resid_move <= RESID_MOVE_BAR;
    let p_ghost = ghost_excess <= GHOST_BAR;
    let pass = p_resid && p_spark && p_tvar && p_rmove && p_ghost;

    println!("\n[ordeal] ===== REAL-IMAGE BAR ===== ({:.1}s)", t0.elapsed().as_secs_f64());
    let row = |name: &str, v: f64, bar: f64, ok: bool, lower: bool| {
        let dist = if lower { v - bar } else { bar - v };
        println!(
            "  {name:<14} {v:>12.5}  bar {bar:>10.5}  {}  (distance to bar {:+.5})",
            if ok { "PASS" } else { "FAIL" },
            dist
        );
    };
    row("resid_still", resid_still, RESID_BAR, p_resid, true);
    row("sparkle_still", sparkle_still, SPARKLE_BAR, p_spark, true);
    row("tvar_still", tvar_still, TVAR_BAR, p_tvar, true);
    row("resid_move", resid_move, RESID_MOVE_BAR, p_rmove, true);
    row("ghost_excess", ghost_excess, GHOST_BAR, p_ghost, true);
    println!("  (teacher sparkle {sparkle_teacher:.1}/Mpx — reference floor)");

    let stamp = stamp_path_for(&wpath);
    if pass {
        let metrics = [
            ("resid_still", resid_still),
            ("sparkle_still", sparkle_still),
            ("tvar_still", tvar_still),
            ("resid_move", resid_move),
            ("ghost_excess", ghost_excess),
        ];
        std::fs::write(&stamp, stamp_pass_text(&wbytes, &metrics)).unwrap();
        println!("\n[ordeal] VERDICT: PASS — stamp written {} — weights may present.", stamp.display());
    } else {
        // Remove any stale stamp so a previously-passing file cannot present.
        let _ = std::fs::remove_file(&stamp);
        println!("\n[ordeal] VERDICT: FAIL — NO stamp — the window is BLACK by law (real or black).");
        std::process::exit(1);
    }
}
