//! GHOST AUTOPSY — SKY HISTORY SMEAR, pure-CPU offscreen probe (no GPU, no
//! live app, no world load). Synthetic scene: camera TRANSLATES sideways
//! (pure translation, forward/up/right constant — no rotation), top half of
//! frame is SKY (no-hit: albedo=0, depth<=0), bottom half is a constant
//! ground plane. The sky's LOW-res radiance (the noisy 1-spp taps
//! `pixel_features_split` bilinear-samples) gets FRESH per-frame noise —
//! the real renderer's actual Monte Carlo variance — so a wrongly-accepted
//! history sample is visibly stale, not coincidentally identical.
//!
//! Metric (per the room's ask): mean-abs-diff between the PRESENTED sky
//! pixel (recurrent, history-fed) and a FRESH no-history render of the exact
//! same pose (a length-1 sequence — prev=None, valid=0 unconditionally),
//! run under (a) current default behavior and (b) `GAIA_V7_SKY_HISTORY=reject`.
//!
//! Uses the REAL shipped v7 weights (not random init) so the net's actual
//! learned response is what gets measured.
//!
//! Run: cargo run -j2 --example rdirect_v7_sky_smear_probe

use glam::Vec3;
use scrying_glass::rdirect::{
    deserialize_weights, direct_render_sequence_hist_split, CamPose, HistFrameSplit,
};

const TARGET_W: u32 = 64;
const TARGET_H: u32 = 48;
const LOW_W: u32 = 32;
const LOW_H: u32 = 24;
const N_FRAMES: usize = 8;
const DEPTH_TOL: f32 = 0.05;
const NORMAL_THRESH: f32 = 0.85;

/// Tiny deterministic LCG — no external RNG dependency, reproducible.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32 as f32) / (u32::MAX as f32)
    }
}

/// Build one frame's synthetic gbuffer + low-res radiance. `frame_idx` seeds
/// the sky's per-frame noise (simulating real 1-spp variance); `eye_x` is
/// the camera's sideways translation this frame.
fn make_frame(frame_idx: usize, eye_x: f32) -> (Vec<Vec3>, Vec<Vec3>, Vec<f32>, Vec<Vec3>, Vec<Vec3>, CamPose) {
    let n = (TARGET_W * TARGET_H) as usize;
    let mut hi_albedo = vec![Vec3::ZERO; n];
    let mut hi_normal = vec![Vec3::ZERO; n];
    let mut hi_depth = vec![0.0f32; n];
    for ty in 0..TARGET_H {
        for tx in 0..TARGET_W {
            let i = (ty * TARGET_W + tx) as usize;
            if ty < TARGET_H / 2 {
                // sky: no-hit
                hi_albedo[i] = Vec3::ZERO;
                hi_depth[i] = 0.0;
                hi_normal[i] = Vec3::ZERO;
            } else {
                // ground: constant plane, camera translates PARALLEL to it
                // (forward/up/right fixed) so the geometric depth stays
                // constant frame-to-frame — isolates the sky mechanism.
                hi_albedo[i] = Vec3::new(0.6, 0.55, 0.5);
                hi_depth[i] = 5.0;
                hi_normal[i] = Vec3::new(0.0, 1.0, 0.0);
            }
        }
    }
    let ln = (LOW_W * LOW_H) as usize;
    let mut low_e = vec![Vec3::ZERO; ln];
    let mut low_d = vec![Vec3::ZERO; ln];
    let mut rng = Lcg(0x9E3779B97F4A7C15u64 ^ (frame_idx as u64).wrapping_mul(0xBF58476D1CE4E5B9));
    for ly in 0..LOW_H {
        for lx in 0..LOW_W {
            let li = (ly * LOW_W + lx) as usize;
            if ly < LOW_H / 2 {
                // sky: base sky color + FRESH per-frame stochastic noise
                // (the real renderer's 1-spp variance under a moving/
                // directional sky — sun glow, horizon gradient, dithering).
                let base = Vec3::new(0.4, 0.55, 0.9);
                let noise = Vec3::new(rng.next_f32(), rng.next_f32(), rng.next_f32()) * 0.35;
                low_e[li] = base + noise;
                low_d[li] = Vec3::ZERO;
            } else {
                low_e[li] = Vec3::new(0.5, 0.46, 0.42);
                low_d[li] = Vec3::new(0.05, 0.05, 0.05);
            }
        }
    }
    let cam = CamPose {
        eye: Vec3::new(eye_x, 1.0, 0.0),
        right: Vec3::new(1.0, 0.0, 0.0),
        up: Vec3::new(0.0, 1.0, 0.0),
        forward: Vec3::new(0.0, 0.0, -1.0),
        half_tan: (30.0f32.to_radians()).tan(),
        aspect: TARGET_W as f32 / TARGET_H as f32,
    };
    (hi_albedo, hi_normal, hi_depth, low_e, low_d, cam)
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn write_png(img: &[Vec3], w: u32, h: u32, path: &std::path::Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in img {
        bytes.push((linear_to_srgb(px.x) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.y) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.z) * 255.0 + 0.5) as u8);
    }
    let file = std::fs::File::create(path).unwrap();
    let writer = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(writer, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("[sky-smear-probe] wrote {}", path.display());
}

fn main() {
    let weights_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rdirect-weights-v7.bin");
    let bytes = std::fs::read(&weights_path).expect("read v7 weights");
    let mlp = deserialize_weights(&bytes).expect("deserialize v7 weights (39-in)");

    let n = (TARGET_W * TARGET_H) as usize;
    let sky_idx: Vec<usize> = (0..n).filter(|&i| (i as u32 / TARGET_W) < TARGET_H / 2).collect();

    // Build the N-frame TRANSLATING sequence (owned buffers, reused for all
    // 3 runs below so the input radiance is byte-identical across them).
    let mut owned = Vec::new();
    for f in 0..N_FRAMES {
        let eye_x = f as f32 * 0.15; // sideways translation, 0.15 units/frame
        owned.push(make_frame(f, eye_x));
    }
    let frames: Vec<HistFrameSplit> = owned
        .iter()
        .map(|(hi_albedo, hi_normal, hi_depth, low_e, low_d, cam)| HistFrameSplit {
            low_e,
            low_d,
            low_w: LOW_W,
            low_h: LOW_H,
            hi_albedo,
            hi_normal,
            hi_depth,
            target_w: TARGET_W,
            target_h: TARGET_H,
            cam: *cam,
        })
        .collect();

    // (A) DEFAULT behavior (env unset — current live behavior).
    unsafe { std::env::remove_var("GAIA_V7_SKY_HISTORY") };
    let out_default = direct_render_sequence_hist_split(&mlp, &frames, DEPTH_TOL, NORMAL_THRESH);

    // (B) GAIA_V7_SKY_HISTORY=reject.
    unsafe { std::env::set_var("GAIA_V7_SKY_HISTORY", "reject") };
    let out_reject = direct_render_sequence_hist_split(&mlp, &frames, DEPTH_TOL, NORMAL_THRESH);
    unsafe { std::env::remove_var("GAIA_V7_SKY_HISTORY") };

    // (C) FRESH no-history reference: each frame rendered as its OWN
    // length-1 sequence (prev=None ⇒ valid=0 unconditionally for every
    // pixel, sky included) — "what the net says with no history bias".
    let mut out_fresh: Vec<Vec<Vec3>> = Vec::with_capacity(N_FRAMES);
    for f in 0..N_FRAMES {
        let ff = &frames[f];
        let one = [HistFrameSplit {
            low_e: ff.low_e,
            low_d: ff.low_d,
            low_w: ff.low_w,
            low_h: ff.low_h,
            hi_albedo: ff.hi_albedo,
            hi_normal: ff.hi_normal,
            hi_depth: ff.hi_depth,
            target_w: ff.target_w,
            target_h: ff.target_h,
            cam: ff.cam,
        }];
        out_fresh.push(direct_render_sequence_hist_split(&mlp, &one, DEPTH_TOL, NORMAL_THRESH).remove(0));
    }

    let mean_abs_sky = |a: &[Vec3], b: &[Vec3]| -> f64 {
        let mut s = 0.0f64;
        for &i in &sky_idx {
            let d = (a[i] - b[i]).abs();
            s += (d.x + d.y + d.z) as f64 / 3.0;
        }
        s / sky_idx.len() as f64
    };

    println!("[sky-smear-probe] N_FRAMES={N_FRAMES} sky_px={} (of {n})", sky_idx.len());
    println!("[sky-smear-probe] per-frame mean-abs-diff over SKY pixels vs FRESH no-history render:");
    let mut default_max = 0.0f64;
    let mut reject_max = 0.0f64;
    for f in 0..N_FRAMES {
        let d_default = mean_abs_sky(&out_default[f], &out_fresh[f]);
        let d_reject = mean_abs_sky(&out_reject[f], &out_fresh[f]);
        default_max = default_max.max(d_default);
        reject_max = reject_max.max(d_reject);
        println!(
            "  frame {f}: default-vs-fresh={d_default:.6e}  reject-vs-fresh={d_reject:.6e}"
        );
    }
    println!(
        "[sky-smear-probe] SUMMARY max-over-frames: default-vs-fresh={default_max:.6e} reject-vs-fresh={reject_max:.6e}"
    );
    println!(
        "[sky-smear-probe] (reject-vs-fresh should be ~0 — reject makes sky history == no-history \
         by construction; default-vs-fresh > 0 is the measured smear)"
    );

    // Regression-safety spot check: GROUND pixels must be UNCHANGED by the
    // reject flag (it only touches the is_miss branch).
    let ground_idx: Vec<usize> = (0..n).filter(|&i| (i as u32 / TARGET_W) >= TARGET_H / 2).collect();
    let mut ground_max_diff = 0.0f32;
    for f in 0..N_FRAMES {
        for &i in &ground_idx {
            let d = (out_default[f][i] - out_reject[f][i]).abs();
            ground_max_diff = ground_max_diff.max(d.x.max(d.y).max(d.z));
        }
    }
    println!("[sky-smear-probe] ground px max-diff default-vs-reject={ground_max_diff:.4e} (must be 0.0)");

    let proof = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof/neural-live");
    write_png(&out_default[N_FRAMES - 1], TARGET_W, TARGET_H, &proof.join("sky-smear-default-last.png"));
    write_png(&out_reject[N_FRAMES - 1], TARGET_W, TARGET_H, &proof.join("sky-smear-reject-last.png"));
}
