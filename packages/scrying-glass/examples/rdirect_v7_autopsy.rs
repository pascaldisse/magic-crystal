//! v7 OUTLIER AUTOPSY (diagnosis, not training). Renders ONE val pose
//! ("orbit_-20", same pose the ordeal/trainer use for the val split) at
//! BOTH 480x360 (train res) and 640x480 (ordeal res) with the CURRENT
//! rdirect-weights-v7.bin, finds the top-50 sparkle-outlier pixels using
//! the ORDEAL'S OWN outlier definition (isolated bright dot: net lum -
//! teacher lum > SPARK_DELTA and a strict local max of that signed error
//! over the 3x3 neighbourhood), and for each reports the full channel
//! decomposition of the TARGET construction (E_full exact, D_full RAW,
//! D_full box-blurred — the smoothed target = E_full + blur(D_full) that
//! v7's trainer actually optimizes against) plus the teacher (E_full +
//! D_full RAW, what the ordeal/sparkle metric compares net output to).
//!
//! CLASSIFY each outlier:
//!   COPIED   — the SMOOTHED TARGET itself still has a local spike at that
//!              pixel (blur radius 2 didn't fully kill it) -> the net was
//!              trained to reproduce a bright/dirty pixel there.
//!   INVENTED — the smoothed target is locally clean there -> the net
//!              produced a bright pixel with no corresponding signal in
//!              what it was actually trained against.
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_v7_autopsy
//!   GAIA_V7A_WEIGHTS (default data/rdirect-weights-v7.bin)
//!   GAIA_V7A_TOPN (default 50)

use std::path::Path;
use std::time::Instant;

use glam::{Vec2, Vec3 as GVec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, headless_device, split_aov, trace_headless_aov, trace_headless_split,
};
use scrying_glass::rdirect::{
    HIST_FEATURES_SPLIT, Mlp, OUTPUT_CHANNELS, deserialize_weights, hist_features_split,
    pixel_features_split,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

const SPARK_DELTA: f32 = 0.15; // same constant as real_image_ordeal.rs
const K: u32 = 8; // matches v7d trainer's GAIA_V7_STILL
const REF_FRAMES: u32 = 96; // matches v7d trainer's GAIA_V7_REF default
const BLUR_RADIUS: i32 = 2; // matches v7d trainer's GAIA_V7_BLUR default
// COPIED/INVENTED split threshold on the smoothed target's own local excess
// (lum(px) - mean(lum(3x3 neighbours)) in the smoothed target). Anything
// above this is "the target itself is locally bright there".
const TARGET_LOCAL_EXCESS_COPIED: f32 = 0.05;

fn env_u32(n: &str, d: u32) -> u32 {
    std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_string(n: &str, d: &str) -> String {
    std::env::var(n).unwrap_or_else(|_| d.to_string())
}

fn lum(c: GVec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

fn box_blur(img: &[GVec3], w: u32, h: u32, radius: i32) -> Vec<GVec3> {
    if radius <= 0 {
        return img.to_vec();
    }
    let mut out = vec![GVec3::ZERO; img.len()];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut sum = GVec3::ZERO;
            let mut cnt = 0.0f32;
            for dy in -radius..=radius {
                let ny = y + dy;
                if ny < 0 || ny >= h as i32 {
                    continue;
                }
                for dx in -radius..=radius {
                    let nx = x + dx;
                    if nx < 0 || nx >= w as i32 {
                        continue;
                    }
                    sum += img[(ny as u32 * w + nx as u32) as usize];
                    cnt += 1.0;
                }
            }
            out[(y as u32 * w + x as u32) as usize] = sum / cnt.max(1.0);
        }
    }
    out
}

/// Everything the autopsy needs at one resolution: the raw E/D split (exact,
/// ref_frames), the box-blurred D, the resulting teacher (E+D raw) and the
/// smoothed target (E + blur(D)) the trainer actually optimizes against, the
/// K fresh-seed low-res split input frames, and the settled net output.
struct Autopsy {
    tw: u32,
    th: u32,
    net: Vec<GVec3>,
    teacher: Vec<GVec3>, // E_full + D_full RAW (what sparkle/ordeal compare net to)
    smoothed: Vec<GVec3>, // E_full + blur(D_full) (what the trainer's loss targets)
    e_full: Vec<GVec3>,
    d_full: Vec<GVec3>,
    d_blurred: Vec<GVec3>,
}

#[allow(clippy::too_many_arguments)]
fn run_autopsy(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_tris: &[LeafTriangle],
    scene: &RenderScene,
    cam: &Camera,
    mlp: &Mlp,
    tw: u32,
    th: u32,
) -> Autopsy {
    let low_w = tw / 2;
    let low_h = th / 2;
    let bvh = Bvh::build(base_tris, &BvhParams::default());

    // K fresh-seed 1spp split input frames — SAME seed formula as the v7
    // trainer's render_pose (0x7abc + f*131 + 5).
    let mut lows_e = Vec::with_capacity(K as usize);
    let mut lows_d = Vec::with_capacity(K as usize);
    for f in 0..K {
        let np = IntegratorParams { spp: 1, seed: 0x7abc + f * 131 + 5, ..IntegratorParams::default() };
        let (e, d) = trace_headless_split(
            device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np,
        );
        lows_e.push(e);
        lows_d.push(d);
    }

    let aov = trace_headless_aov(device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th);
    let (albedo, normal, depth) = split_aov(&aov);

    let (e_full, d_full) = trace_headless_split(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, REF_FRAMES,
        &IntegratorParams::default(),
    );
    let d_blurred = box_blur(&d_full, tw, th, BLUR_RADIUS);
    let n = (tw * th) as usize;
    let teacher: Vec<GVec3> = (0..n).map(|i| e_full[i] + d_full[i]).collect();
    let smoothed: Vec<GVec3> = (0..n).map(|i| e_full[i] + d_blurred[i]).collect();

    // Settle the recurrent split net over K steps (same as trainer's
    // settle_still / ordeal's sequence_render last frame).
    let mut net = vec![GVec3::ZERO; n];
    for ty in 0..th {
        for tx in 0..tw {
            let px = (ty * tw + tx) as usize;
            let alb = albedo[px];
            let mut prev_dl = [0.0f32; 3];
            let mut valid = 0.0f32;
            let mut dl = [0.0f32; OUTPUT_CHANNELS];
            for step in 0..K as usize {
                let s = step.min(lows_e.len() - 1);
                let base = pixel_features_split(
                    &lows_e[s], &lows_d[s], low_w, low_h, tw, th, tx, ty, alb, normal[px], depth[px], Vec2::ZERO,
                );
                let feat = hist_features_split(&base, prev_dl, valid);
                dl = mlp.forward(&feat);
                prev_dl = dl;
                valid = 1.0;
            }
            let div = if alb.length_squared() > 1e-8 { alb + GVec3::splat(1e-3) } else { GVec3::ONE };
            let expm1 = GVec3::new(dl[0].exp() - 1.0, dl[1].exp() - 1.0, dl[2].exp() - 1.0);
            net[px] = GVec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0)) * div;
        }
    }

    Autopsy { tw, th, net, teacher, smoothed, e_full, d_full, d_blurred }
}

struct Outlier {
    x: u32,
    y: u32,
    err: f32,       // net lum - teacher lum (the ordeal's outlier signal)
    net_lum: f32,
    teacher_lum: f32,
    smoothed_lum: f32,
    e_lum: f32,
    d_raw_lum: f32,
    d_blur_lum: f32,
    target_local_excess: f32, // smoothed_lum - mean(3x3 neighbours of smoothed)
    verdict: &'static str,
}

fn find_outliers(a: &Autopsy, topn: usize) -> Vec<Outlier> {
    let (w, h) = (a.tw, a.th);
    let idx = |x: i32, y: i32| (y as usize) * w as usize + x as usize;
    let err = |x: i32, y: i32| lum(a.net[idx(x, y)]) - lum(a.teacher[idx(x, y)]);
    let mut cands: Vec<(i32, i32, f32)> = Vec::new();
    for y in 1..h as i32 - 1 {
        for x in 1..w as i32 - 1 {
            let e = err(x, y);
            if e <= SPARK_DELTA {
                continue;
            }
            let mut is_peak = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if (dx != 0 || dy != 0) && err(x + dx, y + dy) >= e {
                        is_peak = false;
                    }
                }
            }
            if is_peak {
                cands.push((x, y, e));
            }
        }
    }
    cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    cands.truncate(topn);

    cands
        .into_iter()
        .map(|(x, y, e)| {
            let px = idx(x, y);
            let mut nsum = 0.0f32;
            let mut ncnt = 0.0f32;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    nsum += lum(a.smoothed[idx(x + dx, y + dy)]);
                    ncnt += 1.0;
                }
            }
            let smoothed_lum = lum(a.smoothed[px]);
            let target_local_excess = smoothed_lum - nsum / ncnt.max(1.0);
            let verdict = if target_local_excess > TARGET_LOCAL_EXCESS_COPIED { "COPIED" } else { "INVENTED" };
            Outlier {
                x: x as u32,
                y: y as u32,
                err: e,
                net_lum: lum(a.net[px]),
                teacher_lum: lum(a.teacher[px]),
                smoothed_lum,
                e_lum: lum(a.e_full[px]),
                d_raw_lum: lum(a.d_full[px]),
                d_blur_lum: lum(a.d_blurred[px]),
                target_local_excess,
                verdict,
            }
        })
        .collect()
}

fn sparkle_per_mpx(a: &Autopsy) -> f64 {
    let (w, h) = (a.tw, a.th);
    let idx = |x: i32, y: i32| (y as usize) * w as usize + x as usize;
    let err = |x: i32, y: i32| lum(a.net[idx(x, y)]) - lum(a.teacher[idx(x, y)]);
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
                    if (dx != 0 || dy != 0) && err(x + dx, y + dy) >= e {
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

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[v7-autopsy] no GPU adapter");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();

    let val_poses = scrying_glass::denoiser_dataset::law_poses(&params);
    let cam = val_poses.iter().find(|(n, _)| *n == "orbit_-20").unwrap().1.clone();

    let wrel = env_string("GAIA_V7A_WEIGHTS", "data/rdirect-weights-v7.bin");
    let wpath = Path::new(env!("CARGO_MANIFEST_DIR")).join(&wrel);
    let mlp = deserialize_weights(&std::fs::read(&wpath).expect("read weights")).expect("parse weights");
    assert_eq!(mlp.layer_dims()[0].0 as usize, HIST_FEATURES_SPLIT, "expects 39-in split net");
    let topn = env_u32("GAIA_V7A_TOPN", 50) as usize;

    println!("[v7-autopsy] weights={wrel} val pose=orbit_-20 K={K} ref_frames={REF_FRAMES} blur_r={BLUR_RADIUS} topn={topn}");

    let t0 = Instant::now();
    let resolutions = [(480u32, 360u32), (640u32, 480u32)];
    let mut sparkle_by_res: Vec<(u32, u32, f64)> = Vec::new();

    for (tw, th) in resolutions {
        let a = run_autopsy(&device, &queue, &base_tris, &scene, &cam, &mlp, tw, th);
        let sp = sparkle_per_mpx(&a);
        sparkle_by_res.push((tw, th, sp));
        println!("\n[v7-autopsy] ===== {tw}x{th} (sparkle {sp:.1}/Mpx) =====");

        let outliers = find_outliers(&a, topn);
        let n_copied = outliers.iter().filter(|o| o.verdict == "COPIED").count();
        let n_invented = outliers.iter().filter(|o| o.verdict == "INVENTED").count();
        println!(
            "[v7-autopsy] {tw}x{th}: {} outliers found (requested top {topn}) -> COPIED={n_copied} INVENTED={n_invented}",
            outliers.len()
        );
        println!(
            "{:>4} {:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>9}",
            "x", "y", "err", "net", "teacher", "smooth", "E", "D_raw", "D_blur", "tgt_exc", "verdict"
        );
        for o in outliers.iter().take(10) {
            println!(
                "{:>4} {:>4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>10.4} {:>9}",
                o.x, o.y, o.err, o.net_lum, o.teacher_lum, o.smoothed_lum, o.e_lum, o.d_raw_lum, o.d_blur_lum,
                o.target_local_excess, o.verdict
            );
        }
        if outliers.len() > 10 {
            println!("... ({} more rows omitted, see raw dump)", outliers.len() - 10);
        }
        // full raw dump too (small, cheap)
        for o in &outliers {
            println!(
                "RAW {tw}x{th} x={} y={} err={:.5} net={:.5} teacher={:.5} smoothed={:.5} E={:.5} D_raw={:.5} D_blur={:.5} tgt_excess={:.5} verdict={}",
                o.x, o.y, o.err, o.net_lum, o.teacher_lum, o.smoothed_lum, o.e_lum, o.d_raw_lum, o.d_blur_lum,
                o.target_local_excess, o.verdict
            );
        }
    }

    println!("\n[v7-autopsy] ===== RES MISMATCH =====");
    for (tw, th, sp) in &sparkle_by_res {
        println!("  {tw}x{th}: sparkle {sp:.1}/Mpx");
    }
    if sparkle_by_res.len() == 2 {
        let (w0, h0, s0) = sparkle_by_res[0];
        let (w1, h1, s1) = sparkle_by_res[1];
        println!("  delta: {w1}x{h1} - {w0}x{h0} = {:+.1}/Mpx ({:+.1}%)", s1 - s0, 100.0 * (s1 - s0) / s0.max(1e-9));
    }

    println!("\n[v7-autopsy] done in {:.1}s", t0.elapsed().as_secs_f64());
}
