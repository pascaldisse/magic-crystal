//! v7e GAMMA DERIVATION (diagnosis, not training). Settles the recurrent
//! split net over K steps exactly like the autopsy/trainer, but at EVERY
//! step builds the REAL evidence-clamp ceiling source
//! ([`rdirect::evidence_local_max_image`] over that step's actual low_e/
//! low_d 1-spp taps — the same buffers pixel_features_split reads, NOT the
//! ref_frames-converged E_full/D_full the autopsy used as a proxy) and
//! records, for EVERY pixel of the FINAL settled frame, the ratio
//! net_lum / local_max_evidence_lum. Splits the population into sparkle
//! OUTLIERS (ordeal's own definition) vs everything else, and reports the
//! percentiles of each — the non-outlier population's max ratio is the
//! GAMMA FLOOR (smallest gamma that clamps nothing legitimate); the outlier
//! population's min ratio is the GAMMA CEILING (must be strictly above the
//! floor for a clamp to separate the two).
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_v7e_gamma_derive

use std::path::Path;

use glam::Vec2;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{headless_device, trace_headless_aov, trace_headless_split, split_aov, IntegratorParams};
use scrying_glass::rdirect::{
    bilinear_upsample, deserialize_weights, hist_features_split, pixel_features_split,
    HIST_FEATURES_SPLIT, Mlp, OUTPUT_CHANNELS,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

const SPARK_DELTA: f32 = 0.15;
const K: u32 = 8;
const REF_FRAMES: u32 = 96;

fn lum(c: glam::Vec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[idx]
}

fn run(device: &wgpu::Device, queue: &wgpu::Queue, base_tris: &[LeafTriangle], scene: &RenderScene, cam: &Camera, mlp: &Mlp, tw: u32, th: u32) {
    let low_w = tw / 2;
    let low_h = th / 2;
    let bvh = Bvh::build(base_tris, &BvhParams::default());

    let mut lows_e = Vec::with_capacity(K as usize);
    let mut lows_d = Vec::with_capacity(K as usize);
    for f in 0..K {
        let np = IntegratorParams { spp: 1, seed: 0x7abc + f * 131 + 5, ..IntegratorParams::default() };
        let (e, d) = trace_headless_split(device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1, &np);
        lows_e.push(e);
        lows_d.push(d);
    }
    let aov = trace_headless_aov(device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th);
    let (albedo, normal, depth) = split_aov(&aov);
    let (e_full, d_full) = trace_headless_split(device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, tw, th, REF_FRAMES, &IntegratorParams::default());
    let n = (tw * th) as usize;
    let teacher: Vec<glam::Vec3> = (0..n).map(|i| e_full[i] + d_full[i]).collect();

    // TEMPORAL-MEAN evidence composite across all K settle steps (variance
    // reduction — max-across-time was tried and is USELESS as a ceiling: a
    // single 1-spp specular sample can spike far above the converged value,
    // so max-across-K systematically overshoots even the net's own overshoot.
    // Mean over K taps approximates what temporal accumulation actually
    // estimates), THEN spatial 3x3 (r=1) max-pooled per the task's window.
    let mut composite_sum = vec![glam::Vec3::ZERO; n];
    for step in 0..K as usize {
        let e_up = scrying_glass::rdirect::bilinear_upsample(&lows_e[step], low_w, low_h, tw, th);
        let d_up = scrying_glass::rdirect::bilinear_upsample(&lows_d[step], low_w, low_h, tw, th);
        for px in 0..n {
            composite_sum[px] += e_up[px] + d_up[px];
        }
    }
    let composite_mean: Vec<glam::Vec3> = composite_sum.iter().map(|&s| s / K as f32).collect();
    let mut evidence_accum = vec![glam::Vec3::ZERO; n];
    for y in 0..th as i32 {
        for x in 0..tw as i32 {
            let mut m = glam::Vec3::ZERO;
            for dy in -1..=1 {
                let ny = (y + dy).clamp(0, th as i32 - 1);
                for dx in -1..=1 {
                    let nx = (x + dx).clamp(0, tw as i32 - 1);
                    m = m.max(composite_mean[(ny as u32 * tw + nx as u32) as usize]);
                }
            }
            evidence_accum[(y as u32 * tw + x as u32) as usize] = m;
        }
    }

    let mut net = vec![glam::Vec3::ZERO; n];
    for ty in 0..th {
        for tx in 0..tw {
            let px = (ty * tw + tx) as usize;
            let alb = albedo[px];
            let mut prev_dl = [0.0f32; 3];
            let mut valid = 0.0f32;
            let mut dl = [0.0f32; OUTPUT_CHANNELS];
            for step in 0..K as usize {
                let s = step.min(lows_e.len() - 1);
                let base = pixel_features_split(&lows_e[s], &lows_d[s], low_w, low_h, tw, th, tx, ty, alb, normal[px], depth[px], Vec2::ZERO);
                let feat = hist_features_split(&base, prev_dl, valid);
                dl = mlp.forward(&feat);
                prev_dl = dl;
                valid = 1.0;
            }
            let div = if alb.length_squared() > 1e-8 { alb + glam::Vec3::splat(1e-3) } else { glam::Vec3::ONE };
            let expm1 = glam::Vec3::new(dl[0].exp() - 1.0, dl[1].exp() - 1.0, dl[2].exp() - 1.0);
            net[px] = glam::Vec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0)) * div;
        }
    }
    let last_evidence_max = evidence_accum;

    // sparkle-outlier detection (ordeal's own definition, vs teacher)
    let idx = |x: i32, y: i32| (y as usize) * tw as usize + x as usize;
    let err = |x: i32, y: i32| lum(net[idx(x, y)]) - lum(teacher[idx(x, y)]);
    let mut is_outlier = vec![false; n];
    let mut n_outliers = 0u32;
    for y in 1..th as i32 - 1 {
        for x in 1..tw as i32 - 1 {
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
                is_outlier[idx(x, y)] = true;
                n_outliers += 1;
            }
        }
    }

    // ratio = net_lum / local_max_evidence_lum, per pixel, split by outlier flag.
    // Restrict to "bright enough to matter" (local_max_evidence_lum > 0.01) to
    // avoid near-zero-evidence division noise in dark/sky regions.
    let mut ratios_outlier: Vec<f32> = Vec::new();
    let mut ratios_nonoutlier: Vec<f32> = Vec::new();
    for px in 0..n {
        let ev = lum(last_evidence_max[px]);
        if ev <= 0.01 {
            continue;
        }
        let r = lum(net[px]) / ev;
        if is_outlier[px] {
            ratios_outlier.push(r);
        } else {
            ratios_nonoutlier.push(r);
        }
    }
    ratios_outlier.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ratios_nonoutlier.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("[v7e-gamma] {tw}x{th}: {n_outliers} sparkle outliers, {} bright non-outlier px, {} bright outlier px",
        ratios_nonoutlier.len(), ratios_outlier.len());
    for p in [0.50, 0.90, 0.99, 0.999, 1.0] {
        println!("  non-outlier ratio p{:<5.1} = {:.4}", p * 100.0, percentile(&ratios_nonoutlier, p));
    }
    for p in [0.0, 0.01, 0.10, 0.50] {
        println!("  outlier     ratio p{:<5.1} = {:.4}", p * 100.0, percentile(&ratios_outlier, p));
    }
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[v7e-gamma] no GPU adapter");
    };
    let params = scrying_glass::denoiser_dataset::naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();

    let val_poses = scrying_glass::denoiser_dataset::law_poses(&params);
    let cam = val_poses.iter().find(|(n, _)| *n == "orbit_-20").unwrap().1.clone();

    let wrel = "data/rdirect-weights-v7.bin";
    let wpath = Path::new(env!("CARGO_MANIFEST_DIR")).join(wrel);
    let mlp = deserialize_weights(&std::fs::read(&wpath).expect("read weights")).expect("parse weights");
    assert_eq!(mlp.layer_dims()[0].0 as usize, HIST_FEATURES_SPLIT, "expects 39-in split net");

    println!("[v7e-gamma] weights={wrel} val pose=orbit_-20 K={K} ref_frames={REF_FRAMES}");
    for (tw, th) in [(480u32, 360u32), (640u32, 480u32)] {
        run(&device, &queue, &base_tris, &scene, &cam, &mlp, tw, th);
    }
}
