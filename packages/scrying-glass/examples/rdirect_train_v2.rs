//! R-DIRECT v2 — RETRAIN AT GOD'S RESOLUTION (640×480 native).
//!
//! v1 (rdirect-weights-v1) was trained at 96×64 native → visible speckle when
//! the live present path runs at 640×480. This retrains the SAME architecture
//! (5×64, 23→3) on the SAME naruko poses but at the LIVE resolution: low
//! 320×240 1-spp trace → native 640×480, teacher = long-accumulation reference
//! (the classical lab-equipment path — the same machinery Pleroma replaced).
//!
//! DOCTRINE (data = f(seed)): the dataset is a FUNCTION of the render seed and
//! pose set, not a file. Because 640×480×3 frames is ~0.9M pixels, each epoch
//! trains on a fresh deterministic RANDOM SUBSET (GAIA_RDIRECT_SUBSAMPLE px per
//! frame) — full-frame statistics, budget-bounded wall-clock. Checkpoints the
//! v2 weights every GAIA_RDIRECT_CKPT epochs so a timeout still ships weights.
//!
//! Proof: for a held-out pose it writes single-panel PNGs s19-v1 / s19-v2 /
//! s19-teacher and reports PSNR/MSE of v1 and v2 vs the teacher.
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_train_v2

use std::path::Path;
use std::time::Instant;

use glam::{Vec2, Vec3 as GVec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::rdirect::{
    Adam, Mlp, RdirectConfig, TrainingFrame, assemble_dataset_pairs, direct_render_image,
    serialize_weights, train_epoch_prepared, weights_sha256, INPUT_FEATURES, OUTPUT_CHANNELS,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};

const INIT_SEED: u64 = 0x00d1_5eed_0002;
const UPSCALE_SCALE: u32 = 2;

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn naruko_params() -> scrying_glass::scene::SceneParameters {
    scrying_glass::denoiser_dataset::naruko_params()
}

struct Pose {
    name: &'static str,
    camera: Camera,
    edit: Option<(GVec3, GVec3, GVec3)>,
}

struct Rendered {
    name: &'static str,
    low_noisy: Vec<GVec3>,
    hi_albedo: Vec<GVec3>,
    hi_normal: Vec<GVec3>,
    hi_depth: Vec<f32>,
    reference: Vec<GVec3>,
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
}

#[allow(clippy::too_many_arguments)]
fn render_pose(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_tris: &[LeafTriangle],
    scene: &RenderScene,
    pose: &Pose,
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
    ref_frames: u32,
) -> Rendered {
    let tris: Vec<LeafTriangle> = match pose.edit {
        None => base_tris.to_vec(),
        Some((mn, mx, off)) => base_tris
            .iter()
            .map(|t| {
                let c = (GVec3::from(t.positions[0])
                    + GVec3::from(t.positions[1])
                    + GVec3::from(t.positions[2]))
                    / 3.0;
                if c.x >= mn.x && c.x <= mx.x && c.y >= mn.y && c.y <= mx.y && c.z >= mn.z && c.z <= mx.z {
                    let mut nt = *t;
                    for p in nt.positions.iter_mut() {
                        p[0] += off.x;
                        p[1] += off.y;
                        p[2] += off.z;
                    }
                    nt
                } else {
                    *t
                }
            })
            .collect(),
    };
    let bvh = Bvh::build(&tris, &BvhParams::default());
    let cam = &pose.camera;

    let noisy_params = IntegratorParams { spp: 1, ..IntegratorParams::default() };
    let low_noisy = resolve(&trace_headless(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h, 1,
        &noisy_params, None,
    ));
    let (hi_albedo, hi_normal, hi_depth) = split_aov(&trace_headless_aov(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
    ));
    let reference = resolve(&trace_headless(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
        ref_frames, &IntegratorParams::default(), None,
    ));
    eprintln!("[v2] rendered '{}' low={low_w}x{low_h} native={target_w}x{target_h}", pose.name);
    Rendered {
        name: pose.name,
        low_noisy,
        hi_albedo,
        hi_normal,
        hi_depth,
        reference,
        low_w,
        low_h,
        target_w,
        target_h,
    }
}

fn net_output(r: &Rendered, mlp: &Mlp) -> Vec<GVec3> {
    let motion = vec![Vec2::ZERO; (r.target_w * r.target_h) as usize];
    direct_render_image(
        mlp, &r.low_noisy, r.low_w, r.low_h, &r.hi_albedo, &r.hi_normal, &r.hi_depth, &motion,
        r.target_w, r.target_h,
    )
}

fn to_training_frame(r: &Rendered) -> TrainingFrame {
    TrainingFrame {
        low_radiance: r.low_noisy.clone(),
        low_w: r.low_w,
        low_h: r.low_h,
        hi_albedo: r.hi_albedo.clone(),
        hi_normal: r.hi_normal.clone(),
        hi_depth: r.hi_depth.clone(),
        hi_motion: vec![Vec2::ZERO; (r.target_w * r.target_h) as usize],
        reference: r.reference.clone(),
        target_w: r.target_w,
        target_h: r.target_h,
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// PSNR (dB) on the tonemapped sRGB 8-bit values — the eye's space, MAX=255.
fn psnr_srgb(a: &[GVec3], b: &[GVec3], exposure: f32) -> f64 {
    let mut se = 0.0f64;
    let n = a.len();
    for i in 0..n {
        let pa = a[i];
        let pb = b[i];
        for ch in 0..3 {
            let ca = (linear_to_srgb(pa[ch] * exposure) * 255.0).round();
            let cb = (linear_to_srgb(pb[ch] * exposure) * 255.0).round();
            let d = (ca - cb) as f64;
            se += d * d;
        }
    }
    let mse = se / (n as f64 * 3.0);
    if mse <= 0.0 { return 99.0; }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

fn write_panel(panel: &[GVec3], w: u32, h: u32, exposure: f32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let px = panel[(y * w + x) as usize];
            bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
            bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
            bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
        }
    }
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("[v2] wrote {}", path.display());
}

// deterministic SplitMix64 for per-epoch pixel subsampling (data = f(seed)).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[v2] no GPU adapter — cannot forge the dataset");
    };

    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();
    eprintln!("[v2] naruko: {} static leaf tris", base_tris.len());

    // native 640×480 (GOD'S RESOLUTION — the live present path); low = 320×240.
    let target_w = env_u32("GAIA_RDIRECT_TARGET_W", 640);
    let target_h = env_u32("GAIA_RDIRECT_TARGET_H", 480);
    let low_w = target_w / UPSCALE_SCALE;
    let low_h = target_h / UPSCALE_SCALE;
    let ref_frames = env_u32("GAIA_RDIRECT_REF_FRAMES", 128);
    eprintln!("[v2] native={target_w}x{target_h} low={low_w}x{low_h} ref_frames={ref_frames}");

    let edit_min = GVec3::new(-9.0, -1.0, -6.0);
    let edit_max = GVec3::new(9.0, 24.0, 18.0);
    let edit_off = GVec3::new(6.0, 3.0, 0.0);

    let cam = |name, camera| Pose { name, camera, edit: None };
    let poses_all = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| poses_all.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();

    let train: Vec<Pose> = vec![
        cam("front", find("front")),
        cam("wide", find("wide")),
        cam("orbit_+20", find("orbit_+20")),
    ];
    let val: Vec<Pose> = vec![cam("orbit_-20", find("orbit_-20")), cam("orbit_+40", find("orbit_+40"))];
    let edit = Pose { name: "front_edit", camera: find("front"), edit: Some((edit_min, edit_max, edit_off)) };

    let r = |p: &Pose| render_pose(&device, &queue, &base_tris, &scene, p, low_w, low_h, target_w, target_h, ref_frames);
    let t_render = Instant::now();
    let train_r: Vec<Rendered> = train.iter().map(|p| r(p)).collect();
    let val_r: Vec<Rendered> = val.iter().map(|p| r(p)).collect();
    let edit_r = r(&edit);
    eprintln!("[v2] dataset rendered in {:.1}s", t_render.elapsed().as_secs_f64());

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");

    // ── train R-Direct v2 (same arch as v1: 5×64) ───────────────────────────
    let config = RdirectConfig {
        hidden_layers: env_u32("GAIA_RDIRECT_LAYERS", 5) as usize,
        hidden_width: env_u32("GAIA_RDIRECT_WIDTH", 64) as usize,
    };
    // RESUME: load the v2 checkpoint and continue (training runs in <=300s
    // segments; each invocation does `epochs` more steps from a passed offset).
    let resume = matches!(std::env::var("GAIA_RDIRECT_RESUME").as_deref(), Ok("1" | "true"));
    let wpath = data_dir.join("rdirect-weights-v2.bin");
    let mut mlp = if resume && wpath.exists() {
        let m = scrying_glass::rdirect::deserialize_weights(&std::fs::read(&wpath).unwrap())
            .expect("resume v2 weights");
        eprintln!("[v2] RESUMED from {}", wpath.display());
        m
    } else {
        Mlp::new_random(config, INIT_SEED)
    };
    let lr0 = env_f32("GAIA_RDIRECT_LR", 0.002);
    let epochs = env_u32("GAIA_RDIRECT_EPOCHS", 140);
    // global schedule position: frac = (epoch_start + e) / epoch_total.
    let epoch_start = env_u32("GAIA_RDIRECT_EPOCH_START", 0);
    let epoch_total = env_u32("GAIA_RDIRECT_EPOCH_TOTAL", epochs).max(1);
    let batch = env_u32("GAIA_RDIRECT_BATCH", 64) as usize;
    let subsample = env_u32("GAIA_RDIRECT_SUBSAMPLE", 50_000) as usize; // px/frame/epoch
    let ckpt_every = env_u32("GAIA_RDIRECT_CKPT", 20);
    let mut adam = Adam::new(&mlp, lr0, 0.9, 0.999, 1e-8);

    let train_frames: Vec<TrainingFrame> = train_r.iter().map(to_training_frame).collect();
    let (inputs, targets) = assemble_dataset_pairs(&train_frames);
    let n_pairs = inputs.len();
    let per_epoch = (subsample * train_frames.len()).min(n_pairs);
    eprintln!(
        "[v2] full pairs: {n_pairs} (config {:?}, macs/px={}); per-epoch subsample: {per_epoch}",
        config, mlp.macs()
    );

    let write_ckpt = |mlp: &Mlp| {
        std::fs::write(&wpath, serialize_weights(mlp)).unwrap();
    };

    // seed the epoch RNG off the global position so resumed segments draw
    // fresh subsets (data = f(seed), never the same subset twice).
    let mut rng = Rng(INIT_SEED ^ 0xABCD_1234 ^ ((epoch_start as u64) << 20));
    let t_train = Instant::now();
    let mut sub_in: Vec<[f32; INPUT_FEATURES]> = Vec::with_capacity(per_epoch);
    let mut sub_tg: Vec<[f32; OUTPUT_CHANNELS]> = Vec::with_capacity(per_epoch);
    for epoch in 0..epochs {
        let frac = (epoch_start + epoch) as f32 / epoch_total as f32;
        adam.set_lr(lr0 / (1.0 + 2.0 * frac));
        // fresh random subset this epoch (deterministic — data = f(seed))
        sub_in.clear();
        sub_tg.clear();
        for _ in 0..per_epoch {
            let idx = (rng.next() as usize) % n_pairs;
            sub_in.push(inputs[idx]);
            sub_tg.push(targets[idx]);
        }
        let loss = train_epoch_prepared(&mut mlp, &mut adam, &sub_in, &sub_tg, batch);
        if epoch % 10 == 0 || epoch + 1 == epochs {
            println!(
                "[v2] epoch {}/{} (seg {epoch}/{epochs}) train_mse(out-space)={loss:.6} (elapsed {:.1}s)",
                epoch_start + epoch, epoch_total,
                t_train.elapsed().as_secs_f64()
            );
        }
        if (epoch + 1) % ckpt_every == 0 || epoch + 1 == epochs {
            write_ckpt(&mlp);
            eprintln!("[v2] checkpoint @ epoch {} → {}", epoch + 1, wpath.display());
        }
    }
    eprintln!("[v2] training done in {:.1}s", t_train.elapsed().as_secs_f64());

    // ── load v1 for the A/B ─────────────────────────────────────────────────
    let v1 = scrying_glass::rdirect::deserialize_weights(
        &std::fs::read(data_dir.join("rdirect-weights-v1.bin")).unwrap(),
    )
    .expect("v1 weights");

    // ── QUALITY: PSNR/MSE vs teacher, net(v1) vs net(v2), all held-out ──────
    let exposure = env_f32("GAIA_RDIRECT_EXPOSURE", 1.6);
    println!("\n[v2] === QUALITY vs teacher (held-out) — RMSE(linear) / PSNR(sRGB dB) ===");
    println!("{:<12} {:>12} {:>12} {:>10} {:>10}", "pose", "rmse_v1", "rmse_v2", "psnr_v1", "psnr_v2");
    let mut worst_v2 = 0.0f64;
    let report = |rr: &Rendered, worst: &mut f64| {
        let out_v1 = net_output(rr, &v1);
        let out_v2 = net_output(rr, &mlp);
        let e1 = rmse(&out_v1, &rr.reference);
        let e2 = rmse(&out_v2, &rr.reference);
        let p1 = psnr_srgb(&out_v1, &rr.reference, exposure);
        let p2 = psnr_srgb(&out_v2, &rr.reference, exposure);
        println!("{:<12} {:>12.6} {:>12.6} {:>10.2} {:>10.2}", rr.name, e1, e2, p1, p2);
        *worst = worst.max(e2 as f64);
    };
    for rr in &val_r { report(rr, &mut worst_v2); }
    report(&edit_r, &mut worst_v2);
    println!("[v2] --- train poses (informational) ---");
    for rr in &train_r { report(rr, &mut worst_v2); }
    println!("[v2] worst held-out v2 RMSE (pinned bound) = {worst_v2:.6}");

    // ── proof PNGs: same pose, three panels ─────────────────────────────────
    let proof_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("proof/neural-live");
    let shot_pose = std::env::var("GAIA_RDIRECT_SHOT").unwrap_or_else(|_| "orbit_-20".into());
    let rr = val_r.iter().chain(std::iter::once(&edit_r)).chain(train_r.iter())
        .find(|r| r.name == shot_pose).unwrap_or(&val_r[0]);
    let out_v1 = net_output(rr, &v1);
    let out_v2 = net_output(rr, &mlp);
    write_panel(&out_v1, target_w, target_h, exposure, &proof_dir.join("s19-v1.png"));
    write_panel(&out_v2, target_w, target_h, exposure, &proof_dir.join("s19-v2.png"));
    write_panel(&rr.reference, target_w, target_h, exposure, &proof_dir.join("s19-teacher.png"));
    println!("[v2] proof pose '{}': s19-v1 / s19-v2 / s19-teacher written", rr.name);

    // ── ship weights + provenance ───────────────────────────────────────────
    write_ckpt(&mlp);
    let wsha = weights_sha256(&mlp);
    let wbytes = serialize_weights(&mlp);
    println!("[v2] wrote {} ({} bytes) sha256={wsha}", wpath.display(), wbytes.len());

    let prov = serde_json::json!({
        "artifact": "rdirect-weights-v2.bin",
        "weights_sha256": wsha,
        "supersedes": "rdirect-weights-v1.bin",
        "architecture": {
            "kind": "direct-render per-target-pixel MLP (denoise+upscale fused, absolute output) — SAME arch as v1",
            "input_features": INPUT_FEATURES,
            "output_channels": OUTPUT_CHANNELS,
            "hidden_layers": config.hidden_layers,
            "hidden_width": config.hidden_width,
            "macs_per_pixel": mlp.macs(),
        },
        "training": {
            "epochs": epochs, "batch": batch, "lr0": lr0,
            "lr_schedule": "harmonic decay lr0/(1+2*frac)",
            "subsample_px_per_frame_per_epoch": subsample,
            "init_seed": INIT_SEED, "optimizer": "adam",
            "note": "per-epoch random pixel subset (data=f(seed)) to bound wall-clock at 640x480"
        },
        "dataset": {
            "realm": "naruko", "low": [low_w, low_h], "native": [target_w, target_h],
            "scale": UPSCALE_SCALE, "ref_frames": ref_frames,
            "train": ["front", "wide", "orbit_+20"], "val": ["orbit_-20", "orbit_+40"],
            "scene_edit_gate": "front_edit (block displaced)"
        },
        "pinned_bound": { "description": "worst held-out (val+edit) v2 RMSE vs reference", "value": worst_v2 },
    });
    let ppath = data_dir.join("rdirect-weights-v2.provenance.json");
    std::fs::write(&ppath, serde_json::to_string_pretty(&prov).unwrap()).unwrap();
    println!("[v2] wrote {}", ppath.display());
}
