//! R-DIRECT SPIKE — the net that RENDERS DIRECTLY, weighed against the shipped
//! denoise+upscale CHAIN at the SAME ray budget (one 1-spp low-res trace).
//!
//! For each naruko pose we render: a LOW-res 1-spp traced frame (the sparse
//! guide), the LOW-res + NATIVE-res G-buffer (albedo/normal/depth; motion = 0
//! in this static dataset), and the NATIVE-res converged reference (the
//! teacher, long accumulation). Then:
//!   - train the R-Direct MLP on the TRAIN split (low 1-spp → native image);
//!   - evaluate on held-out VAL poses + a held-out SCENE EDIT (a block of
//!     geometry displaced — never seen in that configuration);
//!   - build the shipped CHAIN output (VIII-1 denoiser at low res → VIII-3
//!     upscaler to native) from the SAME low 1-spp frame;
//!   - report RMSE (net / chain / bilinear-of-1spp) vs the reference,
//!     per-pixel MAC cost, and CPU inference ms for both;
//!   - write weights + provenance + a QUAD proof PNG per pose:
//!     1spp-input | net | ground-truth | chain.
//!
//! Run: cargo run -p scrying-glass --release --example rdirect_spike

use std::path::Path;
use std::time::Instant;

use glam::{Vec2, Vec3 as GVec3};

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::{self as den};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::rdirect::{
    Adam, Mlp, RdirectConfig, TrainingFrame, assemble_dataset_pairs, bilinear_upsample,
    direct_render_image, serialize_weights, train_epoch_prepared, weights_sha256,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene};
use scrying_glass::upscaler::{self as up};

const INIT_SEED: u64 = 0x00d1_5eed;
const UPSCALE_SCALE: u32 = 2;

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

// ── dataset poses (naruko; same split family as VIII-1/VIII-3) ──────────────
fn naruko_params() -> scrying_glass::scene::SceneParameters {
    scrying_glass::denoiser_dataset::naruko_params()
}

struct Pose {
    name: &'static str,
    camera: Camera,
    /// If set, displace leaf-tri centroids inside this AABB by `edit_offset`
    /// before building the BVH — the "scene edit" generalization gate.
    edit: Option<(GVec3, GVec3, GVec3)>, // (aabb_min, aabb_max, offset)
}

struct Rendered {
    name: &'static str,
    low_noisy: Vec<GVec3>,
    low_albedo: Vec<GVec3>,
    low_normal: Vec<GVec3>,
    low_depth: Vec<f32>,
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
    // Apply the optional scene edit (displace a block of geometry).
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
    let (low_albedo, low_normal, low_depth) = split_aov(&trace_headless_aov(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, low_w, low_h,
    ));
    let (hi_albedo, hi_normal, hi_depth) = split_aov(&trace_headless_aov(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
    ));
    let reference = resolve(&trace_headless(
        device, queue, &bvh, cam, &scene.sun, scene.sky_top, scene.sky_horizon, target_w, target_h,
        ref_frames, &IntegratorParams::default(), None,
    ));
    eprintln!("[rdirect] rendered '{}' low={low_w}x{low_h} native={target_w}x{target_h}", pose.name);
    Rendered {
        name: pose.name,
        low_noisy,
        low_albedo,
        low_normal,
        low_depth,
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

/// The shipped CHAIN: VIII-1 denoiser at LOW res → VIII-3 upscaler to native.
fn chain_output(r: &Rendered, denoiser: &den::Mlp, upscaler: &up::Mlp) -> Vec<GVec3> {
    let denoised_low = den::denoise_image(
        denoiser, &r.low_noisy, &r.low_albedo, &r.low_normal, &r.low_depth,
    );
    up::upscale_image(
        upscaler, &denoised_low, r.low_w, r.low_h, &r.hi_albedo, &r.hi_normal, &r.hi_depth,
        r.target_w, r.target_h,
    )
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

// ── PNG (quad panel) ────────────────────────────────────────────────────────
fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn write_quad(panels: [&[GVec3]; 4], w: u32, h: u32, exposure: f32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((4 * w * h * 3) as usize);
    for y in 0..h {
        for panel in panels {
            for x in 0..w {
                let px = panel[(y * w + x) as usize];
                bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
                bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
                bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
            }
        }
    }
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 4 * w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("[rdirect] wrote {}", path.display());
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[rdirect] no GPU adapter — cannot forge the dataset");
    };

    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let base_tris = scene.leaf_triangles();
    eprintln!("[rdirect] naruko: {} static leaf tris", base_tris.len());

    // scene AABB (to place the edit block honestly inside the geometry)
    let mut mn = GVec3::splat(f32::INFINITY);
    let mut mx = GVec3::splat(f32::NEG_INFINITY);
    for t in &base_tris {
        for p in &t.positions {
            let v = GVec3::from(*p);
            mn = mn.min(v);
            mx = mx.max(v);
        }
    }
    eprintln!("[rdirect] scene aabb min={mn:?} max={mx:?}");

    // dataset resolution: low 48×32 → native 96×64 (where VIII-1/VIII-3 pin).
    let low_w = env_u32("GAIA_RDIRECT_LOW_W", 48);
    let low_h = env_u32("GAIA_RDIRECT_LOW_H", 32);
    let target_w = low_w * UPSCALE_SCALE;
    let target_h = low_h * UPSCALE_SCALE;
    let ref_frames = env_u32("GAIA_RDIRECT_REF_FRAMES", 128);

    // A block of VISIBLE geometry near the front camera's view (eye≈(0,2,22)
    // looking -z; the lighthouse/vessels cluster near origin), displaced —
    // the "crate moved" gate. The scene AABB is dominated by a ±1000 ground
    // plane, so a centre-of-AABB block lands off-screen; this region is
    // hand-placed on the visible structures and its effect is asserted below.
    let edit_min = GVec3::new(-9.0, -1.0, -6.0);
    let edit_max = GVec3::new(9.0, 24.0, 18.0);
    let edit_off = GVec3::new(6.0, 3.0, 0.0);
    let n_edited = base_tris.iter().filter(|t| {
        let c = (GVec3::from(t.positions[0]) + GVec3::from(t.positions[1]) + GVec3::from(t.positions[2])) / 3.0;
        c.x >= edit_min.x && c.x <= edit_max.x && c.y >= edit_min.y && c.y <= edit_max.y && c.z >= edit_min.z && c.z <= edit_max.z
    }).count();
    eprintln!("[rdirect] scene-edit block: {n_edited} leaf tris displaced by {edit_off:?}");

    let cam = |name, camera| Pose { name, camera, edit: None };
    let poses_all = scrying_glass::denoiser_dataset::law_poses(&params);
    let find = |n: &str| poses_all.iter().find(|(pn, _)| *pn == n).unwrap().1.clone();

    // TRAIN = front, wide, orbit_+20 ; VAL = orbit_-20, orbit_+40 ; EDIT = front+moved-block
    let train: Vec<Pose> = vec![
        cam("front", find("front")),
        cam("wide", find("wide")),
        cam("orbit_+20", find("orbit_+20")),
    ];
    let val: Vec<Pose> = vec![cam("orbit_-20", find("orbit_-20")), cam("orbit_+40", find("orbit_+40"))];
    let edit = Pose { name: "front_edit", camera: find("front"), edit: Some((edit_min, edit_max, edit_off)) };

    let r = |p: &Pose| render_pose(&device, &queue, &base_tris, &scene, p, low_w, low_h, target_w, target_h, ref_frames);
    let train_r: Vec<Rendered> = train.iter().map(|p| r(p)).collect();
    let val_r: Vec<Rendered> = val.iter().map(|p| r(p)).collect();
    let edit_r = r(&edit);
    let front_ref = &train_r[0].reference;
    let edit_delta = rmse(&edit_r.reference, front_ref);
    eprintln!("[rdirect] scene-edit gate: rmse(front_edit_ref, front_ref)={edit_delta:.6} (must be >0)");
    assert!(edit_delta > 1e-4, "scene edit did not change the visible frame");

    // ── load the shipped chain nets ─────────────────────────────────────────
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let denoiser = den::deserialize_weights(&std::fs::read(data_dir.join("denoiser-weights-v1.bin")).unwrap())
        .expect("denoiser weights");
    let upscaler = up::deserialize_weights(&std::fs::read(data_dir.join("upscaler-weights-v1.bin")).unwrap())
        .expect("upscaler weights");

    // ── train R-Direct ──────────────────────────────────────────────────────
    let config = RdirectConfig {
        hidden_layers: env_u32("GAIA_RDIRECT_LAYERS", 5) as usize,
        hidden_width: env_u32("GAIA_RDIRECT_WIDTH", 64) as usize,
    };
    let mut mlp = Mlp::new_random(config, INIT_SEED);
    let lr0 = env_f32("GAIA_RDIRECT_LR", 0.002);
    let epochs = env_u32("GAIA_RDIRECT_EPOCHS", 200);
    let batch = env_u32("GAIA_RDIRECT_BATCH", 64) as usize;
    let mut adam = Adam::new(&mlp, lr0, 0.9, 0.999, 1e-8);

    let train_frames: Vec<TrainingFrame> = train_r.iter().map(to_training_frame).collect();
    let (inputs, targets) = assemble_dataset_pairs(&train_frames);
    eprintln!("[rdirect] train pairs: {} (config {:?}, macs/pixel={})", inputs.len(), config, mlp.macs());

    for epoch in 0..epochs {
        // harmonic lr-decay (NRC lesson): lr0 / (1 + 0.5·epoch_frac)
        let frac = epoch as f32 / epochs.max(1) as f32;
        adam.set_lr(lr0 / (1.0 + 2.0 * frac));
        let loss = train_epoch_prepared(&mut mlp, &mut adam, &inputs, &targets, batch);
        if epoch % 20 == 0 || epoch + 1 == epochs {
            println!("[rdirect] epoch {epoch}/{epochs} train_mse(out-space)={loss:.6}");
        }
    }

    // ── evaluate: RMSE vs reference (net / chain / bilinear-of-1spp) ─────────
    println!("\n[rdirect] === GATE (a) QUALITY — RMSE vs converged reference ===");
    println!("{:<12} {:>10} {:>10} {:>10} {:>12}", "pose", "bilinear", "chain", "net", "net<chain?");
    let mut worst_net_val = 0.0f64;
    let mut all_net_beats_chain = true;
    let mut proof_rows: Vec<(String, Vec<GVec3>, Vec<GVec3>, Vec<GVec3>, Vec<GVec3>)> = Vec::new();

    let eval = |label: &str, rr: &Rendered, worst: &mut f64, all: &mut bool,
                rows: &mut Vec<(String, Vec<GVec3>, Vec<GVec3>, Vec<GVec3>, Vec<GVec3>)>, is_val: bool| {
        let bilinear = bilinear_upsample(&rr.low_noisy, rr.low_w, rr.low_h, rr.target_w, rr.target_h);
        let chain = chain_output(rr, &denoiser, &upscaler);
        let net = net_output(rr, &mlp);
        let e_bil = rmse(&bilinear, &rr.reference);
        let e_chain = rmse(&chain, &rr.reference);
        let e_net = rmse(&net, &rr.reference);
        let net_beats = e_net < e_chain;
        println!("{:<12} {:>10.6} {:>10.6} {:>10.6} {:>12}", label, e_bil, e_chain, e_net,
            if net_beats { "yes" } else { "NO" });
        if is_val {
            *worst = worst.max(e_net);
            if !net_beats { *all = false; }
        }
        rows.push((label.to_string(), bilinear, net, rr.reference.clone(), chain));
    };

    for rr in &val_r {
        eval(rr.name, rr, &mut worst_net_val, &mut all_net_beats_chain, &mut proof_rows, true);
    }
    println!("[rdirect] --- generalization: held-out SCENE EDIT (block displaced) ---");
    eval("front_edit", &edit_r, &mut worst_net_val, &mut all_net_beats_chain, &mut proof_rows, true);
    println!("[rdirect] --- train poses (informational) ---");
    for rr in &train_r {
        eval(rr.name, rr, &mut worst_net_val, &mut all_net_beats_chain, &mut proof_rows, false);
    }

    // ── cost accounting ─────────────────────────────────────────────────────
    println!("\n[rdirect] === GATE (b) COST ===");
    let den_macs = 10u64 * 32 + 32 * 32 * 3 + 32 * 3; // VIII-1 denoiser
    let up_macs = 21u64 * 64 + 64 * 64 * 3 + 64 * 3; // VIII-3 upscaler
    // chain per-native-pixel: denoiser runs at low res (1/scale² native px) + upscaler at native
    let chain_macs_per_native = den_macs / (UPSCALE_SCALE * UPSCALE_SCALE) as u64 + up_macs;
    let net_macs = mlp.macs();
    println!("[rdirect] MAC/native-pixel: chain(denoise@low + upscale@native)={chain_macs_per_native}  net={net_macs}  ratio={:.2}x",
        net_macs as f64 / chain_macs_per_native as f64);

    // CPU wall-clock, median of 5, on the largest val pose (native res).
    let bench = &val_r[0];
    let mut t_net = Vec::new();
    let mut t_chain = Vec::new();
    for _ in 0..5 {
        let s = Instant::now(); let _ = net_output(bench, &mlp); t_net.push(s.elapsed().as_secs_f64() * 1e3);
        let s = Instant::now(); let _ = chain_output(bench, &denoiser, &upscaler); t_chain.push(s.elapsed().as_secs_f64() * 1e3);
    }
    t_net.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_chain.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("[rdirect] CPU ref inference @ {}x{}: net={:.2}ms  chain={:.2}ms (median of 5, CPU f32 reference — NOT the GPU number)",
        target_w, target_h, t_net[2], t_chain[2]);
    // GPU projection anchored to VIII-2's measured 26.5ms @ 900×600 for 3488 MACs.
    let px_900x600 = 900.0 * 600.0;
    let px_native = (target_w * target_h) as f64;
    let ms_per_mac_px = 26.5 / (3488.0 * px_900x600);
    let net_gpu_900 = ms_per_mac_px * net_macs as f64 * px_900x600;
    let chain_gpu_900 = ms_per_mac_px * chain_macs_per_native as f64 * px_900x600;
    println!("[rdirect] GPU PROJECTION (naive-port scaling from VIII-2's 26.5ms/3488-MAC/540kpx, UNVERIFIED extrapolation):");
    println!("[rdirect]   @900x600: net~{net_gpu_900:.1}ms  chain~{chain_gpu_900:.1}ms  (60fps budget=16.67ms; both need the fp16/subgroup opt VIII-2 flagged)");
    let _ = (px_native,);

    // ── verdict summary ─────────────────────────────────────────────────────
    println!("\n[rdirect] === VERDICT INPUTS ===");
    println!("[rdirect] net beats chain on EVERY held-out pose+edit: {}", if all_net_beats_chain { "YES" } else { "NO" });
    println!("[rdirect] worst held-out net RMSE (pinned bound) = {worst_net_val:.6}");

    // ── write weights + provenance ──────────────────────────────────────────
    let wbytes = serialize_weights(&mlp);
    let wpath = data_dir.join("rdirect-weights-v1.bin");
    std::fs::write(&wpath, &wbytes).unwrap();
    let wsha = weights_sha256(&mlp);
    println!("[rdirect] wrote {} ({} bytes) sha256={wsha}", wpath.display(), wbytes.len());

    let prov = serde_json::json!({
        "artifact": "rdirect-weights-v1.bin",
        "weights_sha256": wsha,
        "architecture": {
            "kind": "direct-render per-target-pixel MLP (denoise+upscale fused, absolute output)",
            "input_features": scrying_glass::rdirect::INPUT_FEATURES,
            "feature_layout": "2x2 low-res 1spp demod-log radiance taps(12) + subpixel(2) + hi-res albedo(3) + normal(3) + log-depth(1) + motion(2, zero in static dataset)",
            "output_channels": scrying_glass::rdirect::OUTPUT_CHANNELS,
            "hidden_layers": config.hidden_layers,
            "hidden_width": config.hidden_width,
            "macs_per_pixel": net_macs,
        },
        "training": { "epochs": epochs, "batch": batch, "lr0": lr0, "lr_schedule": "harmonic decay lr0/(1+2*frac)", "init_seed": INIT_SEED, "optimizer": "adam" },
        "dataset": { "realm": "naruko", "low": [low_w, low_h], "native": [target_w, target_h], "scale": UPSCALE_SCALE, "ref_frames": ref_frames,
            "train": ["front", "wide", "orbit_+20"], "val": ["orbit_-20", "orbit_+40"], "scene_edit_gate": "front_edit (block displaced)" },
        "gate_b_cost": { "chain_mac_per_native_px": chain_macs_per_native, "net_mac_per_native_px": net_macs,
            "cpu_ref_ms_net": t_net[2], "cpu_ref_ms_chain": t_chain[2] },
        "pinned_bound": { "description": "worst held-out (val+edit) net RMSE vs reference", "value": worst_net_val },
        "net_beats_chain_all_heldout": all_net_beats_chain,
    });
    let ppath = data_dir.join("rdirect-weights-v1.provenance.json");
    std::fs::write(&ppath, serde_json::to_string_pretty(&prov).unwrap()).unwrap();
    println!("[rdirect] wrote {}", ppath.display());

    // ── proof PNGs (quad: 1spp | net | truth | chain) ───────────────────────
    let exposure = env_f32("GAIA_RDIRECT_EXPOSURE", 1.6);
    let proof_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");
    for (name, bil, net, truth, chain) in &proof_rows {
        let path = proof_dir.join(format!("rdirect-{name}.png"));
        write_quad([bil, net, truth, chain], target_w, target_h, exposure, &path);
    }
    println!("[rdirect] proof quads (1spp-input | net | ground-truth | chain) written to proof/rdirect-*.png");
}
