//! RITE VIII-1 — THE DREAM-DENOISER: the ordeals. See
//! docs/proposals/RITE-VIII-THE-DREAM-DENOISER.md §VIII-1.
//!
//!   (a) inference byte-identical — same frame twice, cold.
//!   (b) denoised RMSE ≤ pinned bound (plus a small DERIVED nondeterminism
//!       margin — see `measure_render_nondeterminism_margin`) on every
//!       validation frame — the bound ships WITH the committed weights (its
//!       provenance sidecar), derived at train time, NEVER chosen here;
//!       this ordeal replays validation deterministically (re-renders the
//!       SAME two held-out poses `viii1_train.rs` used, via the shared
//!       `denoiser_dataset` module) and checks the committed number.
//!   (c) denoised strictly beats noisy on every validation frame (the raw
//!       pinned bound, no margin — this is the quality signal and stays
//!       exact).
//!   (d) THE BAN: `src/denoiser.rs` is picked up by the VIII-0 grep-gate's
//!       forward-proof scope mechanism (confirmed directly, not assumed) +
//!       EVERY `pub fn` signature in `src/denoiser.rs` is scanned for a
//!       fixed list of forbidden parameter-name substrings, not just
//!       `denoise_image` — a second entry point cannot slip past unexamined.
//!       Stated precisely (adversary finding A1, night-2 review): this is a
//!       real, repeatable check against a fixed vocabulary/parameter-name
//!       list, not a proof that no temporal state could ever be smuggled in
//!       under an unlisted name.
//!   (e) senses unchanged — a REAL pipeline seam (adversary finding M1,
//!       night-2 review; the first version of this ordeal captured its
//!       "truth" hash BEFORE the denoiser ran and never used the result, so
//!       passing it proved nothing): render → optionally denoise the
//!       presentation image → THEN gaze the world/AOV truth from the SAME
//!       live (bvh, camera, sun, device, queue) the denoise step had
//!       access to. Two assertions: (i) the gaze hash, captured AFTER the
//!       toggle point, is byte-equal denoiser-on vs denoiser-off; (ii) the
//!       presentation images DIFFER on vs off (the discriminating half —
//!       proves the toggle actually toggles, so (i) isn't comparing two
//!       identical no-op paths).
//!   (A4) the committed weights bytes hash-pin: sha256(committed .bin) ==
//!       provenance.weights_sha256, actually asserted (not just printed).
//!
//! All GPU ordeals print + return early (never a false green) on a host
//! without an adapter, matching `viii0_ordeals.rs`'s convention.

use std::fs;
use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::{Mlp, denoise_image, deserialize_weights, sha256_hex};
use scrying_glass::denoiser_dataset::{
    DATASET_HEIGHT, DATASET_REF_FRAMES, DATASET_WIDTH, VALIDATION_POSE_NAMES, law_poses,
    naruko_params,
};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene, SunLight};

/// The same small fixed pose `viii0_ordeals.rs` uses — fast, non-vacuous
/// (the triangle fills a good part of the frame), no full-realm load.
fn fixed_pose() -> (Bvh, Camera, SunLight, [f32; 4], [f32; 4], u32, u32) {
    let tri = LeafTriangle {
        positions: [[-4.0, -2.0, -8.0], [4.0, -2.0, -8.0], [0.0, 4.0, -8.0]],
        albedo: [0.7, 0.4, 0.2],
        emission: [0.0, 0.0, 0.0],
        metallic: 0.0,
        roughness: 0.6,
    };
    let bvh = Bvh::build(&[tri], &BvhParams::default());
    let camera = Camera {
        eye: GVec3::new(0.0, 0.0, 2.0),
        yaw: 0.0,
        pitch: 0.0,
        fov_y_radians: 50f32.to_radians(),
        near: 0.05,
        far: 1000.0,
    };
    let sun = SunLight {
        direction: GVec3::new(0.3, 1.0, 0.4).normalize().to_array(),
        color: [1.0, 0.95, 0.85],
        intensity: 2.0,
        ambient_intensity: 0.15,
    };
    let sky_top = [0.1, 0.12, 0.22, 1.0];
    let sky_horizon = [0.55, 0.42, 0.5, 1.0];
    (bvh, camera, sun, sky_top, sky_horizon, 64, 48)
}

fn weights_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data")
}

fn read_committed_weights_bytes() -> Vec<u8> {
    fs::read(weights_dir().join("denoiser-weights-v1.bin")).expect(
        "read committed denoiser-weights-v1.bin — run `cargo run -p scrying-glass --release \
         --example viii1_train` to forge it first",
    )
}

fn load_committed_weights() -> Mlp {
    deserialize_weights(&read_committed_weights_bytes())
        .expect("deserialize committed weights artifact")
}

fn load_provenance() -> serde_json::Value {
    let text = fs::read_to_string(weights_dir().join("denoiser-weights-v1.provenance.json"))
        .expect("read committed provenance sidecar");
    serde_json::from_str(&text).expect("parse provenance JSON")
}

#[test]
fn a_inference_is_byte_identical_same_frame_twice_cold() {
    let Some((device, queue)) = headless_device() else {
        eprintln!(
            "[VIII-1 inference determinism] no GPU adapter on this host — ordeal could not run"
        );
        return;
    };
    let (bvh, camera, sun, sky_top, sky_horizon, w, h) = fixed_pose();
    let noisy_params = IntegratorParams {
        spp: 1,
        ..IntegratorParams::default()
    };
    let noisy = resolve(&trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
        &sun,
        sky_top,
        sky_horizon,
        w,
        h,
        1,
        &noisy_params,
        None,
    ));
    let raw_aov = trace_headless_aov(
        &device,
        &queue,
        &bvh,
        &camera,
        &sun,
        sky_top,
        sky_horizon,
        w,
        h,
    );
    let (albedo, normal, depth) = split_aov(&raw_aov);

    let mlp = load_committed_weights();
    // Two entirely independent (cold) inference calls over the SAME inputs.
    let a = denoise_image(&mlp, &noisy, &albedo, &normal, &depth);
    let b = denoise_image(&mlp, &noisy, &albedo, &normal, &depth);
    assert_eq!(
        a, b,
        "denoiser inference is not byte-identical across two cold runs on the same frame"
    );
}

/// (A4) The committed weights bytes must actually hash to the sha256 pinned
/// in the committed provenance sidecar — an asserted check, not merely a
/// number the training script printed once and everyone trusted.
#[test]
fn a4_committed_weights_bytes_match_pinned_provenance_sha256() {
    let bytes = read_committed_weights_bytes();
    let provenance = load_provenance();
    let expected = provenance["weights_sha256"]
        .as_str()
        .expect("weights_sha256 string in provenance JSON");
    let actual = sha256_hex(&bytes);
    assert_eq!(
        actual, expected,
        "committed denoiser-weights-v1.bin does not hash to the sha256 pinned in its provenance sidecar — \
         the artifact and its provenance have drifted apart"
    );
}

/// Build the naruko realm scene + static BVH once (shared by validation
/// replay and the nondeterminism-margin measurement below) — pose/scene
/// definitions themselves come from `denoiser_dataset` (adversary finding
/// A5: ONE shared source with `viii1_train.rs`, never duplicated here).
fn build_naruko_scene() -> (Bvh, RenderScene) {
    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());
    (bvh, scene)
}

/// One validation pose's rendered buffers: (name, noisy, albedo, normal,
/// depth, reference).
type ValidationPose = (
    &'static str,
    Vec<GVec3>,
    Vec<GVec3>,
    Vec<GVec3>,
    Vec<f32>,
    Vec<GVec3>,
);

/// Re-render the SAME two held-out validation poses `examples/viii1_train.rs`
/// used (`denoiser_dataset::VALIDATION_POSE_NAMES` — see that module's docs
/// for the full documented dataset scope), from the same naruko realm, same
/// fixed seed/params, so the pinned bound can be replayed deterministically
/// here.
fn render_validation_poses(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    scene: &RenderScene,
) -> Vec<ValidationPose> {
    let params = naruko_params();
    let (w, h) = (DATASET_WIDTH, DATASET_HEIGHT);
    let ref_frames = DATASET_REF_FRAMES;

    law_poses(&params)
        .into_iter()
        .filter(|(name, _)| VALIDATION_POSE_NAMES.contains(name))
        .map(|(name, camera)| {
            let noisy_params = IntegratorParams {
                spp: 1,
                ..IntegratorParams::default()
            };
            let noisy = resolve(&trace_headless(
                device,
                queue,
                bvh,
                &camera,
                &scene.sun,
                scene.sky_top,
                scene.sky_horizon,
                w,
                h,
                1,
                &noisy_params,
                None,
            ));
            let reference = resolve(&trace_headless(
                device,
                queue,
                bvh,
                &camera,
                &scene.sun,
                scene.sky_top,
                scene.sky_horizon,
                w,
                h,
                ref_frames,
                &IntegratorParams::default(),
                None,
            ));
            let raw_aov = trace_headless_aov(
                device,
                queue,
                bvh,
                &camera,
                &scene.sun,
                scene.sky_top,
                scene.sky_horizon,
                w,
                h,
            );
            let (albedo, normal, depth) = split_aov(&raw_aov);
            (name, noisy, albedo, normal, depth, reference)
        })
        .collect()
}

/// (A2) Derive a small tolerance margin from a REAL measurement rather than
/// asserting the pinned bound with zero slack (a zero-margin bound is
/// tautological the instant the renderer has ANY nondeterminism, however
/// tiny — adversary finding A2, night-2 review). Re-renders one validation
/// pose's reference `REPEATS` times at the SAME (seed, coords) and takes
/// the worst pairwise relative RMSE spread as the measured floor; `m` = 10×
/// that spread (a derived-not-chosen safety gate, per this atom's "gate
/// ~10×" convention). If the renderer is bit-exact across repeats (the
/// expected case — see ordeal (a) and VIII-0's own reference-
/// reproducibility ordeal), the spread is exactly zero and `m` falls back
/// to a float-epsilon-scaled allowance derived from this frame's total
/// float-channel count, guarding against any future non-bit-exact backend
/// without ever being a chosen magic number.
fn measure_render_nondeterminism_margin(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &Camera,
    scene: &RenderScene,
) -> f64 {
    const REPEATS: usize = 3;
    const MARGIN_GATE_FACTOR: f64 = 10.0;
    let (w, h) = (DATASET_WIDTH, DATASET_HEIGHT);
    // A smaller frame count than the full dataset reference (16 vs 128) —
    // this measurement only needs to detect whether repeats diverge AT
    // ALL, not reproduce proof-quality convergence.
    let measure_frames = 16u32;
    let params = IntegratorParams::default();

    let renders: Vec<Vec<GVec3>> = (0..REPEATS)
        .map(|_| {
            resolve(&trace_headless(
                device,
                queue,
                bvh,
                camera,
                &scene.sun,
                scene.sky_top,
                scene.sky_horizon,
                w,
                h,
                measure_frames,
                &params,
                None,
            ))
        })
        .collect();

    let zero = vec![GVec3::ZERO; renders[0].len()];
    let magnitude = rmse(&renders[0], &zero).max(1e-12);

    let mut max_relative_spread = 0.0f64;
    for i in 0..renders.len() {
        for j in (i + 1)..renders.len() {
            let spread = rmse(&renders[i], &renders[j]) / magnitude;
            max_relative_spread = max_relative_spread.max(spread);
        }
    }

    let margin = if max_relative_spread > 0.0 {
        MARGIN_GATE_FACTOR * max_relative_spread
    } else {
        (f32::EPSILON as f64) * (w * h * 3) as f64
    };
    println!(
        "[VIII-1 margin] {REPEATS} repeat renders, max_relative_spread={max_relative_spread:.3e}, derived margin m={margin:.3e}"
    );
    margin
}

#[test]
fn b_and_c_denoised_beats_pinned_bound_and_beats_noisy_on_every_validation_frame() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[VIII-1 validation replay] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let mlp = load_committed_weights();
    let provenance = load_provenance();
    let pinned_bound = provenance["pinned_bound"]["value"]
        .as_f64()
        .expect("pinned_bound.value in provenance JSON");

    let (bvh, scene) = build_naruko_scene();
    let val_frames = render_validation_poses(&device, &queue, &bvh, &scene);
    assert_eq!(
        val_frames.len(),
        2,
        "expected exactly the 2 documented validation poses"
    );

    // (A2) derived margin, measured once against the first validation
    // pose's camera (the margin characterizes RENDERER nondeterminism, not
    // a per-pose quantity — one measurement covers the replay below).
    let margin = measure_render_nondeterminism_margin(
        &device,
        &queue,
        &bvh,
        &law_poses(&naruko_params())[3].1,
        &scene,
    );
    let bound_with_margin = pinned_bound * (1.0 + margin);
    println!(
        "[VIII-1 validation replay] pinned_bound={pinned_bound:.6} margin={margin:.3e} bound_with_margin={bound_with_margin:.6}"
    );

    for (name, noisy, albedo, normal, depth, reference) in val_frames {
        let denoised = denoise_image(&mlp, &noisy, &albedo, &normal, &depth);
        let noisy_rmse = rmse(&noisy, &reference);
        let denoised_rmse = rmse(&denoised, &reference);
        println!(
            "[VIII-1 validation replay] pose={name} noisy_rmse={noisy_rmse:.6} denoised_rmse={denoised_rmse:.6}"
        );
        // (b) derived, pinned bound (plus the derived margin) — never
        // chosen here.
        assert!(
            denoised_rmse <= bound_with_margin,
            "pose '{name}': denoised RMSE {denoised_rmse:.6} exceeds the pinned bound-with-margin \
             {bound_with_margin:.6} (pinned={pinned_bound:.6}, margin={margin:.3e})"
        );
        // (c) strictly beats noisy, against the RAW pinned bound's own
        // comparison target (noisy) — the quality signal, no margin.
        assert!(
            denoised_rmse < noisy_rmse,
            "pose '{name}': denoised RMSE {denoised_rmse:.6} does not beat noisy RMSE {noisy_rmse:.6}"
        );
    }
}

/// (d) THE BAN, part 1: confirm `src/denoiser.rs` is actually picked up by
/// the VIII-0 grep-gate's forward-proof scope mechanism (glob
/// `src/denoiser*.rs` OR `// BAN-SCOPED` marker) — a real check, not an
/// assumption that the marker "must" work.
#[test]
fn d_ban_denoiser_module_is_picked_up_by_the_forward_proof_grep_gate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("src/denoiser.rs");
    let text = fs::read_to_string(&path).expect("read src/denoiser.rs");
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let is_denoiser_glob = name.starts_with("denoiser") && name.ends_with(".rs");
    let is_marked = text.contains("// BAN-SCOPED");
    assert!(
        is_denoiser_glob && is_marked,
        "src/denoiser.rs must match the denoiser*.rs glob AND carry the // BAN-SCOPED marker so \
         the VIII-0 grep-gate's forward-proof scope mechanism picks it up"
    );

    // Direct re-check of the same forbidden vocabulary the VIII-0 gate uses
    // (kept in sync manually; the canonical gate itself lives in
    // viii0_ordeals.rs and already scans this file via the mechanism above —
    // this is a second, redundant witness that the module text itself is
    // clean, not a replacement for that gate).
    let forbidden = [
        "previous_frame",
        "history",
        "motion_vector",
        "temporal",
        "reproject",
        "warp",
        "feedback",
        "recurrent",
        "accum_prev",
        "prev_",
        "last_frame",
        "frame_history",
        "velocity",
    ];
    for word in forbidden {
        assert!(
            !text.to_lowercase().contains(word),
            "forbidden temporal vocabulary '{word}' found in src/denoiser.rs"
        );
    }
}

/// Every `pub fn NAME(...) -> ... {` signature found in `text`, as
/// (name, full signature text INCLUDING the return-type arrow, EXCLUDING
/// the opening `{`). A plain textual scan — fine for this purpose (the
/// file has no nested-brace generic bounds in any signature, confirmed by
/// direct read), not a real Rust parser.
fn public_fn_signatures(text: &str) -> Vec<(String, String)> {
    let mut sigs = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find("pub fn ") {
        let start = search_from + rel;
        let name_start = start + "pub fn ".len();
        let name_end = text[name_start..]
            .find('(')
            .map(|e| name_start + e)
            .unwrap_or(name_start);
        let name = text[name_start..name_end].trim().to_string();
        let body_start = text[start..]
            .find('{')
            .map(|e| start + e)
            .unwrap_or(text.len());
        sigs.push((name, text[start..body_start].to_string()));
        search_from = (body_start + 1).max(start + 1);
    }
    sigs
}

/// (d) THE BAN, part 2 (adversary finding A1: extended from `denoise_image`
/// only to EVERY `pub fn` in the module, so a second entry point cannot
/// slip past unexamined): each public function's signature (grepped
/// directly) takes current-frame buffers only — no frame-index/history
/// parameter anywhere, checked against a fixed forbidden-substring list.
#[test]
fn d_ban_every_public_fn_signature_in_denoiser_takes_current_frame_inputs_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let text = fs::read_to_string(root.join("src/denoiser.rs")).expect("read src/denoiser.rs");
    let sigs = public_fn_signatures(&text);
    assert!(
        sigs.len() >= 10,
        "expected at least 10 `pub fn`s in src/denoiser.rs (found {}) — the textual scan itself may be broken",
        sigs.len()
    );
    let forbidden_params = [
        "frames",
        "frame_index",
        "prev",
        "history",
        "samples_before",
        "last_",
    ];
    for (name, signature) in &sigs {
        for forbidden in forbidden_params {
            assert!(
                !signature.to_lowercase().contains(forbidden),
                "denoiser.rs `pub fn {name}` signature contains '{forbidden}' — every public entry \
                 point must take current-frame buffers only"
            );
        }
    }
}

/// (e) senses unchanged — see module docs for the full rationale (adversary
/// finding M1, night-2 review). The minimal honest pipeline seam: render →
/// optionally denoise the presentation image → THEN gaze the world/AOV
/// truth from the SAME live (bvh, camera, sun, device, queue) the denoise
/// step had access to. Returns (presentation radiance, world-truth hash
/// bytes captured AFTER the toggle point).
#[allow(clippy::too_many_arguments)]
fn render_present_then_gaze(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &Camera,
    sun: &SunLight,
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    w: u32,
    h: u32,
    mlp: Option<&Mlp>,
) -> (Vec<GVec3>, Vec<u8>) {
    let noisy_params = IntegratorParams {
        spp: 1,
        ..IntegratorParams::default()
    };
    let noisy = resolve(&trace_headless(
        device,
        queue,
        bvh,
        camera,
        sun,
        sky_top,
        sky_horizon,
        w,
        h,
        1,
        &noisy_params,
        None,
    ));
    let raw_aov = trace_headless_aov(device, queue, bvh, camera, sun, sky_top, sky_horizon, w, h);
    let (albedo, normal, depth) = split_aov(&raw_aov);

    // THE TOGGLE POINT: presentation = denoised (ON) or raw noisy (OFF).
    let presentation = match mlp {
        Some(mlp) => denoise_image(mlp, &noisy, &albedo, &normal, &depth),
        None => noisy,
    };

    // THE GAZE, always AFTER the toggle point, from the SAME live (bvh,
    // camera, sun, device, queue) the denoiser step just had access to (it
    // took none of them as arguments — this independent fresh AOV read is
    // what proves it didn't reach for them another way): the world's own
    // truth (ruling 5, senses read solver truth).
    let gaze_aov = trace_headless_aov(device, queue, bvh, camera, sun, sky_top, sky_horizon, w, h);
    let mut truth_bytes = Vec::new();
    for c in &gaze_aov {
        truth_bytes.extend_from_slice(bytemuck::cast_slice(c));
    }
    (presentation, truth_bytes)
}

#[test]
fn e_senses_unchanged_oracle_sha_equal_pass_on_or_off() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[VIII-1 senses unchanged] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let (bvh, camera, sun, sky_top, sky_horizon, w, h) = fixed_pose();
    let mlp = load_committed_weights();

    let (presentation_off, truth_off) = render_present_then_gaze(
        &device,
        &queue,
        &bvh,
        &camera,
        &sun,
        sky_top,
        sky_horizon,
        w,
        h,
        None,
    );
    let (presentation_on, truth_on) = render_present_then_gaze(
        &device,
        &queue,
        &bvh,
        &camera,
        &sun,
        sky_top,
        sky_horizon,
        w,
        h,
        Some(&mlp),
    );

    // (ii) the discriminating half, checked FIRST: the toggle must
    // actually toggle, or (i) below would vacuously compare two identical
    // no-op paths.
    assert_ne!(
        presentation_off, presentation_on,
        "presentation image is IDENTICAL with the denoiser on vs off — this ordeal cannot \
         discriminate a real toggle from two no-op paths"
    );

    // (i) the world/gaze truth, captured AFTER the toggle point in BOTH
    // branches, is byte-equal.
    let hash_off = sha256_hex(&truth_off);
    let hash_on = sha256_hex(&truth_on);
    assert_eq!(
        hash_off, hash_on,
        "the world's own gaze truth (read AFTER the denoise step ran) differs between denoiser \
         on and off — the net must filter pixels, never the world"
    );
}
