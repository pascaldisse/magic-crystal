//! RITE VIII-0 — THE NOISE AND THE TRUTH: the ordeals for the AOV export +
//! reference-oracle wave (before any net lands). See
//! docs/proposals/RITE-VIII-THE-DREAM-DENOISER.md §VIII-0.
//!
//!   (a) AOV determinism — two cold runs, same seed/pose, byte-identical.
//!   (b) Reference reproducibility — same (seed, pose, N) → byte-identical
//!       resolved reference, run twice.
//!   (c) Error-metric self-test discrimination at the integration boundary
//!       (the exact-zero self-test itself lives as a unit test beside the
//!       metric in `src/error_metric.rs` — this ordeal proves it against a
//!       REAL resolved GPU image, not a hand-built Vec3 array).
//!   (d) AOV-off bit-identical to pre-change rendering.
//!   (5) THE BAN grep-gate — no temporal vocabulary in the new AOV/error
//!       module, and the AOV export signature takes current-frame inputs
//!       only.
//!
//! All GPU ordeals print + return early (never a false green) on a host
//! without an adapter, matching the existing `medium_parity.rs` convention.

use std::fs;
use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::error_metric::{mae, rmse};
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, LeafTriangle, SunLight};

/// The fixed law pose shared by every VIII-0 ordeal: a small scene (one
/// triangle, so the primary hit AOVs are non-trivial — not an empty-BVH sky
/// shot) under a fixed camera and sun. Deliberately NOT the full Naruko realm
/// (that lives in `examples/viii0_truth.rs`, which is the proof render, not
/// an ordeal) — ordeals stay fast so the suite stays fast.
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

#[test]
fn aov_export_is_deterministic_cold_run() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[VIII-0 AOV determinism] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let (bvh, camera, sun, sky_top, sky_horizon, w, h) = fixed_pose();

    let a = trace_headless_aov(
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
    let b = trace_headless_aov(
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

    assert_eq!(a.len(), b.len());
    assert_eq!(
        a, b,
        "AOV export is not deterministic across two cold runs at the same seed/pose"
    );
    // Discriminating: the triangle fills a good part of the frame, so this is
    // not a vacuously-all-miss (all-zero) buffer.
    let any_hit = a.iter().step_by(2).any(|c| c[3] > 0.0);
    assert!(
        any_hit,
        "fixed pose has no primary hits — ordeal is vacuous"
    );
}

#[test]
fn reference_accumulation_is_reproducible() {
    let Some((device, queue)) = headless_device() else {
        eprintln!(
            "[VIII-0 reference reproducibility] no GPU adapter on this host — ordeal could not run"
        );
        return;
    };
    let (bvh, camera, sun, sky_top, sky_horizon, w, h) = fixed_pose();
    let params = IntegratorParams {
        seed: 0x5eed,
        ..IntegratorParams::default()
    };
    let n = 16u32;

    let a = trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
        &sun,
        sky_top,
        sky_horizon,
        w,
        h,
        n,
        &params,
        None,
    );
    let b = trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
        &sun,
        sky_top,
        sky_horizon,
        w,
        h,
        n,
        &params,
        None,
    );
    assert_eq!(
        a, b,
        "reference accumulation is not byte-identical from (seed, coords) alone, run twice"
    );

    // The resolved image (not just the raw accum, which trivially matches
    // since `a == b`) is also self-error-zero — this is ordeal (c): the
    // metric's zero-self-test against a REAL resolved GPU image.
    let ra = resolve(&a);
    let rb = resolve(&b);
    assert_eq!(rmse(&ra, &rb), 0.0f64, "rmse(ref, ref) must be exactly 0e0");
    assert_eq!(mae(&ra, &rb), 0.0f64, "mae(ref, ref) must be exactly 0e0");
}

#[test]
fn error_metric_discriminates_a_real_gpu_difference() {
    let Some((device, queue)) = headless_device() else {
        eprintln!(
            "[VIII-0 error metric discrimination] no GPU adapter on this host — ordeal could not run"
        );
        return;
    };
    let (bvh, camera, sun, sky_top, sky_horizon, w, h) = fixed_pose();
    let noisy_params = IntegratorParams {
        seed: 0x5eed,
        spp: 1,
        ..IntegratorParams::default()
    };
    let ref_params = IntegratorParams {
        seed: 0x5eed,
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
    let reference = resolve(&trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
        &sun,
        sky_top,
        sky_horizon,
        w,
        h,
        64,
        &ref_params,
        None,
    ));
    let e = rmse(&noisy, &reference);
    assert!(
        e > 0.0,
        "1-spp noisy image scored exactly 0 rmse against the converged reference — the metric \
         degenerated to a trivial always-zero case, or the scene has no variance to discriminate"
    );
}

/// (d) AOV-off bit-identical to pre-change rendering.
///
/// HOW THE GOLDEN HASH WAS DERIVED (honest account): before touching
/// `integrator.rs`/`integrator.wgsl` for VIII-0, at commit b3beb17 (the
/// worktree's base, tree clean), a throwaway example
/// (`examples/_tmp_golden_hash.rs`, since deleted) called the UNMODIFIED
/// `trace_headless` with this exact scene/params and printed an FNV-1a 64
/// hash of the raw `Vec<[f32;4]>` accum bytes:
///   `cargo run -p scrying-glass --release --example _tmp_golden_hash`
///   → GOLDEN_HASH_HEX = 0xff1f4d5a29704621
/// The AOV export landed afterward as a SEPARATE pipeline/bind-group-layout
/// (@group(1)); `trace_headless`, `Integrator::compute_pipeline`, and
/// `compute_layout` were not edited (see the "VIII-0 AOV EXPORT" markers in
/// integrator.rs/.wgsl — everything outside them is byte-identical to
/// b3beb17). This ordeal re-runs the SAME scene against the CURRENT code
/// (AOV export never invoked) and checks the hash still matches — a real,
/// verified comparison, not an assumed one.
#[test]
fn aov_off_matches_pre_atom_golden_hash() {
    let Some((device, queue)) = headless_device() else {
        eprintln!(
            "[VIII-0 AOV-off golden hash] no GPU adapter on this host — ordeal could not run"
        );
        return;
    };
    let tris: Vec<LeafTriangle> = Vec::new();
    let bvh = Bvh::build(&tris, &BvhParams::default());
    let camera = Camera {
        eye: GVec3::new(0.0, 2.0, 22.0),
        yaw: 0.0,
        pitch: 0.0,
        fov_y_radians: 60f32.to_radians(),
        near: 0.1,
        far: 4000.0,
    };
    let sun = SunLight {
        direction: [0.3, 1.0, 0.4],
        color: [1.0, 0.95, 0.85],
        intensity: 1.1,
        ambient_intensity: 0.32,
    };
    let params = IntegratorParams::default();
    let accum = trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
        &sun,
        [0.1, 0.1, 0.2, 1.0],
        [0.6, 0.4, 0.5, 1.0],
        64,
        48,
        4,
        &params,
        None,
    );
    let bytes = bytemuck::cast_slice::<[f32; 4], u8>(&accum);
    let hash = fnv1a64(bytes);
    const GOLDEN: u64 = 0xff1f4d5a29704621;
    assert_eq!(
        hash, GOLDEN,
        "AOV-off rendering diverged from the pre-VIII-0 golden hash (derived at b3beb17) — the \
         AOV dial being off must not perturb the existing accum image at all"
    );
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// THE BAN grep-gate (VIII-0 plants it, VIII-1 extends it): the NEW VIII-0
/// module(s) — `src/error_metric.rs` in full, and the "VIII-0 AOV EXPORT"
/// marked block inside `src/integrator.rs` and `src/integrator.wgsl` — must
/// carry no forbidden temporal vocabulary. Scope choice, stated honestly:
/// the WHOLE files `integrator.rs`/`integrator.wgsl` are NOT new (they
/// predate this rite and are full of legitimate pre-existing accumulation
/// code — `samples_before`, `accum`, multi-frame accumulation for the MAIN
/// path — none of which is temporal-frame-SYNTHESIS and none of which this
/// gate should flag), so the gate is scoped to the literal text between the
/// `VIII-0 AOV EXPORT BEGIN`/`END` markers this atom introduced, which is
/// grep-able and exhaustive over every line this wave added to those files.
///
/// VOCABULARY, NARROWED (monad ruling 07-21, whip 219 merge — supersedes the
/// night-2 widen below): the 07-18 ban intent (main.rs `THE DESIGN IS THE
/// LAW` comment, NEURAL.md ‘hand-heuristics (gates/clamps/thresholds) never
/// ship’) targets the DEAD SVGF/TAA-style hand-tuned reprojection-denoiser
/// (NR1/NR2, demoted to lab equipment 07-18) — motion-vector reprojection,
/// its accept/reject gates, and its blend-with-clamp accumulation — never
/// Pleroma's OWN learned act, which the ONE-RENDER LAW explicitly charters
/// to consume ‘temporal accumulation = substrate … its history input’
/// (NEURAL.md ★ THE ONE-RENDER LAW). The night-2 list banned bare generic
/// words (`history`, `temporal`, `reproject`, `recurrent`, `prev_`) that
/// Pleroma's own net module (`rdirect*.rs` — reprojected-history input
/// features, its recurrent split-net design, `prev_dl` demod state) uses
/// legitimately and constantly; scanning bare true would forbid Pleroma's
/// own charter. Narrowed to the literal heresy identifiers — the hand-
/// heuristic's own vocabulary, which has no legitimate reason to appear in
/// ANY ban-scoped file: the dead present-path names (`raw_accum`,
/// `raw-accum`, `reset_on_move`, `reset-on-move`), the dead alpha-blend gate
/// (`blend_alpha`, `blend-alpha`), its literal gate/clamp/threshold knobs
/// (`alpha_min`, `clamp_k`, `normal_tol`, `still_px` — `TemporalParams`'
/// own field names, integrator.rs; `depth_tol` excluded — Pleroma's own
/// recurrent-net reprojection validity guard reuses that exact disocclusion
/// primitive legitimately, rdirect.rs), and the original
/// unambiguous SVGF-family terms that stay zero-collision in Pleroma's own
/// vocabulary (`previous_frame`, `motion_vector`, `optical_flow`, `warp`,
/// `feedback`, `accum_prev`, `last_frame`, `frame_history`, `velocity`).
/// `history` is explicitly ALLOWED in BAN-SCOPED files as Pleroma's net
/// input-feature term (monad ruling 07-21); a genuine heresy term inserted
/// anywhere in a ban-scoped file (e.g. `raw_accum`, `motion_vector`) must
/// still fail this ordeal — proven by adversary sabotage-check, 07-21.
///
/// SCOPE IS FORWARD-PROOF, so VIII-1 does not require editing this test to
/// be covered: a file is "ban-scoped" (and therefore scanned whole) if
/// EITHER (a) its name matches the glob `src/denoiser*.rs` (a placeholder
/// for the VIII-1 net module), OR (b) it contains the literal marker
/// comment `// BAN-SCOPED` anywhere in its text. **THE VIII-1 MODULE MUST
/// CARRY A `// BAN-SCOPED` HEADER COMMENT (or be named `denoiser*.rs`) OR IT
/// WILL NOT BE CHECKED BY THIS GATE — this is not automatic from merely
/// adding a new file.** `src/error_metric.rs` itself carries the marker so
/// the forward-proof mechanism is exercised (not just declared) by this
/// very ordeal.
#[test]
fn ban_no_temporal_vocabulary_in_the_new_aov_error_module() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    // NARROWED 07-21 (monad ruling, whip 219 merge): the ban targets the DEAD
    // hand-tuned SVGF/TAA reprojection-denoiser (NR1/NR2, 07-18 lab-equipment
    // demotion) — never Pleroma's own learned act, which is chartered
    // (NEURAL.md ★ ONE-RENDER LAW) to take temporal accumulation/history as
    // its substrate. `history`/`temporal`/`reproject`/`recurrent`/`prev_`
    // dropped bare (Pleroma's own net-module vocabulary, `rdirect*.rs`);
    // replaced by the literal dead-heuristic identifiers below, which have
    // no legitimate reason to appear in any ban-scoped file.
    let forbidden = [
        "previous_frame",
        "motion_vector",
        "optical_flow",
        "reset_on_move",
        "reset-on-move",
        "raw_accum",
        "raw-accum",
        "blend_alpha",
        "blend-alpha",
        "alpha_min",
        "clamp_k",
        // depth_tol dropped (07-21): Pleroma's own recurrent-net reprojection
        // validity guard (rdirect.rs `direct_render_sequence_hist`) reuses
        // the identical disocclusion-gate concept legitimately — not a
        // collision with the dead heuristic, a shared/necessary primitive.
        "normal_tol",
        "still_px",
        "warp",
        "feedback",
        "accum_prev",
        "last_frame",
        "frame_history",
        "velocity",
    ];

    // Forward-proof scope: walk src/ once, collect every file that is
    // EITHER named `denoiser*.rs` OR carries a `// BAN-SCOPED` marker
    // comment anywhere in its text. A future VIII-1 module is captured
    // automatically the moment it adds that marker — no edit to this test
    // required.
    let src_dir = root.join("src");
    let mut ban_scoped_files: Vec<std::path::PathBuf> = Vec::new();
    let mut stack = vec![src_dir.clone()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("read src dir") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let is_denoiser_glob = name.starts_with("denoiser") && name.ends_with(".rs");
            let is_marked = fs::read_to_string(&path)
                .map(|t| t.contains("// BAN-SCOPED"))
                .unwrap_or(false);
            if is_denoiser_glob || is_marked {
                ban_scoped_files.push(path);
            }
        }
    }
    assert!(
        ban_scoped_files
            .iter()
            .any(|p| p.ends_with("error_metric.rs")),
        "forward-proof scope mechanism did not pick up src/error_metric.rs via its \
         `// BAN-SCOPED` marker — the scan itself is broken"
    );
    for path in &ban_scoped_files {
        let text = fs::read_to_string(path).expect("read ban-scoped module");
        for word in forbidden {
            assert!(
                !text.to_lowercase().contains(word),
                "forbidden temporal vocabulary '{word}' found in ban-scoped module {}",
                path.display()
            );
        }
    }

    // Marked-block scope: pre-existing files this atom only ADDED lines to
    // (see the honest scope rationale above) — these predate the BAN-SCOPED
    // marker convention, so they use the narrower BEGIN/END block markers
    // instead of being scanned whole.
    let marked_scope = ["src/integrator.rs", "src/integrator.wgsl"];
    for rel in marked_scope {
        let text = fs::read_to_string(root.join(rel)).expect("read pre-existing file");
        let begin = "VIII-0 AOV EXPORT BEGIN";
        let end = "VIII-0 AOV EXPORT END";
        let mut search_from = 0usize;
        let mut found_any_block = false;
        while let Some(b) = text[search_from..].find(begin) {
            let b_abs = search_from + b;
            let e_abs = text[b_abs..]
                .find(end)
                .map(|e| b_abs + e)
                .unwrap_or(text.len());
            let block = &text[b_abs..e_abs];
            found_any_block = true;
            for word in forbidden {
                assert!(
                    !block.to_lowercase().contains(word),
                    "forbidden temporal vocabulary '{word}' found in the VIII-0 AOV EXPORT block of {rel}"
                );
            }
            search_from = e_abs + end.len();
        }
        assert!(
            found_any_block,
            "{rel} has no VIII-0 AOV EXPORT marked block to scan — scope check itself is broken"
        );
    }
}

/// Architecture guarantee: `trace_headless_aov`'s signature (grep the source
/// directly, not just its runtime behavior) takes only current-frame inputs
/// — no frame-index-minus-one parameter, no accumulation-across-frames
/// parameter for AOVs. Contrast with `trace_headless`, whose PRE-EXISTING
/// `frames: u32` parameter is fine (main accum buffer's multi-frame
/// accumulation predates this rite and is untouched).
#[test]
fn ban_aov_signature_takes_current_frame_inputs_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let text = fs::read_to_string(root.join("src/integrator.rs")).expect("read integrator.rs");
    let sig_start = text
        .find("pub fn trace_headless_aov(")
        .expect("trace_headless_aov signature not found");
    let sig_end = text[sig_start..]
        .find(") -> Vec<[f32; 4]> {")
        .map(|e| sig_start + e)
        .expect("trace_headless_aov signature close paren not found");
    let signature = &text[sig_start..sig_end];
    for forbidden_param in ["frames", "frame_index", "prev", "history", "samples_before"] {
        assert!(
            !signature.to_lowercase().contains(forbidden_param),
            "trace_headless_aov signature contains '{forbidden_param}' — AOV export must take \
             current-frame inputs only, no cross-frame parameter"
        );
    }
}
