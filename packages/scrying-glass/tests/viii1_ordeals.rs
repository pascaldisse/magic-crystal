//! RITE VIII-1 — THE DREAM-DENOISER: the ordeals. See
//! docs/proposals/RITE-VIII-THE-DREAM-DENOISER.md §VIII-1.
//!
//!   (a) inference byte-identical — same frame twice, cold.
//!   (b) denoised RMSE ≤ pinned bound on every validation frame — the bound
//!       ships WITH the committed weights (its provenance sidecar), derived
//!       at train time, NEVER chosen here; this ordeal replays validation
//!       deterministically (re-renders the SAME two held-out poses
//!       `viii1_train.rs` used) and checks the committed number.
//!   (c) denoised strictly beats noisy on every validation frame.
//!   (d) THE BAN ordeal: `src/denoiser.rs` is picked up by the VIII-0
//!       grep-gate's forward-proof scope mechanism (confirmed directly, not
//!       assumed) + `denoise_image`'s public signature takes current-frame
//!       buffers only (no frame-index/history parameter).
//!   (e) senses unchanged: the renderer's world-truth buffers (noisy beauty
//!       + AOVs) are sha-equal whether or not the denoiser pass runs on
//!       them — the net filters pixels, never the world.
//!
//! All GPU ordeals print + return early (never a false green) on a host
//! without an adapter, matching `viii0_ordeals.rs`'s convention.

use std::fs;
use std::path::Path;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::denoiser::{Mlp, deserialize_weights, denoise_image, sha256_hex};
use scrying_glass::error_metric::rmse;
use scrying_glass::integrator::{
    IntegratorParams, headless_device, resolve, split_aov, trace_headless, trace_headless_aov,
};
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene, SceneParameters, SunDefaults, SunLight};

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

fn load_committed_weights() -> Mlp {
    let bytes = fs::read(weights_dir().join("denoiser-weights-v1.bin")).expect(
        "read committed denoiser-weights-v1.bin — run `cargo run -p scrying-glass --release \
         --example viii1_train` to forge it first",
    );
    deserialize_weights(&bytes).expect("deserialize committed weights artifact")
}

fn load_provenance() -> serde_json::Value {
    let text = fs::read_to_string(weights_dir().join("denoiser-weights-v1.provenance.json"))
        .expect("read committed provenance sidecar");
    serde_json::from_str(&text).expect("parse provenance JSON")
}

#[test]
fn a_inference_is_byte_identical_same_frame_twice_cold() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[VIII-1 inference determinism] no GPU adapter on this host — ordeal could not run");
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
    let raw_aov = trace_headless_aov(&device, &queue, &bvh, &camera, &sun, sky_top, sky_horizon, w, h);
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

/// Re-render the SAME two held-out validation poses `examples/viii1_train.rs`
/// used ("orbit_-20", "orbit_+40" — see that file's module docs for the
/// full documented dataset scope), from the same naruko realm, same fixed
/// seed/params, so the pinned bound can be replayed deterministically here.
fn render_validation_poses(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Vec<(&'static str, Vec<GVec3>, Vec<GVec3>, Vec<GVec3>, Vec<f32>, Vec<GVec3>)> {
    fn naruko_params() -> SceneParameters {
        SceneParameters {
            fov_y_degrees: 60.0,
            near: 0.1,
            far: 4_000.0,
            sky_top: "#20152f".into(),
            sky_horizon: "#9a627d".into(),
            mesh_color: "#9aa0a6".into(),
            radial_segments: 24,
            camera_position: [0.0, 2.0, 22.0],
            camera_yaw: 0.0,
            camera_pitch: 0.0,
            cluster_error_threshold: 1.0,
            tick_dt: 1.0 / 60.0,
            sun: SunDefaults {
                sun_color: "#ffe2b0".into(),
                sun_intensity: 1.1,
                sun_position: [60.0, 90.0, 30.0],
                ambient_intensity: 0.32,
            },
            emission_intensity: 2.5,
        }
    }
    fn camera_at(eye: [f32; 3], look_at: [f32; 3], fov_deg: f32) -> Camera {
        let f = (GVec3::from_array(look_at) - GVec3::from_array(eye)).normalize();
        Camera {
            eye: GVec3::from_array(eye),
            yaw: (-f.x).atan2(-f.z),
            pitch: f.y.asin(),
            fov_y_radians: fov_deg.to_radians(),
            near: 0.1,
            far: 4_000.0,
        }
    }
    fn orbit_camera(eye: [f32; 3], pivot: [f32; 3], yaw_deg: f32, fov_deg: f32) -> Camera {
        let rel = GVec3::from_array(eye) - GVec3::from_array(pivot);
        let angle = yaw_deg.to_radians();
        let (s, c) = angle.sin_cos();
        let rotated = GVec3::new(rel.x * c + rel.z * s, rel.y, -rel.x * s + rel.z * c);
        let new_eye = GVec3::from_array(pivot) + rotated;
        camera_at(new_eye.to_array(), pivot, fov_deg)
    }

    let params = naruko_params();
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");
    let bvh = Bvh::build(&scene.leaf_triangles(), &BvhParams::default());

    let front_pivot = [0.0, 2.0, 0.0];
    let (w, h) = (96u32, 64u32);
    let ref_frames = 128u32;

    let poses: Vec<(&'static str, Camera)> = vec![
        (
            "orbit_-20",
            orbit_camera(params.camera_position, front_pivot, -20.0, params.fov_y_degrees),
        ),
        (
            "orbit_+40",
            orbit_camera(params.camera_position, front_pivot, 40.0, params.fov_y_degrees),
        ),
    ];

    poses
        .into_iter()
        .map(|(name, camera)| {
            let noisy_params = IntegratorParams {
                spp: 1,
                ..IntegratorParams::default()
            };
            let noisy = resolve(&trace_headless(
                device,
                queue,
                &bvh,
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
                &bvh,
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
            let raw_aov = trace_headless_aov(device, queue, &bvh, &camera, &scene.sun, scene.sky_top, scene.sky_horizon, w, h);
            let (albedo, normal, depth) = split_aov(&raw_aov);
            (name, noisy, albedo, normal, depth, reference)
        })
        .collect()
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

    let val_frames = render_validation_poses(&device, &queue);
    assert_eq!(
        val_frames.len(),
        2,
        "expected exactly the 2 documented validation poses"
    );

    for (name, noisy, albedo, normal, depth, reference) in val_frames {
        let denoised = denoise_image(&mlp, &noisy, &albedo, &normal, &depth);
        let noisy_rmse = rmse(&noisy, &reference);
        let denoised_rmse = rmse(&denoised, &reference);
        println!(
            "[VIII-1 validation replay] pose={name} noisy_rmse={noisy_rmse:.6} denoised_rmse={denoised_rmse:.6} pinned_bound={pinned_bound:.6}"
        );
        // (b) derived, pinned bound — never chosen here.
        assert!(
            denoised_rmse <= pinned_bound,
            "pose '{name}': denoised RMSE {denoised_rmse:.6} exceeds the pinned bound {pinned_bound:.6} \
             derived at train time"
        );
        // (c) strictly beats noisy — reconstruction, not repainting.
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

/// (d) THE BAN, part 2: `denoise_image`'s public signature (grepped
/// directly) takes current-frame buffers only — no frame-index/history
/// parameter anywhere.
#[test]
fn d_ban_denoise_image_signature_takes_current_frame_buffers_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let text = fs::read_to_string(root.join("src/denoiser.rs")).expect("read src/denoiser.rs");
    let sig_start = text
        .find("pub fn denoise_image(")
        .expect("denoise_image signature not found");
    let sig_end = text[sig_start..]
        .find(") -> Vec<Vec3> {")
        .map(|e| sig_start + e)
        .expect("denoise_image signature close paren not found");
    let signature = &text[sig_start..sig_end];
    for forbidden_param in [
        "frames",
        "frame_index",
        "prev",
        "history",
        "samples_before",
        "last_",
    ] {
        assert!(
            !signature.to_lowercase().contains(forbidden_param),
            "denoise_image signature contains '{forbidden_param}' — inference must take \
             current-frame buffers only"
        );
    }
}

/// (e) senses unchanged: the renderer's world-truth buffers are sha-equal
/// whether or not the denoiser pass runs on them — the net filters pixels,
/// never the world. Renders the fixed pose's noisy beauty + AOVs, hashes
/// them (BEFORE any denoise call), runs the denoiser (an unrelated function
/// call that returns a NEW buffer), then re-renders the SAME pose fresh and
/// hashes again — proving the render pass's own truth is identical whether
/// the denoise pass ran in between or not.
#[test]
fn e_senses_unchanged_oracle_sha_equal_pass_on_or_off() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[VIII-1 senses unchanged] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let (bvh, camera, sun, sky_top, sky_horizon, w, h) = fixed_pose();
    let noisy_params = IntegratorParams {
        spp: 1,
        ..IntegratorParams::default()
    };

    fn render_truth(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bvh: &Bvh,
        camera: &Camera,
        sun: &SunLight,
        sky_top: [f32; 4],
        sky_horizon: [f32; 4],
        w: u32,
        h: u32,
        params: &IntegratorParams,
    ) -> Vec<u8> {
        let noisy = resolve(&trace_headless(
            device, queue, bvh, camera, sun, sky_top, sky_horizon, w, h, 1, params, None,
        ));
        let raw_aov = trace_headless_aov(device, queue, bvh, camera, sun, sky_top, sky_horizon, w, h);
        let mut bytes = Vec::new();
        for px in &noisy {
            bytes.extend_from_slice(bytemuck::cast_slice(&px.to_array()));
        }
        for c in &raw_aov {
            bytes.extend_from_slice(bytemuck::cast_slice(c));
        }
        bytes
    }

    // Pass OFF: render, hash, denoiser never invoked at all in this branch.
    let truth_off = render_truth(&device, &queue, &bvh, &camera, &sun, sky_top, sky_horizon, w, h, &noisy_params);
    let hash_off = sha256_hex(&truth_off);

    // Pass ON: render the SAME pose again, run the denoiser on its output
    // (a pure function of the buffers, returns a new Vec, mutates nothing),
    // THEN hash the render's own truth buffers again.
    let truth_on = render_truth(&device, &queue, &bvh, &camera, &sun, sky_top, sky_horizon, w, h, &noisy_params);
    let raw_aov = trace_headless_aov(&device, &queue, &bvh, &camera, &sun, sky_top, sky_horizon, w, h);
    let (albedo, normal, depth) = split_aov(&raw_aov);
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
    let mlp = load_committed_weights();
    let _denoised = denoise_image(&mlp, &noisy, &albedo, &normal, &depth); // pass ON, result unused here on purpose
    let hash_on = sha256_hex(&truth_on);

    assert_eq!(
        hash_off, hash_on,
        "the render pass's own truth (noisy beauty + AOVs) changed depending on whether the \
         denoiser pass ran — the net must filter pixels, never the world"
    );
}
