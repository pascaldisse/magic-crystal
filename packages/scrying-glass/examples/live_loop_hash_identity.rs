//! LIVE-LOOP HASH-IDENTITY ORDEAL — the LEVER 2 semantics gate for the
//! PRODUCTION window loop shape (window-playable lane).
//!
//! Proves that wiring the CPU/GPU overlap into `run_render_loop` is a
//! SCHEDULING change only — frame N still traces exactly frame N's splice, so
//! the pixels are bit-identical to the serial loop, frame for frame. This
//! MIRRORS `perf_audit`'s ATOM B FNV hash-identity (commit 84bbc8f, "the
//! player-shaped frame number"), extended from a single final-frame hash to a
//! PER-FRAME hash of the accumulation buffer across M frames.
//!
//! Method — both paths replay a FRESH scene to the SAME warmed start tick
//! (deterministic `command_bodies_walked`/`tick` sequence), then run M frames:
//!   SERIAL  — skin/tick/splice/upload; dispatch + copy accum→readback in ONE
//!             encoder; submit; `poll(wait)`; map + FNV-hash that frame's accum.
//!   OVERLAP — identical CPU stages and the identical `dispatch + copy` encoder,
//!             but the accum readback rides frame N's OWN submission (so it
//!             captures exactly frame N's trace regardless of the pipeline),
//!             and the previous submission is completed AFTER this frame's CPU
//!             stages (the production `run_render_loop` shape). Each frame's
//!             readback is mapped + hashed once its submission has completed.
//! Gate: serial_hash[i] == overlap_hash[i] for every measured frame i.
//!
//! Run:  cargo run -p scrying-glass --release --example live_loop_hash_identity

use std::path::Path;

use crystal::{Core, load_world_dir};
use glam::Vec3 as GVec3;
use sama::GaitParams;
use scrying_glass::bvh::{Bvh, BvhParams, DynamicSplice, RefitParams};
use scrying_glass::integrator::{Integrator, IntegratorParams, IntegratorUniform, headless_device};
use scrying_glass::scene::{
    Camera, RenderScene, SceneParameters, SunDefaults, WalkerPose, contact_passing_ticks,
};
use vessel::{Body, Preset};

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Naruko authoring dials — verbatim from `live_loop_audit::naruko_params`.
fn naruko_params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: env_f32("GAIA_NATIVE_FOV", 60.0),
        near: 0.1,
        far: 4_000.0,
        sky_top: "#20152f".into(),
        sky_horizon: "#9a627d".into(),
        mesh_color: "#9aa0a6".into(),
        radial_segments: 24,
        camera_position: [
            env_f32("GAIA_NATIVE_SPAWN_X", 0.0),
            env_f32("GAIA_NATIVE_SPAWN_Y", 1.7),
            env_f32("GAIA_NATIVE_SPAWN_Z", 24.0),
        ],
        camera_yaw: env_f32("GAIA_NATIVE_CAMERA_YAW", 0.0),
        camera_pitch: env_f32("GAIA_NATIVE_CAMERA_PITCH", 0.0),
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

/// 64-bit FNV-1a — mirrors `perf_audit::fnv` (examples-don't-share-code
/// precedent already established in perf_audit / refit_parity).
fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[hash-identity] no GPU adapter on this host — cannot run the ordeal");
    };

    let w = env_u32("GAIA_NATIVE_RENDER_W", 640);
    let h = env_u32("GAIA_NATIVE_RENDER_H", 480);
    let frames = env_u32("GAIA_ORDEAL_FRAMES", 24);
    let accum_bytes = (w as u64) * (h as u64) * 16;

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let params = naruko_params();

    let mut int_params = IntegratorParams::default();
    int_params.spp = env_u32("GAIA_NATIVE_SPP", 2);
    int_params.max_bounces = env_u32("GAIA_NATIVE_MAX_BOUNCES", 4);
    let bvh_params = BvhParams::default();
    let refit_params = RefitParams::default();

    // Warm-target tick — verbatim from live_loop_audit (composed mid-stride).
    let body = Body::from_preset(&Preset::nari());
    let gait = GaitParams::walk();
    let cycle = (1.0 / (gait.cadence * gait.dt)).round().max(1.0) as u64;
    let (_c, passing_tick) = contact_passing_ticks(&body, &gait);
    let mut target = 150u64;
    while target < passing_tick + cycle || target % cycle != passing_tick % cycle {
        target += 1;
    }
    let vista = WalkerPose {
        position: GVec3::from_array(params.camera_position),
        yaw: params.camera_yaw,
    };
    let camera = Camera {
        eye: GVec3::from_array(params.camera_position),
        yaw: params.camera_yaw,
        pitch: params.camera_pitch,
        fov_y_radians: params.fov_y_degrees.to_radians(),
        near: params.near,
        far: params.far,
    };

    // Build a fresh scene warmed to `target`, and the static BVH over it.
    let build_scene = || {
        let mut core = Core::default();
        load_world_dir(&world_path, &mut core.world).expect("load naruko");
        let mut scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params)
            .expect("render scene");
        for _ in 0..target {
            scene.command_bodies_walked(0.0, Some(vista));
            scene.tick();
        }
        scene
    };

    let seed_scene = build_scene();
    let static_bvh = Bvh::build(&seed_scene.leaf_triangles(), &bvh_params);
    let sun = seed_scene.sun.clone();
    let sky_top = seed_scene.sky_top;
    let sky_horizon = seed_scene.sky_horizon;
    drop(seed_scene);

    let uniform_for = |integrator: &Integrator| {
        // samples_before = 0: moving dynamics reset accumulation every frame in
        // the live surface path (the 2spp-live tradeoff) — matches both loops.
        IntegratorUniform::build(
            &camera, &sun, sky_top, sky_horizon, w, h,
            integrator.node_count, integrator.tri_count, 0, &int_params, None,
        )
    };
    let make_readback = || {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ordeal accum readback"),
            size: accum_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let hash_of = |rb: &wgpu::Buffer| -> u64 {
        let mapped = rb.get_mapped_range(..).expect("mapped ordeal readback");
        let hh = fnv(&mapped);
        drop(mapped);
        rb.unmap();
        hh
    };

    eprintln!(
        "[hash-identity] {w}x{h} internal, spp {}, vista eye={:?} yaw={} — {} frames per path",
        int_params.spp, params.camera_position, params.camera_yaw, frames,
    );

    // ── SERIAL path — the pre-lever loop shape, per-frame accum hash ─────────
    let serial_hashes: Vec<u64> = {
        let mut scene = build_scene();
        let mut splice = DynamicSplice::build(
            &static_bvh, &scene.dynamic_leaf_triangles(), &bvh_params.dynamic(), refit_params,
        );
        let mut integrator =
            Integrator::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, &splice.merged, None);
        let accum = integrator.make_accum(&device, w, h);
        let readback = make_readback();
        let mut hashes = Vec::with_capacity(frames as usize);
        for _ in 0..frames {
            scene.command_bodies_walked(0.0, Some(vista));
            scene.tick();
            splice.update(&static_bvh, &scene.dynamic_leaf_triangles());
            integrator.update_bvh(&device, &splice.merged);
            let compute_bg = integrator.compute_bind_group(&device, &accum);
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("serial trace+readback"),
            });
            integrator.dispatch(&queue, &mut enc, &uniform_for(&integrator), &compute_bg, w, h);
            enc.copy_buffer_to_buffer(&accum, 0, &readback, 0, accum_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            enc.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
                let _ = tx.send(r.map(|_| ()));
            });
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().expect("serial readback chan").expect("serial map");
            hashes.push(hash_of(&readback));
        }
        hashes
    };

    // ── OVERLAP path — the production `run_render_loop` shape, per-frame hash ─
    // Each frame's accum readback rides that frame's OWN submission, so it
    // captures exactly frame N's trace; the previous submission completes AFTER
    // this frame's CPU stages (skin/tick/splice/upload) — the pipelined shape.
    let overlap_hashes: Vec<u64> = {
        let mut scene = build_scene();
        let mut splice = DynamicSplice::build(
            &static_bvh, &scene.dynamic_leaf_triangles(), &bvh_params.dynamic(), refit_params,
        );
        let mut integrator =
            Integrator::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, &splice.merged, None);
        let accum = integrator.make_accum(&device, w, h);
        let mut hashes = vec![0u64; frames as usize];
        let mut pending: Option<wgpu::SubmissionIndex> = None;
        // (frame index, its own readback, its own map-completion receiver)
        let mut in_flight: Option<(usize, wgpu::Buffer, std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>)> = None;
        for frame in 0..frames as usize {
            scene.command_bodies_walked(0.0, Some(vista));
            scene.tick();
            splice.update(&static_bvh, &scene.dynamic_leaf_triangles());
            integrator.update_bvh(&device, &splice.merged);
            let compute_bg = integrator.compute_bind_group(&device, &accum);

            let readback = make_readback();
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("overlap trace+readback"),
            });
            integrator.dispatch(&queue, &mut enc, &uniform_for(&integrator), &compute_bg, w, h);
            enc.copy_buffer_to_buffer(&accum, 0, &readback, 0, accum_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            enc.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
                let _ = tx.send(r.map(|_| ()));
            });
            // Submit THIS frame WITHOUT waiting — its trace overlaps the NEXT
            // iteration's CPU stages.
            let idx = queue.submit(Some(enc.finish()));

            // Complete the PREVIOUS frame's submission (pipelined), then harvest
            // its already-captured readback.
            if let Some(prev) = pending.take() {
                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: Some(prev),
                    timeout: None,
                });
            }
            if let Some((pf, prb, prx)) = in_flight.take() {
                prx.recv().expect("overlap readback chan").expect("overlap map");
                hashes[pf] = hash_of(&prb);
            }
            pending = Some(idx);
            in_flight = Some((frame, readback, rx));
        }
        // Drain the final in-flight frame.
        if let Some(prev) = pending.take() {
            let _ = device.poll(wgpu::PollType::Wait { submission_index: Some(prev), timeout: None });
        }
        if let Some((pf, prb, prx)) = in_flight.take() {
            prx.recv().expect("overlap readback chan").expect("overlap map");
            hashes[pf] = hash_of(&prb);
        }
        hashes
    };

    // ── VERDICT — per-frame bit-identity ────────────────────────────────────
    let mut mismatches = 0u32;
    for i in 0..frames as usize {
        let ok = serial_hashes[i] == overlap_hashes[i];
        if !ok {
            mismatches += 1;
        }
        if i < 4 || !ok {
            println!(
                "[hash-identity] frame {i:3}: serial={:016x} overlap={:016x} {}",
                serial_hashes[i],
                overlap_hashes[i],
                if ok { "MATCH" } else { "DIFFER" },
            );
        }
    }
    println!(
        "[hash-identity] {} / {} frames bit-identical",
        frames - mismatches,
        frames,
    );
    if mismatches == 0 {
        println!("[hash-identity] ORDEAL PASS — LEVER 2 is scheduling-only, content unchanged.");
    } else {
        println!("[hash-identity] ORDEAL FAIL — {mismatches} frame(s) diverged.");
        std::process::exit(1);
    }
}
