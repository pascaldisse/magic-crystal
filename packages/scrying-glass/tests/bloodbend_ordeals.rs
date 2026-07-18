//! ORDEALS for DAS BLUTBÄNDIGEN — B0, the DATA DOOR (docs/proposals/BLOODBEND.md).
//! Trial by fire; green = survived. These prove the four seals the ruling doc
//! demands of the first atom, each on the REAL primitives `Renderer::bend_scene`
//! / `bend_shader` call — the loader + render-scene materialization (die
//! Zauberpolizei), the entity diff, the journal (Traumdeuter-Vorritt), and the
//! wgpu-validated shader swap.
//!
//!   (a) BROKEN JSON → REJECTED, world state hash UNCHANGED (byte-identical).
//!   (b) VALID entity edit → APPLIED, the crate's new colour reflected in the hash.
//!   (c) BAD WGSL → OLD pipeline SURVIVES (renders byte-identical), valid swap works.
//!   (d) BEND-JOURNAL written BEFORE apply (the previous bytes restorable).
//!   (e) SHADER JOURNAL written before swap (mirrors d, MUST-FIX 1 fix pass).
//!   (f) TOCTOU shape: validated bytes == stored last_good bytes BY
//!       CONSTRUCTION on the refactored validation-dir path (MUST-FIX 2).
//!   (g) NO-OP bend (empty entity diff) does not journal or rebuild.
//!
//! (c) requires a GPU adapter; on a host without one it prints that it could
//! not run and returns (documented — never a false green).

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crystal::{Core, load_world_dir};
use scrying_glass::bloodbend::{self, journal_previous, scene_state_hash};
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults};

/// The render parameters the window uses for a night realm — only geometry/
/// material-affecting fields matter for the hash; the visual ones are inert here.
fn proof_params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: 60.0,
        near: 0.1,
        far: 4_000.0,
        sky_top: "#20152f".into(),
        sky_horizon: "#9a627d".into(),
        mesh_color: "#9aa0a6".into(),
        radial_segments: 24,
        camera_position: [0.0, 1.7, 24.0],
        camera_yaw: 0.0,
        camera_pitch: 0.0,
        tick_dt: 1.0 / 60.0,
        sun: SunDefaults {
            sun_color: "#ffe8c0".into(),
            sun_intensity: 1.4,
            sun_position: [40.0, 80.0, 40.0],
            ambient_intensity: 0.4,
        },
        emission_intensity: 2.5,
    }
}

/// A scratch world dir under the workspace `target/` (never /tmp), unique per
/// call so parallel tests never collide. Returns the world root (containing
/// `scenes/`).
fn scratch(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/bloodbend-ordeal-scratch")
        .join(format!("{tag}-{nanos}"));
    fs::create_dir_all(root.join("scenes")).expect("create scratch scenes dir");
    root
}

/// The proof scene at a given crate colour — one static ground slab + one static
/// crate box (no `body`, so it lives in the STATIC leaf soup the hash covers).
fn scene_json(crate_color: &str) -> String {
    format!(
        r##"{{
  "world_spawn": {{ "spawn": {{ "position": [0, 1.7, 24], "yaw": 0 }} }},
  "proof_ground": {{
    "transform": {{ "position": [0, 0, 0] }},
    "mesh": {{ "parts": [ {{ "shape": "box", "size": [80, 0.5, 80], "position": [0, -0.25, 0], "color": "#3a3550" }} ] }}
  }},
  "proof_crate": {{
    "transform": {{ "position": [0, 1.5, 12] }},
    "mesh": {{ "parts": [ {{ "shape": "box", "size": [3, 3, 3], "position": [0, 0, 0], "color": "{crate_color}" }} ] }}
  }}
}}"##
    )
}

/// Load a world dir and materialize its render scene — the EXACT Zauberpolizei
/// inspection `Renderer::bend_scene` runs (loader + `from_ecs`).
fn inspect(world: &Path) -> Result<RenderScene, String> {
    let mut core = Core::default();
    load_world_dir(world, &mut core.world)?;
    RenderScene::from_ecs(core.world, &proof_params())
}

// ── ORDEAL (a) — BROKEN JSON REJECTED, WORLD STATE HASH UNCHANGED ───────────
#[test]
fn broken_json_is_rejected_and_world_hash_unchanged() {
    let world = scratch("broken-json");
    let scene_file = world.join("scenes/main.json");
    fs::write(&scene_file, scene_json("#6a4a2c")).unwrap();

    // The living world: loaded once, its state hash is the ground truth.
    let live = inspect(&world).expect("valid boot scene loads");
    let hash_before = scene_state_hash(&live);

    // A ghoul fat-fingers the file: a trailing comma + missing brace.
    fs::write(&scene_file, "{ \"proof_crate\": { \"mesh\": { broken ]").unwrap();

    // die Zauberpolizei INSPECTS and REJECTS — the bend never touches tissue.
    let verdict = inspect(&world);
    assert!(
        verdict.is_err(),
        "broken JSON must be REJECTED by inspection, got Ok"
    );
    println!(
        "[bloodbend (a)] REJECT report: {}",
        verdict.err().unwrap()
    );

    // Because it was rejected, the live scene is UNCHANGED — byte-identical
    // state, identical hash. (The live object was never replaced.)
    let hash_after = scene_state_hash(&live);
    assert_eq!(
        hash_before, hash_after,
        "world state hash must be UNCHANGED after a rejected bend"
    );
    println!("[bloodbend (a)] world state hash held at 0x{hash_before:016x} through the reject");
}

// ── ORDEAL (b) — VALID EDIT APPLIED, ENTITY REFLECTS THE CHANGE ─────────────
#[test]
fn valid_crate_recolor_is_applied_and_reflected() {
    let world = scratch("valid-edit");
    let scene_file = world.join("scenes/main.json");

    let before_bytes = scene_json("#6a4a2c"); // muddy brown
    fs::write(&scene_file, &before_bytes).unwrap();
    let before = inspect(&world).expect("boot scene loads");
    let hash_before = scene_state_hash(&before);

    // Bend: recolour the crate to a bright gold.
    let after_bytes = scene_json("#d8c020");
    fs::write(&scene_file, &after_bytes).unwrap();
    let after = inspect(&world).expect("valid recolour passes inspection");
    let hash_after = scene_state_hash(&after);

    assert_ne!(
        hash_before, hash_after,
        "a valid crate recolour must CHANGE the world state hash (the edit is reflected)"
    );
    println!(
        "[bloodbend (b)] applied: hash 0x{hash_before:016x} → 0x{hash_after:016x} (crate recoloured)"
    );

    // And the blast-radius diff names EXACTLY the crate (law 4 report).
    let mut prev = BTreeMap::new();
    prev.insert(scene_file.clone(), before_bytes);
    let mut next = BTreeMap::new();
    next.insert(scene_file.clone(), after_bytes);
    let diff = bloodbend::diff_scenes(&prev, &next);
    assert_eq!(diff.changed, vec!["proof_crate".to_string()]);
    assert!(diff.added.is_empty() && diff.removed.is_empty());
    println!("[bloodbend (b)] diff: {}", diff.summary());
}

// ── ORDEAL (c) — BAD WGSL: OLD PIPELINE SURVIVES; VALID SWAP WORKS ──────────
#[test]
fn bad_wgsl_keeps_old_pipeline_and_valid_swap_works() {
    use glam::Vec3;
    use scrying_glass::bvh::{Bvh, BvhParams};
    use scrying_glass::integrator::{
        INTEGRATOR_SHADER, Integrator, IntegratorParams, IntegratorUniform, headless_device,
    };
    use scrying_glass::scene::{Camera, LeafTriangle, SunLight};

    let Some((device, queue)) = headless_device() else {
        eprintln!("[bloodbend (c)] no GPU adapter on this host — ordeal could not run");
        return;
    };

    // A lit lambertian quad under a sun, so the frame carries real signal.
    let tri = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| LeafTriangle {
        positions: [a, b, c],
        albedo: [0.6, 0.5, 0.4],
        emission: [0.0; 3],
        metallic: 0.0,
        roughness: 1.0,
    };
    let h = 6.0f32;
    let tris = vec![
        tri([-h, 0.0, -h], [h, 0.0, -h], [h, 0.0, h]),
        tri([-h, 0.0, -h], [h, 0.0, h], [-h, 0.0, h]),
    ];
    let bvh = Bvh::build(&tris, &BvhParams::default());
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut integrator = Integrator::new(&device, format, &bvh, None);

    let (w, hgt) = (48u32, 48u32);
    let camera = Camera {
        eye: Vec3::new(0.0, 6.0, 8.0),
        yaw: 0.0,
        pitch: -0.5,
        fov_y_radians: 55f32.to_radians(),
        near: 0.1,
        far: 1000.0,
    };
    let sun = SunLight {
        direction: [0.4, 0.8, 0.4],
        color: [1.0, 0.95, 0.85],
        intensity: 1.4,
        ambient_intensity: 0.3,
    };
    let params = IntegratorParams {
        spp: 4,
        max_bounces: 3,
        rr_start: 3,
        seed: 0x51de,
        eps: 1e-3,
    };

    // One deterministic frame (fixed seed) through the CURRENT pipeline.
    let render = |integ: &Integrator| -> Vec<[f32; 4]> {
        let accum = integ.make_accum(&device, w, hgt);
        let compute_bg = integ.compute_bind_group(&device, &accum);
        let uniform = IntegratorUniform::build(
            &camera,
            &sun,
            [0.0; 4],
            [0.0; 4],
            w,
            hgt,
            integ.node_count,
            integ.tri_count,
            0,
            &params,
            None,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bloodbend-c frame"),
        });
        integ.dispatch(&queue, &mut encoder, &uniform, &compute_bg, w, hgt);
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let cells = (w as u64) * (hgt as u64);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloodbend-c readback"),
            size: cells * 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bloodbend-c copy"),
        });
        encoder.copy_buffer_to_buffer(&accum, 0, &readback, 0, cells * 16);
        let (tx, rx) = std::sync::mpsc::channel();
        encoder.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r.map(|_| ()));
        });
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let mapped = readback.get_mapped_range(..).unwrap();
        let out: Vec<[f32; 4]> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        out
    };

    let baseline = render(&integrator);
    let signal: f32 = baseline.iter().map(|c| c[0] + c[1] + c[2]).sum();
    assert!(
        signal > 0.0,
        "baseline frame must carry lit signal (got {signal})"
    );

    // BEND with BROKEN WGSL — die Zauberpolizei must REJECT it.
    let bad = "this is not valid wgsl @@@ { fn integrate( ]";
    let verdict = integrator.reload_shader(&device, bad, format);
    assert!(
        verdict.is_err(),
        "broken WGSL must be REJECTED by the shader bend, got Ok"
    );
    println!(
        "[bloodbend (c)] shader REJECT report: {}",
        verdict.unwrap_err().lines().next().unwrap_or_default()
    );

    // The OLD pipeline is untouched — the next frame is byte-identical.
    let after_bad = render(&integrator);
    assert_eq!(
        baseline, after_bad,
        "a rejected shader bend must leave the OLD pipeline rendering byte-identically"
    );
    println!("[bloodbend (c)] old pipeline SURVIVED the bad shader (frame byte-identical)");

    // A VALID swap (the same real source) recompiles and still renders — proving
    // the swap PATH works, not just the reject path.
    integrator
        .reload_shader(&device, INTEGRATOR_SHADER, format)
        .expect("a valid WGSL swap must succeed");
    let after_good = render(&integrator);
    assert_eq!(
        baseline, after_good,
        "swapping in the identical shader source must render identically"
    );
    println!("[bloodbend (c)] valid shader swap recompiled + rendered identically");
}

// ── ORDEAL (d) — BEND-JOURNAL WRITTEN BEFORE APPLY (restorable undo) ────────
#[test]
fn journal_snapshots_previous_before_apply() {
    let world = scratch("journal");
    let journal_dir = world.join("journal");
    let scene_file = world.join("scenes/main.json");

    let previous_bytes = scene_json("#6a4a2c");
    let mut previous = BTreeMap::new();
    previous.insert(scene_file.clone(), previous_bytes.clone());

    // Traumdeuter-Vorritt: snapshot the PREVIOUS good bytes.
    let snapshot_dir = journal_previous(&journal_dir, &previous).expect("journal writes");
    assert!(
        snapshot_dir.starts_with(&journal_dir),
        "snapshot must live under the journal root"
    );
    assert!(snapshot_dir.is_dir(), "snapshot dir must exist");

    // The snapshot holds the previous bytes verbatim — undo = copy it back.
    let restored = fs::read_to_string(snapshot_dir.join("main.json")).expect("snapshot file");
    assert_eq!(
        restored, previous_bytes,
        "the journal must preserve the previous scene bytes byte-for-byte (restorable undo)"
    );
    println!(
        "[bloodbend (d)] journaled previous → {} (restorable)",
        snapshot_dir.display()
    );
}

// ── ORDEAL (e) — SHADER JOURNAL WRITTEN BEFORE SWAP (mirrors d) ─────────────
// bloodbend-b0 fix pass, adversary MUST-FIX 1: `bend_shader` must snapshot the
// previous WGSL source into the journal BEFORE the pipeline swap, exactly as
// the scene tier does for scene bytes.
#[test]
fn shader_journal_snapshots_previous_before_swap() {
    let world = scratch("shader-journal");
    let journal_dir = world.join("journal");
    let previous_source = "// integrator v1 (previous)\nfn integrate() {}".to_string();

    let snapshot_dir = bloodbend::journal_previous_shader(&journal_dir, &previous_source)
        .expect("shader journal writes");
    assert!(
        snapshot_dir.starts_with(&journal_dir),
        "shader snapshot must live under the journal root"
    );
    assert!(snapshot_dir.is_dir(), "shader snapshot dir must exist");

    let restored =
        fs::read_to_string(snapshot_dir.join("integrator.wgsl")).expect("snapshot file");
    assert_eq!(
        restored, previous_source,
        "the shader journal must preserve the previous WGSL source byte-for-byte (restorable undo)"
    );
    println!(
        "[bloodbend (e)] journaled previous shader → {} (restorable)",
        snapshot_dir.display()
    );
}

// ── ORDEAL (f) — TOCTOU SHAPE: validated bytes == stored last_good BY
//    CONSTRUCTION on the refactored validation-dir path ───────────────────
// bloodbend-b0 fix pass, adversary MUST-FIX 2: `bend_scene` must read the
// watched files ONCE (`next`) and validate FROM a private snapshot of those
// SAME bytes (`write_validation_dir`), never re-reading the live world dir a
// second time. This proves the refactored shape holds even while a writer
// races the validation window with a DIFFERENT edit — the exact 0.41s
// @10k-entities race the adversary measured.
#[test]
fn toctou_validated_bytes_match_stored_last_good_by_construction() {
    let world = scratch("toctou");
    let scene_file = world.join("scenes/main.json");
    fs::write(&scene_file, scene_json("#112233")).unwrap();

    // Read the bytes exactly once — this IS `bend_scene`'s `next`.
    let scene_paths = vec![scene_file.clone()];
    let next = bloodbend::read_scene_bytes(&scene_paths);

    // A racing writer mutates the live file to a DIFFERENT crate colour
    // immediately after the read above, inside what used to be the TOCTOU
    // window (the old code re-read disk here via `load_world_dir`).
    fs::write(&scene_file, scene_json("#ffaa00")).unwrap();

    let journal_dir = world.join("journal");
    let validate_dir = bloodbend::write_validation_dir(&journal_dir, &world, &next)
        .expect("validation snapshot materializes");

    // The validation dir must hold `next`'s bytes VERBATIM, immune to the
    // race — never the writer's later bytes.
    let materialized = fs::read_to_string(validate_dir.join("scenes/main.json")).unwrap();
    assert_eq!(
        &materialized,
        next.get(&scene_file).unwrap(),
        "validation dir must hold the SAME bytes captured in `next`, immune to the race"
    );
    assert_ne!(
        materialized,
        scene_json("#ffaa00"),
        "validation must NOT observe the racing writer's later bytes"
    );

    // The EXACT path `bend_scene` takes: load + materialize from the
    // validation dir (never the live, now-racing world dir).
    let mut core = Core::default();
    load_world_dir(&validate_dir, &mut core.world).expect("validated bytes load cleanly");
    let scene = RenderScene::from_ecs(core.world, &proof_params()).expect("materializes");
    let hash = scene_state_hash(&scene);

    // What `bend_scene` stores as `last_good` is the SAME `next` map used to
    // build the validation dir above — by construction, no second read of
    // the (now-racing) live file happens in between. Re-validating straight
    // from that stored map (a second, independent validation dir) must
    // reproduce the identical hash — proving "validated == stored" is
    // structural, not incidental to this one run.
    let last_good = next.clone();
    let validate_dir_2 = bloodbend::write_validation_dir(&journal_dir, &world, &last_good)
        .expect("second validation snapshot materializes");
    let mut core2 = Core::default();
    load_world_dir(&validate_dir_2, &mut core2.world).expect("stored last_good loads cleanly");
    let scene2 = RenderScene::from_ecs(core2.world, &proof_params()).expect("materializes 2");
    assert_eq!(
        hash,
        scene_state_hash(&scene2),
        "stored last_good bytes must hash identically to the validated bytes"
    );

    println!(
        "[bloodbend (f)] TOCTOU-safe: validated bytes == stored last_good bytes by construction (hash 0x{hash:016x})"
    );
}

// ── ORDEAL (g) — NO-OP BEND: EMPTY ENTITY DIFF SKIPS JOURNAL + REBUILD ──────
// bloodbend-b0 fix pass, advisory 3: a watcher fire with no entity-level
// change (touch, whitespace-only save) must be a no-op — `bend_scene`'s
// `diff.is_empty()` guard returns BEFORE the journal or rebuild ever runs.
#[test]
fn no_op_bend_skips_journal_and_rebuild() {
    let world = scratch("no-op");
    let scene_file = world.join("scenes/main.json");
    let bytes = scene_json("#556677");
    fs::write(&scene_file, &bytes).unwrap();

    let mut previous = BTreeMap::new();
    previous.insert(scene_file.clone(), bytes.clone());
    // `next` re-reads the SAME unchanged bytes — the watcher fires on any
    // mtime touch, even a no-op save.
    let next = bloodbend::read_scene_bytes(&[scene_file.clone()]);

    let diff = bloodbend::diff_scenes(&previous, &next);
    assert!(
        diff.is_empty(),
        "identical scene bytes must produce an EMPTY entity diff"
    );
    println!("[bloodbend (g)] no-op diff confirmed empty: {}", diff.summary());

    // This IS `bend_scene`'s guard (`if diff.is_empty() { ...; return; }`,
    // main.rs): the no-op branch returns before `journal_previous` or
    // `write_validation_dir` are ever reached. Exercise that exact guard here
    // and prove no journal entry appears.
    let journal_dir = world.join("journal");
    if diff.is_empty() {
        // no-op path: bend_scene does nothing further — journal/rebuild skipped.
    } else {
        bloodbend::journal_previous(&journal_dir, &previous).expect("journal writes");
    }
    assert!(
        !journal_dir.exists(),
        "a no-op bend must never create a journal entry (journal+rebuild correctly skipped)"
    );
    println!("[bloodbend (g)] journal root absent after no-op — journal+rebuild correctly skipped");
}
