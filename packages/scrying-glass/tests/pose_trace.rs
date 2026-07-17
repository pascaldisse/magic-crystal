//! Pose-trace parity harness for the `/walk` + `/pose` debug organs.
//!
//! Replicates the exact organ code path (`respond_walk`: set yaw/pitch/keys,
//! then `player.step` per tick, emitting `pose_json` each tick) against a
//! fixed input script over a flat deterministic floor. The stream it prints is
//! the ground truth for the sama-consumption refactor: it must be
//! byte-identical pre/post (the embodiment does not depend on the retired
//! `homunculus::walk`, so this is the proof it was never on that path).
//!
//! `cargo test -p scrying-glass --test pose_trace -- --nocapture pose_trace_script`
//! prints one `POSE-TRACE <line>` per tick; diff the two runs for 0e0.
//!
//! F3 — THE GUARD HAS A CANON. The full deterministic 200-tick stream is frozen
//! byte-exact in `tests/canon/pose_trace.txt` (committed like
//! `sama/tests/canon/walk_cycle.bin`) and the test asserts the live stream is
//! byte-identical to it — so a movement regression FAILS the suite, never slips
//! through green. RE-DERIVATION (a LEGITIMATE, deliberate movement change only):
//! run `BLESS_POSE_TRACE=1 cargo test -p scrying-glass --test pose_trace
//! pose_trace_script`, which rewrites the canon from the current player math;
//! review the diff and commit it as a conscious act. The canon must NEVER move
//! silently.

use std::fs;
use std::path::Path;

use glam::Vec3;
use scrying_glass::player::{Ground, Key, Player, PlayerParams, Pose};

/// The exact `/pose` organ serialization (mirrors `main::pose_json`).
fn pose_json(pose: &Pose) -> String {
    format!(
        "{{\"position\":[{},{},{}],\"yaw\":{},\"pitch\":{},\"eyeHeight\":{},\"feetY\":{},\"grounded\":{},\"vy\":{}}}",
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.yaw,
        pose.pitch,
        pose.eye_height,
        pose.position.y - pose.eye_height,
        pose.grounded,
        pose.vy,
    )
}

/// A flat floor quad at y=0 spanning [-half, half] in x/z (two triangles).
fn flat_floor(half: f32) -> Ground {
    let positions = [
        [-half, 0.0, -half],
        [half, 0.0, -half],
        [half, 0.0, half],
        [-half, 0.0, -half],
        [half, 0.0, half],
        [-half, 0.0, half],
    ];
    Ground::from_positions(&positions)
}

/// One scripted segment: hold `keys` for `ticks` ticks with a set look.
struct Segment {
    keys: &'static [&'static str],
    yaw: f32,
    pitch: f32,
    ticks: u32,
}

#[test]
fn pose_trace_script() {
    // Deterministic feel constants (defaults; no env dependency at test time).
    let params = PlayerParams::from_env().expect("player params");
    let tick_dt = 1.0f32 / 60.0; // the native default (GAIA_NATIVE_TICK_DT)
    let ground = flat_floor(64.0);

    let spawn_eye = Vec3::new(0.0, params.eye_stand, 0.0);
    let mut player = Player::new(params, spawn_eye, 0.0);

    // idle -> forward -> forward+run -> strafe -> jump+forward -> crouch+back.
    let script = [
        Segment {
            keys: &[],
            yaw: 0.0,
            pitch: 0.0,
            ticks: 20,
        },
        Segment {
            keys: &["w"],
            yaw: 0.3,
            pitch: -0.1,
            ticks: 40,
        },
        Segment {
            keys: &["w", "shift"],
            yaw: 0.3,
            pitch: -0.1,
            ticks: 40,
        },
        Segment {
            keys: &["d"],
            yaw: 1.2,
            pitch: 0.0,
            ticks: 30,
        },
        Segment {
            keys: &["w", "space"],
            yaw: -0.5,
            pitch: 0.2,
            ticks: 40,
        },
        Segment {
            keys: &["s", "c"],
            yaw: 2.0,
            pitch: -0.3,
            ticks: 30,
        },
    ];

    let mut total = 0u32;
    let mut trace = String::new();
    for seg in &script {
        // Mirror respond_walk: set yaw/pitch, then the held key set.
        player.yaw = seg.yaw;
        player.pitch = seg
            .pitch
            .clamp(-player.params.pitch_limit, player.params.pitch_limit);
        player.keys = seg.keys.iter().filter_map(|t| Key::from_token(t)).collect();
        for _ in 0..seg.ticks {
            player.step(tick_dt, &ground);
            total += 1;
            let line = format!("POSE-TRACE {} {}", total, pose_json(&player.pose()));
            println!("{line}");
            trace.push_str(&line);
            trace.push('\n');
        }
    }
    println!("POSE-TRACE-END ticks={total}");

    // F3 — assert the live stream against the frozen canon (byte-exact). The
    // BLESS path rewrites the canon for a deliberate movement change (see the
    // module doc); the default path is the guard.
    let canon_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/canon/pose_trace.txt");
    if std::env::var_os("BLESS_POSE_TRACE").is_some() {
        fs::create_dir_all(canon_path.parent().unwrap()).expect("create canon dir");
        fs::write(&canon_path, &trace).expect("write pose-trace canon");
        println!(
            "POSE-TRACE-BLESSED {} bytes -> {}",
            trace.len(),
            canon_path.display()
        );
        return;
    }
    let canon = fs::read_to_string(&canon_path).expect(
        "pose-trace canon missing — bless it once with BLESS_POSE_TRACE=1 (see module doc)",
    );
    assert_eq!(
        trace, canon,
        "POSE-TRACE stream diverged from the frozen canon ({} ticks). A movement \
         regression is caught here. If this change is DELIBERATE, re-derive the canon \
         with BLESS_POSE_TRACE=1 and commit the diff (see module doc).",
        total,
    );
    println!("POSE-TRACE parity: {} ticks byte-identical to canon", total);
}
