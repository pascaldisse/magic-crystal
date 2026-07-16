//! Live N1 ordeal — interpolation + resilience against a wired-owned server.
//!
//! Owns the reference server's lifecycle so the reconnect is deterministic:
//! spawns `bun server/index.js` (from the READ-ONLY reference worktree named
//! by `WIRED_SERVER_DIR`), kills it mid-session, restarts it, and proves the
//! client resyncs with the right typed events.
//!
//!   WIRED_SERVER_DIR=/path/to/GAIA-World-Engine-wired \
//!   GAIA_PORT=8426 cargo run -p wired --example n1
//!
//! Drives: two-client smoke → remote interpolation (sampled vs slam) →
//! kill+restart the server → resync + typed link events → clean close.

use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use wired::{ClientEvent, Config, Wired};

fn spawn_server(dir: &str, port: u16, save: &str) -> Child {
    Command::new("bun")
        .arg("server/index.js")
        .current_dir(dir)
        .env("GAIA_PORT", port.to_string())
        .env("GAIA_SAVE", save)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn bun server")
}

#[tokio::main]
async fn main() {
    let dir = std::env::var("WIRED_SERVER_DIR")
        .expect("set WIRED_SERVER_DIR to the reference GAIA worktree");
    let port: u16 = std::env::var("GAIA_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8426);
    let save = "wired_n1_ordeal";

    println!("[n1] server dir={dir} port={port}");
    let mut server = spawn_server(&dir, port, save);
    tokio::time::sleep(Duration::from_millis(2500)).await;

    // --- observer A: collect typed link events in the background ---
    let a = Wired::connect(Config::with_port(port).presence("player-rust-a"));
    assert!(a.wait_live().await, "A never went live");
    let log: Arc<Mutex<Vec<ClientEvent>>> = Arc::new(Mutex::new(Vec::new()));
    {
        let mut rx = a.client_events();
        let log = log.clone();
        tokio::spawn(async move {
            while let Ok(ev) = rx.recv().await {
                println!("[n1] A client-event: {ev:?}");
                log.lock().unwrap().push(ev);
            }
        });
    }
    a.spawn_presence([0.0, 2.0, 22.0], 0.0).expect("A spawn");
    tokio::time::sleep(Duration::from_millis(300)).await;

    // --- mover B: smooth straight-line path streamed at ~10Hz ---
    let b = Wired::connect(Config::with_port(port).presence("player-rust-b"));
    assert!(b.wait_live().await, "B never went live");
    b.spawn_presence([-6.0, 2.0, 22.0], 0.0).expect("B spawn");
    tokio::time::sleep(Duration::from_millis(300)).await;

    println!(
        "[n1] two-client smoke: A sees B={} | B sees A={}",
        a.has_entity("player-rust-b"),
        b.has_entity("player-rust-a")
    );

    // B walks +x at ~5 m/s, one merge every 100ms; A samples at 60Hz.
    let speed = 5.0;
    let mut max_smooth = 0.0_f64;
    let mut max_slam = 0.0_f64;
    let mut prev_smooth: Option<f64> = None;
    let mut prev_slam: Option<f64> = None;
    let mut nonempty_frames = 0;
    for step in 0..=20 {
        let t = step as f64 * 0.1;
        let x = -6.0 + speed * t;
        b.move_presence([x, 2.0, 22.0], 0.0).expect("B move");
        // six render frames per feed tick
        for _ in 0..6 {
            tokio::time::sleep(Duration::from_millis(1000 / 60)).await;
            if let Some(s) = a.sample("player-rust-b") {
                nonempty_frames += 1;
                if let Some(p) = prev_smooth {
                    max_smooth = max_smooth.max((s.position[0] - p).abs());
                }
                prev_smooth = Some(s.position[0]);
            }
            if let Some(s) = a.slam("player-rust-b") {
                if let Some(p) = prev_slam {
                    max_slam = max_slam.max((s.position[0] - p).abs());
                }
                prev_slam = Some(s.position[0]);
            }
        }
    }
    println!(
        "[n1] A interpolated B over {nonempty_frames} frames: smooth_max_step={max_smooth:.4} slam_max_step={max_slam:.4}"
    );
    if let Some(s) = a.sample("player-rust-b") {
        println!("[n1] A final sampled B.x={:.3}", s.position[0]);
    }

    // --- RESILIENCE: kill the server, let A drop, restart, resync ---
    println!("[n1] killing server (pid may differ) ...");
    let _ = server.kill();
    let _ = server.wait();
    tokio::time::sleep(Duration::from_millis(800)).await;
    println!("[n1] A status after kill: {:?}", a.status());

    server = spawn_server(&dir, port, save);
    // wait for A to reconnect + resync
    tokio::time::sleep(Duration::from_millis(4000)).await;
    println!("[n1] A status after restart: {:?}", a.status());

    let events = log.lock().unwrap().clone();
    let saw_reconnect = events.iter().any(|e| matches!(e, ClientEvent::Reconnected));
    let saw_resync = events
        .iter()
        .any(|e| matches!(e, ClientEvent::Resync { resync: true, .. }));
    println!("[n1] typed events after restart: reconnect={saw_reconnect} resync={saw_resync}");

    // WorldView consistent: fresh snapshot repopulated the scene entities.
    let entities = a.view().entities.len();
    println!("[n1] A view entities after resync: {entities}");
    assert!(entities > 0, "resync produced an empty world view");

    // Server lost the runtime presences; re-spawn A and confirm the echo.
    a.spawn_presence([0.0, 2.0, 22.0], 0.0).expect("A re-spawn");
    tokio::time::sleep(Duration::from_millis(400)).await;
    println!(
        "[n1] A re-spawned presence, A sees itself={}",
        a.has_entity("player-rust-a")
    );

    // --- clean close: server reaps on socket close ---
    b.close().await;
    a.close().await;
    tokio::time::sleep(Duration::from_millis(200)).await;

    let _ = server.kill();
    let _ = server.wait();
    println!(
        "[n1] closed clients, killed server. done. reconnect={saw_reconnect} resync={saw_resync}"
    );
    assert!(
        saw_reconnect && saw_resync,
        "expected Reconnected + Resync typed events"
    );
}
