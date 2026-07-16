//! Live ordeal driver — run against a wired-owned GAIA server.
//!
//!   GAIA_PORT=8425 bun server/index.js   # from the worktree
//!   GAIA_PORT=8425 cargo run -p wired --example ordeal
//!
//! Drives: connect → spawn presence → move → second client → clean close.
//! HTTP checks (/sense/query, /events) are done by the calling shell.

use std::time::Duration;
use wired::{Config, Wired};

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("GAIA_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8420);

    println!("[ordeal] port={port}");

    // --- client A: connect + spawn presence ---
    let a = Wired::connect(Config::with_port(port).presence("player-rust-a"));
    assert!(a.wait_live().await, "client A never went live");
    println!(
        "[ordeal] A live; snapshot entities={}",
        a.view().entities.len()
    );
    a.spawn_presence([0.0, 2.0, 22.0], 0.0).expect("A spawn");
    tokio::time::sleep(Duration::from_millis(400)).await;
    println!("[ordeal] A spawned presence player-rust-a");

    // --- set-op roundtrip: drop a marker entity, wait for the echo ---
    a.send_ops(vec![wired::spawn_presence_op(
        "wired-marker",
        [5.0, 2.0, 20.0],
        0.0,
    )])
    .expect("A marker spawn");
    tokio::time::sleep(Duration::from_millis(300)).await;
    println!(
        "[ordeal] A sees wired-marker back: {}",
        a.has_entity("wired-marker")
    );

    // --- move presence, echo folds back into A's view ---
    a.move_presence([3.0, 2.0, 18.0], 1.57).expect("A move");
    tokio::time::sleep(Duration::from_millis(400)).await;
    if let Some(doc) = a.view().entities.get("player-rust-a") {
        if let Some(pos) = doc.transform.as_ref().and_then(|t| t.position.as_ref()) {
            println!(
                "[ordeal] A presence pos after move: [{}, {}, {}]",
                pos[0], pos[1], pos[2]
            );
        }
    }

    // --- client B: sees A's presence in its own snapshot/stream ---
    let b = Wired::connect(Config::with_port(port).presence("player-rust-b"));
    assert!(b.wait_live().await, "client B never went live");
    b.spawn_presence([-2.0, 2.0, 22.0], 0.0).expect("B spawn");
    tokio::time::sleep(Duration::from_millis(500)).await;
    println!(
        "[ordeal] B sees A: {} | A sees B: {}",
        b.has_entity("player-rust-a"),
        a.has_entity("player-rust-b")
    );

    // --- clean shutdown: close B, server should reap player-rust-b ---
    b.close().await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    println!(
        "[ordeal] B closed; A still sees B (should reap → false): {}",
        a.has_entity("player-rust-b")
    );

    a.close().await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    println!("[ordeal] A closed. done.");
}
