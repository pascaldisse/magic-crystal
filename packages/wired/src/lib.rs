//! # wired — the DreamForge presence protocol client
//!
//! An async WebSocket link from a Rust process to the LIVE GAIA world server
//! (`server/index.js`). The Glass speaks to the Bun/Node server through this:
//! connect, receive the world snapshot, hold a crystal-compatible entity map
//! ([`crystal::EntityMap`]), send op batches and presence moves, disconnect
//! cleanly (the server reaps the hello'd presence on socket close).
//!
//! ```no_run
//! # async fn demo() -> Result<(), String> {
//! use wired::{Config, Wired};
//! let client = Wired::connect(Config::with_port(8425).presence("player-rust"));
//! client.wait_live().await;
//! client.spawn_presence([0.0, 2.0, 22.0], 0.0)?;
//! client.move_presence([1.0, 2.0, 22.0], 1.57)?;
//! client.close().await;
//! # Ok(()) }
//! ```
//!
//! Protocol recon lives in [`codec`]'s module docs.

pub mod client;
pub mod codec;

pub use client::{default_url, fresh_client_id, Config, ReconnectPolicy, Status, Wired};
pub use codec::{
    apply_op, despawn_op, hello, merge_op, move_presence_ops, ops_batch, set_op, spawn_presence_op,
    WorldView,
};

// Re-export the protocol vocabulary so downstreams need only depend on wired.
pub use crystal::{EntityDoc, EntityMap, Op, WsMessage};
