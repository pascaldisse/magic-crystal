//! Async WebSocket client — connect, hold a live [`WorldView`], send op
//! batches, reconnect, disconnect cleanly.
//!
//! One manager task owns the socket. Outbound messages ride an mpsc channel
//! so the handle is cheap to clone-send from anywhere; inbound frames fold
//! into a shared [`WorldView`] behind a mutex. Reconnect mirrors the JS
//! client (net.js): 500ms backoff doubling to 5s, hello re-sent on every
//! (re)connect so the presence stays reap-bound.

use crate::codec::{self, WorldView};
use crystal::{Op, OpBatch, WsMessage};
use futures_util::{SinkExt, StreamExt};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc, watch};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;

/// Connection lifecycle, mirrored to a `watch` channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Dialing / redialing the server.
    Connecting,
    /// Socket open, snapshot expected, ops flow.
    Live,
    /// Between attempts, waiting out the backoff.
    Reconnecting,
    /// Terminal — `close()` called or reconnect disabled after a drop.
    Closed,
}

/// Reconnect backoff, matching the JS client defaults.
#[derive(Clone, Copy, Debug)]
pub struct ReconnectPolicy {
    /// When false, a dropped socket ends the client (goes [`Status::Closed`]).
    pub enabled: bool,
    /// First backoff after a drop.
    pub initial: Duration,
    /// Backoff ceiling (doubles up to this).
    pub max: Duration,
}

impl Default for ReconnectPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            initial: Duration::from_millis(500),
            max: Duration::from_millis(5000),
        }
    }
}

/// Where and how to connect.
#[derive(Clone, Debug)]
pub struct Config {
    /// WebSocket URL. Defaults to `ws://127.0.0.1:<GAIA_PORT|8420>`.
    pub url: String,
    /// Presence id to `hello` on every (re)connect. `None` skips hello.
    pub presence: Option<String>,
    /// `from` tag on outbound op batches. Defaults to a fresh client id.
    pub client_id: String,
    /// Reconnect policy.
    pub reconnect: ReconnectPolicy,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            url: default_url(),
            presence: None,
            client_id: fresh_client_id(),
            reconnect: ReconnectPolicy::default(),
        }
    }
}

impl Config {
    /// `ws://127.0.0.1:<port>` with the given port.
    pub fn with_port(port: u16) -> Self {
        Self {
            url: format!("ws://127.0.0.1:{port}"),
            ..Self::default()
        }
    }

    /// Set the presence id (fluent).
    pub fn presence(mut self, presence: impl Into<String>) -> Self {
        self.presence = Some(presence.into());
        self
    }
}

/// `ws://127.0.0.1:<port>`, honoring `GAIA_PORT` (default 8420).
pub fn default_url() -> String {
    let port = std::env::var("GAIA_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(8420);
    format!("ws://127.0.0.1:{port}")
}

/// Process-unique client id, shaped like the JS `c<base36>` tag.
pub fn fresh_client_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!("c{:x}{:x}", nanos & 0xffff_ffff, n)
}

enum Command {
    Send(Box<WsMessage>),
    Close,
}

/// A live link to the GAIA world server.
pub struct Wired {
    config: Config,
    view: Arc<Mutex<WorldView>>,
    tx: mpsc::UnboundedSender<Command>,
    status: watch::Receiver<Status>,
    events: broadcast::Sender<serde_json::Value>,
    batches: broadcast::Sender<OpBatch>,
    task: Option<JoinHandle<()>>,
}

impl Wired {
    /// Spawn the manager task and return a handle immediately. Connection
    /// proceeds in the background; await [`Wired::wait_live`] to gate on it.
    pub fn connect(config: Config) -> Self {
        let view = Arc::new(Mutex::new(WorldView {
            presence: config.presence.clone(),
            ..WorldView::default()
        }));
        let (tx, rx) = mpsc::unbounded_channel();
        let (status_tx, status_rx) = watch::channel(Status::Connecting);
        let (events_tx, _) = broadcast::channel(256);
        let (batches_tx, _) = broadcast::channel(1024);
        let task = tokio::spawn(run(
            config.clone(),
            view.clone(),
            rx,
            status_tx,
            events_tx.clone(),
            batches_tx.clone(),
        ));
        Self {
            config,
            view,
            tx,
            status: status_rx,
            events: events_tx,
            batches: batches_tx,
            task: Some(task),
        }
    }

    /// The `from` tag stamped on this client's op batches.
    pub fn client_id(&self) -> &str {
        &self.config.client_id
    }

    /// The presence id bound via `hello`, if any.
    pub fn presence(&self) -> Option<String> {
        self.config.presence.clone()
    }

    /// Current connection status.
    pub fn status(&self) -> Status {
        *self.status.borrow()
    }

    /// Await the socket becoming [`Status::Live`]. Returns false if the client
    /// reaches [`Status::Closed`] first.
    pub async fn wait_live(&self) -> bool {
        let mut status = self.status.clone();
        loop {
            match *status.borrow_and_update() {
                Status::Live => return true,
                Status::Closed => return false,
                _ => {}
            }
            if status.changed().await.is_err() {
                return false;
            }
        }
    }

    /// A snapshot clone of the current world view.
    pub fn view(&self) -> WorldView {
        self.view.lock().expect("view mutex").clone()
    }

    /// True when the entity map holds `id`.
    pub fn has_entity(&self, id: &str) -> bool {
        self.view
            .lock()
            .expect("view mutex")
            .entities
            .contains_key(id)
    }

    /// Subscribe to transient `event` ops as they arrive.
    pub fn events(&self) -> broadcast::Receiver<serde_json::Value> {
        self.events.subscribe()
    }

    /// The op event source: subscribe to every applied inbound op batch, in
    /// arrival order, exactly as it folded into the [`WorldView`]. This is the
    /// clean seam a journaling consumer (steiner's live tap) reads — wired
    /// publishes; the consumer decides tick coordinates and persistence, so
    /// wired never depends on the recorder (acyclic: steiner -> wired).
    pub fn op_batches(&self) -> broadcast::Receiver<OpBatch> {
        self.batches.subscribe()
    }

    /// Send a raw message. Errors only if the manager task is gone.
    pub fn send_message(&self, message: WsMessage) -> Result<(), String> {
        self.tx
            .send(Command::Send(Box::new(message)))
            .map_err(|_| "wired client is closed".to_string())
    }

    /// Send a gameplay op batch (tagged with this client's id).
    pub fn send_ops(&self, ops: Vec<Op>) -> Result<(), String> {
        self.send_message(codec::ops_batch(ops, self.config.client_id.clone(), false))
    }

    /// Send a dev op batch — the server writes these through to scene files.
    pub fn send_dev_ops(&self, ops: Vec<Op>) -> Result<(), String> {
        self.send_message(codec::ops_batch(ops, self.config.client_id.clone(), true))
    }

    /// Spawn this client's presence entity.
    pub fn spawn_presence(&self, position: [f64; 3], yaw: f64) -> Result<(), String> {
        let id = self
            .config
            .presence
            .clone()
            .ok_or("config has no presence id")?;
        self.send_ops(vec![codec::spawn_presence_op(&id, position, yaw)])
    }

    /// Move this client's presence (position + facing).
    pub fn move_presence(&self, position: [f64; 3], yaw: f64) -> Result<(), String> {
        let id = self
            .config
            .presence
            .clone()
            .ok_or("config has no presence id")?;
        self.send_ops(codec::move_presence_ops(&id, position, yaw))
    }

    /// Close the socket and stop reconnecting; awaits the manager task.
    pub async fn close(mut self) {
        let _ = self.tx.send(Command::Close);
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

impl Drop for Wired {
    fn drop(&mut self) {
        // Best-effort: signal close so a dropped handle doesn't leave the
        // socket (and its server-side presence) lingering.
        let _ = self.tx.send(Command::Close);
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

async fn run(
    config: Config,
    view: Arc<Mutex<WorldView>>,
    mut rx: mpsc::UnboundedReceiver<Command>,
    status: watch::Sender<Status>,
    events: broadcast::Sender<serde_json::Value>,
    batches: broadcast::Sender<OpBatch>,
) {
    let mut backoff = config.reconnect.initial;
    loop {
        let _ = status.send(Status::Connecting);
        let stream = match tokio_tungstenite::connect_async(&config.url).await {
            Ok((stream, _)) => stream,
            Err(_) => {
                if !config.reconnect.enabled {
                    let _ = status.send(Status::Closed);
                    return;
                }
                let _ = status.send(Status::Reconnecting);
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(config.reconnect.max);
                continue;
            }
        };
        backoff = config.reconnect.initial;
        let _ = status.send(Status::Live);
        let (mut write, mut read) = stream.split();

        // hello binds the presence for server-side reaping
        if let Some(presence) = &config.presence {
            let hello = serde_json::to_string(&codec::hello(presence.clone())).unwrap();
            if write.send(Message::Text(hello)).await.is_err() {
                continue;
            }
        }

        let graceful = loop {
            tokio::select! {
                command = rx.recv() => match command {
                    Some(Command::Send(message)) => {
                        let text = match serde_json::to_string(&*message) {
                            Ok(text) => text,
                            Err(_) => continue,
                        };
                        if write.send(Message::Text(text)).await.is_err() {
                            break false;
                        }
                    }
                    Some(Command::Close) | None => {
                        let _ = write.send(Message::Close(None)).await;
                        let _ = write.flush().await;
                        break true;
                    }
                },
                inbound = read.next() => match inbound {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(message) = serde_json::from_str::<WsMessage>(&text) {
                            if let WsMessage::Ops(ops) = &message {
                                let _ = batches.send(OpBatch {
                                    dev: ops.dev,
                                    ops: ops.ops.clone(),
                                    from: ops.from.clone(),
                                    extra: ops.extra.clone(),
                                });
                            }
                            let emitted = view.lock().expect("view mutex").apply(&message);
                            for event in emitted {
                                let _ = events.send(event);
                            }
                        }
                    }
                    Some(Ok(Message::Ping(payload))) => {
                        let _ = write.send(Message::Pong(payload)).await;
                    }
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break false,
                    Some(Ok(_)) => {}
                },
            }
        };

        if graceful || !config.reconnect.enabled {
            let _ = status.send(Status::Closed);
            return;
        }
        let _ = status.send(Status::Reconnecting);
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(config.reconnect.max);
    }
}
