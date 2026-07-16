//! Async WebSocket client — connect, hold a live [`WorldView`], send op
//! batches, reconnect, disconnect cleanly.
//!
//! One manager task owns the socket. Outbound messages ride an mpsc channel
//! so the handle is cheap to clone-send from anywhere; inbound frames fold
//! into a shared [`WorldView`] behind a mutex. Reconnect mirrors the JS
//! client (net.js): 500ms backoff doubling to 5s, hello re-sent on every
//! (re)connect so the presence stays reap-bound.

use crate::codec::{self, WorldView};
use crate::interp::{sample_of, InterpConfig, Interpolator, Sample};
use crystal::{Op, OpBatch, SetOp, WsMessage};
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc, watch};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;

/// A typed lifecycle signal about the LINK (distinct from world `event` ops).
/// Delivered on [`Wired::client_events`]; the consumer reacts to resyncs,
/// dropped ordering, and dead links without parsing raw frames.
#[derive(Clone, Debug, PartialEq)]
pub enum ClientEvent {
    /// A fresh snapshot atomically replaced the world view. `resync` is false
    /// for the very first snapshot, true for every one after (i.e. after a
    /// reconnect) — the interpolation buffers were cleared as part of it.
    Resync { entities: usize, resync: bool },
    /// The socket re-opened after a drop (before its resync snapshot lands).
    Reconnected,
    /// An ops batch's sequence counter skipped or went backwards — frames were
    /// lost or reordered on the wire. Only fires when the server stamps a `seq`
    /// (or `counter`) on its ops broadcasts.
    CounterGap { previous: u64, got: u64 },
    /// No inbound frame within the staleness window — the link may be dead.
    /// Fires once per stale spell; cleared by the next inbound frame.
    Stale { elapsed_ms: u64 },
}

/// Heartbeat + staleness tuning. Both knobs are optional (`None` disables).
#[derive(Clone, Copy, Debug)]
pub struct Resilience {
    /// Send a WebSocket Ping every interval to keep an idle link measured
    /// (the server pongs, which resets staleness). `None` sends no pings.
    pub heartbeat: Option<Duration>,
    /// Emit [`ClientEvent::Stale`] when no inbound frame (of any kind) arrives
    /// within this window. `None` disables staleness detection.
    pub staleness: Option<Duration>,
}

impl Default for Resilience {
    fn default() -> Self {
        Self {
            heartbeat: Some(Duration::from_secs(2)),
            staleness: Some(Duration::from_secs(6)),
        }
    }
}

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
    /// Presence interpolation tuning (remote entities).
    pub interp: InterpConfig,
    /// Heartbeat + staleness tuning.
    pub resilience: Resilience,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            url: default_url(),
            presence: None,
            client_id: fresh_client_id(),
            reconnect: ReconnectPolicy::default(),
            interp: InterpConfig::default(),
            resilience: Resilience::default(),
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

    /// Set the interpolation tuning (fluent).
    pub fn interp(mut self, interp: InterpConfig) -> Self {
        self.interp = interp;
        self
    }

    /// Set the resilience tuning (fluent).
    pub fn resilience(mut self, resilience: Resilience) -> Self {
        self.resilience = resilience;
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
    interp: Arc<Mutex<Interpolator>>,
    start: Instant,
    tx: mpsc::UnboundedSender<Command>,
    status: watch::Receiver<Status>,
    events: broadcast::Sender<serde_json::Value>,
    client_events: broadcast::Sender<ClientEvent>,
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
        let interp = Arc::new(Mutex::new(Interpolator::new(
            config.interp,
            config.presence.clone(),
        )));
        let start = Instant::now();
        let (tx, rx) = mpsc::unbounded_channel();
        let (status_tx, status_rx) = watch::channel(Status::Connecting);
        let (events_tx, _) = broadcast::channel(256);
        let (client_events_tx, _) = broadcast::channel(256);
        let (batches_tx, _) = broadcast::channel(1024);
        let task = tokio::spawn(run(
            config.clone(),
            view.clone(),
            interp.clone(),
            start,
            rx,
            status_tx,
            events_tx.clone(),
            client_events_tx.clone(),
            batches_tx.clone(),
        ));
        Self {
            config,
            view,
            interp,
            start,
            tx,
            status: status_rx,
            events: events_tx,
            client_events: client_events_tx,
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

    /// Subscribe to typed link lifecycle signals ([`ClientEvent`]): resyncs,
    /// reconnects, counter gaps, staleness.
    pub fn client_events(&self) -> broadcast::Receiver<ClientEvent> {
        self.client_events.subscribe()
    }

    /// The interpolation tuning in force.
    pub fn interp_config(&self) -> InterpConfig {
        self.config.interp
    }

    /// Interpolated position/facing for a REMOTE entity at the current render
    /// clock (stutter-free). `None` for the own presence or an unbuffered id —
    /// read the own body from [`Wired::view`] (local authority).
    pub fn sample(&self, id: &str) -> Option<Sample> {
        let now = self.start.elapsed().as_secs_f64();
        self.interp.lock().expect("interp mutex").sample(id, now)
    }

    /// Every buffered remote entity, interpolated at the current render clock.
    pub fn sampled_view(&self) -> BTreeMap<String, Sample> {
        let now = self.start.elapsed().as_secs_f64();
        self.interp.lock().expect("interp mutex").sampled_view(now)
    }

    /// The raw slam value the reference engine would show for `id` (newest
    /// state, no delay, no interpolation) — exposed for smooth-vs-slam checks.
    pub fn slam(&self, id: &str) -> Option<Sample> {
        let now = self.start.elapsed().as_secs_f64();
        self.interp
            .lock()
            .expect("interp mutex")
            .slam(id, now - self.config.interp.delay)
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

#[allow(clippy::too_many_arguments)]
async fn run(
    config: Config,
    view: Arc<Mutex<WorldView>>,
    interp: Arc<Mutex<Interpolator>>,
    start: Instant,
    mut rx: mpsc::UnboundedReceiver<Command>,
    status: watch::Sender<Status>,
    events: broadcast::Sender<serde_json::Value>,
    client_events: broadcast::Sender<ClientEvent>,
    batches: broadcast::Sender<OpBatch>,
) {
    // Timers fire always; a disabled knob is parked at a far-future period and
    // guarded at the emission site, so the select stays branch-static.
    let never = Duration::from_secs(365 * 24 * 3600);
    let heartbeat_period = config.resilience.heartbeat.unwrap_or(never);
    let staleness = config.resilience.staleness;
    let staleness_tick = staleness
        .map(|d| (d / 2).max(Duration::from_millis(250)))
        .unwrap_or(never);

    let mut backoff = config.reconnect.initial;
    let mut connections: u64 = 0;
    let mut snapshots: u64 = 0;
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
        if connections > 0 {
            let _ = client_events.send(ClientEvent::Reconnected);
        }
        connections += 1;
        let (mut write, mut read) = stream.split();

        // hello binds the presence for server-side reaping
        if let Some(presence) = &config.presence {
            let hello = serde_json::to_string(&codec::hello(presence.clone())).unwrap();
            if write.send(Message::Text(hello)).await.is_err() {
                continue;
            }
        }

        // per-connection resilience state
        let mut last_inbound = Instant::now();
        let mut stale_flagged = false;
        let mut last_seq: Option<u64> = None;
        let mut heartbeat = tokio::time::interval(heartbeat_period);
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        heartbeat.tick().await; // consume the immediate first tick
        let mut stale_timer = tokio::time::interval(staleness_tick);
        stale_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        stale_timer.tick().await;

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
                _ = heartbeat.tick() => {
                    if config.resilience.heartbeat.is_some()
                        && write.send(Message::Ping(Vec::new())).await.is_err() {
                        break false;
                    }
                }
                _ = stale_timer.tick() => {
                    if let Some(window) = staleness {
                        let elapsed = last_inbound.elapsed();
                        if elapsed > window && !stale_flagged {
                            stale_flagged = true;
                            let _ = client_events.send(ClientEvent::Stale {
                                elapsed_ms: elapsed.as_millis() as u64,
                            });
                        }
                    }
                }
                inbound = read.next() => match inbound {
                    Some(Ok(Message::Text(text))) => {
                        last_inbound = Instant::now();
                        stale_flagged = false;
                        if let Ok(message) = serde_json::from_str::<WsMessage>(&text) {
                            // steiner's live tap: publish the op batch BEFORE
                            // folding into the view — the acyclic seam.
                            if let WsMessage::Ops(ops) = &message {
                                let _ = batches.send(OpBatch {
                                    dev: ops.dev,
                                    ops: ops.ops.clone(),
                                    from: ops.from.clone(),
                                    extra: ops.extra.clone(),
                                });
                            }
                            let now = start.elapsed().as_secs_f64();
                            fold_inbound(
                                &message, &view, &interp, now,
                                &mut snapshots, &mut last_seq,
                                &events, &client_events,
                            );
                        }
                    }
                    Some(Ok(Message::Ping(payload))) => {
                        last_inbound = Instant::now();
                        stale_flagged = false;
                        let _ = write.send(Message::Pong(payload)).await;
                    }
                    Some(Ok(_)) => {
                        last_inbound = Instant::now();
                        stale_flagged = false;
                    }
                    Some(Err(_)) | None => break false,
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

/// Fold one inbound message: update the world view, record interpolation
/// samples, and emit world `event` ops plus typed [`ClientEvent`]s (resync,
/// counter gaps).
#[allow(clippy::too_many_arguments)]
fn fold_inbound(
    message: &WsMessage,
    view: &Arc<Mutex<WorldView>>,
    interp: &Arc<Mutex<Interpolator>>,
    now: f64,
    snapshots: &mut u64,
    last_seq: &mut Option<u64>,
    events: &broadcast::Sender<serde_json::Value>,
    client_events: &broadcast::Sender<ClientEvent>,
) {
    // Detect a sequence gap on ops BEFORE folding, so `previous` is honest.
    if let WsMessage::Ops(ops) = message {
        if let Some(seq) = ops_seq(ops) {
            if let Some(prev) = *last_seq {
                if seq != prev + 1 {
                    let _ = client_events.send(ClientEvent::CounterGap {
                        previous: prev,
                        got: seq,
                    });
                }
            }
            *last_seq = Some(seq);
        }
    }

    let emitted = view.lock().expect("view mutex").apply(message);
    for event in emitted {
        let _ = events.send(event);
    }

    // Record interpolation samples from the freshly-updated view.
    {
        let world = view.lock().expect("view mutex");
        let mut ip = interp.lock().expect("interp mutex");
        record_interp(&mut ip, message, &world, now);
    }

    if let WsMessage::Snapshot(snap) = message {
        *snapshots += 1;
        *last_seq = None; // fresh session — don't gap-fire across the resync
        let _ = client_events.send(ClientEvent::Resync {
            entities: snap.entities.len(),
            resync: *snapshots > 1,
        });
    }
}

/// A monotonic per-broadcast sequence from an ops message, if the server
/// stamps one (`seq` preferred, else `counter`). The reference server sends
/// neither, so gap detection stays dormant against it.
fn ops_seq(ops: &crystal::WsOpsMessage) -> Option<u64> {
    ops.extra
        .get("seq")
        .or_else(|| ops.extra.get("counter"))
        .and_then(Value::as_u64)
}

/// Buffer interpolation samples for the entities an inbound message touched.
/// A snapshot atomically clears then re-seeds every buffer (fresh session).
fn record_interp(interp: &mut Interpolator, message: &WsMessage, world: &WorldView, now: f64) {
    match message {
        WsMessage::Snapshot(_) => {
            interp.clear();
            interp.observe(&world.entities, now);
        }
        WsMessage::Ops(ops) => {
            for op in &ops.ops {
                match op {
                    Op::Set(SetOp { id, .. }) => record_one(interp, world, id, now),
                    Op::Other { op, fields } => match op.as_str() {
                        "spawn" => {
                            if let Some(id) = fields.get("id").and_then(Value::as_str) {
                                record_one(interp, world, id, now);
                            }
                        }
                        "despawn" => {
                            if let Some(id) = fields.get("id").and_then(Value::as_str) {
                                interp.forget(id);
                            }
                        }
                        "clear" => interp.clear(),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

fn record_one(interp: &mut Interpolator, world: &WorldView, id: &str, now: f64) {
    if let Some(doc) = world.entities.get(id) {
        if let Some(sample) = sample_of(doc) {
            interp.record(id, sample, now);
        }
    }
}
