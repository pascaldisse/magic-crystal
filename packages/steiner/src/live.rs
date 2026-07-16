//! The live tap — a running session becomes a worldline.
//!
//! ENTROPY law: `state = f(seed, journal)`. A LIVE session, though, does not
//! start at genesis — it starts from the server snapshot the client received
//! on connect. This module closes that gap: seed a [`Recorder`] from a wired
//! [`WorldView`] snapshot (journal frame 0), then journal every op batch the
//! client applies, stamped with a monotonic entropy tick. Offline, the same
//! `(seed, journal)` rebuilds the ECS to any tick.
//!
//! ## The seam (chosen, documented)
//! `steiner` depends on `wired` behind the `live` feature; `wired` never
//! depends on `steiner`, so the graph stays acyclic (`steiner -> wired ->
//! crystal`). `wired` publishes the op event source
//! ([`Wired::op_batches`](../../wired/struct.Wired.html)); the tap consumes it
//! and owns every journaling decision (tick assignment, persistence, replay).
//! Keeping the dependency optional leaves steiner's default build \u2014 the
//! determinism ordeals \u2014 free of tokio and the network stack.
//!
//! The tap is deliberately transport-agnostic: [`LiveTap::record_batch`] takes
//! a plain [`OpBatch`], so it journals a live `wired` stream, a replayed one,
//! or a scripted test feed identically.

use crate::hash::{hash_state, StateMap};
use crate::journal::SnapshotFrame;
use crate::{Recorder, SteinerError};
use crystal::{EntityMap, OpBatch};
use std::collections::BTreeMap;
use wired::WorldView;

/// A live session recorder: a [`Recorder`] seeded from a server snapshot, plus
/// a monotonic entropy clock advanced one tick per applied batch.
pub struct LiveTap {
    recorder: Recorder,
    tick: u64,
}

impl LiveTap {
    /// Open a live worldline seeded by a wired [`WorldView`] snapshot. The
    /// snapshot becomes journal frame 0; the entropy clock starts at 0 (the
    /// first recorded batch lands at tick 1).
    pub fn from_world_view(seed: u64, view: &WorldView) -> Result<Self, SteinerError> {
        Self::from_snapshot(seed, snapshot_of(&view.entities))
    }

    /// Open a live worldline from an explicit base [`SnapshotFrame`].
    pub fn from_snapshot(seed: u64, snapshot: SnapshotFrame) -> Result<Self, SteinerError> {
        Ok(Self {
            recorder: Recorder::from_snapshot(seed, snapshot)?,
            tick: 0,
        })
    }

    /// Journal one applied op batch at the next entropy tick; returns the tick
    /// it was stamped with. This is the tap's whole contract: every batch the
    /// client applied, recorded in arrival order.
    pub fn record_batch(&mut self, batch: &OpBatch) -> Result<u64, SteinerError> {
        self.tick += 1;
        self.recorder.record(batch, self.tick)?;
        Ok(self.tick)
    }

    /// The current entropy tick (count of batches recorded).
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Borrow the underlying recorder.
    pub fn recorder(&self) -> &Recorder {
        &self.recorder
    }

    /// The complete `(seed, journal)` bytes — snapshot frame 0 and every batch.
    pub fn journal_bytes(&self) -> &[u8] {
        self.recorder.journal_bytes()
    }

    /// The live ECS digest — equals [`world_view_hash`] of the client's view
    /// once every applied batch has been recorded.
    pub fn state_hash(&self) -> u64 {
        self.recorder.state_hash()
    }
}

/// Canonicalize a wired [`EntityMap`] into a [`SnapshotFrame`] (`gaia id ->
/// component -> value`) for journal frame 0.
pub fn snapshot_of(entities: &EntityMap) -> SnapshotFrame {
    SnapshotFrame {
        entities: entities_state(entities),
    }
}

/// The canonical [`StateMap`] of a wired [`WorldView`] — the same digest domain
/// a [`Recorder`] exposes, so live and replayed states compare directly.
pub fn world_view_hash(view: &WorldView) -> u64 {
    hash_state(&entities_state(&view.entities))
}

/// The canonical [`StateMap`] of a wired [`WorldView`] (for diffing/inspection).
pub fn world_view_state(view: &WorldView) -> StateMap {
    entities_state(&view.entities)
}

/// Flatten each typed [`EntityDoc`] back to its `component -> value` map.
fn entities_state(entities: &EntityMap) -> StateMap {
    let mut state = StateMap::new();
    for (id, doc) in entities {
        if let Ok(serde_json::Value::Object(map)) = serde_json::to_value(doc) {
            let components: BTreeMap<String, serde_json::Value> = map.into_iter().collect();
            state.insert(id.clone(), components);
        }
    }
    state
}
