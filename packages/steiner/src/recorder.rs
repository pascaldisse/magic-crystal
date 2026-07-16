//! The recorder: a [`Core`] wrapped by its journal.
//!
//! Every op batch applied through [`Recorder::record`] mutates the ECS AND lands
//! in the journal, stamped with its entropy tick. From `(seed, journal)` the ECS
//! is rebuilt bit-for-bit — [`Recorder::replay`] reconstructs any point on the
//! entropy x-axis, and [`Recorder::fork`] branches a new worldline that shares
//! the parent's past exactly.
//!
//! Op semantics (J0): a `set` op writes a component onto the gaia-bound entity
//! for its id (creating the entity and registering the component the first time
//! it is seen); `set` with a `null` value removes the component. This matches
//! `crystal`'s data-driven world loading, where unknown components are opaque
//! JSON buffers. Non-`set` ops are recorded verbatim but leave the ECS state
//! untouched — they are ECS-neutral protocol traffic for J0.

use crate::error::SteinerError;
use crate::journal::{fork_journal, read_journal, JournalEntry, JournalWriter, ReadOutcome};
use crystal::{Core, Op, OpBatch, WorldOptions};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A live worldline: the ECS core plus the append-only journal that records it.
pub struct Recorder {
    core: Core,
    journal: JournalWriter,
    /// gaia id -> its live component names. The authoritative index the recorder
    /// owns, so hashing walks the ECS in a fully deterministic order.
    index: BTreeMap<String, BTreeSet<String>>,
}

impl Recorder {
    /// Open a fresh worldline for `seed` with default world options.
    pub fn new(seed: u64) -> Self {
        Self::with_options(seed, WorldOptions::default())
    }

    /// Open a fresh worldline for `seed` with explicit ECS options.
    pub fn with_options(seed: u64, options: WorldOptions) -> Self {
        Self {
            core: Core::new(options),
            journal: JournalWriter::new(seed),
            index: BTreeMap::new(),
        }
    }

    /// The world seed of this worldline.
    pub fn seed(&self) -> u64 {
        self.journal.seed()
    }

    /// Borrow the wrapped core.
    pub fn core(&self) -> &Core {
        &self.core
    }

    /// Mutably borrow the wrapped core.
    pub fn core_mut(&mut self) -> &mut Core {
        &mut self.core
    }

    /// The encoded journal bytes — the complete `(seed, journal)` save.
    pub fn journal_bytes(&self) -> &[u8] {
        self.journal.as_bytes()
    }

    /// Apply an op batch to the ECS and append it to the journal at `tick`.
    pub fn record(&mut self, batch: &OpBatch, tick: u64) -> Result<(), SteinerError> {
        self.apply_ops(&batch.ops)?;
        self.journal.append(&JournalEntry {
            tick,
            source: batch.from.clone(),
            ops: batch.ops.clone(),
        })
    }

    /// Rebuild a worldline from `(seed, journal)`, replaying every frame whose
    /// tick is `<= until` (or all frames when `until` is `None`). The returned
    /// recorder re-journals what it replays, so a full replay of a `Complete`
    /// journal yields byte-identical journal bytes (the determinism ordeal).
    /// A torn tail is skipped cleanly; the [`ReadOutcome`] reports it.
    pub fn replay(bytes: &[u8], until: Option<u64>) -> Result<(Self, ReadOutcome), SteinerError> {
        let decoded = read_journal(bytes)?;
        let mut recorder = Recorder::new(decoded.seed);
        for entry in &decoded.entries {
            if until.is_some_and(|t| entry.tick > t) {
                continue;
            }
            recorder.record(
                &OpBatch {
                    dev: false,
                    ops: entry.ops.clone(),
                    from: entry.source.clone(),
                    extra: Default::default(),
                },
                entry.tick,
            )?;
        }
        Ok((recorder, decoded.outcome))
    }

    /// Resume a worldline from an intact journal buffer WITHOUT re-journaling:
    /// the ECS is rebuilt from the frames and the existing (truncated-to-valid)
    /// bytes are kept, so subsequent [`record`](Self::record) calls extend the
    /// same ledger. This is how a fork continues its parent's file.
    pub fn resume(bytes: &[u8]) -> Result<(Self, ReadOutcome), SteinerError> {
        let decoded = read_journal(bytes)?;
        let mut recorder = Recorder {
            core: Core::new(WorldOptions::default()),
            journal: JournalWriter::from_prefix(bytes[..decoded.valid_len].to_vec(), decoded.seed),
            index: BTreeMap::new(),
        };
        for entry in &decoded.entries {
            recorder.apply_ops(&entry.ops)?;
        }
        Ok((recorder, decoded.outcome))
    }

    /// Branch a new worldline at entropy `at_tick`: the child shares the parent's
    /// past bit-for-bit (identical prefix bytes and prefix state) and can then be
    /// recorded onto independently.
    pub fn fork(&self, at_tick: u64) -> Result<Self, SteinerError> {
        let child_bytes = fork_journal(self.journal_bytes(), at_tick)?;
        let (recorder, _) = Recorder::resume(&child_bytes)?;
        Ok(recorder)
    }

    /// A deterministic 64-bit digest of the entire ECS state (every gaia-bound
    /// entity and each of its component values, in sorted order). Two worldlines
    /// hash equal iff their observable component state is identical.
    pub fn state_hash(&self) -> u64 {
        let mut hash = FNV_OFFSET;
        for (id, components) in &self.index {
            hash = fnv(hash, id.as_bytes());
            hash = fnv(hash, &[0x1e]);
            let entity = self
                .core
                .world
                .entity_for_gaia(id)
                .expect("indexed gaia id must be bound");
            for name in components {
                let component = self
                    .core
                    .world
                    .component_id(name)
                    .expect("indexed component must be registered");
                let value = self
                    .core
                    .world
                    .get_component(entity, component)
                    .expect("indexed component must be present");
                hash = fnv(hash, name.as_bytes());
                hash = fnv(hash, &[0x1f]);
                hash = fnv(hash, &serde_json::to_vec(&value).unwrap_or_default());
                hash = fnv(hash, &[0x00]);
            }
            hash = fnv(hash, &[0xff]);
        }
        hash
    }

    /// Apply ops to the ECS + index without touching the journal.
    fn apply_ops(&mut self, ops: &[Op]) -> Result<(), SteinerError> {
        for op in ops {
            if let Op::Set(set) = op {
                self.apply_set(&set.id, &set.component, &set.value)?;
            }
            // Non-set ops are ECS-neutral for J0: recorded, not applied.
        }
        Ok(())
    }

    fn apply_set(&mut self, id: &str, component: &str, value: &Value) -> Result<(), SteinerError> {
        let world = &mut self.core.world;
        let entity = match world.entity_for_gaia(id) {
            Some(entity) => entity,
            None => {
                let entity = world.create_entity(vec![]).map_err(SteinerError::Apply)?;
                world
                    .bind_gaia_id(id, entity)
                    .map_err(SteinerError::Apply)?;
                self.index.entry(id.to_owned()).or_default();
                entity
            }
        };
        let component_id = match world.component_id(component) {
            Some(id) => id,
            None => world
                .register_component(opaque_component(component))
                .map_err(SteinerError::Apply)?,
        };
        let present = world.has_component(entity, component_id);
        if value.is_null() {
            if present {
                world
                    .remove_component(entity, component_id)
                    .map_err(SteinerError::Apply)?;
                if let Some(set) = self.index.get_mut(id) {
                    set.remove(component);
                }
            }
            return Ok(());
        }
        if present {
            world
                .set_component(entity, component_id, value.clone())
                .map_err(SteinerError::Apply)?;
        } else {
            world
                .add_component(entity, component_id, value.clone())
                .map_err(SteinerError::Apply)?;
        }
        self.index
            .entry(id.to_owned())
            .or_default()
            .insert(component.to_owned());
        Ok(())
    }
}

/// An opaque JSON-buffer component, matching `crystal`'s world loader for
/// unknown component names.
fn opaque_component(name: &str) -> crystal::ComponentDescriptor {
    crystal::ComponentDescriptor {
        name: name.to_owned(),
        fields: BTreeMap::new(),
        enableable: false,
        buffer: true,
        default: None,
    }
}

fn fnv(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
