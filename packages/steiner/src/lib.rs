//! # Reading Steiner — the world-history organ
//!
//! Steiner makes DreamForge's entropy law executable. From `ENTROPY.md`:
//! `state = f(seed, journal)` — a worldline is one `(seed, journal)` pair, the
//! arrow of time is the growing journal, and there is no other save. This crate
//! is the ledger and the reader of it.
//!
//! - [`journal`] — the append-only op journal: a versioned, seed-carrying header
//!   followed by length-prefixed, CRC-guarded frames (one recorded op batch per
//!   frame, stamped with its entropy tick). Torn writes are detected, never
//!   partially applied.
//! - [`Recorder`] — wraps a [`crystal::Core`]: every applied op batch lands in
//!   the journal; [`Recorder::replay`] rebuilds the ECS at any entropy point;
//!   [`Recorder::fork`] branches a new worldline that shares the parent's past.
//!
//! Determinism is the law's trial: a full replay of an intact journal produces
//! byte-identical journal bytes and a bit-identical ECS state hash.

pub mod error;
pub mod hash;
pub mod journal;
#[cfg(feature = "live")]
pub mod live;
pub mod realm;
mod recorder;

pub use error::{SteinerError, TornKind};
pub use hash::{hash_state, StateMap};
pub use journal::{
    encode_header_v2, fork_journal, read_header, read_header_meta, read_journal, DecodedJournal,
    Header, JournalEntry, JournalWriter, ReadOutcome, SnapshotFrame, FORMAT_VERSION,
    FORMAT_VERSION_SNAPSHOT, HEADER_LEN, HEADER_LEN_V2, MAGIC,
};
pub use realm::{materialize_state, RealmScenes, SceneDocument};
pub use recorder::Recorder;

#[cfg(feature = "live")]
pub use live::LiveTap;
