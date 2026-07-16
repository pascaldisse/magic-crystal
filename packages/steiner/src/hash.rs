//! The one canonical ECS-state digest, shared by every reader of a worldline.
//!
//! A worldline's observable state is `id -> component -> value`. Two paths
//! reach it — the [`Recorder`](crate::Recorder)'s live [`crystal::Core`] and a
//! wired [`WorldView`](../wired) snapshot — and both must agree bit-for-bit or
//! the live-record ordeal is meaningless. So the digest is defined ONCE here,
//! over a normalized component map, and every caller routes through it.
//!
//! Normalization: each entity's `{component: value}` map is round-tripped
//! through [`crystal::EntityDoc`] and re-serialized. That collapses the two
//! representations to one canonical JSON form (typed fields ordered by the
//! struct, unknown components preserved in `extra`), so a raw op value stored
//! by the recorder and the same value folded incrementally into a WorldView
//! hash identically. Keys are walked in sorted (`BTreeMap`) order — the digest
//! never depends on insertion order.

use crystal::EntityDoc;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A canonical worldline state: `gaia id -> component name -> component value`,
/// both levels sorted. The digest domain both readers build before hashing.
pub type StateMap = BTreeMap<String, BTreeMap<String, Value>>;

/// Digest a canonical [`StateMap`]. Each entity's component map is normalized
/// through [`EntityDoc`] first, so two encodings of the same state agree.
pub fn hash_state(state: &StateMap) -> u64 {
    let mut hash = FNV_OFFSET;
    for (id, components) in state {
        hash = fnv(hash, id.as_bytes());
        hash = fnv(hash, &[0x1e]);
        for (name, value) in normalize(components) {
            hash = fnv(hash, name.as_bytes());
            hash = fnv(hash, &[0x1f]);
            hash = fnv(hash, &serde_json::to_vec(&value).unwrap_or_default());
            hash = fnv(hash, &[0x00]);
        }
        hash = fnv(hash, &[0xff]);
    }
    hash
}

/// Round-trip one entity's component map through [`EntityDoc`] and return the
/// normalized, sorted `component -> value` pairs. Components that fail the
/// round-trip fall back to their raw value, so nothing is silently dropped.
fn normalize(components: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    let mut object = Map::new();
    for (name, value) in components {
        object.insert(name.clone(), value.clone());
    }
    let normalized = match serde_json::from_value::<EntityDoc>(Value::Object(object.clone())) {
        Ok(doc) => match serde_json::to_value(&doc) {
            Ok(Value::Object(map)) => map,
            _ => object,
        },
        Err(_) => object,
    };
    normalized.into_iter().collect()
}

fn fnv(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
