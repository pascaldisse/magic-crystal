//! The append-only op journal: the physical ledger of a worldline.
//!
//! A journal is a versioned header (carrying the world seed) followed by
//! length-prefixed, CRC-guarded frames. Each frame is one recorded op batch
//! stamped with the entropy tick at which it was cast. The CRC per frame
//! (and per header) makes torn writes detectable: replay stops cleanly at the
//! last frame that survives intact, never applying a partial one.
//!
//! Layout (all integers little-endian). Two header versions coexist; the
//! reader accepts both, so v1 files (genesis worldlines) never break:
//! ```text
//! v1 header = MAGIC[8] version:u16(=1) seed:u64 crc32(header[0..18]):u32   (22 bytes)
//! v2 header = MAGIC[8] version:u16(=2) seed:u64 snapshot_hash:u64
//!             crc32(header[0..26]):u32                                     (30 bytes)
//! frame     = len:u32 payload[len] crc32(payload):u32
//! payload   = serde_json(JournalEntry)   -- op frame
//!           | serde_json(SnapshotFrame)  -- v2 frame 0 ONLY (the base state)
//! ```
//! A v2 journal opens with exactly one SnapshotFrame (frame 0 — the server
//! snapshot a live session started from), then op frames. `snapshot_hash` in
//! the header pins that base state's identity. A v1 journal has no snapshot
//! frame and replays from genesis, unchanged.

use crate::error::{SteinerError, TornKind};
use crystal::Op;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Magic tag opening every Steiner journal.
pub const MAGIC: &[u8; 8] = b"STEINERJ";
/// On-disk format version for genesis (no-snapshot) worldlines.
pub const FORMAT_VERSION: u16 = 1;
/// On-disk format version for snapshot-prefixed (live-session) worldlines.
pub const FORMAT_VERSION_SNAPSHOT: u16 = 2;
/// v1 header length: MAGIC(8) + version(2) + seed(8) + crc(4).
pub const HEADER_LEN: usize = 22;
/// v2 header length: v1 fields + snapshot_hash(8).
pub const HEADER_LEN_V2: usize = 30;

/// Frame 0 of a v2 journal: the base ECS state a live session started from,
/// as `gaia id -> component name -> value`. Serialized deterministically
/// (sorted `BTreeMap`s) so re-writing it is byte-identical.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SnapshotFrame {
    /// The base world state, keyed by gaia id then component name.
    pub entities: BTreeMap<String, BTreeMap<String, Value>>,
}

/// A decoded, verified journal header. `snapshot_hash` is `Some` iff v2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// Format version (1 or 2).
    pub version: u16,
    /// The world seed.
    pub seed: u64,
    /// The base-snapshot digest (v2 only).
    pub snapshot_hash: Option<u64>,
    /// Header length in bytes (where frame 0 begins).
    pub len: usize,
}

/// One recorded op batch: the entropy tick, the source that cast it, and the ops.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JournalEntry {
    /// The entropy coordinate (tick) at which this batch was cast.
    pub tick: u64,
    /// The presence/daemon id that authored the batch, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// The op batch, applied verbatim on replay.
    pub ops: Vec<Op>,
}

/// How a journal read terminated. `Complete` means a clean frame boundary at
/// EOF; `Torn` means a truncated or CRC-broken tail was skipped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadOutcome {
    /// Read reached EOF exactly on a frame boundary.
    Complete,
    /// The tail was torn; `valid_frames` intact frames preceded it.
    Torn {
        /// What broke the final (skipped) frame.
        kind: TornKind,
        /// Count of intact frames applied before the tear.
        valid_frames: usize,
    },
}

/// The entries a journal decoded to, plus the tail status and the byte length
/// of the intact prefix (header + all valid frames).
#[derive(Clone, Debug)]
pub struct DecodedJournal {
    /// The world seed recovered from the header.
    pub seed: u64,
    /// The base snapshot (v2 frame 0), or `None` for a genesis (v1) journal.
    pub snapshot: Option<SnapshotFrame>,
    /// The snapshot digest from the header (v2 only).
    pub snapshot_hash: Option<u64>,
    /// Every intact op frame, in append order.
    pub entries: Vec<JournalEntry>,
    /// Whether the tail was clean or torn.
    pub outcome: ReadOutcome,
    /// Byte length of the intact prefix (header + valid frames). Truncating a
    /// buffer to this length yields a clean, torn-free journal.
    pub valid_len: usize,
}

/// Append-only journal writer over an in-memory byte buffer.
#[derive(Clone, Debug)]
pub struct JournalWriter {
    buf: Vec<u8>,
    seed: u64,
}

impl JournalWriter {
    /// Open a fresh v1 journal for `seed` — writes the genesis header.
    pub fn new(seed: u64) -> Self {
        Self {
            buf: encode_header(seed),
            seed,
        }
    }

    /// Open a fresh v2 journal for `seed` seeded by `snapshot` — writes the
    /// snapshot-carrying header, then the snapshot as frame 0. Subsequent
    /// [`append`](Self::append)s land after it.
    pub fn new_with_snapshot(seed: u64, snapshot: &SnapshotFrame) -> Result<Self, SteinerError> {
        let hash = crate::hash::hash_state(&snapshot.entities);
        let mut writer = Self {
            buf: encode_header_v2(seed, hash),
            seed,
        };
        writer.append_frame(&serde_json::to_vec(snapshot)?)?;
        Ok(writer)
    }

    /// Resume writing onto an existing intact byte prefix produced by this
    /// module (header included). Callers pass the `valid_len` prefix so torn
    /// tails are never reopened.
    pub fn from_prefix(bytes: Vec<u8>, seed: u64) -> Self {
        Self { buf: bytes, seed }
    }

    /// The world seed carried in the header.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Append one entry as a length-prefixed, CRC-guarded frame.
    pub fn append(&mut self, entry: &JournalEntry) -> Result<(), SteinerError> {
        self.append_frame(&serde_json::to_vec(entry)?)
    }

    /// Append a raw payload as one length-prefixed, CRC-guarded frame.
    fn append_frame(&mut self, payload: &[u8]) -> Result<(), SteinerError> {
        let len: u32 = payload
            .len()
            .try_into()
            .map_err(|_| SteinerError::FrameTooLarge(payload.len()))?;
        self.buf.extend_from_slice(&len.to_le_bytes());
        self.buf.extend_from_slice(payload);
        self.buf.extend_from_slice(&crc32(payload).to_le_bytes());
        Ok(())
    }

    /// Borrow the encoded journal bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Consume the writer and take its bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// Encode a v1 (genesis) header carrying `seed`.
pub fn encode_header(seed: u64) -> Vec<u8> {
    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    header.extend_from_slice(&seed.to_le_bytes());
    let crc = crc32(&header);
    header.extend_from_slice(&crc.to_le_bytes());
    debug_assert_eq!(header.len(), HEADER_LEN);
    header
}

/// Encode a v2 (snapshot-prefixed) header carrying `seed` and `snapshot_hash`.
pub fn encode_header_v2(seed: u64, snapshot_hash: u64) -> Vec<u8> {
    let mut header = Vec::with_capacity(HEADER_LEN_V2);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&FORMAT_VERSION_SNAPSHOT.to_le_bytes());
    header.extend_from_slice(&seed.to_le_bytes());
    header.extend_from_slice(&snapshot_hash.to_le_bytes());
    let crc = crc32(&header);
    header.extend_from_slice(&crc.to_le_bytes());
    debug_assert_eq!(header.len(), HEADER_LEN_V2);
    header
}

/// Parse and verify a journal header (v1 or v2). Returns the seed only — use
/// [`read_header_meta`] for the version and snapshot hash.
pub fn read_header(bytes: &[u8]) -> Result<u64, SteinerError> {
    Ok(read_header_meta(bytes)?.seed)
}

/// Parse and verify a journal header of either version, returning full metadata.
pub fn read_header_meta(bytes: &[u8]) -> Result<Header, SteinerError> {
    if bytes.len() < HEADER_LEN {
        return Err(SteinerError::TruncatedHeader);
    }
    if &bytes[0..8] != MAGIC {
        return Err(SteinerError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[8], bytes[9]]);
    let seed = u64::from_le_bytes([
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17],
    ]);
    match version {
        FORMAT_VERSION => {
            let stored = u32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);
            if crc32(&bytes[0..18]) != stored {
                return Err(SteinerError::HeaderCrc);
            }
            Ok(Header {
                version,
                seed,
                snapshot_hash: None,
                len: HEADER_LEN,
            })
        }
        FORMAT_VERSION_SNAPSHOT => {
            if bytes.len() < HEADER_LEN_V2 {
                return Err(SteinerError::TruncatedHeader);
            }
            let stored = u32::from_le_bytes([bytes[26], bytes[27], bytes[28], bytes[29]]);
            if crc32(&bytes[0..26]) != stored {
                return Err(SteinerError::HeaderCrc);
            }
            let snapshot_hash = u64::from_le_bytes([
                bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23], bytes[24],
                bytes[25],
            ]);
            Ok(Header {
                version,
                seed,
                snapshot_hash: Some(snapshot_hash),
                len: HEADER_LEN_V2,
            })
        }
        other => Err(SteinerError::UnsupportedVersion(other)),
    }
}

/// One intact frame's byte span, discovered by the frame walker.
struct Frame {
    payload_start: usize,
    payload_end: usize,
    frame_end: usize,
}

/// Read the next frame at `offset`. `Ok(Some(frame))` for an intact frame,
/// `Ok(None)` for a clean boundary at EOF, `Err(kind)` for a torn tail.
fn next_frame(bytes: &[u8], offset: usize) -> Result<Option<Frame>, TornKind> {
    let remaining = bytes.len() - offset;
    if remaining == 0 {
        return Ok(None);
    }
    if remaining < 4 {
        return Err(TornKind::Truncated);
    }
    let len = u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]) as usize;
    let payload_start = offset + 4;
    let crc_start = payload_start + len;
    let frame_end = crc_start + 4;
    if frame_end > bytes.len() {
        return Err(TornKind::Truncated);
    }
    let stored = u32::from_le_bytes([
        bytes[crc_start],
        bytes[crc_start + 1],
        bytes[crc_start + 2],
        bytes[crc_start + 3],
    ]);
    if crc32(&bytes[payload_start..crc_start]) != stored {
        return Err(TornKind::FrameCrc);
    }
    Ok(Some(Frame {
        payload_start,
        payload_end: crc_start,
        frame_end,
    }))
}

/// Decode a whole journal buffer (v1 or v2). The header must verify (typed
/// error otherwise); a torn frame tail is reported, not fatal — replay stops
/// at the last intact frame. A v2 journal's frame 0 is the base snapshot; a
/// torn snapshot frame yields `snapshot: None` and zero valid entries.
pub fn read_journal(bytes: &[u8]) -> Result<DecodedJournal, SteinerError> {
    let header = read_header_meta(bytes)?;
    let mut entries = Vec::new();
    let mut offset = header.len;
    let mut outcome = ReadOutcome::Complete;
    let mut snapshot = None;

    // v2: frame 0 is the base snapshot, read before any op frame.
    if header.version == FORMAT_VERSION_SNAPSHOT {
        match next_frame(bytes, offset) {
            Ok(Some(frame)) => match serde_json::from_slice::<SnapshotFrame>(
                &bytes[frame.payload_start..frame.payload_end],
            ) {
                Ok(snap) => {
                    snapshot = Some(snap);
                    offset = frame.frame_end;
                }
                Err(_) => {
                    outcome = ReadOutcome::Torn {
                        kind: TornKind::FrameCrc,
                        valid_frames: 0,
                    };
                    return Ok(DecodedJournal {
                        seed: header.seed,
                        snapshot: None,
                        snapshot_hash: header.snapshot_hash,
                        entries,
                        outcome,
                        valid_len: header.len,
                    });
                }
            },
            Ok(None) | Err(_) => {
                // A v2 journal with a torn/absent snapshot frame: no base state.
                outcome = ReadOutcome::Torn {
                    kind: TornKind::Truncated,
                    valid_frames: 0,
                };
                return Ok(DecodedJournal {
                    seed: header.seed,
                    snapshot: None,
                    snapshot_hash: header.snapshot_hash,
                    entries,
                    outcome,
                    valid_len: header.len,
                });
            }
        }
    }

    loop {
        match next_frame(bytes, offset) {
            Ok(None) => break,
            Ok(Some(frame)) => {
                match serde_json::from_slice::<JournalEntry>(
                    &bytes[frame.payload_start..frame.payload_end],
                ) {
                    Ok(entry) => entries.push(entry),
                    Err(_) => {
                        outcome = ReadOutcome::Torn {
                            kind: TornKind::FrameCrc,
                            valid_frames: entries.len(),
                        };
                        break;
                    }
                }
                offset = frame.frame_end;
            }
            Err(kind) => {
                outcome = ReadOutcome::Torn {
                    kind,
                    valid_frames: entries.len(),
                };
                break;
            }
        }
    }

    Ok(DecodedJournal {
        seed: header.seed,
        snapshot,
        snapshot_hash: header.snapshot_hash,
        entries,
        outcome,
        valid_len: offset,
    })
}

/// Fork a journal at entropy `at_tick`: copy the header and every frame whose
/// tick is `<= at_tick`, byte-for-byte, into a new buffer. The shared prefix is
/// therefore bit-identical to the parent — a new worldline that remembers the
/// same past.
pub fn fork_journal(bytes: &[u8], at_tick: u64) -> Result<Vec<u8>, SteinerError> {
    let header = read_header_meta(bytes)?;
    let mut out = bytes[0..header.len].to_vec();
    let mut offset = header.len;

    // v2: copy frame 0 (the base snapshot) verbatim — every fork shares it.
    if header.version == FORMAT_VERSION_SNAPSHOT {
        match next_frame(bytes, offset) {
            Ok(Some(frame)) => {
                out.extend_from_slice(&bytes[offset..frame.frame_end]);
                offset = frame.frame_end;
            }
            _ => return Ok(out),
        }
    }

    while let Ok(Some(frame)) = next_frame(bytes, offset) {
        match serde_json::from_slice::<JournalEntry>(&bytes[frame.payload_start..frame.payload_end])
        {
            Ok(entry) => {
                if entry.tick <= at_tick {
                    out.extend_from_slice(&bytes[offset..frame.frame_end]);
                }
            }
            Err(_) => break,
        }
        offset = frame.frame_end;
    }
    Ok(out)
}

/// CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320).
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}
