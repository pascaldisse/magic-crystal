//! The append-only op journal: the physical ledger of a worldline.
//!
//! A journal is a versioned header (carrying the world seed) followed by
//! length-prefixed, CRC-guarded frames. Each frame is one recorded op batch
//! stamped with the entropy tick at which it was cast. The CRC per frame
//! (and per header) makes torn writes detectable: replay stops cleanly at the
//! last frame that survives intact, never applying a partial one.
//!
//! Layout (all integers little-endian):
//! ```text
//! header  = MAGIC[8] version:u16 seed:u64 crc32(header[0..18]):u32   (22 bytes)
//! frame   = len:u32 payload[len] crc32(payload):u32
//! payload = serde_json(JournalEntry)
//! ```

use crate::error::{SteinerError, TornKind};
use crystal::Op;
use serde::{Deserialize, Serialize};

/// Magic tag opening every Steiner journal.
pub const MAGIC: &[u8; 8] = b"STEINERJ";
/// On-disk format version, carried in the header.
pub const FORMAT_VERSION: u16 = 1;
/// Fixed header length in bytes: MAGIC(8) + version(2) + seed(8) + crc(4).
pub const HEADER_LEN: usize = 22;

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
    /// Every intact frame, in append order.
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
    /// Open a fresh journal for `seed` — writes the versioned header.
    pub fn new(seed: u64) -> Self {
        Self {
            buf: encode_header(seed),
            seed,
        }
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
        let payload = serde_json::to_vec(entry)?;
        let len: u32 = payload
            .len()
            .try_into()
            .map_err(|_| SteinerError::FrameTooLarge(payload.len()))?;
        self.buf.extend_from_slice(&len.to_le_bytes());
        self.buf.extend_from_slice(&payload);
        self.buf.extend_from_slice(&crc32(&payload).to_le_bytes());
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

/// Encode a versioned header carrying `seed`.
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

/// Parse and verify a journal header. Returns the seed.
pub fn read_header(bytes: &[u8]) -> Result<u64, SteinerError> {
    if bytes.len() < HEADER_LEN {
        return Err(SteinerError::TruncatedHeader);
    }
    if &bytes[0..8] != MAGIC {
        return Err(SteinerError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[8], bytes[9]]);
    if version != FORMAT_VERSION {
        return Err(SteinerError::UnsupportedVersion(version));
    }
    let stored = u32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);
    if crc32(&bytes[0..18]) != stored {
        return Err(SteinerError::HeaderCrc);
    }
    let seed = u64::from_le_bytes([
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17],
    ]);
    Ok(seed)
}

/// Decode a whole journal buffer. The header must verify (typed error otherwise);
/// a torn frame tail is reported, not fatal — replay stops at the last intact frame.
pub fn read_journal(bytes: &[u8]) -> Result<DecodedJournal, SteinerError> {
    let seed = read_header(bytes)?;
    let mut entries = Vec::new();
    let mut offset = HEADER_LEN;
    let mut outcome = ReadOutcome::Complete;

    loop {
        let remaining = bytes.len() - offset;
        if remaining == 0 {
            break; // clean frame boundary at EOF
        }
        if remaining < 4 {
            outcome = ReadOutcome::Torn {
                kind: TornKind::Truncated,
                valid_frames: entries.len(),
            };
            break;
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
            outcome = ReadOutcome::Torn {
                kind: TornKind::Truncated,
                valid_frames: entries.len(),
            };
            break;
        }
        let payload = &bytes[payload_start..crc_start];
        let stored = u32::from_le_bytes([
            bytes[crc_start],
            bytes[crc_start + 1],
            bytes[crc_start + 2],
            bytes[crc_start + 3],
        ]);
        if crc32(payload) != stored {
            outcome = ReadOutcome::Torn {
                kind: TornKind::FrameCrc,
                valid_frames: entries.len(),
            };
            break;
        }
        // A torn payload can be intact-CRC only by astronomical accident; a
        // decode error here is treated as a torn tail, never a panic.
        match serde_json::from_slice::<JournalEntry>(payload) {
            Ok(entry) => entries.push(entry),
            Err(_) => {
                outcome = ReadOutcome::Torn {
                    kind: TornKind::FrameCrc,
                    valid_frames: entries.len(),
                };
                break;
            }
        }
        offset = frame_end;
    }

    Ok(DecodedJournal {
        seed,
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
    read_header(bytes)?;
    let mut out = bytes[0..HEADER_LEN].to_vec();
    let mut offset = HEADER_LEN;
    loop {
        let remaining = bytes.len() - offset;
        if remaining < 4 {
            break;
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
            break;
        }
        let payload = &bytes[payload_start..crc_start];
        let stored = u32::from_le_bytes([
            bytes[crc_start],
            bytes[crc_start + 1],
            bytes[crc_start + 2],
            bytes[crc_start + 3],
        ]);
        if crc32(payload) != stored {
            break;
        }
        let entry: JournalEntry = match serde_json::from_slice(payload) {
            Ok(entry) => entry,
            Err(_) => break,
        };
        if entry.tick <= at_tick {
            out.extend_from_slice(&bytes[offset..frame_end]);
        }
        offset = frame_end;
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
