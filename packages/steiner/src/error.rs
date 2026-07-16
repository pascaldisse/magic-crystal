//! Typed failures for the journal organ. Torn-write handling never panics and
//! never partially applies — corruption surfaces as a value, not a crash.

use std::fmt;

/// Why the final frame of a journal was rejected (its ops are never applied).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TornKind {
    /// The tail ended before a full frame (length prefix or payload/CRC missing).
    Truncated,
    /// A frame's CRC did not match its payload (or the payload failed to decode).
    FrameCrc,
}

impl fmt::Display for TornKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TornKind::Truncated => f.write_str("truncated frame"),
            TornKind::FrameCrc => f.write_str("frame CRC mismatch"),
        }
    }
}

/// All ways a Steiner operation can fail.
#[derive(Debug)]
pub enum SteinerError {
    /// I/O against a journal file failed.
    Io(std::io::Error),
    /// Buffer was shorter than a full header.
    TruncatedHeader,
    /// The magic tag was absent — not a Steiner journal.
    BadMagic,
    /// The header declared a format version this build cannot read.
    UnsupportedVersion(u16),
    /// The header CRC did not verify.
    HeaderCrc,
    /// A single frame's payload exceeded `u32` bytes.
    FrameTooLarge(usize),
    /// Serializing or deserializing an entry payload failed.
    Serde(serde_json::Error),
    /// Applying an op batch to the ECS failed.
    Apply(String),
}

impl fmt::Display for SteinerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SteinerError::Io(error) => write!(f, "journal I/O: {error}"),
            SteinerError::TruncatedHeader => f.write_str("journal header is truncated"),
            SteinerError::BadMagic => f.write_str("not a Steiner journal (bad magic)"),
            SteinerError::UnsupportedVersion(version) => {
                write!(f, "unsupported journal format version {version}")
            }
            SteinerError::HeaderCrc => f.write_str("journal header CRC mismatch"),
            SteinerError::FrameTooLarge(len) => write!(f, "journal frame too large: {len} bytes"),
            SteinerError::Serde(error) => write!(f, "journal entry codec: {error}"),
            SteinerError::Apply(message) => write!(f, "op apply: {message}"),
        }
    }
}

impl std::error::Error for SteinerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SteinerError::Io(error) => Some(error),
            SteinerError::Serde(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SteinerError {
    fn from(error: std::io::Error) -> Self {
        SteinerError::Io(error)
    }
}

impl From<serde_json::Error> for SteinerError {
    fn from(error: serde_json::Error) -> Self {
        SteinerError::Serde(error)
    }
}
