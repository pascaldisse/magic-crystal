//! # Jörmungandr — the residency ring (DreamForge R0)
//!
//! The World Serpent that holds the world in FINITE memory. In a universe with
//! zero loading (DREAMFORGE.md), the serpent encircling each observer IS the
//! streaming ring — tail-in-mouth OUROBOROS: cluster-pages recycled around a
//! ring of fixed byte budget. This crate is the R0 groundwork: keep the pages
//! required by the observer's CURRENT cut resident, over a hard byte budget,
//! evicting the pages FARTHEST from the observer first.
//!
//! It reads transmutation's chunked `.cbdg` artifact (packages/transmute/
//! FORMAT.md): a bounded INDEX (header + directory) loads EAGERLY on
//! [`Ring::mount`] — no geometry — and page payloads load LAZILY, exactly the
//! ones the cut needs, on [`Ring::update`].
//!
//! ## The three invariants (the serpent's law — exercised in `tests/ordeals.rs`)
//! 1. **Budget never exceeded** — resident bytes ≤ `budget` after EVERY update.
//! 2. **Required always resident** — every page in the current cut is resident
//!    after the update that names it (it is PINNED — never an eviction victim).
//! 3. **Determinism** — an identical observer flight replays an identical
//!    load/evict sequence.
//!
//! ## What a "page" costs
//! Budget accounting uses each page's on-disk chunk length ([`transmutation::
//! PageRef::len`]) — known from the INDEX alone, so the ring can plan a load or
//! eviction WITHOUT decoding any geometry. Eviction distance uses a page's
//! world-space AABB, computed once from its cluster bounds when the page is
//! read and then cached on the resident record.
//!
//! ## Read seam (this phase)
//! File reads are SYNCHRONOUS range reads (`seek` + `read_exact`) issued inside
//! [`Ring::update`] — the header, the directory, and each needed page range,
//! never the whole file. Async streaming is a later phase (tracked in
//! transmutation `serialize.rs`); the byte layout (absolute offsets) already
//! supports it, only the read seam is synchronous here.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use transmutation::{read_header, Directory, Page, HEADER_LEN};

/// Opaque handle to a mounted artifact (index = mount order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactId(pub u32);

/// A residency request key: exactly one cluster-page of one artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageKey {
    pub artifact: ArtifactId,
    pub page: u32,
}

impl PageKey {
    pub fn new(artifact: ArtifactId, page: u32) -> Self {
        Self { artifact, page }
    }
}

/// A world-space axis-aligned box — a page's spatial extent, merged from its
/// cluster bounding spheres. Used ONLY for eviction distance ordering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    /// Squared Euclidean distance from a point to the box (0 inside). f64
    /// accumulation for a stable, deterministic ordering.
    pub fn distance2(&self, p: [f32; 3]) -> f64 {
        let mut acc = 0.0f64;
        for ((&lo, &hi), &v) in self.min.iter().zip(&self.max).zip(&p) {
            let (lo, hi, v) = (lo as f64, hi as f64, v as f64);
            let d = if v < lo {
                lo - v
            } else if v > hi {
                v - hi
            } else {
                0.0
            };
            acc += d * d;
        }
        acc
    }
}

/// A resident page: its byte cost (on-disk chunk length) and cached AABB.
#[derive(Clone, Copy, Debug)]
pub struct ResidentPage {
    pub key: PageKey,
    /// On-disk page chunk length — the budget cost of holding it.
    pub bytes: u32,
    /// World-space extent (eviction distance metric).
    pub aabb: Aabb,
}

/// Typed failure — the ring NEVER panics on bad input or a torn artifact.
#[derive(Debug)]
pub enum RingError {
    /// A page id is not present in the artifact's directory.
    UnknownPage { artifact: ArtifactId, page: u32 },
    /// A key referenced an artifact that was never mounted.
    UnknownArtifact(ArtifactId),
    /// The current cut's total page bytes exceed the whole budget — the two
    /// invariants (fit the budget / keep the cut resident) cannot both hold.
    BudgetTooSmall { required_bytes: u64, budget: u64 },
    /// The artifact index (header/directory) is malformed or the version is
    /// unsupported.
    BadIndex(String),
    /// A page's byte range runs past EOF — a torn/truncated artifact.
    TornPage { artifact: ArtifactId, page: u32 },
    /// A page chunk failed to decode (corrupt payload).
    Corrupt { artifact: ArtifactId, page: u32 },
    /// Underlying filesystem error.
    Io(String),
}

impl std::fmt::Display for RingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RingError::UnknownPage { artifact, page } => {
                write!(f, "unknown page {page} in artifact {}", artifact.0)
            }
            RingError::UnknownArtifact(a) => write!(f, "unknown artifact {}", a.0),
            RingError::BudgetTooSmall {
                required_bytes,
                budget,
            } => write!(
                f,
                "current cut needs {required_bytes} B but budget is only {budget} B"
            ),
            RingError::BadIndex(e) => write!(f, "bad artifact index: {e}"),
            RingError::TornPage { artifact, page } => {
                write!(
                    f,
                    "torn page {page} in artifact {} (range past EOF)",
                    artifact.0
                )
            }
            RingError::Corrupt { artifact, page } => {
                write!(f, "corrupt page {page} in artifact {}", artifact.0)
            }
            RingError::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for RingError {}

/// One mounted artifact: its file path + the eagerly-loaded bounded index.
struct Artifact {
    path: PathBuf,
    dir: Directory,
}

/// Cumulative ring counters (for ordeals and telemetry).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RingStats {
    pub loads: u64,
    pub loaded_bytes: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
    /// High-water mark of resident bytes across all updates (≤ budget always).
    pub peak_resident_bytes: u64,
}

/// The outcome of one [`Ring::update`] tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tick {
    /// Every page resident after this update, in stable key order.
    pub resident: Vec<PageKey>,
    /// Pages read in from disk this tick (stable key order).
    pub loaded_this_tick: Vec<PageKey>,
    /// Pages evicted this tick, in eviction order (farthest first).
    pub evicted_this_tick: Vec<PageKey>,
    /// Resident bytes after this tick (≤ budget, always).
    pub resident_bytes: u64,
}

/// The residency ring: a fixed byte budget, a resident page pool, and the
/// mounted artifacts' indices.
pub struct Ring {
    budget: u64,
    resident: BTreeMap<PageKey, ResidentPage>,
    resident_bytes: u64,
    artifacts: Vec<Artifact>,
    stats: RingStats,
}

/// Default byte budget when a caller does not size the ring itself: 64 MiB.
/// A deliberately modest pool — the whole point is that a universe-scale
/// artifact is FAR larger than the ring, so residency (not "load it all") is
/// mandatory. Sized so a healthy multi-LOD cut fits with headroom; tune per
/// platform via [`Ring::new`].
pub const DEFAULT_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

impl Ring {
    /// A ring over a fixed byte `budget`.
    pub fn new(budget: u64) -> Self {
        Self {
            budget,
            resident: BTreeMap::new(),
            resident_bytes: 0,
            artifacts: Vec::new(),
            stats: RingStats::default(),
        }
    }

    /// A ring over [`DEFAULT_BUDGET_BYTES`].
    pub fn with_default_budget() -> Self {
        Self::new(DEFAULT_BUDGET_BYTES)
    }

    pub fn budget(&self) -> u64 {
        self.budget
    }
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
    pub fn stats(&self) -> RingStats {
        self.stats
    }
    pub fn is_resident(&self, key: PageKey) -> bool {
        self.resident.contains_key(&key)
    }
    /// Resident record for a page (bytes + cached AABB), if resident.
    pub fn resident_page(&self, key: PageKey) -> Option<&ResidentPage> {
        self.resident.get(&key)
    }
    /// Page count of a mounted artifact (from its index).
    pub fn page_count(&self, artifact: ArtifactId) -> Option<usize> {
        self.artifact(artifact).ok().map(|a| a.dir.pages.len())
    }

    /// Mount an artifact: EAGERLY read its bounded index (header + directory)
    /// — no geometry — and return a handle. Page payloads stay on disk until a
    /// cut requires them.
    pub fn mount<P: AsRef<Path>>(&mut self, path: P) -> Result<ArtifactId, RingError> {
        let path = path.as_ref().to_path_buf();
        let dir = read_index(&path)?;
        let id = ArtifactId(self.artifacts.len() as u32);
        self.artifacts.push(Artifact { path, dir });
        Ok(id)
    }

    fn artifact(&self, id: ArtifactId) -> Result<&Artifact, RingError> {
        self.artifacts
            .get(id.0 as usize)
            .ok_or(RingError::UnknownArtifact(id))
    }

    /// Advance the ring one tick: given the observer's position and the set of
    /// pages the CURRENT cut requires, make every required page resident while
    /// never exceeding the budget. Non-required resident pages are kept for
    /// reuse and evicted FARTHEST-first only to make room.
    ///
    /// Order of operations guarantees the budget is never transiently exceeded:
    /// page sizes come from the INDEX, so the ring frees space (evicting
    /// farthest non-required pages) BEFORE any read.
    pub fn update(
        &mut self,
        observer_pos: [f32; 3],
        required_pages: &[PageKey],
    ) -> Result<Tick, RingError> {
        // Dedup + validate the cut; sum its byte cost and pick the new loads.
        let mut required: std::collections::BTreeSet<PageKey> = std::collections::BTreeSet::new();
        for &k in required_pages {
            let art = self.artifact(k.artifact)?;
            if art.dir.pages.get(k.page as usize).is_none() {
                return Err(RingError::UnknownPage {
                    artifact: k.artifact,
                    page: k.page,
                });
            }
            required.insert(k);
        }

        let mut required_bytes: u64 = 0;
        let mut needed_new_bytes: u64 = 0;
        let mut to_load: Vec<PageKey> = Vec::new();
        for &k in &required {
            let len = self.page_len(k);
            required_bytes += len as u64;
            if !self.resident.contains_key(&k) {
                needed_new_bytes += len as u64;
                to_load.push(k);
            }
        }
        // If the cut itself cannot fit, both invariants are unsatisfiable.
        if required_bytes > self.budget {
            return Err(RingError::BudgetTooSmall {
                required_bytes,
                budget: self.budget,
            });
        }

        // Free space: evict FARTHEST-from-observer non-required pages until the
        // new loads fit. Because required_bytes ≤ budget, evicting every
        // non-required page always frees enough — the loop can never fail.
        let mut evicted: Vec<PageKey> = Vec::new();
        while self.resident_bytes + needed_new_bytes > self.budget {
            let victim = self
                .resident
                .values()
                .filter(|rp| !required.contains(&rp.key))
                // Farthest first; deterministic tie-break by key.
                .max_by(|a, b| {
                    a.aabb
                        .distance2(observer_pos)
                        .partial_cmp(&b.aabb.distance2(observer_pos))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.key.cmp(&b.key))
                })
                .map(|rp| rp.key);
            let Some(victim) = victim else { break };
            let rp = self.resident.remove(&victim).expect("victim resident");
            self.resident_bytes -= rp.bytes as u64;
            self.stats.evictions += 1;
            self.stats.evicted_bytes += rp.bytes as u64;
            evicted.push(victim);
        }

        // Load the new required pages (space is already reserved).
        to_load.sort_unstable();
        for &k in &to_load {
            let rp = self.load_page(k)?;
            self.resident_bytes += rp.bytes as u64;
            self.stats.loads += 1;
            self.stats.loaded_bytes += rp.bytes as u64;
            self.resident.insert(k, rp);
        }

        if self.resident_bytes > self.stats.peak_resident_bytes {
            self.stats.peak_resident_bytes = self.resident_bytes;
        }

        Ok(Tick {
            resident: self.resident.keys().copied().collect(),
            loaded_this_tick: to_load,
            evicted_this_tick: evicted,
            resident_bytes: self.resident_bytes,
        })
    }

    fn page_len(&self, k: PageKey) -> u32 {
        // Caller has already validated existence.
        self.artifacts[k.artifact.0 as usize].dir.pages[k.page as usize].len
    }

    /// Range-read one page from disk and compute its cached AABB.
    fn load_page(&self, k: PageKey) -> Result<ResidentPage, RingError> {
        let art = self.artifact(k.artifact)?;
        let pr = art
            .dir
            .pages
            .get(k.page as usize)
            .ok_or(RingError::UnknownPage {
                artifact: k.artifact,
                page: k.page,
            })?;
        let bytes = read_range(&art.path, pr.offset, pr.len).map_err(|e| match e {
            RangeError::Short => RingError::TornPage {
                artifact: k.artifact,
                page: k.page,
            },
            RangeError::Io(m) => RingError::Io(m),
        })?;
        let page: Page = bincode::deserialize(&bytes).map_err(|_| RingError::Corrupt {
            artifact: k.artifact,
            page: k.page,
        })?;
        Ok(ResidentPage {
            key: k,
            bytes: pr.len,
            aabb: page_aabb(&page),
        })
    }
}

/// Merge a page's cluster bounding spheres into a world-space AABB. An empty
/// page collapses to a point at the origin (it has no geometry to place).
fn page_aabb(page: &Page) -> Aabb {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for c in &page.clusters {
        let (ctr, r) = (c.bounds.center, c.bounds.radius);
        for ((mn, mx), &ct) in min.iter_mut().zip(max.iter_mut()).zip(&ctr) {
            *mn = mn.min(ct - r);
            *mx = mx.max(ct + r);
        }
    }
    if !min[0].is_finite() {
        return Aabb {
            min: [0.0; 3],
            max: [0.0; 3],
        };
    }
    Aabb { min, max }
}

/// Read + validate a `.cbdg` header, then range-read and decode ONLY the
/// bounded directory — never the geometry.
fn read_index(path: &Path) -> Result<Directory, RingError> {
    let mut head = [0u8; HEADER_LEN];
    read_exact_at(path, 0, &mut head).map_err(|e| match e {
        RangeError::Short => RingError::BadIndex("truncated header".into()),
        RangeError::Io(m) => RingError::Io(m),
    })?;
    let h = read_header(&head).map_err(|e| RingError::BadIndex(e.to_string()))?;
    let dir_bytes = read_range(path, h.dir_offset, h.dir_len).map_err(|e| match e {
        RangeError::Short => RingError::BadIndex("truncated directory".into()),
        RangeError::Io(m) => RingError::Io(m),
    })?;
    bincode::deserialize::<Directory>(&dir_bytes)
        .map_err(|e| RingError::BadIndex(format!("directory decode: {e}")))
}

enum RangeError {
    Short,
    Io(String),
}

fn read_range(path: &Path, offset: u64, len: u32) -> Result<Vec<u8>, RangeError> {
    let mut buf = vec![0u8; len as usize];
    read_exact_at(path, offset, &mut buf)?;
    Ok(buf)
}

fn read_exact_at(path: &Path, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
    let mut f = File::open(path).map_err(|e| RangeError::Io(e.to_string()))?;
    f.seek(SeekFrom::Start(offset))
        .map_err(|e| RangeError::Io(e.to_string()))?;
    match f.read_exact(buf) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Err(RangeError::Short),
        Err(e) => Err(RangeError::Io(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_distance_zero_inside_and_grows_outside() {
        let b = Aabb {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };
        assert_eq!(b.distance2([0.0, 0.0, 0.0]), 0.0);
        assert_eq!(b.distance2([4.0, 0.0, 0.0]), 9.0); // (4-1)^2
    }
}
