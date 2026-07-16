//! Versioned, CHUNKED binary (de)serialization of a transmuted `Dag` — the `.cbdg`
//! format (see FORMAT.md).
//!
//! WHY CHUNKED (finding 4): a whole-file bincode blob forces the entire Great
//! Chain resident to read anything — a non-starter at universe scale. This
//! layout is:
//!   header  →  fixed, always at offset 0
//!   pages   →  independently bincode-decodable geometry chunks (group- and
//!              leaf-granular; a page is exactly what RENDER.md streams)
//!   directory → bounded index: levels, group records, and a PageRef table
//!              (offset/len/level/deps) so ANY page is range-readable and the
//!              ROOT is loadable WITHOUT touching the rest.
//!
//! VERSIONING (finding 4): the header is stable forever; ANY layout or semantic
//! change bumps `FORMAT_VERSION`. "Additive serde field ⇒ no bump" is a LIE for
//! a range-indexed format — adding a field shifts every page offset — so it is
//! not the policy here.

use crate::dag::{Cluster, Dag, Group};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MAGIC: [u8; 4] = *b"CBDG";
/// v2 = chunked (header + pages + directory). v1 was the whole-file blob.
pub const FORMAT_VERSION: u16 = 2;
/// Fixed header size in bytes.
pub const HEADER_LEN: usize = 24;
const NO_ROOT: u32 = u32::MAX;

#[derive(Debug)]
pub enum SerdeError {
    BadMagic,
    UnsupportedVersion(u16),
    Truncated,
    BadOffset,
    Bincode(String),
}

impl std::fmt::Display for SerdeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerdeError::BadMagic => write!(f, "bad magic (not a CBDG cluster Great Chain)"),
            SerdeError::UnsupportedVersion(v) => write!(f, "unsupported format version {v}"),
            SerdeError::Truncated => write!(f, "truncated file/header"),
            SerdeError::BadOffset => write!(f, "chunk offset/length out of bounds"),
            SerdeError::Bincode(e) => write!(f, "bincode: {e}"),
        }
    }
}
impl std::error::Error for SerdeError {}

/// One page's index entry: a byte range + level + dependency page ids. Lets a
/// loader range-read exactly this page and follow the dependency graph without
/// decoding any geometry it does not need.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PageRef {
    pub id: u32,
    pub offset: u64,
    pub len: u32,
    pub level: u32,
    /// Cluster ids stored in this page.
    pub clusters: Vec<u32>,
    /// Pages this page's clusters depend on (children live there).
    pub deps: Vec<u32>,
}

/// The bounded directory: everything needed to plan residency WITHOUT reading
/// geometry. Range-read from `dir_offset`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Directory {
    pub input_tri_count: u32,
    pub partitioner: String,
    pub levels: Vec<Vec<u32>>,
    /// Group records (child/parent sets, shared LOD sphere + error).
    pub groups: Vec<Group>,
    /// Page index table (order = page id).
    pub pages: Vec<PageRef>,
    /// Coarsest-level page ids — load these first, render, then stream down.
    pub roots: Vec<u32>,
    /// cluster id → owning page id.
    pub cluster_page: Vec<u32>,
    /// Total cluster count (for reassembly bounds).
    pub cluster_count: u32,
}

impl Directory {
    pub fn page_ref(&self, id: u32) -> Option<&PageRef> {
        self.pages.get(id as usize)
    }
    /// Transitive closure of a page's dependencies (the subtree's pages),
    /// planned from the index alone — no geometry decode.
    pub fn subtree_pages(&self, root: u32) -> Vec<u32> {
        let mut seen = std::collections::BTreeSet::new();
        let mut stack = vec![root];
        while let Some(p) = stack.pop() {
            if !seen.insert(p) {
                continue;
            }
            if let Some(pr) = self.page_ref(p) {
                for &d in &pr.deps {
                    stack.push(d);
                }
            }
        }
        seen.into_iter().collect()
    }
}

/// One geometry page — independently bincode-decodable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Page {
    pub id: u32,
    pub clusters: Vec<Cluster>,
}

fn bc<T: Serialize>(v: &T) -> Result<Vec<u8>, SerdeError> {
    bincode::serialize(v).map_err(|e| SerdeError::Bincode(e.to_string()))
}
fn dc<'a, T: Deserialize<'a>>(b: &'a [u8]) -> Result<T, SerdeError> {
    bincode::deserialize(b).map_err(|e| SerdeError::Bincode(e.to_string()))
}

/// Assign every cluster to exactly one page: produced parents ride their
/// producing group's page; leaves ride their consuming (parent) group's leaf
/// page; ungrouped leaves ride a single orphan page. Deterministic ordering
/// (finding 8) so identical DAGs serialize byte-identically.
fn plan_pages(dag: &Dag) -> (Vec<Vec<u32>>, Vec<u32>, Vec<u32>, Vec<u32>) {
    // page rows (cluster ids), page level, page dep set — grown in a stable order.
    let mut rows: Vec<Vec<u32>> = Vec::new();
    let mut page_level: Vec<u32> = Vec::new();
    let mut cluster_page = vec![u32::MAX; dag.clusters.len()];

    // one page per group, holding that group's produced parents (id order).
    for g in &dag.groups {
        let pid = rows.len() as u32;
        let mut parents = g.parents.clone();
        parents.sort_unstable();
        for &c in &parents {
            cluster_page[c as usize] = pid;
        }
        // parents live at level+1 (children are at group.level).
        page_level.push(g.level + 1);
        rows.push(parents);
    }

    // leaves (cluster.group == None) bucketed by consuming group (stable order).
    let mut leaf_buckets: BTreeMap<i64, Vec<u32>> = BTreeMap::new();
    for c in &dag.clusters {
        if c.group.is_none() {
            let key = c.parent_group.map(|g| g as i64).unwrap_or(-1);
            leaf_buckets.entry(key).or_default().push(c.id);
        }
    }
    for (_k, mut ids) in leaf_buckets {
        ids.sort_unstable();
        let pid = rows.len() as u32;
        let level = ids
            .first()
            .map(|&c| dag.clusters[c as usize].level)
            .unwrap_or(0);
        for &c in &ids {
            cluster_page[c as usize] = pid;
        }
        page_level.push(level);
        rows.push(ids);
    }

    // roots = pages holding the coarsest (last) level's clusters.
    let mut roots: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    if let Some(top) = dag.levels.last() {
        for &c in top {
            let cp = cluster_page[c as usize];
            if cp != u32::MAX {
                roots.insert(cp);
            }
        }
    }
    (rows, page_level, cluster_page, roots.into_iter().collect())
}

/// Serialize a DAG to the chunked `.cbdg` blob.
pub fn serialize(dag: &Dag) -> Result<Vec<u8>, SerdeError> {
    let (rows, page_level, cluster_page, roots) = plan_pages(dag);

    // Build pages, computing offsets relative to the end of the header.
    let mut page_body = Vec::new();
    let mut page_refs: Vec<PageRef> = Vec::with_capacity(rows.len());
    // deps computed in plan_pages? recompute alongside for the PageRef table.
    let mut deps: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
    for row in &rows {
        let mut set = std::collections::BTreeSet::new();
        for &cid in row {
            for &child in &dag.clusters[cid as usize].children {
                let cp = cluster_page[child as usize];
                if cp != u32::MAX {
                    set.insert(cp);
                }
            }
        }
        deps.push(set.into_iter().collect());
    }

    for (pid, row) in rows.iter().enumerate() {
        let clusters: Vec<Cluster> = row.iter().map(|&c| dag.clusters[c as usize].clone()).collect();
        let page = Page {
            id: pid as u32,
            clusters,
        };
        let bytes = bc(&page)?;
        let offset = (HEADER_LEN + page_body.len()) as u64;
        let len = bytes.len() as u32;
        page_body.extend_from_slice(&bytes);
        page_refs.push(PageRef {
            id: pid as u32,
            offset,
            len,
            level: page_level[pid],
            clusters: row.clone(),
            deps: deps[pid].clone(),
        });
    }

    let directory = Directory {
        input_tri_count: dag.input_tri_count,
        partitioner: dag.partitioner.clone(),
        levels: dag.levels.clone(),
        groups: dag.groups.clone(),
        pages: page_refs,
        roots: roots.clone(),
        cluster_page,
        cluster_count: dag.clusters.len() as u32,
    };
    let dir_bytes = bc(&directory)?;
    let dir_offset = (HEADER_LEN + page_body.len()) as u64;

    let mut out = Vec::with_capacity(HEADER_LEN + page_body.len() + dir_bytes.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // flags
    out.extend_from_slice(&dir_offset.to_le_bytes());
    out.extend_from_slice(&(dir_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&roots.first().copied().unwrap_or(NO_ROOT).to_le_bytes());
    debug_assert_eq!(out.len(), HEADER_LEN);
    out.extend_from_slice(&page_body);
    out.extend_from_slice(&dir_bytes);
    Ok(out)
}

/// Parsed header (finding 4 — read this, then the directory, then only the
/// pages you need).
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub version: u16,
    pub flags: u16,
    pub dir_offset: u64,
    pub dir_len: u32,
    pub root_page: u32,
}

/// Read + validate the fixed header without touching the body.
pub fn read_header(bytes: &[u8]) -> Result<Header, SerdeError> {
    if bytes.len() < HEADER_LEN {
        return Err(SerdeError::Truncated);
    }
    if bytes[0..4] != MAGIC {
        return Err(SerdeError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != FORMAT_VERSION {
        return Err(SerdeError::UnsupportedVersion(version));
    }
    let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
    let dir_offset = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let dir_len = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let root_page = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
    Ok(Header {
        version,
        flags,
        dir_offset,
        dir_len,
        root_page,
    })
}

/// Read the directory (bounded index) — enough to plan residency, NO geometry.
pub fn read_directory(bytes: &[u8]) -> Result<Directory, SerdeError> {
    let h = read_header(bytes)?;
    let start = h.dir_offset as usize;
    let end = start
        .checked_add(h.dir_len as usize)
        .ok_or(SerdeError::BadOffset)?;
    if end > bytes.len() {
        return Err(SerdeError::BadOffset);
    }
    dc(&bytes[start..end])
}

/// Range-read a single page by its `PageRef` — decodes independently of any
/// other page (finding 4).
///
/// TRACKED (advisory, not built this pass): this and [`read_root`] take a full
/// in-memory `&[u8]`. For real file/HTTP residency streaming they want a
/// `ReadAt`/range-fetch adapter (`read_at(offset, len) -> Bytes`) so the header,
/// directory, and each page range load WITHOUT mapping the whole `.cbdg` — the
/// byte layout already supports it (absolute offsets), only the read seam does
/// not yet.
pub fn read_page(bytes: &[u8], pr: &PageRef) -> Result<Page, SerdeError> {
    let start = pr.offset as usize;
    let end = start
        .checked_add(pr.len as usize)
        .ok_or(SerdeError::BadOffset)?;
    if end > bytes.len() {
        return Err(SerdeError::BadOffset);
    }
    dc(&bytes[start..end])
}

/// Load ONLY the root (coarsest) pages — the "render something immediately
/// without the rest of the file" path (finding 4). Returns directory + the
/// root clusters.
pub fn read_root(bytes: &[u8]) -> Result<(Directory, Vec<Cluster>), SerdeError> {
    let dir = read_directory(bytes)?;
    let mut clusters = Vec::new();
    for &pid in &dir.roots {
        let pr = dir.page_ref(pid).ok_or(SerdeError::BadOffset)?;
        clusters.extend(read_page(bytes, pr)?.clusters);
    }
    Ok((dir, clusters))
}

/// Deserialize a full DAG (reads directory + every page, reassembles by id).
pub fn deserialize(bytes: &[u8]) -> Result<Dag, SerdeError> {
    let dir = read_directory(bytes)?;
    let mut clusters: Vec<Option<Cluster>> = vec![None; dir.cluster_count as usize];
    for pr in &dir.pages {
        let page = read_page(bytes, pr)?;
        for c in page.clusters {
            let idx = c.id as usize;
            if idx >= clusters.len() {
                return Err(SerdeError::BadOffset);
            }
            clusters[idx] = Some(c);
        }
    }
    let clusters: Vec<Cluster> = clusters
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(SerdeError::Truncated)?;
    Ok(Dag {
        clusters,
        groups: dir.groups,
        levels: dir.levels,
        input_tri_count: dir.input_tri_count,
        partitioner: dir.partitioner,
    })
}
