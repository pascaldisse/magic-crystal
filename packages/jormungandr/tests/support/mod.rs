//! The committed GENERATOR — forges a large `.cbdg` artifact from PROCEDURAL
//! geometry at test time (never a committed blob; DREAMFORGE.md forbids
//! authored/pre-baked streaming data). All scratch lands under `target/` via
//! cargo's `CARGO_TARGET_TMPDIR` — NEVER `/tmp`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use transmutation::{
    read_directory, read_page, serialize, subdivided_cube, transmute_default, uv_sphere, Mesh,
    TransmuteParams, Vertex,
};

/// A page's world-space AABB derived (test-side) from its cluster bounds — the
/// same metric the ring caches internally, recomputed here so the flight can
/// choose a spatially-coherent cut and the ordeals can predict eviction order.
#[derive(Clone, Copy, Debug)]
pub struct PageBox {
    pub page: u32,
    pub len: u32,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl PageBox {
    pub fn center(&self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }
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

/// Weld several procedural primitives, scattered across space, into ONE mesh so
/// the transmutation produces MANY spatially-distinct pages — a stand-in for a
/// chunk of universe. Purely procedural, fully deterministic.
fn forge_mesh() -> Mesh {
    // A 3x3x3 lattice of subdivided cubes + spheres, spread far apart so pages
    // land at distinct world positions (good eviction-distance signal).
    let mut verts: Vec<Vertex> = Vec::new();
    let mut inds: Vec<u32> = Vec::new();
    let mut push = |m: &Mesh, off: [f32; 3]| {
        let base = verts.len() as u32;
        for v in &m.vertices {
            verts.push(Vertex::new(
                [
                    v.position[0] + off[0],
                    v.position[1] + off[1],
                    v.position[2] + off[2],
                ],
                v.normal,
                v.uv,
            ));
        }
        for &i in &m.indices {
            inds.push(base + i);
        }
    };
    let cube = subdivided_cube(1.6, 18);
    let ball = uv_sphere(1.0, 40, 28);
    let spacing = 24.0f32;
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                let off = [x as f32 * spacing, y as f32 * spacing, z as f32 * spacing];
                if (x + y + z) % 2 == 0 {
                    push(&cube, off);
                } else {
                    push(&ball, off);
                }
            }
        }
    }
    Mesh::new(verts, inds)
}

/// Suite-reliability fix (adversary review, VII-0b): `ordeals.rs` calls
/// `generate_artifact` from THREE tests (`ordeals_flight`,
/// `ordeal_eviction_order_honors_distance`, `ordeal_unknown_page_is_typed_error`)
/// that all share the SAME `dir` (`CARGO_TARGET_TMPDIR`, one directory for the
/// whole test binary). `cargo test` runs a binary's tests in parallel threads
/// by default, so three threads writing the SAME fixed filename
/// (`serpent.cbdg`) concurrently is a real torn-write/torn-read race — the
/// root cause of the observed flake. Fixed at the source: a unique filename
/// per call (process id + a monotonic in-process counter — the same "make
/// the shared name unique" pattern `packages/scrying-glass/tests/
/// vii0b_terrain.rs`'s temp-world helpers already use for the same reason).
/// Every caller uses the RETURNED path, never a hardcoded name, so this is a
/// pure implementation-detail fix — no call site changes needed.
static ARTIFACT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Forge the large artifact and write it to a per-call-unique
/// `dir/serpent_<pid>_<n>.cbdg`. Returns the path. Deterministic: the
/// artifact BYTES are identical every run (finding 8 byte-identity holds);
/// only the filename varies, to keep concurrent test threads from tearing
/// each other's writes.
pub fn generate_artifact(dir: &Path) -> PathBuf {
    let mesh = forge_mesh();
    let dag = transmute_default(&mesh, &TransmuteParams::default()).expect("transmute");
    let bytes = serialize(&dag).expect("serialize");
    let unique = ARTIFACT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("serpent_{}_{unique}.cbdg", std::process::id()));
    std::fs::write(&path, &bytes).expect("write artifact");
    path
}

/// Read the full page-box table (test-side spatial index) straight from the
/// file. Reads geometry — this is TEST scaffolding for choosing the cut, NOT
/// how the ring itself plans.
pub fn page_boxes(path: &Path) -> Vec<PageBox> {
    let bytes = std::fs::read(path).expect("read artifact");
    let dir = read_directory(&bytes).expect("directory");
    let mut out = Vec::with_capacity(dir.pages.len());
    for pr in &dir.pages {
        let page = read_page(&bytes, pr).expect("page");
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
            min = [0.0; 3];
            max = [0.0; 3];
        }
        out.push(PageBox {
            page: pr.id,
            len: pr.len,
            min,
            max,
        });
    }
    out
}
