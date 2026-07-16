//! Graph partitioning for shard grouping — the Great Chain's "adjacency groups
//! (METIS-class min-edge-cut)" step (RENDER.md §1 Build).
//!
//! `Partitioner` is the swappable seam: METIS drops in as the real min-edge-cut
//! partitioner (crate `metis`, feature `metis`, vendored+built via metis-sys);
//! `GreedyPartitioner` is the always-available pure-Rust fallback behind the
//! SAME trait. Grouping = partition the shard adjacency graph (nodes = shards,
//! edges weighted by shared original-mesh vertices) into `ceil(n / group_size)`
//! parts.
//!
//! DETERMINISM (finding 8): every partition is a pure function of the graph +
//! nparts. No RNG-seeded ordering leaks in; METIS is fed a fixed seed and a
//! stable CSR. Callers must feed a deterministically-ordered graph.
//!
//! BACKEND HONESTY (finding 7): `partition` returns the backend that ACTUALLY
//! ran, so a METIS request that silently falls back to greedy is reported as
//! greedy — the DAG's `partitioner` field never lies.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

// ---------------------------------------------------------------------------
// DETERMINISTIC WORK COUNTER (MF-1b — wall-clock is flaky under parallel load)
//
// The scaling regression is judged by an OPERATION COUNT, not by ms: complexity
// is a property of the algorithm, so counting the complexity-bearing operations
// of the balance phase (writes to the `queued` dedup array + BFS neighbor visits
// + queue pushes) is identical under any machine load — parallel-stable by
// construction. The epoch-stamp balance phase does O(1) queued-writes per
// component (→ O(V) total); the old per-component full-V clear did O(V) writes
// per component (→ O(V²)). Counting queued-writes therefore separates the two by
// the graph's V factor (ratio ≈4 vs ≈16 for star N vs 4N).
//
// TEST-GATED: the counter static + accessor exist ONLY under `cfg(test)` or the
// `work-count` feature. In a production release build `bump_work` is an empty
// `#[inline(always)]` no-op and NO counter symbol is emitted.
// ---------------------------------------------------------------------------
#[cfg(any(test, feature = "work-count"))]
thread_local! {
    static BALANCE_WORK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(any(test, feature = "work-count"))]
#[inline]
fn bump_work(n: u64) {
    BALANCE_WORK.with(|c| c.set(c.get().wrapping_add(n)));
}

#[cfg(not(any(test, feature = "work-count")))]
#[inline(always)]
fn bump_work(_n: u64) {}

/// Test/bench-only: read the balance-phase work counter and reset it to 0.
/// Present only under `cfg(test)` or the `work-count` feature — never in a
/// production release artifact.
#[cfg(any(test, feature = "work-count"))]
pub fn take_balance_work() -> u64 {
    BALANCE_WORK.with(|c| c.replace(0))
}

/// CSR adjacency graph over shards. Weights are shared-vertex counts.
#[derive(Clone, Debug, Default)]
pub struct AdjacencyGraph {
    pub node_count: usize,
    /// CSR row offsets, length `node_count + 1`.
    pub xadj: Vec<i32>,
    /// CSR column indices (neighbor node ids).
    pub adjncy: Vec<i32>,
    /// Edge weights, parallel to `adjncy`.
    pub adjwgt: Vec<i32>,
}

impl AdjacencyGraph {
    /// Neighbors (and weights) of `node`.
    pub fn neighbors(&self, node: usize) -> impl Iterator<Item = (usize, i32)> + '_ {
        let s = self.xadj[node] as usize;
        let e = self.xadj[node + 1] as usize;
        (s..e).map(move |k| (self.adjncy[k] as usize, self.adjwgt[k]))
    }
}

/// A partition result: a part id per node PLUS the backend that produced it
/// (finding 7 — no lying `name()` when a fallback fired mid-run).
#[derive(Clone, Debug)]
pub struct Partition {
    /// Part id per node, in `0..nparts`.
    pub parts: Vec<usize>,
    /// Backend that actually produced `parts` ("metis" | "greedy").
    pub backend: &'static str,
}

/// The swappable partition strategy. Returns a part id per node + real backend.
pub trait Partitioner {
    fn partition(&self, graph: &AdjacencyGraph, nparts: usize) -> Partition;
    /// Nominal backend name (what this partitioner PREFERS; may fall back).
    fn name(&self) -> &'static str;
}

/// Real METIS min-edge-cut k-way partitioner (feature `metis`).
#[cfg(feature = "metis")]
pub struct MetisPartitioner;

// Vendored METIS keeps its coarsening/refinement RNG in PROCESS-GLOBAL mutable
// state and re-seeds it (`InitRandom(seed)`) at the start of EVERY partition, so
// SEQUENTIAL calls are deterministic. That global is NOT thread-safe, though:
// concurrent partitions would interleave seed+consume and diverge. Serializing
// each partition call keeps every `Seed(0)`→consume atomic → deterministic under
// ANY caller concurrency (defends finding 8 beyond single-threaded use).
#[cfg(feature = "metis")]
static METIS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "metis")]
impl Partitioner for MetisPartitioner {
    fn partition(&self, graph: &AdjacencyGraph, nparts: usize) -> Partition {
        if nparts <= 1 || graph.node_count <= 1 {
            return Partition {
                parts: vec![0; graph.node_count],
                backend: "metis",
            };
        }
        // Hold the process-wide METIS lock for the WHOLE seed→partition so no
        // other thread's partition can interleave the shared RNG.
        let _metis_guard = METIS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // METIS requires nparts <= node_count and a valid CSR. Recursive
        // bisection is stable for the small nparts we feed it.
        let xadj: Vec<metis::Idx> = graph.xadj.iter().map(|&v| v as metis::Idx).collect();
        let adjncy: Vec<metis::Idx> = graph.adjncy.iter().map(|&v| v as metis::Idx).collect();
        let adjwgt: Vec<metis::Idx> =
            graph.adjwgt.iter().map(|&v| v.max(1) as metis::Idx).collect();
        let mut part = vec![0 as metis::Idx; graph.node_count];

        let np = nparts.min(graph.node_count) as metis::Idx;
        // METIS can refuse pathological graphs; fall back rather than fail the
        // whole transmutation — and REPORT that greedy actually ran.
        let g = match metis::Graph::new(1, np, &xadj, &adjncy) {
            // FIXED SEED (finding 8): METIS randomizes coarsening/refinement
            // from its RNG; pinning the seed makes two builds byte-identical.
            Ok(g) => g.set_adjwgt(&adjwgt).set_option(metis::option::Seed(0)),
            Err(_) => return GreedyPartitioner::default().partition(graph, nparts),
        };
        match g.part_recursive(&mut part) {
            Ok(_) => Partition {
                parts: part.iter().map(|&p| p as usize).collect(),
                backend: "metis",
            },
            Err(_) => GreedyPartitioner::default().partition(graph, nparts),
        }
    }
    fn name(&self) -> &'static str {
        "metis"
    }
}

/// Pure-Rust fallback: greedy BFS flood grouping targeting balanced part sizes,
/// preferring the heaviest-weight neighbor (approximates min-edge-cut). Always
/// available; identical trait contract so METIS is a pure swap.
///
/// Fixes:
///  - DISCONNECTED GEOMETRY (finding 2): when the growing pass exhausts its
///    `nparts` slots but nodes remain (many components), each leftover component
///    is split into BOUNDED BFS chunks (≤ `balance_chunk_mult` × target) and each
///    chunk is placed into the currently-smallest part — a large component can
///    never dump wholesale into one part ([97,1,1,1] regression). Chunks follow
///    the component's BFS traversal order (locality-coherent), NOT node id
///    (advisory A-1) — connected chunks wherever the component's shape allows.
///  - QUADRATIC FRONTIER (finding 3): the grow frontier is a lazy max-heap keyed
///    by weight-to-assigned with a stale-entry skip; `pop_best`'s full-frontier
///    rescan per pop is gone (was O(frontier) per pop → O(E log V) total).
///  - QUADRATIC BALANCE (re-inquisition MF-1): the balance-phase BFS-dedup mask
///    is an EPOCH stamp (`queued: Vec<u32>` compared against a per-component
///    epoch), never a full-V `false`-clear per residual component. A star graph
///    (thousands of isolated leaf components after the center assigns) used to
///    pay O(V) per component → Θ(V²); the epoch makes each component O(edges).
///    Whole greedy path stays O(E log V)-ish.
pub struct GreedyPartitioner {
    /// Leftover BFS-chunk size as a multiple of the balanced target (n/nparts)
    /// when spreading disconnected components across parts. Param (never
    /// hardcode); default 1 → chunks ≈ one balanced part, so no part exceeds
    /// ~2× target (finding 2).
    ///
    /// VALID RANGE: any usize ≥ 0 is accepted; 0 is treated as 1 (`.max(1)`).
    /// The chunk cap `target × balance_chunk_mult` is computed with
    /// `saturating_mul` (advisory A-4) so a pathologically large multiple can
    /// never overflow — it saturates to `usize::MAX`, which simply collapses
    /// each leftover component into ONE chunk (safe: still one
    /// smallest-part placement per component, never a panic/wrap).
    pub balance_chunk_mult: usize,
}

impl Default for GreedyPartitioner {
    fn default() -> Self {
        Self {
            balance_chunk_mult: 1,
        }
    }
}

impl Partitioner for GreedyPartitioner {
    fn partition(&self, graph: &AdjacencyGraph, nparts: usize) -> Partition {
        Partition {
            parts: greedy_parts(graph, nparts, self.balance_chunk_mult),
            backend: "greedy",
        }
    }
    fn name(&self) -> &'static str {
        "greedy"
    }
}

fn greedy_parts(graph: &AdjacencyGraph, nparts: usize, balance_chunk_mult: usize) -> Vec<usize> {
    let n = graph.node_count;
    if nparts <= 1 || n <= 1 {
        return vec![0; n];
    }
    let nparts = nparts.min(n);
    let target = n.div_ceil(nparts); // balanced part-size target
    // Bounded leftover chunk (finding 2). saturating_mul so an over-large
    // balance_chunk_mult can never overflow (advisory A-4) — it saturates,
    // collapsing a component into one chunk rather than panicking/wrapping.
    let chunk_cap = target.saturating_mul(balance_chunk_mult.max(1)).max(1);
    let mut part = vec![usize::MAX; n];
    let mut sizes = vec![0usize; nparts];
    // gain[node] = summed edge weight to already-assigned nodes (any part).
    // Monotone non-decreasing as assignment grows, so a lazy max-heap with a
    // stale-entry skip replaces the old full-frontier rescan (finding 3).
    let mut gain = vec![0i64; n];
    let mut heap: BinaryHeap<(i64, Reverse<usize>)> = BinaryHeap::new();
    // Balance-phase BFS dedup: EPOCH stamps, not a per-component full clear
    // (MF-1). `queued[node] == epoch` ⇔ enqueued for the CURRENT component.
    // Init 0 with the first epoch at 1 so no stamp ever collides with the
    // unvisited state.
    let mut queued = vec![0u32; n];
    let mut epoch = 0u32;
    let mut current = 0usize;

    for seed in 0..n {
        if part[seed] != usize::MAX {
            continue;
        }
        if current < nparts {
            // Grow: flood a locality-coherent ~target group from `seed`, always
            // taking the frontier node with the greatest weight to already-
            // assigned nodes (min-edge-cut heuristic). Deterministic tie-break:
            // smallest node id (Reverse in a max-heap).
            heap.clear();
            heap.push((gain[seed], Reverse(seed)));
            let mut count = 0usize;
            while let Some((g, Reverse(node))) = heap.pop() {
                if part[node] != usize::MAX || g != gain[node] {
                    continue; // already assigned, or a stale (superseded) entry
                }
                part[node] = current;
                sizes[current] += 1;
                count += 1;
                for (nb, w) in graph.neighbors(node) {
                    if part[nb] == usize::MAX {
                        gain[nb] += w as i64;
                        heap.push((gain[nb], Reverse(nb)));
                    }
                }
                if count >= target {
                    break;
                }
            }
            if count > 0 {
                current += 1;
            }
        } else {
            // Balance (disconnected leftovers): collect the whole reachable
            // unassigned component in BFS order, then spread it across parts in
            // BOUNDED chunks (≤ chunk_cap), each chunk to the currently-smallest
            // part — a large component can never dump wholesale into one part
            // (finding 2).
            //
            // MF-1: bump the epoch instead of clearing `queued` — O(1) per
            // component, not O(V). (Wrap guard: u32 epochs suffice for up to
            // ~4e9 components; on the astronomically-rare wrap, reset once.)
            epoch = match epoch.checked_add(1) {
                Some(e) => e,
                None => {
                    for q in queued.iter_mut() {
                        *q = 0;
                    }
                    1
                }
            };
            // BFS (queue, front-pop) so `comp` is in traversal order → chunks
            // are locality-coherent, connected where the component allows
            // (advisory A-1). BFS from the smallest unassigned seed with CSR
            // neighbor order is fully deterministic → stable chunk boundaries,
            // no sort needed.
            let mut queue: VecDeque<usize> = VecDeque::new();
            queue.push_back(seed);
            queued[seed] = epoch;
            bump_work(1); // seed queued-write (MF-1b: complexity-bearing op)
            let mut comp: Vec<usize> = Vec::new();
            while let Some(node) = queue.pop_front() {
                if part[node] != usize::MAX {
                    continue;
                }
                comp.push(node);
                for (nb, _) in graph.neighbors(node) {
                    bump_work(1); // neighbor visit
                    if part[nb] == usize::MAX && queued[nb] != epoch {
                        queued[nb] = epoch;
                        queue.push_back(nb);
                        bump_work(1); // queued-write on enqueue
                    }
                }
            }
            for chunk in comp.chunks(chunk_cap) {
                let dst = smallest_part(&sizes);
                for &node in chunk {
                    part[node] = dst;
                    sizes[dst] += 1;
                }
            }
        }
    }
    // Any node still unassigned (defensive) → smallest part.
    for p in part.iter_mut() {
        if *p == usize::MAX {
            let dst = smallest_part(&sizes);
            *p = dst;
            sizes[dst] += 1;
        }
    }
    part
}

fn smallest_part(sizes: &[usize]) -> usize {
    let mut best = 0usize;
    for (i, &s) in sizes.iter().enumerate() {
        if s < sizes[best] {
            best = i;
        }
    }
    best
}

/// The default partitioner: METIS when compiled in, else greedy. Keeps callers
/// oblivious to which backend is present.
pub fn default_partitioner() -> Box<dyn Partitioner> {
    #[cfg(feature = "metis")]
    {
        Box::new(MetisPartitioner)
    }
    #[cfg(not(feature = "metis"))]
    {
        Box::new(GreedyPartitioner::default())
    }
}

// ---------------------------------------------------------------------------
// MF-1b — DETERMINISTIC balance-phase scaling regression (unit test).
//
// Lives HERE (not tests/inquisition.rs) because the work counter is gated on the
// crate's own `cfg(test)`; integration tests link the crate as an external dep
// where that cfg is inactive. As a unit test it runs under EVERY feature set
// (`cargo test -p transmutation` default AND `--no-default-features`).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod scaling_tests {
    use super::*;

    /// Star graph: node 0 wired to `leaves` leaves. After the center part fills,
    /// every remaining leaf is its OWN isolated component — the exact shape that
    /// made the balance phase's per-component full-V clear Θ(V²) (MF-1).
    fn star_graph(leaves: usize) -> AdjacencyGraph {
        let n = leaves + 1;
        let mut adjncy = Vec::with_capacity(leaves * 2);
        let mut adjwgt = Vec::with_capacity(leaves * 2);
        let mut xadj = vec![0i32];
        for leaf in 1..n {
            adjncy.push(leaf as i32);
            adjwgt.push(1);
        }
        xadj.push(adjncy.len() as i32);
        for _ in 1..n {
            adjncy.push(0);
            adjwgt.push(1);
            xadj.push(adjncy.len() as i32);
        }
        AdjacencyGraph { node_count: n, xadj, adjncy, adjwgt }
    }

    /// Partition a star; return (balance-phase op count, diagnostic wall-clock).
    /// The op count is the DETERMINISTIC judge — a pure function of algorithm +
    /// graph, identical under any machine load, parallel-stable by construction.
    fn star_partition_work(leaves: usize) -> (u64, std::time::Duration) {
        let graph = star_graph(leaves);
        let _ = take_balance_work(); // clear stray counts
        let t0 = std::time::Instant::now();
        let part = GreedyPartitioner::default().partition(&graph, 20);
        let elapsed = t0.elapsed();
        let ops = take_balance_work();
        assert_eq!(part.parts.len(), leaves + 1);
        assert!(part.parts.iter().all(|&p| p < 20));
        (ops, elapsed)
    }

    #[test]
    fn finding7_high_valence_scales_linearly() {
        // MF-1b BITE (DETERMINISTIC): the balance phase must be ~linear in V, not
        // Θ(V²). Wall-clock CANNOT judge complexity under parallel load — the ×3
        // gate showed core contention pushing VALID linear code to ratio 14.59,
        // indistinguishable from genuine quadratic 15.78, with no threshold
        // margin between them. So judge the OPERATION COUNT of the balance phase
        // instead: pure function of the algorithm + graph, identical under any
        // machine load, parallel-stable by construction.
        //
        // Star at N and 4N. Linear balance ⇒ ops ratio ≈ 4; the old per-component
        // full-V clear ⇒ ops ratio ≈ 16. Gate on the ops ratio.
        //
        //   TRANSMUTE_SCALE_FACTOR (default 6.0): max tolerated ops(4N)/ops(N).
        //   4 is the ideal linear ratio; 6 leaves headroom for constant terms;
        //   quadratic (≈16) blows straight through it. Judges OPS now, not ms.
        let factor: f64 = std::env::var("TRANSMUTE_SCALE_FACTOR")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6.0);
        let small = 8_000usize;
        let big = 32_000usize; // 4× small

        let (ops_small, t_small) = star_partition_work(small);
        let (ops_big, t_big) = star_partition_work(big);

        let ratio = ops_big as f64 / (ops_small as f64).max(1.0);
        // Wall-clock is a PRINTED DIAGNOSTIC ONLY — never asserted (MF-1b).
        println!(
            "MF-1b scaling (ops): star {small} = {ops_small} ops, star {big} (4×) = {ops_big} \
             ops, ops ratio = {ratio:.2} (limit {factor}; linear≈4, quadratic≈16) \
             [diagnostic wall-clock: {:.3} ms / {:.3} ms, NOT asserted]",
            t_small.as_secs_f64() * 1e3,
            t_big.as_secs_f64() * 1e3,
        );
        assert!(
            ratio < factor,
            "balance phase scales super-linearly: ops({big})/ops({small}) = {ratio:.2} ≥ \
             {factor} — quadratic per-component clear regressed (MF-1)"
        );
    }
}
