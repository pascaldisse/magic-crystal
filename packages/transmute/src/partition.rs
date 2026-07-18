//! The Great Chain's sole graph partitioner: vendored METIS min-edge-cut.
//!
//! There is no feature gate and no fallback. A build without METIS is unsupported;
//! a METIS error fails transmutation loudly. Every METIS seed derives from the
//! ENTROPY coordinate: `hash(world_seed, entropy, canonical_geometry_identity)`.

use crate::dag::TransmuteError;

/// CSR adjacency graph over canonical-geometry shards. Weights are
/// shared-vertex counts; callers construct it in stable canonical order.
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

/// The complete deterministic coordinate for a single METIS invocation.
/// `geometry_identity` is a hash of the canonical geometry being partitioned,
/// not an allocation address or traversal accident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartitionEntropy {
    pub world_seed: u64,
    pub entropy: u64,
    pub geometry_identity: u64,
}

impl PartitionEntropy {
    /// Stable non-cryptographic mixing: all METIS-visible randomness comes from
    /// this exact `(world_seed, entropy, canonical_geometry_identity)` tuple.
    fn metis_seed(self) -> metis::Idx {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for word in [self.world_seed, self.entropy, self.geometry_identity] {
            h ^= word;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
            h ^= h >> 32;
        }
        // METIS accepts a signed idx seed. Keep it non-negative and never use a
        // magic fixed value: even a zero world coordinate has geometry identity.
        (h & 0x7fff_ffff) as metis::Idx
    }
}

/// Partition result from the sole backend.
#[derive(Clone, Debug)]
pub struct Partition {
    /// Part id per node, in `0..nparts`.
    pub parts: Vec<usize>,
}

/// Real vendored-METIS partitioner. Its RNG is process-global, so the mutex
/// holds the seed→partition interval atomically across concurrent callers.
pub struct MetisPartitioner;

static METIS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

impl MetisPartitioner {
    /// Partition exactly with METIS or return a typed transmutation failure.
    pub fn partition(
        &self,
        graph: &AdjacencyGraph,
        nparts: usize,
        entropy: PartitionEntropy,
    ) -> Result<Partition, TransmuteError> {
        if nparts <= 1 || graph.node_count <= 1 {
            return Ok(Partition {
                parts: vec![0; graph.node_count],
            });
        }
        let _metis_guard = METIS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xadj: Vec<metis::Idx> = graph.xadj.iter().map(|&v| v as metis::Idx).collect();
        let adjncy: Vec<metis::Idx> = graph.adjncy.iter().map(|&v| v as metis::Idx).collect();
        let adjwgt: Vec<metis::Idx> = graph
            .adjwgt
            .iter()
            .map(|&v| v.max(1) as metis::Idx)
            .collect();
        let mut part = vec![0 as metis::Idx; graph.node_count];
        let np = nparts.min(graph.node_count) as metis::Idx;
        let graph = metis::Graph::new(1, np, &xadj, &adjncy)
            .map_err(|error| TransmuteError::Partition(format!("METIS graph: {error}")))?
            .set_adjwgt(&adjwgt)
            .set_option(metis::option::Seed(entropy.metis_seed()));
        graph
            .part_recursive(&mut part)
            .map_err(|error| TransmuteError::Partition(format!("METIS partition: {error}")))?;
        Ok(Partition {
            parts: part.into_iter().map(|p| p as usize).collect(),
        })
    }
}

/// The one permitted partitioner; no alternate implementation exists.
pub fn default_partitioner() -> MetisPartitioner {
    MetisPartitioner
}

#[cfg(test)]
mod tests {
    use super::PartitionEntropy;

    #[test]
    fn entropy_seed_is_a_function_of_all_required_coordinates() {
        let base = PartitionEntropy {
            world_seed: 11,
            entropy: 22,
            geometry_identity: 33,
        };
        assert_ne!(
            base.metis_seed(),
            PartitionEntropy {
                world_seed: 12,
                ..base
            }
            .metis_seed()
        );
        assert_ne!(
            base.metis_seed(),
            PartitionEntropy {
                entropy: 23,
                ..base
            }
            .metis_seed()
        );
        assert_ne!(
            base.metis_seed(),
            PartitionEntropy {
                geometry_identity: 34,
                ..base
            }
            .metis_seed()
        );
    }
}
