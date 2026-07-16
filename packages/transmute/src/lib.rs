//! transmutation — DreamForge offline mesh → cluster Great Chain (TRUE NAME:
//! GRIMOIRE, gaia-dreamforge repo; the b-word is forbidden vocabulary).
//!
//! CPU-only (this lane, no GPU code). Implements the offline/import half of
//! RENDER.md §1 (the sole geometry pipeline): triangles → shards → adjacency
//! groups → simplify → repartition → repeat to root, DAG with monotone error.
//! BUY: meshopt (shard build + simplifier) + METIS (grouping). BUILD: the Great
//! Chain (grouping, boundary locking, staged commit, chunked paging).
//!
//! Entry point: [`transmute`]. Serialize with [`serialize()`] / [`deserialize`]
//! (chunked format; [`read_root`] / [`read_page`] for residency-scale reads).

pub mod dag;
pub mod mesh;
pub mod partition;
pub mod serialize;

pub use dag::{
    transmute, Bounds, Cluster, Dag, Group, MeshletParams, TransmuteError, TransmuteParams,
    WeldParams, MAX_TRIANGLES_CEIL, MAX_VERTICES_CEIL, TRI_MULTIPLE,
};
pub use mesh::{
    cylinder, subdivided_cube, uv_sphere, GaiaPrimitive, Mesh, Vertex, POSITION_OFFSET,
    VERTEX_STRIDE,
};
#[cfg(feature = "metis")]
pub use partition::MetisPartitioner;
pub use partition::{
    default_partitioner, AdjacencyGraph, GreedyPartitioner, Partition, Partitioner,
};
pub use serialize::{
    deserialize, read_directory, read_header, read_page, read_root, serialize, Directory, Header,
    Page, PageRef, FORMAT_VERSION, HEADER_LEN, MAGIC,
};

/// Convenience: transmute with the default partitioner (METIS when compiled in).
pub fn transmute_default(mesh: &Mesh, params: &TransmuteParams) -> Result<Dag, TransmuteError> {
    let p = default_partitioner();
    transmute(mesh, params, p.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budgets_ok(dag: &Dag, p: &MeshletParams) -> bool {
        dag.clusters
            .iter()
            .all(|c| c.vertices.len() <= p.max_vertices && c.tri_count() <= p.max_triangles)
    }

    #[test]
    fn sphere_transmutes_multi_level_chain() {
        let mesh = uv_sphere(1.0, 128, 96);
        let params = TransmuteParams::default();
        let dag = transmute_default(&mesh, &params).unwrap();
        assert!(
            dag.level_count() > 1,
            "expected N>1 chain levels, got {}",
            dag.level_count()
        );
    }

    #[test]
    fn leaf_tri_sum_equals_input() {
        let mesh = uv_sphere(1.0, 96, 64);
        let input = mesh.tri_count();
        let dag = transmute_default(&mesh, &TransmuteParams::default()).unwrap();
        assert_eq!(dag.leaf_tri_sum(), input, "leaf tri sum must equal input");
        assert_eq!(dag.input_tri_count as usize, input);
    }

    #[test]
    fn every_shard_within_budgets() {
        let params = TransmuteParams::default();
        for mesh in [
            uv_sphere(1.0, 96, 64),
            subdivided_cube(2.0, 40),
            GaiaPrimitive::Cylinder {
                radius_top: 0.5,
                radius_bottom: 1.0,
                height: 3.0,
                radial_segments: 48,
                height_segments: 32,
            }
            .tessellate(),
        ] {
            let dag = transmute(&mesh, &params, default_partitioner().as_ref()).unwrap();
            assert!(
                budgets_ok(&dag, &params.meshlet),
                "a cluster exceeded vert/tri budgets"
            );
        }
    }

    #[test]
    fn serialize_roundtrip_equality() {
        let mesh = uv_sphere(1.0, 64, 48);
        let dag = transmute_default(&mesh, &TransmuteParams::default()).unwrap();
        let bytes = serialize(&dag).expect("serialize");
        let back = deserialize(&bytes).expect("deserialize");
        assert_eq!(dag, back, "roundtrip must be identity");
    }

    #[test]
    fn header_rejects_garbage() {
        assert!(deserialize(b"nope").is_err());
        let mut bytes = serialize(
            &transmute_default(&uv_sphere(1.0, 16, 12), &TransmuteParams::default()).unwrap(),
        )
        .unwrap();
        bytes[4] = 0xFF; // corrupt version
        assert!(deserialize(&bytes).is_err());
    }

    #[test]
    fn gaia_primitives_tessellate_nonempty() {
        let prims = [
            GaiaPrimitive::Box {
                size: [1.0, 2.0, 3.0],
                subdivisions: 4,
            },
            GaiaPrimitive::Sphere {
                radius: 1.0,
                width_segments: 24,
                height_segments: 16,
            },
            GaiaPrimitive::Cone {
                radius: 1.0,
                height: 2.0,
                radial_segments: 24,
                height_segments: 8,
            },
        ];
        for p in prims {
            let m = p.tessellate();
            assert!(m.tri_count() > 0 && !m.vertices.is_empty());
        }
    }

    #[test]
    fn greedy_fallback_also_transmutes() {
        let mesh = uv_sphere(1.0, 96, 64);
        let dag = transmute(
            &mesh,
            &TransmuteParams::default(),
            &GreedyPartitioner::default(),
        )
        .unwrap();
        assert!(dag.level_count() > 1);
        assert_eq!(dag.leaf_tri_sum(), mesh.tri_count());
        assert!(budgets_ok(&dag, &MeshletParams::default()));
        assert_eq!(dag.partitioner, "greedy", "backend name must not lie");
    }
}
