//! fracture — RITE VI-2's shared seam: bond break -> flood-fill fragments ->
//! transmute re-mesh -> ECS vessel birth. Depends on [`elements`] (the
//! solver whose bonds tear) and [`transmutation`] (THE geometry path — no
//! side-door mesh generator) and [`crystal`] (the ECS fragments are born
//! into). Shared by `scrying-glass` (the real runtime drop/break/settle
//! path) and `oracle`'s tests (the same-wave canon ordeal) so the fragment
//! logic is written exactly once — no cycle, since neither `scrying-glass`
//! nor `oracle` depends on the other (checked: neither crate names the other
//! in its `Cargo.toml`), and both may depend on this crate instead.
//!
//! DESIGN NOTE (fragment granularity, VI-2 OPEN item 5): one fragment = one
//! connected component of the surviving bond graph, no further splitting and
//! no merging — the flood-fill partition IS the fragment set, exactly as
//! read off the physics. No min-fragment-mass / max-fragment-count ceiling
//! is imposed here (an OPEN ruling per the proposal); a future ceiling would
//! filter/re-merge this crate's `Fragment` list, not change how it is built.

use elements::{Solver, Vec3};
use serde_json::json;
use transmutation::{mesh::subdivided_cube, transmute_default, Dag, Mesh, TransmuteParams};

/// One connected component of a bonded body's surviving bond graph after a
/// fracture — a `Solver::fragment_components` group, plus the derived mass
/// and local extent a caller needs to mesh and place it.
#[derive(Clone, Debug)]
pub struct Fragment {
    /// Particle indices belonging to this fragment (see
    /// [`elements::Solver::fragment_components`] for the ordering guarantee
    /// — ascending-start-index BFS, deterministic).
    pub particles: Vec<usize>,
    /// This fragment's mass — the sum of its particles' masses (`1/inv_mass`
    /// each). See [`fragment_masses_exact`] for the bit-exact accounting
    /// this crate actually uses in the Equivalent-Exchange ordeal; this
    /// field is a plain sum, useful for anything that doesn't need the
    /// 0e0-exact guarantee (e.g. logging, physical plausibility checks).
    pub mass: f64,
    /// The mass-weighted centroid of this fragment's CURRENT particle
    /// positions — where its render/sense transform places it.
    pub centroid: Vec3,
    /// Half-extents of this fragment's particle AABB about its centroid —
    /// sized for both the sensing `mesh: {shape:"box"}` component and the
    /// render cube union's bounding footprint.
    pub half_extent: Vec3,
}

/// Flood-fill the fracture-time fragments of a bonded body whose particle
/// indices are `whole` (the `Vec<usize>` `Solver::spawn_bonded_box`
/// returned). Call AFTER `Solver::step` (which runs `fracture_pass` before
/// returning) so `solver.constraints` reflects the post-break bond graph.
/// One [`Fragment`] per connected component; components (and each
/// component's particle order) are exactly `Solver::fragment_components`'s
/// deterministic BFS order — the bedrock the byte-determinism ordeals stand
/// on.
pub fn compute_fragments(solver: &Solver, whole: &[usize]) -> Vec<Fragment> {
    solver
        .fragment_components(whole)
        .into_iter()
        .map(|particles| {
            let mut mass_sum = 0.0;
            let mut weighted = Vec3::ZERO;
            let mut min = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut max = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
            for &i in &particles {
                let inv_m = solver.particles.inv_mass[i];
                let m = if inv_m > 0.0 { 1.0 / inv_m } else { 0.0 };
                let pos = solver.particles.pos[i];
                mass_sum += m;
                weighted = weighted + pos.scale(m);
                min = Vec3::new(min.x.min(pos.x), min.y.min(pos.y), min.z.min(pos.z));
                max = Vec3::new(max.x.max(pos.x), max.y.max(pos.y), max.z.max(pos.z));
            }
            let centroid = if mass_sum > 0.0 {
                weighted.scale(1.0 / mass_sum)
            } else {
                Vec3::ZERO
            };
            let half_extent = if particles.len() > 1 {
                Vec3::new(
                    (max.x - min.x) * 0.5,
                    (max.y - min.y) * 0.5,
                    (max.z - min.z) * 0.5,
                )
            } else {
                Vec3::ZERO
            };
            Fragment {
                particles,
                mass: mass_sum,
                centroid,
                half_extent,
            }
        })
        .collect()
}

/// EQUIVALENT EXCHANGE, exact-partition strategy: rather than compare two
/// independently-accumulated floating sums (whose add ORDER differs between
/// "walk the whole body" and "walk fragment-by-fragment", which IEEE 754
/// addition is not guaranteed to agree on bit-for-bit even for equal-valued
/// terms), reduce both sides to the SAME canonical ascending-particle-index
/// summation order before summing at all. Returns `(whole_mass,
/// fragments_mass)` computed by summing `1/inv_mass` over
/// `whole.sorted()` and over `fragments.flatten().sorted()` respectively —
/// two Vecs that are asserted equal (the actual partition-completeness
/// check) before the identical-order sums are taken, so the two totals are
/// literally the same sequence of additions and compare bit-exact by
/// construction. This is the exactness strategy `ordeal_equivalent_
/// exchange_mass_exact` documents and relies on.
pub fn fragment_masses_exact(solver: &Solver, whole: &[usize], fragments: &[Fragment]) -> (f64, f64) {
    let mut whole_sorted: Vec<usize> = whole.to_vec();
    whole_sorted.sort_unstable();
    let mut frag_particles: Vec<usize> = fragments.iter().flat_map(|f| f.particles.iter().copied()).collect();
    frag_particles.sort_unstable();
    assert_eq!(
        whole_sorted, frag_particles,
        "fragment partition lost or duplicated a particle — every particle's mass must go to \
         exactly one fragment"
    );
    let sum_of = |indices: &[usize]| -> f64 {
        indices
            .iter()
            .map(|&i| {
                let inv_m = solver.particles.inv_mass[i];
                if inv_m > 0.0 {
                    1.0 / inv_m
                } else {
                    0.0
                }
            })
            .sum()
    };
    (sum_of(&whole_sorted), sum_of(&frag_particles))
}

/// The cube edge length one particle-cube gets in a fragment's render mesh —
/// DERIVED from the bonded body's lattice spacing (never invented per-call):
/// the mean of the box's per-axis step lengths, `(dims / (counts-1))`
/// component-wise, so adjoining particle-cubes touch without gapping or
/// overlapping past their shared face. `counts` uses the same "1 if only one
/// particle on that axis" convention `spawn_bonded_box` does.
pub fn lattice_cube_size(dims: Vec3, counts: (usize, usize, usize)) -> f64 {
    let step = |extent: f64, n: usize| if n > 1 { extent / (n - 1) as f64 } else { extent };
    let sx = step(dims.x, counts.0.max(1));
    let sy = step(dims.y, counts.1.max(1));
    let sz = step(dims.z, counts.2.max(1));
    (sx + sy + sz) / 3.0
}

/// Build a fragment's render mesh: one axis-aligned cube per particle,
/// centred on the particle's CURRENT world position, edge length
/// `cube_size`, unioned (vertex/index concatenation — no dedup needed,
/// `transmute_default` welds coincident wedges itself per its own `weld`
/// pass) into a single [`Mesh`], then run through [`transmute_default`] —
/// THE geometry path, no side-door custom generator. `subdivisions = 1`
/// (the coarsest legal tessellation of a cube's six faces): a single
/// particle-cube IS already the finest authored primitive at this stage: no
/// LOD is invented here, only clusters transmute already builds.
pub fn fragment_mesh(solver: &Solver, fragment: &Fragment, cube_size: f64) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for &i in &fragment.particles {
        let p = solver.particles.pos[i];
        let cube = subdivided_cube(cube_size as f32, 1);
        let offset = indices_offset(&vertices);
        for v in &cube.vertices {
            vertices.push(transmutation::mesh::Vertex::new(
                [
                    v.position[0] + p.x as f32,
                    v.position[1] + p.y as f32,
                    v.position[2] + p.z as f32,
                ],
                v.normal,
                v.uv,
            ));
        }
        for idx in &cube.indices {
            indices.push(idx + offset);
        }
    }
    Mesh::new(vertices, indices)
}

fn indices_offset(vertices: &[transmutation::mesh::Vertex]) -> u32 {
    vertices.len() as u32
}

/// Transmute a fragment's per-particle-cube union into its Great Chain —
/// the one geometry pipeline, no exceptions.
pub fn fragment_dag(mesh: &Mesh, params: &TransmuteParams) -> Result<Dag, transmutation::TransmuteError> {
    transmute_default(mesh, params)
}

/// The three ECS component names a fragment vessel carries (registered
/// idempotently — see [`ensure_component`]): the fracture-time transform
/// (position only — VI-2 design note: a bonded body's fragments carry no
/// fitted rotation, see the crate-level doc for why), a sensing bounding-box
/// mesh (derived from the SAME live particle data the render mesh comes
/// from — same-wave, one source), and the parent trace (no orphan matter).
pub const FRAGMENT_TRANSFORM: &str = "transform";
pub const FRAGMENT_MESH: &str = "mesh";
pub const FRAGMENT_OF: &str = "fragment_of";

/// Register a component (idempotent: returns the existing id if already
/// registered) using the SAME `{"fields": {"v": "object"}}` generic-sense
/// schema `oracle::model::World::load` registers scene components with, so
/// a fragment vessel is legible to the SAME `component_value`/`geometry`
/// read path a loaded realm's vessels are (no bespoke schema for fragments).
pub fn ensure_component(world: &mut crystal::EcsWorld, name: &str) -> crystal::ComponentId {
    if let Some(id) = world.component_id(name) {
        return id;
    }
    world
        .register_component_json(&format!(r#"{{"name":{},"fields":{{"v":"object"}}}}"#, json!(name)))
        .expect("register fragment component")
}

/// Birth every fragment in `fragments` as a real ECS entity, traced to
/// `parent_id` (no orphan matter — [`FRAGMENT_OF`] resolves back to the
/// vessel that broke). gaia ids are `"{parent_id}.fragment.{i}"` in
/// ASCENDING fragment order (the same order `compute_fragments` returns,
/// itself deterministic) — a second identical run births identical ids.
/// Returns the gaia ids assigned, in that order.
pub fn birth_fragment_entities(
    world: &mut crystal::EcsWorld,
    parent_id: &str,
    fragments: &[Fragment],
) -> Vec<String> {
    let transform_id = ensure_component(world, FRAGMENT_TRANSFORM);
    let mesh_id = ensure_component(world, FRAGMENT_MESH);
    let fragment_of_id = ensure_component(world, FRAGMENT_OF);
    let mut ids = Vec::with_capacity(fragments.len());
    for (i, fragment) in fragments.iter().enumerate() {
        let gaia_id = format!("{parent_id}.fragment.{i}");
        let size = [
            (fragment.half_extent.x * 2.0) as f32,
            (fragment.half_extent.y * 2.0) as f32,
            (fragment.half_extent.z * 2.0) as f32,
        ];
        let c = fragment.centroid;
        let entity = world
            .create_entity(vec![
                (
                    transform_id,
                    json!({ "v": { "position": [c.x, c.y, c.z] } }),
                ),
                (
                    mesh_id,
                    json!({ "v": { "parts": [{ "shape": "box", "size": size }] } }),
                ),
                (fragment_of_id, json!({ "v": { "parent": parent_id } })),
            ])
            .expect("create fragment entity");
        world
            .bind_gaia_id(gaia_id.clone(), entity)
            .expect("bind fragment gaia id");
        ids.push(gaia_id);
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use elements::{DistanceConstraint, SolverConfig, LOVE};

    fn broken_lattice() -> (Solver, Vec<usize>) {
        // A 2x2x2x... wait: a 1D chain of 4 bonded particles, the middle
        // bond authored weak so it fractures under gravity's hanging load —
        // the smallest lattice `spawn_bonded_box` could ever produce is a
        // box; for a crisp two-fragment split we author the chain directly
        // (same DistanceConstraint API `spawn_bonded_box` itself uses).
        let cfg = SolverConfig {
            dt: 1.0 / 120.0,
            substeps: 12,
            fracture_threshold: 50.0,
            ..SolverConfig::default()
        };
        let mut s = Solver::new(cfg);
        let seg = 0.2;
        let mass = 1.0;
        let mut idx = Vec::new();
        idx.push(s.particles.add(Vec3::new(0.0, 5.0, 0.0), 0.0)); // anchor
        for i in 1..4 {
            idx.push(s.particles.add_mass(Vec3::new(0.0, 5.0 - seg * i as f64, 0.0), mass));
        }
        for i in 1..4 {
            let love = if i == 2 { 0.05 } else { LOVE };
            s.constraints
                .push(DistanceConstraint::new(idx[i - 1], idx[i], seg, 1.0e-6, love));
        }
        for _ in 0..600 {
            s.step();
            if !s.fractures.is_empty() {
                break;
            }
        }
        (s, idx)
    }

    #[test]
    fn fragments_split_the_chain_in_two() {
        let (s, whole) = broken_lattice();
        let fragments = compute_fragments(&s, &whole);
        assert_eq!(fragments.len(), 2, "expected exactly two fragments after the weak link tore");
    }

    #[test]
    fn fragment_dag_round_trips_transmute() {
        let (s, whole) = broken_lattice();
        let fragments = compute_fragments(&s, &whole);
        let cube = 0.2; // matches `seg` above
        for f in &fragments {
            let mesh = fragment_mesh(&s, f, cube);
            assert!(!mesh.indices.is_empty());
            let dag = fragment_dag(&mesh, &TransmuteParams::default()).expect("transmute a fragment");
            assert_eq!(dag.leaf_tri_sum(), mesh.tri_count());
        }
    }

    /// (c) fragment chains byte-deterministic — run the SAME fracture
    /// scenario twice (a fresh `broken_lattice()` build each time — the
    /// "double build" the spec asks for, not just a re-run of one Solver),
    /// assert the fragment partition is identical AND the re-meshed chain
    /// bytes are identical (`transmutation::serialize`, hashed via a plain
    /// byte-equality — the strongest possible check, stronger than a hash).
    #[test]
    fn ordeal_fragment_chains_byte_deterministic() {
        let cube = 0.2;
        let params = TransmuteParams::default();

        let build_serialized_chains = || -> (Vec<Vec<usize>>, Vec<Vec<u8>>) {
            let (s, whole) = broken_lattice();
            let fragments = compute_fragments(&s, &whole);
            let partitions: Vec<Vec<usize>> = fragments.iter().map(|f| f.particles.clone()).collect();
            let bytes: Vec<Vec<u8>> = fragments
                .iter()
                .map(|f| {
                    let mesh = fragment_mesh(&s, f, cube);
                    let dag = fragment_dag(&mesh, &params).expect("transmute a fragment");
                    transmutation::serialize(&dag).expect("serialize a fragment's chain")
                })
                .collect();
            (partitions, bytes)
        };

        let (partitions_a, bytes_a) = build_serialized_chains();
        let (partitions_b, bytes_b) = build_serialized_chains();

        assert_eq!(
            partitions_a, partitions_b,
            "the fragment partition diverged between two independent builds of the identical \
             fracture scenario"
        );
        assert_eq!(bytes_a.len(), bytes_b.len(), "fragment count diverged");
        for (i, (a, b)) in bytes_a.iter().zip(bytes_b.iter()).enumerate() {
            assert_eq!(
                a, b,
                "fragment {i}'s serialized chain bytes diverged between two independent builds"
            );
        }
        println!(
            "ORDEAL fragment-chains-byte-deterministic: {} fragments, {} total serialized bytes, \
             byte-identical across two independent builds",
            bytes_a.len(),
            bytes_a.iter().map(|b| b.len()).sum::<usize>()
        );
    }
}
