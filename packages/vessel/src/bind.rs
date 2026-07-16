//! Skin binding and pose deformation.
//!
//! Vertex weights come from homunculus's automatic capsule-falloff skinning
//! ([`homunculus::compute_weights`]) — no manual weight painting (a DreamForge
//! law). Posing is linear-blend skinning: each vertex is transformed by the
//! weighted blend of its bones' skinning matrices `world_pose * inv_bind`, and
//! its normal by the same blended rotation part, renormalized.

use crate::mesh::Mesh;
use glam::{Affine3A, Mat3A, Vec3, Vec3A};
use homunculus::{compute_weights, BoneCapsule, Pose, Skeleton, SkinWeights};

/// Compute automatic skin weights binding `vertices` to `capsules` (the
/// bind-pose bone capsules), keeping the `max_influences` strongest bones each.
pub fn skin(
    capsules: &[BoneCapsule],
    vertices: &[Vec3],
    max_influences: Option<usize>,
) -> SkinWeights {
    compute_weights(capsules, vertices, max_influences)
}

/// Linear-blend-skin `mesh` (authored in bind pose) into `pose`.
///
/// `bind_world` and `posed_world` are the per-bone forward-kinematics outputs
/// of the bind pose and the target pose respectively. Positions are the
/// weighted blend of `posed_world[i] * inv(bind_world[i]) * v`; normals use the
/// same blended linear part, renormalized. Indices are carried through
/// unchanged (topology is invariant under posing).
pub fn deform(
    mesh: &Mesh,
    weights: &SkinWeights,
    bind_world: &[Affine3A],
    posed_world: &[Affine3A],
) -> Mesh {
    assert_eq!(
        mesh.positions.len(),
        weights.per_vertex.len(),
        "weights must cover every vertex"
    );
    assert_eq!(
        bind_world.len(),
        posed_world.len(),
        "bind and posed world must share bone count"
    );

    // Skinning matrix per bone: move a bind-space point into posed space.
    let skin_mats: Vec<Affine3A> = bind_world
        .iter()
        .zip(posed_world.iter())
        .map(|(bind, posed)| *posed * bind.inverse())
        .collect();

    let mut positions = Vec::with_capacity(mesh.positions.len());
    let mut normals = Vec::with_capacity(mesh.normals.len());

    for (vi, (&p, &n)) in mesh.positions.iter().zip(mesh.normals.iter()).enumerate() {
        let mut m = Mat3A::ZERO;
        let mut t = Vec3A::ZERO;
        for &(bone, w) in &weights.per_vertex[vi] {
            let sm = skin_mats[bone];
            m += sm.matrix3 * w;
            t += sm.translation * w;
        }
        let pos = m * Vec3A::from(p) + t;
        let nrm = m * Vec3A::from(n);
        let nlen = nrm.length();
        let nrm = if nlen > 1.0e-20 {
            nrm / nlen
        } else {
            Vec3A::from(n)
        };
        positions.push(Vec3::from(pos));
        normals.push(Vec3::from(nrm));
    }

    Mesh {
        positions,
        normals,
        indices: mesh.indices.clone(),
    }
}

/// Convenience: the bind-pose world transforms for a skeleton.
pub fn bind_world(skeleton: &Skeleton) -> Vec<Affine3A> {
    Pose::bind(skeleton).forward_kinematics(skeleton)
}
