//! # Vessel — DreamForge V0 body vessel
//!
//! Turns a homunculus [`Skeleton`] into a renderable, auto-skinned body mesh —
//! the geometry nari and the naruko cat get once they need real surfaces. The
//! whole pipeline is procedural and parameter-driven; nothing is hand-authored:
//!
//! 1. [`sdf`] — the body signed-distance field: a smooth union of the
//!    skeleton's bone capsules (blend radius [`VesselParams::smooth_k`]).
//! 2. [`mesh`] — marching cubes over the field ([`VesselParams::resolution`])
//!    into a watertight triangle mesh with positions + SDF-gradient normals.
//!    NO UVs — texturing is virtual (DreamForge law).
//! 3. [`bind`] — automatic per-vertex weights via [`homunculus::compute_weights`]
//!    (no manual rigging — law) and linear-blend pose deformation.
//!
//! [`Vessel::build`] runs the pipeline once; [`Vessel::posed`] deforms the same
//! bound mesh into any [`Pose`]. Building is deterministic: identical params
//! yield a byte-identical mesh (an ENTROPY-law ordeal), for humans and
//! quadrupeds alike.

pub mod bind;
pub mod mesh;
pub mod sdf;
pub mod tables;

pub use bind::{bind_world, deform, skin};
pub use mesh::{marching_cubes, Mesh};
pub use sdf::{capsule_sdf, smin, BodySdf};

use glam::Affine3A;
use homunculus::skin::capsules as bone_capsules;
use homunculus::{BoneCapsule, Pose, Skeleton, SkinWeights};

/// Parameters for building a vessel. Every field has a default; the defaults
/// mesh both the humanoid and quadruped presets cleanly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VesselParams {
    /// SDF smooth-union blend radius in metres — how strongly limbs fuse.
    pub smooth_k: f32,
    /// Marching-cubes cells along the longest bounds axis.
    pub resolution: usize,
    /// Iso level contoured (0 = the capsule surfaces).
    pub iso: f32,
    /// Bounds padding beyond the smooth-union reach, as a multiple of the base
    /// capsule radius — keeps the surface strictly interior (watertightness).
    pub padding: f32,
    /// Max bones influencing one vertex (`None` = all).
    pub max_influences: Option<usize>,
}

impl Default for VesselParams {
    fn default() -> Self {
        Self {
            smooth_k: 0.04,
            resolution: 48,
            iso: 0.0,
            padding: 1.5,
            max_influences: Some(4),
        }
    }
}

/// A built body vessel: the bind-pose mesh, its automatic skin weights, and the
/// bind-pose bone transforms needed to deform it into any pose.
#[derive(Clone, Debug)]
pub struct Vessel {
    /// The bind-pose triangle mesh (positions + normals, no UVs).
    pub mesh: Mesh,
    /// Automatic per-vertex skin weights (via homunculus).
    pub weights: SkinWeights,
    /// Bind-pose bone capsules used for the SDF and skinning.
    pub capsules: Vec<BoneCapsule>,
    /// Bind-pose world transforms per bone.
    pub bind_world: Vec<Affine3A>,
}

impl Vessel {
    /// Build a vessel from a skeleton and parameters.
    pub fn build(skeleton: &Skeleton, params: &VesselParams) -> Vessel {
        let bind_world = bind::bind_world(skeleton);
        let capsules = bone_capsules(skeleton, &bind_world);

        let field = BodySdf::new(capsules.clone(), params.smooth_k);
        // Margin encloses the smooth-union reach plus padding, and a couple of
        // cells so the surface never touches the grid boundary.
        let base_radius = capsules.iter().map(|c| c.radius).fold(0.0f32, f32::max);
        let margin = params.smooth_k + params.padding * base_radius;
        let (lo, hi) = field.bounds(margin);

        let mesh = mesh::marching_cubes(&field, lo, hi, params.resolution, params.iso);
        let weights = bind::skin(&capsules, &mesh.positions, params.max_influences);

        Vessel {
            mesh,
            weights,
            capsules,
            bind_world,
        }
    }

    /// Deform the bound mesh into `pose` via linear-blend skinning.
    pub fn posed(&self, skeleton: &Skeleton, pose: &Pose) -> Mesh {
        let posed_world = pose.forward_kinematics(skeleton);
        bind::deform(&self.mesh, &self.weights, &self.bind_world, &posed_world)
    }
}
