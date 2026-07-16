//! RIGID BODIES (P2) — matter that holds its shape, built on the same
//! particle substrate as everything else. A rigid body is a CLUSTER of
//! particles plus a SHAPE-MATCHING constraint (Müller, *Meshless Deformations
//! Based on Shape Matching*): each solve, fit the cluster's present pose to a
//! rotated+translated copy of its REST shape (rotation by polar decomposition)
//! and pull every particle toward its goal. `stiffness = 1.0` is perfectly
//! rigid; `< 1.0` is deformable — the loves scale `[0,1]` extends here with no
//! new dial. No new force is invented: shape matching is love toward a rigid
//! goal, the same pull a bond is toward its rest length.

use crate::mat3::{polar_rotation, Mat3, PolarConfig};
use crate::math::Vec3;
use crate::particles::Particles;

/// A rigid (or deformable) cluster: which particles it owns, their rest
/// offsets from the rest centroid, their masses, and the live transform
/// readout (centroid + rotation) refreshed every solve.
#[derive(Clone, Debug)]
pub struct RigidBody {
    /// The particle indices this body owns (into the shared [`Particles`]).
    pub indices: Vec<usize>,
    /// Rest offsets `q_i = x_rest_i − centroid_rest`, in the body frame.
    pub rest: Vec<Vec3>,
    /// Per-particle mass (the shape-match weight; also `1/inv_mass`).
    pub masses: Vec<f64>,
    /// Total mass — Σ masses, the centroid's denominator.
    pub total_mass: f64,
    /// Shape-match stiffness on `[0,1]`: `1.0` = rigid, `<1.0` = deformable.
    pub stiffness: f64,
    /// Polar-decomposition dials (rotation extraction).
    pub polar: PolarConfig,
    /// Live centroid readout `c` (mass-weighted), updated each solve.
    pub centroid: Vec3,
    /// Live rotation readout `R` (rest frame → world), updated each solve.
    pub rotation: Mat3,
}

impl RigidBody {
    /// Build a body from particles already placed in `particles`. Rest offsets
    /// and masses are read from their current state; the rest centroid is the
    /// mass-weighted mean of the given indices RIGHT NOW. `stiffness` is
    /// clamped to `[0,1]`.
    pub fn from_indices(
        particles: &Particles,
        indices: Vec<usize>,
        stiffness: f64,
        polar: PolarConfig,
    ) -> Self {
        let masses: Vec<f64> = indices
            .iter()
            .map(|&i| {
                let w = particles.inv_mass[i];
                if w > 0.0 {
                    1.0 / w
                } else {
                    0.0
                }
            })
            .collect();
        let total_mass: f64 = masses.iter().sum();
        // Rest centroid — mass-weighted mean of the rest positions.
        let mut centroid = Vec3::ZERO;
        if total_mass > 0.0 {
            for (k, &i) in indices.iter().enumerate() {
                centroid = centroid + particles.pos[i].scale(masses[k]);
            }
            centroid = centroid.scale(1.0 / total_mass);
        }
        let rest: Vec<Vec3> = indices
            .iter()
            .map(|&i| particles.pos[i] - centroid)
            .collect();
        RigidBody {
            indices,
            rest,
            masses,
            total_mass,
            stiffness: stiffness.clamp(0.0, 1.0),
            polar,
            centroid,
            rotation: Mat3::IDENTITY,
        }
    }

    /// One shape-matching solve: compute the present centroid, fit the best
    /// rigid rotation of the rest shape onto the cluster (polar decomposition
    /// of the mass-weighted covariance `A = Σ mᵢ (xᵢ−c) ⊗ qᵢ`), then pull each
    /// particle a `stiffness` fraction toward its goal `g_i = c + R·q_i`.
    /// Anchored particles (`inv_mass == 0`) are left where they stand. Updates
    /// the `centroid`/`rotation` readout as a side effect.
    pub fn solve(&mut self, particles: &mut Particles) {
        if self.total_mass <= 0.0 || self.indices.is_empty() {
            return;
        }
        // Present mass-weighted centroid.
        let mut c = Vec3::ZERO;
        for (k, &i) in self.indices.iter().enumerate() {
            c = c + particles.pos[i].scale(self.masses[k]);
        }
        c = c.scale(1.0 / self.total_mass);

        // Covariance frame A = Σ mᵢ (xᵢ − c) ⊗ qᵢ.
        let mut a = Mat3::ZERO;
        for (k, &i) in self.indices.iter().enumerate() {
            let p = particles.pos[i] - c;
            a = a.add(Mat3::outer(p.scale(self.masses[k]), self.rest[k]));
        }
        let r = polar_rotation(a, self.polar);

        // Pull toward the rigid goal.
        for (k, &i) in self.indices.iter().enumerate() {
            if particles.inv_mass[i] == 0.0 {
                continue;
            }
            let goal = c + r.mul_vec(self.rest[k]);
            let delta = (goal - particles.pos[i]).scale(self.stiffness);
            particles.pos[i] = particles.pos[i] + delta;
        }

        self.centroid = c;
        self.rotation = r;
    }
}
