//! FLUID — THE POOL DIORAMA. A parameterized contained pool of Position-Based-
//! Fluid particles built ON the one solver (no new physics, no hardcoded
//! magnitude — every dial a [`FluidPoolSpec`] field with a documented default),
//! plus the authored "hand" that drops a rigid crate into it. Shared by the
//! measurement example (`examples/fluid_measure.rs`), the ordeals
//! (`tests/fluid_ordeals.rs`) and the offline render
//! (`scrying-glass/examples/fluid_diorama.rs`) so all three drive the EXACT
//! same worldline.
//!
//! The pool is a static [`Collider`] box open at the top (floor + four inward-
//! facing walls) — the fluid is contained by the SAME particle-vs-triangle
//! collision every other scene uses, nothing pool-specific. The water is one
//! [`Solver::spawn_fluid_box`] call filling the lower part; its incompressibility
//! is the density constraint. A dropped crate is an ordinary rigid body
//! ([`Solver::spawn_rigid_box`]); it splashes and floats-or-sinks purely by the
//! fluid↔solid contact coupling and the mass ratio — no buoyancy is scripted.
//! Deterministic: the tick index is the only clock.

use crate::collision::{Collider, ContactMaterial, Triangle};
use crate::math::Vec3;
use crate::solver::{Solver, SolverConfig};

/// The pool + fluid + crate dials — every one a parameter with a documented
/// default (never-hardcode law).
#[derive(Clone, Copy, Debug)]
pub struct FluidPoolSpec {
    /// Inner pool full width/depth in metres `(x, z)`. Default `(1.2, 1.2)`.
    pub inner: (f64, f64),
    /// Pool wall height in metres (floor at `y = 0`). Default `1.2`.
    pub wall_height: f64,
    /// Initial fill height of the water column in metres. Default `0.6`
    /// (half-full — room to splash without overtopping). The fluid box is
    /// spawned filling `inner × fill_height` from the floor up.
    pub fill_height: f64,
    /// Fluid particle spacing in metres — the resolution dial that sets the
    /// fluid particle count. Default `0.06` (a ~20×20×10 column ≈ 4000
    /// particles at the default pool; the ordeals/tests use a coarser spacing
    /// for speed). Mass per particle = `rest_density × spacing³` (derived).
    pub spacing: f64,
    /// Physical fluid rest density kg/m³ (sets particle MASS). Default `1000`
    /// (water). The SPH rest density is DERIVED from the packing at calibration.
    pub rest_density: f64,
    /// Smoothing radius as a MULTIPLE of `spacing` (`h = h_factor × spacing`).
    /// Default `3.0` — enough neighbours (~27+ in-support) for a smooth SPH
    /// density estimate, the paper's regime.
    pub h_factor: f64,
    /// Fluid particle contact radius (thickness vs walls and crate) in metres.
    /// Default `0.5 × spacing` (particles just touch their lattice neighbours).
    pub fluid_radius_factor: f64,
    /// Solver substeps per tick. Default `4` (fluid is stiff; more substeps =
    /// better incompressibility. Kept at 4 for a tractable bench; the 60 FPS
    /// building lane uses 8).
    pub substeps: usize,
}

impl Default for FluidPoolSpec {
    fn default() -> Self {
        FluidPoolSpec {
            inner: (1.2, 1.2),
            wall_height: 1.2,
            fill_height: 0.6,
            spacing: 0.06,
            rest_density: 1000.0,
            h_factor: 3.0,
            fluid_radius_factor: 0.5,
            substeps: 4,
        }
    }
}

impl FluidPoolSpec {
    /// The fluid particle count this spec produces (`nx × ny × nz` over the
    /// fill volume at `spacing`).
    pub fn fluid_count(&self) -> usize {
        let n = |extent: f64| (extent / self.spacing).floor() as usize + 1;
        n(self.inner.0) * n(self.fill_height) * n(self.inner.1)
    }
}

/// A pool diorama wired into a fresh solver: the solver, the fluid particle
/// indices, and the spec.
pub struct FluidPool {
    pub solver: Solver,
    pub fluid: Vec<usize>,
    pub spec: FluidPoolSpec,
}

/// Build the static pool: a floor and four inward-facing walls (open top),
/// centred on the origin in xz, floor at `y = 0`. Inner full extents
/// `(ix, iz)`, wall height `wh`. Inward normals so a fluid particle inside is
/// pushed back into the pool.
fn pool_collider(ix: f64, iz: f64, wh: f64, material: ContactMaterial) -> Collider {
    let hx = ix * 0.5;
    let hz = iz * 0.5;
    // A quad from four corners (ccw) with an explicit inward normal → 2 tris.
    let quad = |a: Vec3, b: Vec3, c: Vec3, d: Vec3, n: Vec3, out: &mut Vec<Triangle>| {
        out.push(Triangle::with_normal(a, b, c, n));
        out.push(Triangle::with_normal(a, c, d, n));
    };
    let mut tris = Vec::new();
    // Floor (normal +y).
    quad(
        Vec3::new(-hx, 0.0, -hz),
        Vec3::new(hx, 0.0, -hz),
        Vec3::new(hx, 0.0, hz),
        Vec3::new(-hx, 0.0, hz),
        Vec3::new(0.0, 1.0, 0.0),
        &mut tris,
    );
    // Wall x = -hx, inward normal +x.
    quad(
        Vec3::new(-hx, 0.0, -hz),
        Vec3::new(-hx, 0.0, hz),
        Vec3::new(-hx, wh, hz),
        Vec3::new(-hx, wh, -hz),
        Vec3::new(1.0, 0.0, 0.0),
        &mut tris,
    );
    // Wall x = +hx, inward normal -x.
    quad(
        Vec3::new(hx, 0.0, -hz),
        Vec3::new(hx, 0.0, hz),
        Vec3::new(hx, wh, hz),
        Vec3::new(hx, wh, -hz),
        Vec3::new(-1.0, 0.0, 0.0),
        &mut tris,
    );
    // Wall z = -hz, inward normal +z.
    quad(
        Vec3::new(-hx, 0.0, -hz),
        Vec3::new(hx, 0.0, -hz),
        Vec3::new(hx, wh, -hz),
        Vec3::new(-hx, wh, -hz),
        Vec3::new(0.0, 0.0, 1.0),
        &mut tris,
    );
    // Wall z = +hz, inward normal -z.
    quad(
        Vec3::new(-hx, 0.0, hz),
        Vec3::new(hx, 0.0, hz),
        Vec3::new(hx, wh, hz),
        Vec3::new(-hx, wh, hz),
        Vec3::new(0.0, 0.0, -1.0),
        &mut tris,
    );
    Collider { triangles: tris, material }
}

/// Fill the pool: build the walls, spawn the water column, calibrate the SPH
/// rest density. The fluid box fills `inner × fill_height` from the floor,
/// inset half a spacing from the walls so the outermost particles start
/// inside the pool (not clipping the wall skin).
pub fn fill(spec: FluidPoolSpec) -> FluidPool {
    let cfg = SolverConfig {
        dt: 1.0 / 60.0,
        substeps: spec.substeps,
        ..SolverConfig::default()
    };
    let mut solver = Solver::new(cfg);
    solver.collider = Some(pool_collider(
        spec.inner.0,
        spec.inner.1,
        spec.wall_height,
        ContactMaterial::default(),
    ));

    // The water column: inset one spacing from the walls, sitting on the floor.
    let s = spec.spacing;
    let fill_dims = Vec3::new(
        (spec.inner.0 - 2.0 * s).max(s),
        (spec.fill_height - s).max(s),
        (spec.inner.1 - 2.0 * s).max(s),
    );
    // Centre so the column's base sits ~half a spacing above the floor.
    let center = Vec3::new(0.0, fill_dims.y * 0.5 + s, 0.0);
    let radius = spec.fluid_radius_factor * s;
    let fluid = solver.spawn_fluid_box(center, fill_dims, s, spec.rest_density, spec.h_factor, radius);
    solver.calibrate_fluid_rest_density();

    FluidPool { solver, fluid, spec }
}

/// The current top (max y) of the fluid — the surface height witness.
pub fn surface_height(pool: &FluidPool) -> f64 {
    pool.fluid
        .iter()
        .map(|&p| pool.solver.particles.pos[p].y)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Settle the pool `ticks` steps under gravity (no crate) — lets the spawn
/// lattice relax to its hydrostatic rest. Returns the final surface height.
pub fn settle(pool: &mut FluidPool, ticks: u64) -> f64 {
    for _ in 0..ticks {
        pool.solver.step();
    }
    surface_height(pool)
}

/// Drop a rigid crate of full extents `dims`, made of matter at `density`
/// kg/m³, centred at `(0, drop_y, 0)` above the pool with a downward velocity
/// `speed` (m/s). `lattice` is the crate's particle resolution. Returns the
/// rigid body index. The crate's float/sink is NOT scripted — it emerges from
/// `density` vs the fluid's `rest_density` through the contact coupling.
#[allow(clippy::too_many_arguments)]
pub fn drop_crate(
    pool: &mut FluidPool,
    dims: Vec3,
    lattice: (usize, usize, usize),
    density: f64,
    drop_y: f64,
    speed: f64,
    particle_radius: f64,
) -> usize {
    let idx = pool.solver.spawn_rigid_box(
        Vec3::new(0.0, drop_y, 0.0),
        dims,
        lattice,
        density,
        1.0, // rigid stiffness (shape matching) — a hard crate
        particle_radius,
    );
    if speed != 0.0 {
        pool.solver.apply_impulse(idx, Vec3::new(0.0, -speed, 0.0));
    }
    idx
}

/// The mean y of a rigid body's particles — its centre-of-height witness
/// (float/sink readout).
pub fn body_center_y(pool: &FluidPool, rigid_index: usize) -> f64 {
    let body = &pool.solver.rigids[rigid_index];
    let sum: f64 = body.indices.iter().map(|&p| pool.solver.particles.pos[p].y).sum();
    sum / body.indices.len() as f64
}
