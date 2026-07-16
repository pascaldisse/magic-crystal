//! The dense density grid — a rasterized volume of scalar density.
//!
//! Stored as `f32` for the CPU reference but `f16`-convertible ([`crate::half`])
//! for the GPU volume texture the Rite VI port will upload. Sampling is
//! trilinear with zero outside the grid box, so a constant-filled grid reads
//! back exactly constant inside its bounds — the grid-vs-analytic agreement
//! ordeal.

use crate::half::{f16_bits_to_f32, f32_to_f16_bits};
use crate::sources::Density;
use crate::vec::{vec3, Vec3};

/// A regular axis-aligned grid of scalar density values.
///
/// Cell `(i, j, k)` has its *center* at
/// `world_origin + voxel_size · (i+0.5, j+0.5, k+0.5)`. The grid box spans
/// `[world_origin, world_origin + voxel_size · dims]`.
#[derive(Clone, Debug, PartialEq)]
pub struct DensityGrid {
    dims: [usize; 3],
    voxel_size: f64,
    world_origin: Vec3,
    data: Vec<f32>,
}

impl DensityGrid {
    /// A grid of zeros with the given parameters.
    ///
    /// # Panics
    /// If any dimension is zero or `voxel_size` is not strictly positive.
    pub fn new(dims: [usize; 3], voxel_size: f64, world_origin: Vec3) -> Self {
        assert!(
            dims[0] > 0 && dims[1] > 0 && dims[2] > 0,
            "grid dims must be non-zero"
        );
        assert!(voxel_size > 0.0, "voxel_size must be positive");
        let n = dims[0] * dims[1] * dims[2];
        DensityGrid {
            dims,
            voxel_size,
            world_origin,
            data: vec![0.0; n],
        }
    }

    /// Rasterize an analytic density field by sampling it at each cell center.
    /// Deterministic: a field built from a seed produces byte-identical `data`
    /// on every run (ENTROPY law).
    pub fn rasterize<D: Density>(
        dims: [usize; 3],
        voxel_size: f64,
        world_origin: Vec3,
        field: &D,
    ) -> Self {
        let mut grid = DensityGrid::new(dims, voxel_size, world_origin);
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let p = grid.cell_center(i, j, k);
                    let idx = grid.index(i, j, k);
                    grid.data[idx] = field.density(p) as f32;
                }
            }
        }
        grid
    }

    /// Grid resolution in cells.
    #[inline]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Edge length of one cubic cell, in world units.
    #[inline]
    pub fn voxel_size(&self) -> f64 {
        self.voxel_size
    }

    /// World-space minimum corner of the grid box.
    #[inline]
    pub fn world_origin(&self) -> Vec3 {
        self.world_origin
    }

    /// Read-only view of the raw `f32` density buffer (x-fastest, then y, z).
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Flat buffer index of cell `(i, j, k)`.
    #[inline]
    fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.dims[1] + j) * self.dims[0] + i
    }

    /// World-space center of cell `(i, j, k)`.
    #[inline]
    fn cell_center(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.world_origin
            + vec3(
                (i as f64 + 0.5) * self.voxel_size,
                (j as f64 + 0.5) * self.voxel_size,
                (k as f64 + 0.5) * self.voxel_size,
            )
    }

    /// Store this grid as half-precision bits (the GPU upload format).
    pub fn to_f16(&self) -> Vec<u16> {
        self.data.iter().map(|&v| f32_to_f16_bits(v)).collect()
    }

    /// Reconstruct a grid from half-precision bits + parameters (the readback).
    ///
    /// # Panics
    /// If `bits.len()` does not match `dims`.
    pub fn from_f16(dims: [usize; 3], voxel_size: f64, world_origin: Vec3, bits: &[u16]) -> Self {
        assert_eq!(
            bits.len(),
            dims[0] * dims[1] * dims[2],
            "f16 buffer length must match dims"
        );
        let mut grid = DensityGrid::new(dims, voxel_size, world_origin);
        for (dst, &b) in grid.data.iter_mut().zip(bits) {
            *dst = f16_bits_to_f32(b);
        }
        grid
    }
}

impl Density for DensityGrid {
    /// Trilinearly-interpolated density at a world point; `0` outside the box.
    fn density(&self, p: Vec3) -> f64 {
        // Position in "cell-center" space: center of cell 0 sits at coord 0.
        let local = (p - self.world_origin) / self.voxel_size;
        let gx = local.x - 0.5;
        let gy = local.y - 0.5;
        let gz = local.z - 0.5;

        let (nx, ny, nz) = (self.dims[0], self.dims[1], self.dims[2]);
        // Fully outside the sampling domain → vacuum.
        if gx < -0.5 || gy < -0.5 || gz < -0.5 {
            return 0.0;
        }
        if gx > nx as f64 - 0.5 || gy > ny as f64 - 0.5 || gz > nz as f64 - 0.5 {
            return 0.0;
        }

        let sample =
            |i: usize, j: usize, k: usize| -> f64 { self.data[self.index(i, j, k)] as f64 };
        let clampi = |v: f64, n: usize| -> usize {
            if v < 0.0 {
                0
            } else if v as usize >= n {
                n - 1
            } else {
                v as usize
            }
        };

        let i0 = clampi(gx.floor(), nx);
        let j0 = clampi(gy.floor(), ny);
        let k0 = clampi(gz.floor(), nz);
        let i1 = (i0 + 1).min(nx - 1);
        let j1 = (j0 + 1).min(ny - 1);
        let k1 = (k0 + 1).min(nz - 1);

        let tx = (gx - i0 as f64).clamp(0.0, 1.0);
        let ty = (gy - j0 as f64).clamp(0.0, 1.0);
        let tz = (gz - k0 as f64).clamp(0.0, 1.0);

        let c00 = sample(i0, j0, k0) * (1.0 - tx) + sample(i1, j0, k0) * tx;
        let c10 = sample(i0, j1, k0) * (1.0 - tx) + sample(i1, j1, k0) * tx;
        let c01 = sample(i0, j0, k1) * (1.0 - tx) + sample(i1, j0, k1) * tx;
        let c11 = sample(i0, j1, k1) * (1.0 - tx) + sample(i1, j1, k1) * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        c0 * (1.0 - tz) + c1 * tz
    }
}
