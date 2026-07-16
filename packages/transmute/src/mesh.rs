//! Mesh input + test-mesh generators + GAIA primitive-part loader.
//!
//! All generators are param'd (IRON LAW: never hardcode) with defaults that
//! match RENDER.md's "drop any geometry in" doctrine. Vertex layout is a fixed
//! 32-byte interleaved record so it feeds meshopt's `VertexDataAdapter`
//! (position at offset 0) with zero repacking.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};

/// One interleaved vertex: position | normal | uv. `repr(C)`, 32 bytes, Pod so
/// a `&[Vertex]` casts straight to the byte slice meshopt wants.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position,
            normal,
            uv,
        }
    }
}

/// Byte stride of one `Vertex` (meshopt `vertex_stride`).
pub const VERTEX_STRIDE: usize = std::mem::size_of::<Vertex>();
/// Byte offset of the position field within a `Vertex` (meshopt `position_offset`).
pub const POSITION_OFFSET: usize = 0;

/// Indexed triangle mesh. Indices are u32 triples into `vertices`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    /// Triangle count.
    pub fn tri_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Raw vertex bytes for meshopt's `VertexDataAdapter`.
    pub fn vertex_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.vertices)
    }
}

// ---------------------------------------------------------------------------
// Test-mesh generators
// ---------------------------------------------------------------------------

/// UV sphere. `radius`, `segments_u` (longitude bands, min 3), `segments_v`
/// (latitude bands, min 2). Seam/pole vertices are duplicated (welding happens
/// later in the DAG merge step).
pub fn uv_sphere(radius: f32, segments_u: usize, segments_v: usize) -> Mesh {
    let su = segments_u.max(3);
    let sv = segments_v.max(2);
    let mut vertices = Vec::with_capacity((su + 1) * (sv + 1));
    for iy in 0..=sv {
        let v = iy as f32 / sv as f32;
        let theta = v * PI; // 0..PI, y pole to y pole
        let (st, ct) = theta.sin_cos();
        for ix in 0..=su {
            let u = ix as f32 / su as f32;
            let phi = u * TAU;
            let (sp, cp) = phi.sin_cos();
            let n = [st * cp, ct, st * sp];
            let p = [n[0] * radius, n[1] * radius, n[2] * radius];
            vertices.push(Vertex::new(p, n, [u, v]));
        }
    }
    let mut indices = Vec::with_capacity(su * sv * 6);
    let row = su + 1;
    for iy in 0..sv {
        for ix in 0..su {
            let a = (iy * row + ix) as u32;
            let b = a + 1;
            let c = ((iy + 1) * row + ix) as u32;
            let d = c + 1;
            // two tris per quad; poles produce degenerate slivers, harmless.
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    Mesh::new(vertices, indices)
}

/// Subdivided cube. `size` = full edge length, `subdivisions` = grid cells per
/// face edge (min 1). Face-flat normals.
pub fn subdivided_cube(size: f32, subdivisions: usize) -> Mesh {
    let n = subdivisions.max(1);
    let h = size * 0.5;
    let mut mesh = Mesh::default();
    // (origin, u-axis, v-axis, normal) for the six faces, each spanning [-h,h].
    type FaceFrame = ([f32; 3], [f32; 3], [f32; 3], [f32; 3]);
    let faces: [FaceFrame; 6] = [
        (
            [-h, -h, h],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ), // +Z
        (
            [h, -h, -h],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ), // -Z
        (
            [h, -h, h],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ), // +X
        (
            [-h, -h, -h],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ), // -X
        (
            [-h, h, h],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ), // +Y
        (
            [-h, -h, -h],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ), // -Y
    ];
    for (origin, uax, vax, normal) in faces {
        add_grid_face(&mut mesh, origin, uax, vax, size, normal, n);
    }
    mesh
}

fn add_grid_face(
    mesh: &mut Mesh,
    origin: [f32; 3],
    uax: [f32; 3],
    vax: [f32; 3],
    span: f32,
    normal: [f32; 3],
    n: usize,
) {
    let base = mesh.vertices.len() as u32;
    let row = n + 1;
    for iv in 0..=n {
        let fv = iv as f32 / n as f32;
        for iu in 0..=n {
            let fu = iu as f32 / n as f32;
            let p = [
                origin[0] + uax[0] * fu * span + vax[0] * fv * span,
                origin[1] + uax[1] * fu * span + vax[1] * fv * span,
                origin[2] + uax[2] * fu * span + vax[2] * fv * span,
            ];
            mesh.vertices.push(Vertex::new(p, normal, [fu, fv]));
        }
    }
    for iv in 0..n {
        for iu in 0..n {
            let a = base + (iv * row + iu) as u32;
            let b = a + 1;
            let c = base + ((iv + 1) * row + iu) as u32;
            let d = c + 1;
            mesh.indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
}

// ---------------------------------------------------------------------------
// GAIA mesh primitive-part loader
// ---------------------------------------------------------------------------

/// GAIA mesh primitive parts (client/kernel/geometry.js `buildBaseGeometry`),
/// tessellated to indexed triangles so real world content can be transmuted later.
/// Segment counts are params (defaults mirror the JS kernel).
#[derive(Clone, Debug, PartialEq)]
pub enum GaiaPrimitive {
    /// Box with full extents `size` and per-face grid `subdivisions`.
    Box { size: [f32; 3], subdivisions: usize },
    /// Cylinder / cone frustum. `radius_top == 0` yields a cone.
    Cylinder {
        radius_top: f32,
        radius_bottom: f32,
        height: f32,
        radial_segments: usize,
        height_segments: usize,
    },
    /// Sphere (UV). `width_segments` = longitude, `height_segments` = latitude.
    Sphere {
        radius: f32,
        width_segments: usize,
        height_segments: usize,
    },
    /// Cone (radius at base, apex at top). Sugar over `Cylinder { radius_top: 0 }`.
    Cone {
        radius: f32,
        height: f32,
        radial_segments: usize,
        height_segments: usize,
    },
}

impl GaiaPrimitive {
    /// Tessellate to an indexed triangle mesh.
    pub fn tessellate(&self) -> Mesh {
        match *self {
            GaiaPrimitive::Box { size, subdivisions } => {
                // subdivided_cube is uniform-cube; scale per axis for a box.
                let mut m = subdivided_cube(1.0, subdivisions);
                for v in &mut m.vertices {
                    v.position[0] *= size[0];
                    v.position[1] *= size[1];
                    v.position[2] *= size[2];
                }
                m
            }
            GaiaPrimitive::Sphere {
                radius,
                width_segments,
                height_segments,
            } => uv_sphere(radius, width_segments, height_segments),
            GaiaPrimitive::Cylinder {
                radius_top,
                radius_bottom,
                height,
                radial_segments,
                height_segments,
            } => cylinder(
                radius_top,
                radius_bottom,
                height,
                radial_segments,
                height_segments,
            ),
            GaiaPrimitive::Cone {
                radius,
                height,
                radial_segments,
                height_segments,
            } => cylinder(0.0, radius, height, radial_segments, height_segments),
        }
    }
}

/// Cylinder/cone frustum, centered at origin, axis +Y, height `height`.
/// Side wall + optional top/bottom caps (skipped when a radius is 0).
pub fn cylinder(
    radius_top: f32,
    radius_bottom: f32,
    height: f32,
    radial_segments: usize,
    height_segments: usize,
) -> Mesh {
    let rs = radial_segments.max(3);
    let hs = height_segments.max(1);
    let half = height * 0.5;
    let mut mesh = Mesh::default();

    // side wall grid
    let base = 0u32;
    let row = rs + 1;
    let slope = (radius_bottom - radius_top) / height; // dr/dy magnitude
    for iy in 0..=hs {
        let vy = iy as f32 / hs as f32;
        let y = -half + vy * height;
        let r = radius_bottom + (radius_top - radius_bottom) * vy;
        for ix in 0..=rs {
            let u = ix as f32 / rs as f32;
            let phi = u * TAU;
            let (sp, cp) = phi.sin_cos();
            let p = [r * cp, y, r * sp];
            // outward normal accounts for the cone slope.
            let mut n = [cp, slope, sp];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-8);
            n = [n[0] / len, n[1] / len, n[2] / len];
            mesh.vertices.push(Vertex::new(p, n, [u, vy]));
        }
    }
    for iy in 0..hs {
        for ix in 0..rs {
            let a = base + (iy * row + ix) as u32;
            let b = a + 1;
            let c = base + ((iy + 1) * row + ix) as u32;
            let d = c + 1;
            mesh.indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    // caps (fan) — only when the radius is non-zero.
    if radius_bottom > 0.0 {
        add_cap(&mut mesh, radius_bottom, -half, [0.0, -1.0, 0.0], rs, false);
    }
    if radius_top > 0.0 {
        add_cap(&mut mesh, radius_top, half, [0.0, 1.0, 0.0], rs, true);
    }
    mesh
}

fn add_cap(mesh: &mut Mesh, radius: f32, y: f32, normal: [f32; 3], rs: usize, top: bool) {
    let center = mesh.vertices.len() as u32;
    mesh.vertices
        .push(Vertex::new([0.0, y, 0.0], normal, [0.5, 0.5]));
    let rim_base = mesh.vertices.len() as u32;
    for ix in 0..=rs {
        let u = ix as f32 / rs as f32;
        let phi = u * TAU;
        let (sp, cp) = phi.sin_cos();
        mesh.vertices.push(Vertex::new(
            [radius * cp, y, radius * sp],
            normal,
            [(cp * 0.5) + 0.5, (sp * 0.5) + 0.5],
        ));
    }
    for ix in 0..rs {
        let a = rim_base + ix as u32;
        let b = rim_base + ix as u32 + 1;
        // wind so the cap faces `normal`.
        if top {
            mesh.indices.extend_from_slice(&[center, a, b]);
        } else {
            mesh.indices.extend_from_slice(&[center, b, a]);
        }
    }
}
