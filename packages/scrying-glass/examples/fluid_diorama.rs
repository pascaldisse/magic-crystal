//! FLUID — THE POOL DIORAMA, rendered on the GPU path-tracer. ONE deterministic
//! worldline drives the SAME [`elements::fluid`] pool the ordeals measure:
//! fill → settle → drop a dense crate → splash → settle. Three fixed-tick
//! captures, read with eyes to confirm the particles read as WATER:
//!
//!   proof/fluid-rest.png     — the settled pool, a FLAT rest surface
//!   proof/fluid-splash.png   — mid-splash, the crate breaking the surface
//!   proof/fluid-settled.png  — the dense crate sunk, surface recomposed
//!
//! The fluid particles are drawn as small overlapping octahedra (blue), the
//! pool as grey walls (floor + far/side walls, near wall omitted so the camera
//! sees in), the crate as brown octahedra. Determinism: the tick index is the
//! only clock; two runs render identical frames.
//!
//! Run:  cargo run -p scrying-glass --release --example fluid_diorama

use std::path::Path;

use glam::Vec3 as GVec3;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{headless_device, resolve, trace_headless, IntegratorParams};
use scrying_glass::scene::{Camera, LeafTriangle, SunLight};

use elements::fluid::{drop_crate, fill, FluidPoolSpec};
use elements::Vec3;

fn v(p: Vec3) -> [f32; 3] {
    [p.x as f32, p.y as f32, p.z as f32]
}

/// A unit icosphere as a triangle list (subdivided octahedron, `levels`
/// subdivisions, vertices projected to the unit sphere). Built once, then
/// instanced per particle so overlapping spheres merge into a coherent fluid
/// surface (the metaball look) rather than a lattice of facets.
fn unit_sphere(levels: u32) -> Vec<[[f32; 3]; 3]> {
    // Octahedron seed.
    let p = [
        [1.0f32, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let mut tris: Vec<[[f32; 3]; 3]> = vec![
        [p[0], p[2], p[4]],
        [p[2], p[1], p[4]],
        [p[1], p[3], p[4]],
        [p[3], p[0], p[4]],
        [p[2], p[0], p[5]],
        [p[1], p[2], p[5]],
        [p[3], p[1], p[5]],
        [p[0], p[3], p[5]],
    ];
    let norm = |a: [f32; 3]| {
        let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        [a[0] / l, a[1] / l, a[2] / l]
    };
    let mid = |a: [f32; 3], b: [f32; 3]| norm([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]);
    for _ in 0..levels {
        let mut next = Vec::with_capacity(tris.len() * 4);
        for [a, b, c] in tris {
            let ab = mid(a, b);
            let bc = mid(b, c);
            let ca = mid(c, a);
            next.push([a, ab, ca]);
            next.push([ab, b, bc]);
            next.push([ca, bc, c]);
            next.push([ab, bc, ca]);
        }
        tris = next;
    }
    tris
}

/// Instance the unit sphere at `c`, radius `r`.
fn sphere_at(unit: &[[[f32; 3]; 3]], c: Vec3, r: f64, albedo: [f32; 3], out: &mut Vec<LeafTriangle>) {
    let (cx, cy, cz, r) = (c.x as f32, c.y as f32, c.z as f32, r as f32);
    let e = [0.0, 0.0, 0.0];
    for t in unit {
        let f = [
            [cx + t[0][0] * r, cy + t[0][1] * r, cz + t[0][2] * r],
            [cx + t[1][0] * r, cy + t[1][1] * r, cz + t[1][2] * r],
            [cx + t[2][0] * r, cy + t[2][1] * r, cz + t[2][2] * r],
        ];
        out.push(LeafTriangle::lambertian(f, albedo, e));
    }
}

/// A quad (two tris) from four corners with a given albedo.
fn quad(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3], al: [f32; 3], out: &mut Vec<LeafTriangle>) {
    let e = [0.0, 0.0, 0.0];
    out.push(LeafTriangle::lambertian([a, b, c], al, e));
    out.push(LeafTriangle::lambertian([a, c, d], al, e));
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn write_png(img: &[GVec3], w: u32, h: u32, exposure: f32, path: &Path) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for px in img {
        bytes.push((linear_to_srgb(px.x * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.y * exposure) * 255.0 + 0.5) as u8);
        bytes.push((linear_to_srgb(px.z * exposure) * 255.0 + 0.5) as u8);
    }
    let file = std::fs::File::create(path).unwrap();
    let writer = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(writer, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("[fluid] wrote {}", path.display());
}

fn camera_at(eye: [f32; 3], look_at: [f32; 3], fov_deg: f32) -> Camera {
    let f = (GVec3::from_array(look_at) - GVec3::from_array(eye)).normalize();
    Camera {
        eye: GVec3::from_array(eye),
        yaw: (-f.x).atan2(-f.z),
        pitch: f.y.asin(),
        fov_y_radians: fov_deg.to_radians(),
        near: 0.05,
        far: 100.0,
    }
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[fluid] no GPU adapter on this host — cannot render the diorama");
    };

    // The render pool: a modest particle count for a legible surface.
    let spec = FluidPoolSpec {
        spacing: 0.07,
        ..FluidPoolSpec::default()
    };
    let mut pool = fill(spec);
    eprintln!("[fluid] N_fluid = {}, spacing {} m", pool.fluid.len(), spec.spacing);

    let (ix, iz, wh) = (spec.inner.0, spec.inner.1, spec.wall_height);
    let (hx, hz) = (ix * 0.5, iz * 0.5);
    let wall_al = [0.55, 0.55, 0.58];
    let floor_al = [0.35, 0.35, 0.40];
    let fluid_al = [0.10, 0.35, 0.75];
    let crate_al = [0.60, 0.38, 0.16];
    let part_r = spec.spacing * 0.85; // heavy overlap → coherent surface
    let unit = unit_sphere(2); // 128-tri icosphere, instanced per particle

    // Static pool walls. Camera sits at (+x, +z), so the +x and +z walls face
    // it and are OMITTED (they would occlude the water); floor + the two far
    // walls (-x, -z) are kept. Built once.
    let mut walls: Vec<LeafTriangle> = Vec::new();
    quad(
        [-hx as f32, 0.0, -hz as f32],
        [hx as f32, 0.0, -hz as f32],
        [hx as f32, 0.0, hz as f32],
        [-hx as f32, 0.0, hz as f32],
        floor_al,
        &mut walls,
    );
    // far wall z = -hz
    quad(
        [-hx as f32, 0.0, -hz as f32],
        [-hx as f32, wh as f32, -hz as f32],
        [hx as f32, wh as f32, -hz as f32],
        [hx as f32, 0.0, -hz as f32],
        wall_al,
        &mut walls,
    );
    // -x wall (far)
    quad(
        [-hx as f32, 0.0, -hz as f32],
        [-hx as f32, 0.0, hz as f32],
        [-hx as f32, wh as f32, hz as f32],
        [-hx as f32, wh as f32, -hz as f32],
        wall_al,
        &mut walls,
    );
    // A large ground plane under it all for a grounded shot.
    quad(
        [-6.0, -0.001, -6.0],
        [6.0, -0.001, -6.0],
        [6.0, -0.001, 6.0],
        [-6.0, -0.001, 6.0],
        [0.20, 0.20, 0.22],
        &mut walls,
    );

    // Sun/sky (built directly — no ECS).
    let sun = SunLight {
        direction: GVec3::new(0.4, 0.9, 0.5).normalize().into(),
        color: [1.0, 0.95, 0.85],
        intensity: 2.4,
        ambient_intensity: 0.5,
    };
    let sky_top = [0.35, 0.50, 0.80, 1.0];
    let sky_horizon = [0.75, 0.80, 0.88, 1.0];

    let camera = camera_at([1.35, 0.95, 1.65], [0.0, 0.35, 0.0], 50.0);
    let (w, h) = (900u32, 650u32);
    let int_params = IntegratorParams {
        spp: 3,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let exposure = 1.4;
    let bvh_params = BvhParams::default();
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");

    let build_and_trace = |pool: &elements::fluid::FluidPool,
                           crate_idx: Option<usize>,
                           name: &str| {
        let mut tris = walls.clone();
        for &i in &pool.fluid {
            sphere_at(&unit, pool.solver.particles.pos[i], part_r, fluid_al, &mut tris);
        }
        if let Some(ci) = crate_idx {
            for &p in &pool.solver.rigids[ci].indices {
                sphere_at(&unit, pool.solver.particles.pos[p], part_r * 1.1, crate_al, &mut tris);
            }
        }
        eprintln!("[fluid] {name}: {} tris", tris.len());
        let bvh = Bvh::build(&tris, &bvh_params);
        let accum = trace_headless(
            &device, &queue, &bvh, &camera, &sun, sky_top, sky_horizon, w, h, 32,
            &int_params, None,
        );
        write_png(&resolve(&accum), w, h, exposure, &proof.join(name));
    };

    // 1. SETTLE to rest (maxspd falls to ~cm/s by ~300 ticks under the damped
    //    Jacobi step), capture the surface.
    for _ in 0..300 {
        pool.solver.step();
    }
    let surf_rest = pool.fluid.iter().map(|&i| pool.solver.particles.pos[i].y).fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[fluid] rest surface = {surf_rest:.4} m");
    build_and_trace(&pool, None, "fluid-rest.png");

    // 2. DROP a dense crate (sinks); capture mid-splash a few ticks after entry.
    let crate_dims = Vec3::new(0.34, 0.34, 0.34);
    let ci = drop_crate(&mut pool, crate_dims, (5, 5, 5), 1600.0, 1.25, 3.5, 0.034);
    // Advance to the moment the crate meets the water and throws it up.
    for _ in 0..22 {
        pool.solver.step();
    }
    let surf_splash = pool.fluid.iter().map(|&i| pool.solver.particles.pos[i].y).fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[fluid] splash surface = {surf_splash:.4} m");
    build_and_trace(&pool, Some(ci), "fluid-splash.png");

    // 3. SETTLE with the crate sunk; capture the recomposed surface.
    for _ in 0..200 {
        pool.solver.step();
    }
    let cy = {
        let b = &pool.solver.rigids[ci];
        b.indices.iter().map(|&p| pool.solver.particles.pos[p].y).sum::<f64>() / b.indices.len() as f64
    };
    let surf_settled = pool.fluid.iter().map(|&i| pool.solver.particles.pos[i].y).fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[fluid] settled: crate cy = {cy:.4} m, surface = {surf_settled:.4} m");
    build_and_trace(&pool, Some(ci), "fluid-settled.png");

    eprintln!("[fluid] three relics forged — read them with eyes.");
}
