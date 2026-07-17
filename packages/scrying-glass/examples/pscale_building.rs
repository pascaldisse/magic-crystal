//! P-SCALE — THE BUILDING FALLS · the visual proof (scrying-glass offline
//! render). Renders the EXACT anchored bonded building the measurement lane
//! profiles (`elements::building`, base anchored to the ground, knocked down
//! by an authored lateral impulse) at three fixed-tick stops:
//!
//!   proof/pscale-standing.png     — the multi-storey structure at rest
//!   proof/pscale-collapse.png     — mid-collapse, fragments separating
//!   proof/pscale-rubble.png       — the fragments settled into a debris field
//!
//! Geometry is built DIRECTLY from the solver's live particle positions — one
//! small cube per particle, coloured by which fragment (connected bond
//! component) it belongs to, so distinct chunks read as distinct rubble, not
//! a smear. No ECS, no `body` sigil (the ECS path has no anchor field — a
//! physics-recon open problem) — this renders the anchored solver scenario
//! the ordeals and measurement actually exercise. Deterministic: the tick
//! index is the only clock.
//!
//! Run: cargo run -p scrying-glass --release --example pscale_building

use std::path::Path;

use glam::Vec3 as GVec3;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{headless_device, resolve, trace_headless, IntegratorParams};
use scrying_glass::scene::{Camera, LeafTriangle, SunLight};

use elements::building::{erect, settle, topple, BuildingSpec};

const TOPPLE_SPEED: f64 = 30.0;
const TOPPLE_FRACTION: f64 = 0.5;
const SETTLE_TICKS: u64 = 40; // settle to the standing plateau (fracture disarmed)
const COLLAPSE_TICKS: u64 = 360;

fn camera_at(eye: [f32; 3], look_at: [f32; 3], fov_deg: f32) -> Camera {
    let f = (GVec3::from_array(look_at) - GVec3::from_array(eye)).normalize();
    Camera {
        eye: GVec3::from_array(eye),
        yaw: (-f.x).atan2(-f.z),
        pitch: f.y.asin(),
        fov_y_radians: fov_deg.to_radians(),
        near: 0.1,
        far: 4_000.0,
    }
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
    enc.write_header()
        .unwrap()
        .write_image_data(&bytes)
        .unwrap();
    eprintln!("[pscale] wrote {}", path.display());
}

/// A distinct, VIVID colour per fragment index (deterministic hash → hue), so
/// a settled rubble field reads unmistakably as many separate chunks and not
/// a uniform slab. Whole (one fragment) → a warm stone (the intact tower).
fn fragment_color(fi: usize, total: usize) -> [f32; 3] {
    if total <= 1 {
        return [0.62, 0.52, 0.40]; // warm stone — the intact structure
    }
    // Deterministic golden-ratio hue spread → adjacent fragment indices land
    // far apart on the wheel, so neighbouring chunks contrast strongly.
    let hue = ((fi as f32) * 0.61803398875).fract();
    let sat = 0.70;
    let val = 0.80;
    // HSV→RGB.
    let h6 = hue * 6.0;
    let c = val * sat;
    let x = c * (1.0 - ((h6 % 2.0) - 1.0).abs());
    let m = val - c;
    let (r, g, b) = match h6 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

/// Append the 12 triangles of an axis-aligned box centred at `c`, half-extents
/// `he`, with the given albedo.
fn push_box(tris: &mut Vec<LeafTriangle>, c: GVec3, he: GVec3, albedo: [f32; 3]) {
    let e = [0.0f32, 0.0, 0.0]; // no emission — lit by sun/sky
    let v = |sx: f32, sy: f32, sz: f32| {
        [c.x + sx * he.x, c.y + sy * he.y, c.z + sz * he.z]
    };
    // 8 corners
    let p = [
        v(-1.0, -1.0, -1.0),
        v(1.0, -1.0, -1.0),
        v(1.0, 1.0, -1.0),
        v(-1.0, 1.0, -1.0),
        v(-1.0, -1.0, 1.0),
        v(1.0, -1.0, 1.0),
        v(1.0, 1.0, 1.0),
        v(-1.0, 1.0, 1.0),
    ];
    // 6 faces, 2 tris each (winding not critical — the integrator is double-sided)
    let faces = [
        [0, 1, 2, 3], // -z
        [5, 4, 7, 6], // +z
        [4, 0, 3, 7], // -x
        [1, 5, 6, 2], // +x
        [4, 5, 1, 0], // -y
        [3, 2, 6, 7], // +y
    ];
    for f in faces {
        tris.push(LeafTriangle::lambertian([p[f[0]], p[f[1]], p[f[2]]], albedo, e));
        tris.push(LeafTriangle::lambertian([p[f[0]], p[f[2]], p[f[3]]], albedo, e));
    }
}

/// Build the whole render triangle set for the current solver state: a big
/// ground quad + one cube per particle, coloured by fragment.
fn scene_triangles(building: &elements::building::Building) -> (Vec<LeafTriangle>, usize) {
    let s = &building.solver;
    let spec = building.spec;
    let (nx, ny, nz) = spec.lattice;
    // Cube half-extents ≈ 0.48 × the lattice spacing per axis, so the standing
    // tower reads as solid and chunks stay visible when they scatter.
    let sx = if nx > 1 { spec.footprint.0 / (nx - 1) as f64 } else { spec.footprint.0 };
    let sy = if ny > 1 { spec.height / (ny - 1) as f64 } else { spec.height };
    let sz = if nz > 1 { spec.footprint.1 / (nz - 1) as f64 } else { spec.footprint.1 };
    let he = GVec3::new((0.48 * sx) as f32, (0.48 * sy) as f32, (0.48 * sz) as f32);

    let fragments = s.fragment_components(&building.whole);
    let mut owner = std::collections::HashMap::new();
    for (fi, frag) in fragments.iter().enumerate() {
        for &p in frag {
            owner.insert(p, fi);
        }
    }

    let mut tris = Vec::new();
    // Ground plane (dark slate), 300 m square at y = 0.
    let g = 300.0f32;
    let ga = [0.10f32, 0.11, 0.14];
    for t in [
        [[-g, 0.0, -g], [g, 0.0, -g], [g, 0.0, g]],
        [[-g, 0.0, -g], [g, 0.0, g], [-g, 0.0, g]],
    ] {
        tris.push(LeafTriangle::lambertian(t, ga, [0.0, 0.0, 0.0]));
    }

    for &p in &building.whole {
        let pos = s.particles.pos[p];
        let c = GVec3::new(pos.x as f32, pos.y as f32, pos.z as f32);
        let fi = *owner.get(&p).unwrap_or(&0);
        let col = fragment_color(fi, fragments.len());
        push_box(&mut tris, c, he, col);
    }
    (tris, fragments.len())
}

fn render_stop(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    building: &elements::building::Building,
    camera: &Camera,
    sun: &SunLight,
    name: &str,
    proof: &Path,
) {
    let (tris, nfrag) = scene_triangles(building);
    let bvh = Bvh::build(&tris, &BvhParams::default());
    let (w, h) = (1000u32, 700u32);
    let frames = 40u32;
    let params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let sky_top = [0.10f32, 0.13, 0.22, 1.0];
    let sky_horizon = [0.55f32, 0.60, 0.72, 1.0];
    eprintln!(
        "[pscale] {name}: {} tris, {} fragment(s), tick {}",
        tris.len(),
        nfrag,
        building.solver.tick
    );
    let accum = trace_headless(
        device, queue, &bvh, camera, sun, sky_top, sky_horizon, w, h, frames, &params, None,
    );
    write_png(&resolve(&accum), w, h, 1.1, &proof.join(name));
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[pscale] no GPU adapter on this host");
    };

    let spec = BuildingSpec {
        lattice: (8, 16, 8), // 1024 particles — stable at rest, chunky rubble
        ..BuildingSpec::default()
    };
    let mut b = erect(spec);

    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");
    let sun = SunLight {
        direction: GVec3::new(0.5, 0.8, 0.35).normalize().to_array(),
        color: [1.0, 0.93, 0.82],
        intensity: 1.5,
        ambient_intensity: 0.35,
    };

    // Frame the whole standing tower and the ground it scatters onto.
    let look_at = [0.0f32, 7.0, 0.0];
    let camera = camera_at([26.0, 13.0, 34.0], look_at, 52.0);

    // ── STANDING: build the tower at rest (fracture disarmed during settle),
    // then render the intact multi-storey structure.
    let rest_frags = settle(&mut b, SETTLE_TICKS);
    eprintln!("[pscale] settled to rest: {rest_frags} fragment(s) (want 1)");
    render_stop(&device, &queue, &b, &camera, &sun, "pscale-standing.png", &proof);

    // The hand strikes: a lateral shove on the upper storeys.
    topple(&mut b, TOPPLE_SPEED, TOPPLE_FRACTION);

    // ── COLLAPSE: capture the tower MID-FALL — the tick its top has dropped
    // to ~55% of its standing height (leaning/folding, fragments breaking
    // free while still elevated), which reads far better than the already-
    // flattened first-many-fragments tick. Height, not fragment count, is
    // the visual "collapse in progress" signal.
    let stand_h: f64 = b.whole.iter().map(|&p| b.solver.particles.pos[p].y).fold(0.0, f64::max);
    let mut collapse_tick = None;
    for _ in 0..COLLAPSE_TICKS {
        b.solver.step();
        let h: f64 = b.whole.iter().map(|&p| b.solver.particles.pos[p].y).fold(0.0, f64::max);
        let frags = b.solver.fragment_components(&b.whole).len();
        if h < 0.55 * stand_h && frags > 4 {
            collapse_tick = Some(b.solver.tick);
            break;
        }
    }
    eprintln!("[pscale] mid-collapse captured at tick {:?}", collapse_tick);
    render_stop(&device, &queue, &b, &camera, &sun, "pscale-collapse.png", &proof);

    // ── RUBBLE: settle the rest of the way.
    while b.solver.tick < SETTLE_TICKS + COLLAPSE_TICKS {
        b.solver.step();
    }
    let final_frags = b.solver.fragment_components(&b.whole).len();
    render_stop(&device, &queue, &b, &camera, &sun, "pscale-rubble.png", &proof);
    eprintln!(
        "[pscale] settled: {} fragments at tick {}",
        final_frags, b.solver.tick
    );
    assert!(
        final_frags > 1,
        "[pscale] rubble frame shows only {final_frags} fragment(s) — the building must break"
    );
    eprintln!("[pscale] three relics forged — read them with eyes.");
}
