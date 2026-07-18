//! CHAR-EDITOR C0 proof forge — THREE PARAMETRIC BODIES STAND.
//!
//! Three creatures built from `char_editor::CreatureParams` alone — a tall
//! biped, a short biped, and a quadruped, each with its own palette — composed
//! through the EXISTING body path (`vessel::Body::from_preset`) and rendered on
//! the Naruko seawall by the EXISTING traced renderer (`scrying_glass`). The
//! substrate never renders; it only emits presets. Proof:
//!
//!   proof/c0-variants.png — the three variant bodies on the seawall at dusk.
//!
//! Run:  cargo run -p char-editor --release --example c0_variants

use std::path::Path;

use glam::{Mat4, Vec3 as GVec3};

use char_editor::params::{CreatureParams, Morphology, PaletteParams, Proportions, RegionScheme};
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{headless_device, resolve, trace_headless, IntegratorParams};
use scrying_glass::player::Ground;
use scrying_glass::scene::{Camera, LeafTriangle, RenderScene, SceneParameters, SunDefaults};
use vessel::{Blend, Body, Preset};

fn naruko_params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: 55.0,
        near: 0.1,
        far: 4_000.0,
        sky_top: "#20152f".into(),
        sky_horizon: "#9a627d".into(),
        mesh_color: "#9aa0a6".into(),
        radial_segments: 24,
        camera_position: [0.0, 2.0, 22.0],
        camera_yaw: 0.0,
        camera_pitch: 0.0,
        tick_dt: 1.0 / 60.0,
        sun: SunDefaults {
            sun_color: "#ffe2b0".into(),
            sun_intensity: 1.1,
            sun_position: [60.0, 90.0, 30.0],
            ambient_intensity: 0.32,
        },
        emission_intensity: 2.5,
    }
}

/// A tall, slim biped in a red-and-white outfit (the plain humanoid partition).
fn tall_biped() -> CreatureParams {
    CreatureParams {
        morphology: Morphology::Biped,
        region_scheme: RegionScheme::Plain,
        proportions: Proportions {
            height: 1.28,
            girth: 1.0,
            thigh: 1.08,
            shank: 1.12,
            ..Proportions::default()
        },
        palette: PaletteParams {
            slots: vec![
                ("head".into(), "#f3e0d0".into()),
                ("torso".into(), "#c81e3a".into()),
                ("arms".into(), "#e23a52".into()),
                ("hands".into(), "#f3e0d0".into()),
                ("legs".into(), "#f5f5f7".into()),
                ("feet".into(), "#1a1a22".into()),
            ],
            default: "#f3e0d0".into(),
            blend: Blend::Smooth { width: 0.3 },
        },
        mesh: Default::default(),
    }
}

/// A short, stocky biped in forest greens.
fn short_biped() -> CreatureParams {
    CreatureParams {
        morphology: Morphology::Biped,
        region_scheme: RegionScheme::Plain,
        proportions: Proportions {
            height: 0.72,
            girth: 1.5,
            shoulder_width: 1.3,
            hip_width: 1.25,
            ..Proportions::default()
        },
        palette: PaletteParams {
            slots: vec![
                ("head".into(), "#e8c9ac".into()),
                ("torso".into(), "#2e7d32".into()),
                ("arms".into(), "#43a047".into()),
                ("hands".into(), "#e8c9ac".into()),
                ("legs".into(), "#1b5e20".into()),
                ("feet".into(), "#0d3311".into()),
            ],
            default: "#e8c9ac".into(),
            blend: Blend::Smooth { width: 0.3 },
        },
        mesh: Default::default(),
    }
}

/// A teal-coated quadruped (distinct from the pink cat canon).
fn teal_quadruped() -> CreatureParams {
    CreatureParams {
        morphology: Morphology::Quadruped,
        region_scheme: RegionScheme::Plain,
        proportions: Proportions {
            height: 1.7,
            girth: 1.15,
            ..Proportions::default()
        },
        palette: PaletteParams {
            slots: vec![
                ("head".into(), "#4dd0e1".into()),
                ("body".into(), "#0097a7".into()),
                ("legs".into(), "#00838f".into()),
                ("tail".into(), "#006064".into()),
            ],
            default: "#0097a7".into(),
            blend: Blend::Smooth { width: 0.4 },
        },
        mesh: Default::default(),
    }
}

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

/// Skin a composed body at its idle pose, grounded onto the seawall floor at
/// `(x, z)`, into world-space leaf triangles carrying per-triangle albedo.
/// Mirrors the engine's own grounding (lowest contact vertex rests on the
/// floor) using only the public body path.
fn place_body(
    params: &CreatureParams,
    name: &'static str,
    x: f32,
    z: f32,
    ground: &Ground,
) -> Vec<LeafTriangle> {
    let outcome = params.build(name);
    let preset: &Preset = &outcome.preset;
    let body = Body::from_preset(preset);
    let mesh = body.idle_mesh();
    let albedo = body.vertex_albedo();

    // Lowest vertex of the idle mesh (feet) -> contact height.
    let contact = mesh
        .positions
        .iter()
        .map(|p| p.y)
        .fold(f32::INFINITY, f32::min);
    // The seawall floor under this column (fallback to nari's stand height).
    let floor_y = ground.height_at(x, z, f32::INFINITY).unwrap_or(1.4);
    let grounded_y = floor_y - contact;
    let model = Mat4::from_translation(GVec3::new(x, grounded_y, z));

    eprintln!(
        "[c0] {name}: {} verts, floor_y={floor_y:.3}, grounded so feet rest on the seawall",
        mesh.positions.len()
    );

    mesh.indices
        .chunks_exact(3)
        .map(|tri| {
            let corner = |i: u32| {
                model
                    .transform_point3(mesh.positions[i as usize])
                    .to_array()
            };
            let mean = |a: usize, b: usize, c: usize| (albedo[a] + albedo[b] + albedo[c]) / 3.0;
            let a = tri[0] as usize;
            let b = tri[1] as usize;
            let c = tri[2] as usize;
            LeafTriangle::lambertian(
                [corner(tri[0]), corner(tri[1]), corner(tri[2])],
                mean(a, b, c).to_array(),
                [0.0; 3],
            )
        })
        .collect()
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
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
    eprintln!("[c0] wrote {}", path.display());
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[c0] no GPU adapter on this host — cannot forge the relic");
    };

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = crystal::Core::default();
    crystal::load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();

    let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("scene");

    // The realm's own static geometry — the seawall the bodies stand on.
    let mut tris = scene.leaf_triangles();
    let mut static_positions: Vec<[f32; 3]> = Vec::with_capacity(tris.len() * 3);
    for t in &tris {
        static_positions.extend_from_slice(&t.positions);
    }
    let ground = Ground::from_positions(&static_positions);
    eprintln!(
        "[c0] seawall: {} static triangles, {} walkable",
        tris.len(),
        ground.triangle_count()
    );

    // Three parametric bodies, spaced along the seawall.
    tris.extend(place_body(&tall_biped(), "tall_biped", -4.2, 18.0, &ground));
    tris.extend(place_body(
        &short_biped(),
        "short_biped",
        0.0,
        18.0,
        &ground,
    ));
    tris.extend(place_body(
        &teal_quadruped(),
        "teal_quadruped",
        3.0,
        17.8,
        &ground,
    ));

    let bvh = Bvh::build(&tris, &BvhParams::default());
    eprintln!(
        "[c0] naruko + 3 variant bodies: {} leaf triangles",
        tris.len()
    );

    let (w, h) = (1100u32, 620u32);
    let frames = 48u32;
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };

    // Frame all three against the pink dusk band.
    let cam = camera_at([0.0, 3.2, 23.0], [0.0, 2.6, 18.0], 60.0);
    let accum = trace_headless(
        &device,
        &queue,
        &bvh,
        &cam,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        w,
        h,
        frames,
        &int_params,
        None,
    );
    let img = resolve(&accum);
    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");
    write_png(&img, w, h, 1.0, &proof.join("c0-variants.png"));
    eprintln!("[c0] three parametric bodies stand.");
}
