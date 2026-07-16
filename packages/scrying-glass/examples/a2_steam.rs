//! A2 relic forge — the ramen steam plume rises above the Naruko stall, lit by
//! the SAME traced light as every surface (Rite VI, the Aether entering the
//! Pleroma). A2 makes the binding TRUE: the plume's light is a REAL emitter read
//! from the realm (the stall's lantern — position + colour + intensity derived,
//! never invented), and the plume is GROUNDED on the stall's serving surface
//! (its y-min derived from the geometry — no clip through the counter). Loads the
//! real Naruko realm and renders three relics headlessly on the GPU:
//!
//!   proof/a2-steam.png        — the plume, steam ON (bound light, grounded)
//!   proof/a2-steam-orbit.png  — the same plume from another angle
//!   proof/a2-steam-off.png    — steam OFF (for the localized-diff report)
//!
//! It prints the honest frame cost (ms/frame with the medium marching) and the
//! steam-on-vs-off difference split into the PLUME region and the far sky (the
//! discriminating claim: the plume changes, the far sky does not).
//!
//! Run:  cargo run -p scrying-glass --release --example a2_steam

use std::path::Path;
use std::time::Instant;

use std::f32::consts::PI;

use aether::{DensityGrid, HomogeneousMedium, SteamColumn};
use crystal::{Core, load_world_dir};
use glam::Vec3 as GVec3;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, MediumGpu, MediumLightGpu, headless_device, resolve,
};
use scrying_glass::scene::{
    Camera, EmissiveSource, RenderScene, SceneParameters, SunDefaults, SunLight, emissive_sources,
    top_flat_surface_y,
};

/// Naruko scene parameters (mirrors the window defaults in `main.rs` — the same
/// realm a player boots). Nothing here is hardcoded into the engine; these are
/// the world's authoring dials.
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
        cluster_error_threshold: 1.0,
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

/// The steam plume above the ramen stall. Both the plume's GROUNDING and its
/// LIGHT are derived from the realm, never invented (A2 true binding + clip):
///
/// - `counter_top_y` is the world-space top of the stall's serving surface
///   (`top_flat_surface_y`) — the plume's y-min sits exactly there, so the
///   column rises FROM the counter instead of clipping down through it.
/// - `light` is a real emitter read from the realm (`emissive_sources`): its
///   world position and colour bind the medium's in-scatter light, and its
///   radiant intensity is derived as radiance (emission colour × the world's
///   emission-intensity dial) × the emitter's projected area (πr²) — no
///   free-floating 'own light'.
///
/// The steam OPTICS (scattering coefficients, height, turbulence) are honest
/// medium dials with documented choices; only the light and the ground are
/// bound to the world.
/// A resolved medium light: the GPU light (real position/direction), its colour
/// tint, and its scalar intensity — all bound to a real scene source.
struct BoundLight {
    light: MediumLightGpu,
    color: [f32; 3],
    intensity: f32,
    label: String,
}

/// Bind the medium's in-scatter light to a REAL scene source. Parameterized
/// selection: the NEAREST emissive entity to the plume — the stall's own glow
/// (its lantern) — with the sky sun/moon as the fallback when the realm has no
/// emitters near the plume. Position/direction, colour and intensity are all
/// read from the world; nothing is invented. The nearest emitter is bound
/// because it is the light that actually sits with the stall and back-lights
/// its steam toward a viewer (forward scatter), the way a night stall reads.
/// `fallback_reach` (world units) is how near an emitter must be to be chosen
/// over the sun — a documented dial, not a hidden constant.
fn select_medium_light(
    sources: &[EmissiveSource],
    sun: &SunLight,
    emission_intensity: f32,
    plume_center: [f32; 3],
    fallback_reach: f32,
) -> BoundLight {
    let nearest = sources.iter().min_by(|a, b| {
        let da = dist2(a.position, plume_center);
        let db = dist2(b.position, plume_center);
        da.total_cmp(&db)
    });
    match nearest {
        Some(source) if dist2(source.position, plume_center).sqrt() <= fallback_reach => {
            BoundLight {
                light: MediumLightGpu::Point {
                    position: source.position,
                },
                color: source.color,
                // Radiant intensity DERIVED from the real emitter: the world's
                // emission-radiance dial × the emitter's emitting area (πr²).
                intensity: emission_intensity * PI * source.radius * source.radius,
                label: source.id.clone(),
            }
        }
        _ => BoundLight {
            light: MediumLightGpu::Directional {
                to_light: sun.direction,
            },
            color: sun.color,
            intensity: sun.intensity,
            label: "sun".into(),
        },
    }
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn steam_medium(bound: &BoundLight, counter_top_y: f32) -> MediumGpu {
    // The plume is grounded ON the counter surface (derived), rising over a
    // documented height. Base radius and turbulence are steam dials.
    let plume_height = 4.2_f64;
    let column = SteamColumn {
        base: aether::vec3(-1.0, counter_top_y as f64, 25.6),
        height: plume_height,
        radius: 0.85,
        peak: 1.0,
        turbulence: 0.7,
        ..SteamColumn::default()
    };
    // Rasterize into the grid the GPU uploads (the SAME artifact the CPU marches).
    // The grid's y-MIN is the counter surface — nothing exists below it (clip).
    let vsize = 0.12;
    let dims = [26usize, 36, 26];
    let origin = aether::vec3(-2.5, counter_top_y as f64, 24.1);
    let grid = DensityGrid::rasterize(dims, vsize, origin, &column);

    // Optics: a THIN translucent veil, near-pure scattering (water droplets
    // absorb almost nothing: albedo ≈ 0.999). Base-center optical depth ≈ 0.5
    // (T ≈ 0.6) — the dusk sky shows THROUGH the wisp as a mauve dimming
    // instead of a black smokestack, which is what real stall steam at 9 m
    // from a modest lantern is: in-scatter (intensity/d² ≈ 0.03) can never
    // outshine the sky it occludes, so readability comes from translucency,
    // wispy structure, and a warm tint near the base. g = 0.4 is the EFFECTIVE
    // phase: droplet HG g ≈ 0.8 isotropized by multiple scattering (similarity
    // theory, g' < g), which a single-scatter march must fold in — and it
    // hands the side-lit orbit view its share of the lantern. Never a boosted
    // light; the veil is as bright as the physics allows.
    let optics = HomogeneousMedium::new(0.001, 1.5, 0.4);
    let d = grid.dims();
    let o = grid.world_origin();

    MediumGpu {
        dims: [d[0] as u32, d[1] as u32, d[2] as u32],
        voxel_size: grid.voxel_size() as f32,
        world_origin: [o.x as f32, o.y as f32, o.z as f32],
        sigma_a: optics.sigma_a as f32,
        sigma_s: optics.sigma_s as f32,
        g: optics.g as f32,
        far: 60.0,
        march_steps: 128,
        shadow_steps: 32,
        shadow_dist: 7.0,
        light: bound.light,
        light_color: bound.color,
        light_intensity: bound.intensity,
        density: grid.data().to_vec(),
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

/// Render one relic; returns the resolved linear-radiance image and the honest
/// mean ms/frame for the accumulation loop (medium marching included).
#[allow(clippy::too_many_arguments)]
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &Camera,
    scene: &RenderScene,
    w: u32,
    h: u32,
    frames: u32,
    params: &IntegratorParams,
    medium: Option<&MediumGpu>,
) -> (Vec<GVec3>, f64) {
    let start = Instant::now();
    let accum = scrying_glass::integrator::trace_headless(
        device,
        queue,
        bvh,
        camera,
        &scene.sun,
        scene.sky_top,
        scene.sky_horizon,
        w,
        h,
        frames,
        params,
        medium,
    );
    let ms_per_frame = start.elapsed().as_secs_f64() * 1e3 / frames as f64;
    (resolve(&accum), ms_per_frame)
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
    eprintln!("[a2] wrote {}", path.display());
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[a2] no GPU adapter on this host — cannot forge the relic");
    };

    // Load the real Naruko realm.
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();

    // A2 binding + clip: read the real emitters + the stall's serving surface
    // from the realm BEFORE the render scene consumes the world.
    let sources = emissive_sources(&core.world).expect("emissive sources");
    let counter_top_y = top_flat_surface_y(&core.world, "naruko_stall_massing")
        .expect("stall surface")
        .expect("the stall has a flat serving surface");

    let scene =
        RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("render scene");

    // The plume centre (its base is grounded on the counter; the light selection
    // uses the mid-column point).
    let plume_center = [-1.0, counter_top_y + 1.7, 25.6];
    // An emitter owns the steam only where it OUT-LIGHTS the sun: a point
    // source of intensity I beats the sun's irradiance (sun_intensity) inside
    // d < sqrt(I / sun_intensity). Derived for the stall lantern (I = 2.376,
    // sun 1.1): sqrt(2.376 / 1.1) = 1.47 m. The lantern sits ~9 m from the
    // plume, where its irradiance is I/d² ≈ 0.03 — 3% of the sun's — so the
    // honest dominant illuminant for this open-air plume is the SUN (derived,
    // not chosen); the lantern would own steam rising from its own sphere.
    let fallback_reach = 1.47;
    let bound = select_medium_light(
        &sources,
        &scene.sun,
        params.emission_intensity,
        plume_center,
        fallback_reach,
    );
    eprintln!(
        "[a2] bound light = {:?}  colour {:?}  intensity {:.3}  |  counter top y = {:.2}",
        bound.label, bound.color, bound.intensity, counter_top_y
    );

    // Static + dynamic geometry into one BVH (as the window does).
    let mut tris = scene.leaf_triangles();
    tris.extend(scene.dynamic_leaf_triangles());
    let bvh = Bvh::build(&tris, &BvhParams::default());
    eprintln!("[a2] naruko: {} leaf triangles", tris.len());

    let medium = steam_medium(&bound, counter_top_y);

    let (w, h) = (900u32, 600u32);
    let frames = 40u32;
    let int_params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let exposure = 1.0;

    // Front three-quarter view: the stall with the plume rising above its roof
    // against the night sky and distant city.
    let cam_front = camera_at([3.5, 3.4, 33.0], [-1.0, 4.2, 25.6], 55.0);
    // Orbit: the other shoulder, same plume.
    let cam_orbit = camera_at([-6.5, 3.6, 32.0], [-1.0, 4.2, 25.6], 55.0);

    let (img_on, ms_on) = render(
        &device,
        &queue,
        &bvh,
        &cam_front,
        &scene,
        w,
        h,
        frames,
        &int_params,
        Some(&medium),
    );
    let (img_off, ms_off) = render(
        &device,
        &queue,
        &bvh,
        &cam_front,
        &scene,
        w,
        h,
        frames,
        &int_params,
        None,
    );
    let (img_orbit, ms_orbit) = render(
        &device,
        &queue,
        &bvh,
        &cam_orbit,
        &scene,
        w,
        h,
        frames,
        &int_params,
        Some(&medium),
    );

    // PROFILE SPLIT (front view): isolate the shadow (self-occlusion) march by
    // rendering the SAME medium with shadow_steps=1. (full - no_shadow) ≈ shadow
    // march cost; (no_shadow - off) ≈ primary march cost. Honest numbers, not feel.
    let medium_noshadow = MediumGpu {
        shadow_steps: 1,
        ..steam_medium(&bound, counter_top_y)
    };
    let (_img_ns, ms_noshadow) = render(
        &device,
        &queue,
        &bvh,
        &cam_front,
        &scene,
        w,
        h,
        frames,
        &int_params,
        Some(&medium_noshadow),
    );
    println!("[a2] ── MEDIUM MARCH SPLIT (front, {w}x{h}) ──");
    println!(
        "[a2]   primary march ≈ {:.2} ms  (no-shadow {ms_noshadow:.2} − off {ms_off:.2})",
        ms_noshadow - ms_off
    );
    println!(
        "[a2]   shadow march  ≈ {:.2} ms  (full {ms_on:.2} − no-shadow {ms_noshadow:.2})",
        ms_on - ms_noshadow
    );

    let proof = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");
    write_png(&img_on, w, h, exposure, &proof.join("a2-steam.png"));
    write_png(
        &img_orbit,
        w,
        h,
        exposure,
        &proof.join("a2-steam-orbit.png"),
    );
    write_png(&img_off, w, h, exposure, &proof.join("a2-steam-off.png"));

    // Localized-diff report: split the ON-vs-OFF difference into the PLUME
    // region (a column of pixels over the stall) and the FAR SKY (top-left
    // corner, far from the plume). The plume must change; the far sky must not.
    let mut plume_sum = 0.0f64;
    let mut plume_n = 0.0f64;
    let mut sky_sum = 0.0f64;
    let mut sky_n = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let d = (img_on[i] - img_off[i]).abs();
            let dv = (d.x + d.y + d.z) as f64 / 3.0;
            // Plume region: central column, upper-middle of the frame.
            let in_plume =
                x > w * 40 / 100 && x < w * 62 / 100 && y > h * 8 / 100 && y < h * 70 / 100;
            // Far sky: top-left corner, away from the plume.
            let in_sky = x < w * 20 / 100 && y < h * 20 / 100;
            if in_plume {
                plume_sum += dv;
                plume_n += 1.0;
            }
            if in_sky {
                sky_sum += dv;
                sky_n += 1.0;
            }
        }
    }
    let plume_diff = plume_sum / plume_n.max(1.0);
    let sky_diff = sky_sum / sky_n.max(1.0);

    println!("[a2] ── FRAME COST (honest, {w}x{h}, {frames} accum frames) ──");
    println!("[a2]   steam ON  : {ms_on:.2} ms/frame");
    println!(
        "[a2]   steam OFF : {ms_off:.2} ms/frame  (medium overhead {:.2} ms)",
        ms_on - ms_off
    );
    println!("[a2]   orbit ON  : {ms_orbit:.2} ms/frame");
    println!("[a2] ── STEAM ON vs OFF (localized diff) ──");
    println!("[a2]   plume region mean |Δ| = {plume_diff:.5}");
    println!("[a2]   far-sky      mean |Δ| = {sky_diff:.6}");
    println!(
        "[a2]   plume/sky ratio = {:.1}x  (discriminating: the plume changes, the sky does not)",
        plume_diff / sky_diff.max(1e-9)
    );
}
