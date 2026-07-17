//! SCRATCH — pixel-parity gate for the perf-fix lane. NOT for tip.
//! Renders the DYN-ON merged tree deterministically (fixed seeds, fixed frames)
//! at the three law poses and prints a 64-bit FNV hash of the raw accum bytes.
//! Bit-exact wins must leave every hash unchanged.

use std::f32::consts::PI;
use std::path::Path;

use aether::{DensityGrid, HomogeneousMedium, SteamColumn};
use crystal::{Core, load_world_dir};
use glam::Vec3 as GVec3;
use sama::GaitParams;
use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    IntegratorParams, MediumGpu, MediumLightGpu, headless_device, trace_headless,
};
use scrying_glass::scene::{
    Camera, EmissiveSource, RenderScene, SceneParameters, SunDefaults, SunLight,
    contact_passing_ticks, emissive_sources, top_flat_surface_y,
};
use vessel::{Body, Preset};

fn naruko_params() -> SceneParameters {
    SceneParameters {
        fov_y_degrees: 60.0,
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

struct BoundLight {
    light: MediumLightGpu,
    color: [f32; 3],
    intensity: f32,
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn select_medium_light(
    sources: &[EmissiveSource],
    sun: &SunLight,
    emission_intensity: f32,
    plume_center: [f32; 3],
    fallback_reach: f32,
) -> BoundLight {
    let nearest = sources
        .iter()
        .min_by(|a, b| dist2(a.position, plume_center).total_cmp(&dist2(b.position, plume_center)));
    match nearest {
        Some(s) if dist2(s.position, plume_center).sqrt() <= fallback_reach => BoundLight {
            light: MediumLightGpu::Point {
                position: s.position,
            },
            color: s.color,
            intensity: emission_intensity * PI * s.radius * s.radius,
        },
        _ => BoundLight {
            light: MediumLightGpu::Directional {
                to_light: sun.direction,
            },
            color: sun.color,
            intensity: sun.intensity,
        },
    }
}

fn steam_medium(bound: &BoundLight, counter_top_y: f32) -> MediumGpu {
    let column = SteamColumn {
        base: aether::vec3(-1.0, counter_top_y as f64, 25.6),
        height: 4.2,
        radius: 0.85,
        peak: 1.0,
        turbulence: 0.7,
        ..SteamColumn::default()
    };
    let dims = [26usize, 36, 26];
    let origin = aether::vec3(-2.5, counter_top_y as f64, 24.1);
    let grid = DensityGrid::rasterize(dims, 0.12, origin, &column);
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

fn key17(t: &scrying_glass::bvh::GpuTri) -> [OrdF; 17] {
    [
        t.v0[0],
        t.v0[1],
        t.v0[2],
        t.v1[0],
        t.v1[1],
        t.v1[2],
        t.v2[0],
        t.v2[1],
        t.v2[2],
        t.albedo[0],
        t.albedo[1],
        t.albedo[2],
        t.albedo[3],
        t.emission[0],
        t.emission[1],
        t.emission[2],
        t.emission[3],
    ]
    .map(OrdF)
}

#[derive(PartialEq, PartialOrd)]
struct OrdF(f32);

// Standalone Moller-Trumbore mirror for brute force (bvh::tri_hit is private).
fn tri_hit_pub(
    o: [f32; 3],
    d: [f32; 3],
    t: &scrying_glass::bvh::GpuTri,
    t_min: f32,
    t_max: f32,
) -> Option<f32> {
    let v0 = [t.v0[0], t.v0[1], t.v0[2]];
    let e1 = [t.v1[0] - v0[0], t.v1[1] - v0[1], t.v1[2] - v0[2]];
    let e2 = [t.v2[0] - v0[0], t.v2[1] - v0[1], t.v2[2] - v0[2]];
    let cr = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dt = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let p = cr(d, e2);
    let det = dt(e1, p);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv = 1.0 / det;
    let tv = [o[0] - v0[0], o[1] - v0[1], o[2] - v0[2]];
    let u = dt(tv, p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cr(tv, e1);
    let vv = dt(d, q) * inv;
    if vv < 0.0 || u + vv > 1.0 {
        return None;
    }
    let th = dt(e2, q) * inv;
    if th > t_min && th <= t_max {
        Some(th)
    } else {
        None
    }
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn main() {
    let Some((device, queue)) = headless_device() else {
        panic!("[parity] no GPU");
    };
    let (w, h) = (900u32, 600u32);
    let frames = if std::env::var("PRIM_ONLY").is_ok() {
        1
    } else {
        4
    };

    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).expect("load naruko");
    let params = naruko_params();
    let sources = emissive_sources(&core.world).expect("emissive sources");
    let counter_top_y = top_flat_surface_y(&core.world, "naruko_stall_massing")
        .expect("stall surface")
        .expect("flat serving surface");
    let mut scene =
        RenderScene::from_ecs(std::mem::take(&mut core.world), &params).expect("render scene");

    let plume_center = [-1.0, counter_top_y + 1.7, 25.6];
    let bound = select_medium_light(
        &sources,
        &scene.sun,
        params.emission_intensity,
        plume_center,
        1.47,
    );
    let medium = steam_medium(&bound, counter_top_y);

    let bvh_params = BvhParams::default();
    let static_bvh = Bvh::build(&scene.leaf_triangles(), &bvh_params);

    let body = Body::from_preset(&Preset::nari());
    let gait = GaitParams::walk();
    let cycle = (1.0 / (gait.cadence * gait.dt)).round().max(1.0) as u64;
    let (_c, passing_tick) = contact_passing_ticks(&body, &gait);
    let mut target = 150u64;
    while target < passing_tick + cycle || target % cycle != passing_tick % cycle {
        target += 1;
    }
    for _ in 0..target {
        scene.command_bodies(6.0);
        scene.tick();
    }

    let dyn_bvh = Bvh::build(&scene.dynamic_leaf_triangles(), &bvh_params);
    let merged = Bvh::merge(&static_bvh, &dyn_bvh);

    // Depth probe (stack-overflow guard is 64 nodes deep).
    fn max_depth(b: &Bvh, n: usize, d: usize) -> usize {
        let node = &b.nodes[n];
        if node.count > 0 {
            d
        } else {
            let l = max_depth(b, node.left_first as usize, d + 1);
            let r = max_depth(b, node.left_first as usize + 1, d + 1);
            l.max(r)
        }
    }
    eprintln!(
        "[parity] depths: static={} dyn={} merged={} (stack cap 64)",
        max_depth(&static_bvh, 0, 0),
        max_depth(&dyn_bvh, 0, 0),
        max_depth(&merged, 0, 0)
    );

    let mut int_params = IntegratorParams::default();
    if std::env::var("PRIM_ONLY").is_ok() {
        int_params.spp = 1;
        int_params.max_bounces = 0;
    }
    if let Ok(b) = std::env::var("BOUNCES") {
        int_params.max_bounces = b.parse().unwrap();
    }
    if let Ok(s) = std::env::var("SPP") {
        int_params.spp = s.parse().unwrap();
    }
    let poses = [
        (
            "front",
            Camera {
                eye: GVec3::from_array(params.camera_position),
                yaw: params.camera_yaw,
                pitch: params.camera_pitch,
                fov_y_radians: params.fov_y_degrees.to_radians(),
                near: params.near,
                far: params.far,
            },
        ),
        (
            "wide",
            camera_at([-4.5, 8.5, 33.0], [-5.5, 2.0, 15.5], 60.0),
        ),
        (
            "a2_steam",
            camera_at([3.5, 3.4, 33.0], [-1.0, 4.2, 25.6], 55.0),
        ),
    ];

    // CPU diagnosis: does the tree's hit() ever disagree with a brute-force
    // linear nearest-hit (the topology-free ground truth)? Same tiebreak.
    if std::env::var("DIAG").is_ok() {
        let pi: usize = std::env::var("DIAG_POSE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let cam = &poses[pi].1;
        const TE: f32 = 1e-5;
        const TA: f32 = 1e-4;
        let (right, up, forward) = cam.basis();
        let aspect = w as f32 / h as f32;
        let half = (cam.fov_y_radians * 0.5).tan();
        let right = right * (half * aspect);
        let up = up * half;
        let eye = [cam.eye.x, cam.eye.y, cam.eye.z];
        let tris = &merged.tris;
        // Band-aware exhaustive ground truth: min t over all tris, winner =
        // canonical key among all within the tie band of that min.
        let brute = |o: [f32; 3], d: [f32; 3]| -> Option<(f32, usize)> {
            let mut tmin = f32::INFINITY;
            for t in tris.iter() {
                if let Some(th) = tri_hit_pub(o, d, t, 1e-3, f32::INFINITY) {
                    tmin = tmin.min(th);
                }
            }
            if !tmin.is_finite() {
                return None;
            }
            let band = tmin * TE + TA;
            let mut win: Option<usize> = None;
            for (i, t) in tris.iter().enumerate() {
                if let Some(th) = tri_hit_pub(o, d, t, 1e-3, f32::INFINITY)
                    && th <= tmin + band
                    && (win.is_none() || key17(t) < key17(&tris[win.unwrap()]))
                {
                    win = Some(i);
                }
            }
            win.map(|i| (tmin, i))
        };
        let mut mism = 0u32;
        for py in 0..h {
            for px in 0..w {
                let sx = (2.0 * (px as f32 + 0.5) / w as f32) - 1.0;
                let sy = 1.0 - (2.0 * (py as f32 + 0.5) / h as f32);
                let dv = (forward + right * sx + up * sy).normalize();
                let d = [dv.x, dv.y, dv.z];
                let tree = merged.hit(eye, d, 1e-3, f32::INFINITY);
                let bf = brute(eye, d);
                let agree = match (tree, bf) {
                    (None, None) => true,
                    (Some((tt, ti)), Some((bt, bi))) => {
                        tt == bt && key17(&tris[ti as usize]) == key17(&tris[bi])
                    }
                    _ => false,
                };
                if !agree {
                    if mism < 6 {
                        eprintln!("[diag] px({px},{py}) tree={tree:?} brute={bf:?}");
                        if let (Some((_, ti)), Some((_, bi))) = (tree, bf) {
                            eprintln!("        tree_tri={:?}", &tris[ti as usize]);
                            eprintln!("        brute_tri={:?}", &tris[bi]);
                        }
                    }
                    mism += 1;
                }
            }
        }
        eprintln!("[diag] wide center-ray tree-vs-brute mismatches: {mism}");
        return;
    }

    let med_opt = if std::env::var("NO_MED").is_ok() {
        None
    } else {
        Some(&medium)
    };
    for (name, cam) in &poses {
        let accum = trace_headless(
            &device,
            &queue,
            &merged,
            cam,
            &scene.sun,
            scene.sky_top,
            scene.sky_horizon,
            w,
            h,
            frames,
            &int_params,
            med_opt,
        );
        let bytes: &[u8] = bytemuck::cast_slice(&accum);
        println!("[parity] {name:10} hash={:016x}", fnv(bytes));
        if let Ok(tag) = std::env::var("PARITY_DUMP") {
            std::fs::write(format!("/tmp/parity_{tag}_{name}.bin"), bytes).unwrap();
        }
    }
}
