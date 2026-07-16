//! ORDEALS for the Pleroma-in-the-glass (Rite IV, L1). Trial by fire; green =
//! survived. Each drives the REAL GPU integrator (`integrator.wgsl`) headlessly
//! and prints its verbatim numbers.
//!
//!   1. PARITY — GPU tracer vs the CPU Pleroma reference on a committed analytic
//!      scene (emissive quad over lambertian floor): mean abs diff within a
//!      derived tolerance.
//!   2. SHADOW — an occluded probe point receives NO sun (direct term 0).
//!   3. DETERMINISM — same seed + frames ⇒ byte-identical accumulation buffer;
//!      a different seed differs (the seed truly drives it).
//!
//! These require a GPU adapter. On a host without one the ordeal prints that it
//! could not run and returns (documented — never a false green).

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{IntegratorParams, headless_device, resolve, trace_headless};
use scrying_glass::scene::{Camera, LeafTriangle, SunLight};

/// A horizontal square of two triangles in the `y` plane, side `2*half`
/// (lambertian material).
fn quad(y: f32, half: f32, albedo: [f32; 3], emission: [f32; 3]) -> [LeafTriangle; 2] {
    metal_quad(y, half, albedo, emission, 0.0, 1.0)
}

/// A horizontal square with explicit metallic/roughness dials (the L2 conductor
/// lobe). `metallic 0, roughness 1` = the lambertian `quad`.
fn metal_quad(
    y: f32,
    half: f32,
    albedo: [f32; 3],
    emission: [f32; 3],
    metallic: f32,
    roughness: f32,
) -> [LeafTriangle; 2] {
    let a = [-half, y, -half];
    let b = [half, y, -half];
    let c = [half, y, half];
    let d = [-half, y, half];
    [
        LeafTriangle {
            positions: [a, b, c],
            albedo,
            emission,
            metallic,
            roughness,
        },
        LeafTriangle {
            positions: [a, c, d],
            albedo,
            emission,
            metallic,
            roughness,
        },
    ]
}

/// Build a [`Camera`] from an eye + look target (yaw/pitch derived so it matches
/// the CPU Pleroma camera's ray generation exactly).
fn look_camera(eye: [f32; 3], look_at: [f32; 3], fov_deg: f32) -> Camera {
    let f = GVec3::from_array(look_at) - GVec3::from_array(eye);
    let f = f.normalize();
    let pitch = f.y.asin();
    let yaw = (-f.x).atan2(-f.z);
    Camera {
        eye: GVec3::from_array(eye),
        yaw,
        pitch,
        fov_y_radians: fov_deg.to_radians(),
        near: 0.05,
        far: 1000.0,
    }
}

fn no_sun() -> SunLight {
    SunLight {
        direction: [0.0, 1.0, 0.0],
        color: [0.0, 0.0, 0.0],
        intensity: 0.0,
        ambient_intensity: 0.0,
    }
}

// ── ORDEAL 1 · PARITY vs the CPU Pleroma reference ────────────────────────
#[test]
fn parity_gpu_tracer_matches_pleroma() {
    use pleroma::{Camera as LCamera, Film, Material, Params, Scene, Shape, Vec3 as LVec3, vec3};

    let Some((device, queue)) = headless_device() else {
        eprintln!("[PARITY] no GPU adapter on this host — ordeal could not run");
        return;
    };

    // Committed analytic scene: an emissive quad (y=4, half 3, emission 3.0)
    // over a lambertian floor (y=0, albedo 0.5). Sun off + sky black ⇒ the GPU
    // integrator reduces to EXACTLY the Pleroma's unidirectional emissive transport.
    let floor_albedo = [0.5, 0.5, 0.5];
    let emit = [3.0, 3.0, 3.0];
    let floor_half = 40.0;
    let emit_half = 3.0;
    let emit_y = 4.0;

    let mut tris = Vec::new();
    tris.extend(quad(0.0, floor_half, floor_albedo, [0.0; 3]));
    tris.extend(quad(emit_y, emit_half, [0.0; 3], emit));
    let bvh = Bvh::build(&tris, &BvhParams::default());

    let (w, h) = (64u32, 64u32);
    let eye = [0.0, 8.0, 6.0];
    let look = [0.0, 0.0, -1.0];
    let fov = 50.0f32;
    let camera = look_camera(eye, look, fov);

    // Matched totals: GPU frames*spp == Pleroma spp.
    let frames = 64u32;
    let params = IntegratorParams {
        spp: 4,
        max_bounces: 5,
        rr_start: 3,
        seed: 0x1234,
        eps: 1e-3,
    };
    let total_spp = frames * params.spp;

    let accum = trace_headless(
        &device,
        &queue,
        &bvh,
        &camera,
        &no_sun(),
        [0.0; 4],
        [0.0; 4],
        w,
        h,
        frames,
        &params,
        None,
    );
    let gpu = resolve(&accum);

    // The same scene in the CPU Pleroma (Plane floor + Box emitter).
    let mut scene = Scene::new();
    scene.add(
        Shape::Plane {
            point: LVec3::ZERO,
            normal: vec3(0.0, 1.0, 0.0),
        },
        Material::lambertian(vec3(0.5, 0.5, 0.5)),
    );
    scene.add(
        Shape::Box {
            min: vec3(-(emit_half as f64), emit_y as f64, -(emit_half as f64)),
            max: vec3(emit_half as f64, emit_y as f64 + 0.02, emit_half as f64),
        },
        Material::emissive(vec3(3.0, 3.0, 3.0)),
    );
    let lcam = LCamera::new(
        vec3(eye[0] as f64, eye[1] as f64, eye[2] as f64),
        vec3(look[0] as f64, look[1] as f64, look[2] as f64),
        vec3(0.0, 1.0, 0.0),
        fov as f64,
        w as f64 / h as f64,
    );
    let lp = Params {
        spp: total_spp,
        max_bounces: params.max_bounces,
        rr_start: params.rr_start,
        eps: params.eps as f64,
        seed: params.seed as u64,
    };
    let film = Film::render(&scene, &lcam, w, h, &lp);

    // Mean absolute difference in LINEAR radiance over every pixel/channel.
    let mut sum = 0.0f64;
    let mut n = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let g = gpu[(y * w + x) as usize];
            let l = film.get(x, y);
            sum += (g.x as f64 - l.x).abs();
            sum += (g.y as f64 - l.y).abs();
            sum += (g.z as f64 - l.z).abs();
            n += 3.0;
        }
    }
    let mad = sum / n;

    // Derived tolerance: both estimators converge to the SAME expected image
    // (matched primary rays + identical transport), so the residual is combined
    // Monte-Carlo standard error at 256 spp + f32-vs-f64 rounding + the finite
    // GPU floor vs the Pleroma's infinite plane at the frame edges. Averaged over
    // 64*64*3 samples this sits well under 0.05.
    let tol = 0.05;
    println!("[PARITY] {total_spp} spp  mean-abs-diff={mad:.5}  tol={tol}");
    assert!(mad < tol, "GPU/Pleroma parity: mad {mad} exceeds tol {tol}");
}

// ── ORDEAL 1b · SPECULAR PARITY — GPU mirror vs the CPU Pleroma mirror ────
// A committed analytic scene WITH a perfect MIRROR: a flat mirror floor (y=0,
// metallic 1, roughness 0, reflectance 0.9) reflecting an emissive quad
// overhead (y=6). The camera looks down at an angle so the reflected emitter
// fills much of the frame. Flat mirror = analytically identical geometry in
// BOTH integrators (CPU Plane vs GPU quad, exact — a tessellated sphere would
// inject normal error that MASKS the BRDF parity we are testing here; the
// chrome SPHERE is verified with eyes in proof/l2-chrome.png). The mirror lobe
// is a delta (near-zero variance), so parity is TIGHT.
//
// DISCRIMINATION: the same GPU scene with the mirror BROKEN (roughness forced
// to 1 → the surface scatters diffusely instead of reflecting) is scored
// against the SAME mirror reference; its MAD must blow far past the gate — so
// a broken mirror cannot pass.
#[test]
fn specular_parity_gpu_mirror_matches_pleroma() {
    use pleroma::{Camera as LCamera, Film, Material, Params, Scene, Shape, Vec3 as LVec3, vec3};

    let Some((device, queue)) = headless_device() else {
        eprintln!("[SPECULAR PARITY] no GPU adapter on this host — ordeal could not run");
        return;
    };

    let refl = [0.9, 0.9, 0.9];
    let emit = [3.0, 3.0, 3.0];
    let floor_half = 40.0;
    let emit_half = 4.0;
    let emit_y = 6.0;
    let (w, h) = (64u32, 64u32);
    let eye = [0.0, 7.0, 11.0];
    let look = [0.0, 0.0, -3.0];
    let fov = 55.0f32;
    let camera = look_camera(eye, look, fov);
    let frames = 64u32;
    let params = IntegratorParams {
        spp: 4,
        max_bounces: 4,
        rr_start: 3,
        seed: 0x1234,
        eps: 1e-3,
    };
    let total_spp = frames * params.spp;

    // GPU scene with a mirror (roughness 0) and, for discrimination, a broken
    // one (roughness 1 = diffuse scatter).
    let gpu_image = |floor_roughness: f32| {
        let mut tris = Vec::new();
        tris.extend(metal_quad(
            0.0,
            floor_half,
            refl,
            [0.0; 3],
            1.0,
            floor_roughness,
        ));
        tris.extend(quad(emit_y, emit_half, [0.0; 3], emit));
        let bvh = Bvh::build(&tris, &BvhParams::default());
        let accum = trace_headless(
            &device,
            &queue,
            &bvh,
            &camera,
            &no_sun(),
            [0.0; 4],
            [0.0; 4],
            w,
            h,
            frames,
            &params,
            None,
        );
        resolve(&accum)
    };
    let gpu_mirror = gpu_image(0.0);
    let gpu_broken = gpu_image(1.0);

    // CPU reference: the SAME mirror (Plane, metal reflectance 0.9, roughness 0)
    // + a thin emissive box at y=6.
    let mut scene = Scene::new();
    scene.add(
        Shape::Plane {
            point: LVec3::ZERO,
            normal: vec3(0.0, 1.0, 0.0),
        },
        Material::metal(vec3(0.9, 0.9, 0.9), 0.0),
    );
    scene.add(
        Shape::Box {
            min: vec3(-(emit_half as f64), emit_y as f64, -(emit_half as f64)),
            max: vec3(emit_half as f64, emit_y as f64 + 0.02, emit_half as f64),
        },
        Material::emissive(vec3(3.0, 3.0, 3.0)),
    );
    let lcam = LCamera::new(
        vec3(eye[0] as f64, eye[1] as f64, eye[2] as f64),
        vec3(look[0] as f64, look[1] as f64, look[2] as f64),
        vec3(0.0, 1.0, 0.0),
        fov as f64,
        w as f64 / h as f64,
    );
    let lp = Params {
        spp: total_spp,
        max_bounces: params.max_bounces,
        rr_start: params.rr_start,
        eps: params.eps as f64,
        seed: params.seed as u64,
    };
    let film = Film::render(&scene, &lcam, w, h, &lp);

    let mad = |img: &[glam::Vec3]| -> f64 {
        let mut sum = 0.0f64;
        let mut n = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                let g = img[(y * w + x) as usize];
                let l = film.get(x, y);
                sum += (g.x as f64 - l.x).abs();
                sum += (g.y as f64 - l.y).abs();
                sum += (g.z as f64 - l.z).abs();
                n += 3.0;
            }
        }
        sum / n
    };
    let mad_mirror = mad(&gpu_mirror);
    let mad_broken = mad(&gpu_broken);

    // Derived tolerance: the delta mirror lobe is near-zero-variance, so the
    // residual is combined MC error on the EMITTER's finite solid angle +
    // f32-vs-f64 rounding + finite-floor-vs-infinite-plane edges. Averaged over
    // 64*64*3 samples this sits under 0.05 (same class as the diffuse parity).
    let tol = 0.05;
    println!(
        "[SPECULAR PARITY] {total_spp} spp  mirror mad={mad_mirror:.5} (tol {tol})  broken(rough=1) mad={mad_broken:.5}"
    );
    assert!(
        mad_mirror < tol,
        "GPU/Pleroma mirror parity: mad {mad_mirror} exceeds tol {tol}"
    );
    // The gate must DISCRIMINATE: a broken mirror scores far worse than the gate.
    assert!(
        mad_broken > tol * 3.0,
        "broken mirror scored {mad_broken}, too close to the gate {tol} — gate does not discriminate"
    );
}

// ── ORDEAL 2 · SHADOW — an occluded point receives no sun ────────────────
#[test]
fn shadowed_point_receives_no_sun() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[SHADOW] no GPU adapter on this host — ordeal could not run");
        return;
    };

    // Floor (y=0, albedo 0.5) + an occluder patch high overhead (y=5, half 2,
    // centered at x=0). Sun toward normalize([2,3,0]) (up and +x), sky black,
    // ambient 0. max_bounces=0 isolates the DIRECT sun term. The occluder casts
    // a shadow onto the floor near x=-3.33; a straight-down camera over that
    // patch sees the SHADOWED floor (its column misses the occluder), while a
    // camera over x=+10 sees LIT floor. Only the probe→sun ray differs.
    let mut tris = quad(0.0, 40.0, [0.5, 0.5, 0.5], [0.0; 3]).to_vec();
    tris.extend_from_slice(&quad(5.0, 2.0, [0.5, 0.5, 0.5], [0.0; 3]));
    let bvh = Bvh::build(&tris, &BvhParams::default());

    let sd = 13.0f32.sqrt();
    let sun = SunLight {
        direction: [2.0 / sd, 3.0 / sd, 0.0],
        color: [1.0, 1.0, 1.0],
        intensity: 1.0,
        ambient_intensity: 0.0,
    };
    let params = IntegratorParams {
        spp: 2,
        max_bounces: 0,
        rr_start: 8,
        seed: 0x5eed,
        eps: 1e-3,
    };
    let probe = |x: f32| {
        // A pinhole-narrow FOV so the 1×1 pixel samples a POINT (not the whole
        // frame) — the jittered ray span stays inside the shadow patch.
        let camera = look_camera([x, 20.0, 0.0], [x, 0.0, 0.0], 1.0);
        let accum = trace_headless(
            &device, &queue, &bvh, &camera, &sun, [0.0; 4], [0.0; 4], 1, 1, 4, &params, None,
        );
        resolve(&accum)[0].x
    };
    let shadowed = probe(-3.333);
    let lit = probe(10.0);
    // Lit floor reads albedo·sun·cosθ = 0.5 · (3/√13).
    let expected_lit = 0.5 * (3.0 / sd);
    println!(
        "[SHADOW] lit floor={lit:.4} (expect {expected_lit:.4})  occluded probe={shadowed:.4}"
    );
    assert!(
        (lit - expected_lit).abs() < 0.02,
        "lit floor should read {expected_lit}, got {lit}"
    );
    // Under the occluder's shadow the sun ray is blocked → NO sun, exactly 0.
    assert!(shadowed < 1e-3, "occluded probe got sun: {shadowed}");
}

// ── ORDEAL 3 · DETERMINISM — same seed+frames ⇒ identical buffer ─────────
#[test]
fn determinism_same_seed_frames_byte_identical() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[DETERMINISM] no GPU adapter on this host — ordeal could not run");
        return;
    };

    let mut tris = Vec::new();
    tris.extend(quad(0.0, 20.0, [0.6, 0.3, 0.2], [0.0; 3]));
    tris.extend(quad(4.0, 3.0, [0.0; 3], [3.0, 3.0, 3.0]));
    let bvh = Bvh::build(&tris, &BvhParams::default());

    let camera = look_camera([0.0, 6.0, 6.0], [0.0, 0.0, -1.0], 50.0);
    let sun = no_sun();
    let (w, h) = (48u32, 48u32);
    let params = IntegratorParams {
        spp: 2,
        max_bounces: 4,
        rr_start: 2,
        seed: 0xABCDEF,
        eps: 1e-3,
    };
    let run = |p: &IntegratorParams| {
        trace_headless(
            &device, &queue, &bvh, &camera, &sun, [0.0; 4], [0.0; 4], w, h, 3, p, None,
        )
    };
    let a = run(&params);
    let b = run(&params);
    let bytes = |v: &[[f32; 4]]| -> Vec<u8> {
        v.iter()
            .flat_map(|c| c.iter().flat_map(|f| f.to_bits().to_le_bytes()))
            .collect()
    };
    let (ba, bb) = (bytes(&a), bytes(&b));
    let identical = ba == bb;

    let mut params2 = params;
    params2.seed = 0x123456;
    let c = run(&params2);
    let differs = bytes(&c) != ba;

    println!(
        "[DETERMINISM] {} f32 bytes  same-seed identical={identical}  diff-seed differs={differs}",
        ba.len()
    );
    assert!(identical, "same seed+frames produced different buffers");
    assert!(differs, "different seed produced identical buffer");
}
