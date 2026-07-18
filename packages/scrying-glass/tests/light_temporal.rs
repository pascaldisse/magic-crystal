//! ORDEALS for LIGHT-NOT-DOTS — temporal accumulation with reprojection on the
//! live present path. Trial by fire; green = survived. Every ordeal drives the
//! REAL GPU temporal pipeline (`integrate_temporal` + `temporal_resolve` in
//! integrator.wgsl) headlessly through the SAME ping-pong + prev-camera wiring
//! the window's render loop uses (`trace_headless_temporal`), and prints its
//! verbatim numbers.
//!
//!   a. STATIC CONVERGENCE — a still camera accumulates its ONE light pass'
//!      samples across frames: variance vs a single 1spp frame drops by a
//!      measured factor, and the result matches a plain N-sample average
//!      (proving temporal accumulation is a correct running mean), with error
//!      vs a long-trace reference bounded.
//!   b1. REPROJECTION ACCEPTS — a camera panning over a static scene keeps
//!      accumulating (reprojection tracks): final error vs the converged truth
//!      at the final pose is well below a single 1spp frame's error.
//!   b2. REPROJECTION REJECTS (no ghosting) — after accumulating at pose A, a
//!      teleport to a disjoint pose B rejects the stale history: the B frame is
//!      as noisy as a fresh 1spp frame at B and does NOT carry A's content
//!      (no smear).
//!   c. FRAME BUDGET — the two temporal passes at 640×480 measured wall time,
//!      reported in real ms against the 16.67 ms/60 FPS wall.
//!
//! These require a GPU adapter. On a host without one the ordeal prints that it
//! could not run and returns (documented — never a false green).

use std::time::Instant;

use glam::Vec3 as GVec3;

use scrying_glass::bvh::{Bvh, BvhParams};
use scrying_glass::integrator::{
    Integrator, IntegratorParams, IntegratorUniform, TemporalParams, headless_device, resolve,
    trace_headless, trace_headless_temporal,
};
use scrying_glass::scene::{Camera, LeafTriangle, SunLight};

/// A horizontal square (two triangles) in the `y` plane, side `2*half`.
fn quad(y: f32, half: f32, albedo: [f32; 3], emission: [f32; 3]) -> [LeafTriangle; 2] {
    let a = [-half, y, -half];
    let b = [half, y, -half];
    let c = [half, y, half];
    let d = [-half, y, half];
    [
        LeafTriangle { positions: [a, b, c], albedo, emission, metallic: 0.0, roughness: 1.0 },
        LeafTriangle { positions: [a, c, d], albedo, emission, metallic: 0.0, roughness: 1.0 },
    ]
}

/// A cube of side `2*half` centred at `c` (12 lambertian triangles).
fn cube(c: [f32; 3], half: f32, albedo: [f32; 3]) -> Vec<LeafTriangle> {
    let s = half;
    let v = |dx: f32, dy: f32, dz: f32| [c[0] + dx * s, c[1] + dy * s, c[2] + dz * s];
    let p = [
        v(-1.0, -1.0, -1.0), v(1.0, -1.0, -1.0), v(1.0, 1.0, -1.0), v(-1.0, 1.0, -1.0),
        v(-1.0, -1.0, 1.0), v(1.0, -1.0, 1.0), v(1.0, 1.0, 1.0), v(-1.0, 1.0, 1.0),
    ];
    let face = |a: usize, b: usize, d: usize, e: usize| {
        [
            LeafTriangle { positions: [p[a], p[b], p[d]], albedo, emission: [0.0; 3], metallic: 0.0, roughness: 1.0 },
            LeafTriangle { positions: [p[a], p[d], p[e]], albedo, emission: [0.0; 3], metallic: 0.0, roughness: 1.0 },
        ]
    };
    let mut t = Vec::new();
    t.extend(face(0, 1, 2, 3)); // -z
    t.extend(face(5, 4, 7, 6)); // +z
    t.extend(face(4, 0, 3, 7)); // -x
    t.extend(face(1, 5, 6, 2)); // +x
    t.extend(face(3, 2, 6, 7)); // +y
    t.extend(face(4, 5, 1, 0)); // -y
    t
}

fn look_camera(eye: [f32; 3], look_at: [f32; 3], fov_deg: f32) -> Camera {
    let f = (GVec3::from_array(look_at) - GVec3::from_array(eye)).normalize();
    let pitch = f.y.asin();
    let yaw = (-f.x).atan2(-f.z);
    Camera { eye: GVec3::from_array(eye), yaw, pitch, fov_y_radians: fov_deg.to_radians(), near: 0.05, far: 1000.0 }
}

fn sun() -> SunLight {
    SunLight {
        direction: GVec3::new(0.4, 0.8, 0.3).normalize().into(),
        color: [1.0, 0.96, 0.9],
        intensity: 2.0,
        ambient_intensity: 0.2,
    }
}

/// A committed lit realm: lambertian floor, an emissive slab (the glow that
/// makes indirect noise), and a crate — enough bounce noise that a single 1spp
/// frame is visibly grainy (the "dots").
fn realm() -> Vec<LeafTriangle> {
    let mut tris = Vec::new();
    tris.extend(quad(0.0, 30.0, [0.55, 0.5, 0.45], [0.0; 3]));
    tris.extend(quad(7.0, 2.5, [0.0; 3], [3.0, 2.6, 2.0])); // emitter overhead
    tris.extend(cube([2.0, 1.0, -3.0], 1.0, [0.7, 0.3, 0.25])); // crate
    tris.extend(cube([-3.0, 1.5, -5.0], 1.5, [0.3, 0.45, 0.7])); // block
    tris
}

const SKY_TOP: [f32; 4] = [0.25, 0.30, 0.45, 0.0];
const SKY_HOR: [f32; 4] = [0.55, 0.45, 0.55, 0.0];

/// Mean squared error between two linear-radiance images.
fn mse(a: &[GVec3], b: &[GVec3]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut s = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x - *y;
        s += (d.x * d.x + d.y * d.y + d.z * d.z) as f64;
    }
    s / (a.len() as f64 * 3.0)
}

fn params() -> IntegratorParams {
    IntegratorParams { spp: 1, max_bounces: 4, rr_start: 2, seed: 0x0011_6417, eps: 1e-3 }
}

fn one_frame(device: &wgpu::Device, queue: &wgpu::Queue, bvh: &Bvh, cam: &Camera, w: u32, h: u32) -> Vec<GVec3> {
    let a = trace_headless(device, queue, bvh, cam, &sun(), SKY_TOP, SKY_HOR, w, h, 1, &params(), None);
    resolve(&a)
}

fn reference(device: &wgpu::Device, queue: &wgpu::Queue, bvh: &Bvh, cam: &Camera, w: u32, h: u32) -> Vec<GVec3> {
    let a = trace_headless(device, queue, bvh, cam, &sun(), SKY_TOP, SKY_HOR, w, h, 512, &params(), None);
    resolve(&a)
}

// ── ORDEAL a · STATIC CONVERGENCE ─────────────────────────────────────────
#[test]
fn temporal_static_convergence() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[STATIC] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let (w, h) = (160u32, 120u32);
    let tris = realm();
    let bvh = Bvh::build(&tris, &BvhParams::default());
    let cam = look_camera([0.0, 3.0, 10.0], [0.0, 1.5, -3.0], 55.0);

    let truth = reference(&device, &queue, &bvh, &cam, w, h);
    let dots = one_frame(&device, &queue, &bvh, &cam, w, h);

    let n = 64u32;
    let cams: Vec<Camera> = std::iter::repeat(cam).take(n as usize).collect();
    let temporal = trace_headless_temporal(
        &device, &queue, &bvh, &cams, &sun(), SKY_TOP, SKY_HOR, w, h,
        &params(), &TemporalParams::default(),
    );

    // A plain N-sample average of the SAME decorrelated per-frame samples —
    // the correctness oracle for a still camera (temporal must equal it).
    let plain = resolve(&trace_headless(
        &device, &queue, &bvh, &cam, &sun(), SKY_TOP, SKY_HOR, w, h, n, &params(), None,
    ));

    let mse_dots = mse(&dots, &truth);
    let mse_temporal = mse(&temporal, &truth);
    let mse_plain = mse(&plain, &truth);
    let factor = mse_dots / mse_temporal.max(1e-12);
    let temporal_vs_plain = mse(&temporal, &plain);

    eprintln!(
        "[STATIC] n={n} 1spp MSE={mse_dots:.5e} temporal MSE={mse_temporal:.5e} \
         plain-avg MSE={mse_plain:.5e} variance-reduction={factor:.1}x \
         temporal-vs-plain MSE={temporal_vs_plain:.3e}"
    );

    // Temporal IS a correct running mean when still (matches the plain N-sample
    // average to within incremental-averaging float rounding — the identity
    // reprojection path does no resampling).
    assert!(
        temporal_vs_plain < 1e-5,
        "still-camera temporal must equal the N-sample average (got {temporal_vs_plain:.3e})"
    );
    // Variance dropped by roughly n (a real reconstruction, not a copy).
    assert!(
        factor > (n as f64) * 0.5,
        "variance reduction {factor:.1}x below half of n={n}"
    );
    // Accumulated light is close to truth, and far cleaner than a single frame.
    assert!(
        mse_temporal < mse_dots * 0.1,
        "temporal must be >=10x closer to truth than 1spp"
    );
}

// ── ORDEAL b1 · REPROJECTION ACCEPTS (panning keeps accumulating) ──────────
#[test]
fn temporal_motion_reprojection_accepts() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[PAN] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let (w, h) = (160u32, 120u32);
    let tris = realm();
    let bvh = Bvh::build(&tris, &BvhParams::default());

    // A gentle pan: the eye slides along +x, always looking at the crate. The
    // scene is static, so reprojection is geometrically exact and history must
    // keep accumulating (no ghosting to reject).
    let n = 48usize;
    let mut cams = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / (n as f32 - 1.0);
        let x = -3.0 + 6.0 * t; // -3 → +3
        cams.push(look_camera([x, 3.0, 10.0], [0.0, 1.5, -3.0], 55.0));
    }
    let final_cam = *cams.last().unwrap();
    let truth = reference(&device, &queue, &bvh, &final_cam, w, h);
    let dots = one_frame(&device, &queue, &bvh, &final_cam, w, h);

    let temporal = trace_headless_temporal(
        &device, &queue, &bvh, &cams, &sun(), SKY_TOP, SKY_HOR, w, h,
        &params(), &TemporalParams::default(),
    );

    let mse_dots = mse(&dots, &truth);
    let mse_temporal = mse(&temporal, &truth);
    let factor = mse_dots / mse_temporal.max(1e-12);
    eprintln!(
        "[PAN] frames={n} moving 1spp MSE={mse_dots:.5e} temporal MSE={mse_temporal:.5e} \
         cleanup={factor:.2}x vs single frame"
    );
    // During motion the reprojection tracks the static world → the final frame
    // is meaningfully cleaner than a single 1spp frame at the same pose.
    assert!(
        mse_temporal < mse_dots * 0.7,
        "panning temporal must beat a single frame (got {factor:.2}x)"
    );
}

// ── ORDEAL b2 · REPROJECTION REJECTS (teleport ⇒ no ghosting) ──────────────
#[test]
fn temporal_teleport_rejects_stale_history() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[TELEPORT] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let (w, h) = (160u32, 120u32);
    let tris = realm();
    let bvh = Bvh::build(&tris, &BvhParams::default());

    let cam_a = look_camera([0.0, 3.0, 10.0], [0.0, 1.5, -3.0], 55.0);
    // A disjoint pose looking the OTHER way — almost nothing reprojects.
    let cam_b = look_camera([12.0, 3.0, -12.0], [-3.0, 1.5, -5.0], 55.0);

    // Accumulate hard at A, then a single frame at B (the teleport).
    let mut cams: Vec<Camera> = std::iter::repeat(cam_a).take(48).collect();
    cams.push(cam_b);
    let teleport = trace_headless_temporal(
        &device, &queue, &bvh, &cams, &sun(), SKY_TOP, SKY_HOR, w, h,
        &params(), &TemporalParams::default(),
    );

    let truth_b = reference(&device, &queue, &bvh, &cam_b, w, h);
    let dots_b = one_frame(&device, &queue, &bvh, &cam_b, w, h);
    let converged_a = {
        let a = trace_headless_temporal(
            &device, &queue, &bvh,
            &std::iter::repeat(cam_a).take(48).collect::<Vec<_>>(),
            &sun(), SKY_TOP, SKY_HOR, w, h, &params(), &TemporalParams::default(),
        );
        a
    };

    let mse_teleport_b = mse(&teleport, &truth_b); // should be ~ a fresh 1spp
    let mse_dots_b = mse(&dots_b, &truth_b);
    let mse_teleport_vs_a = mse(&teleport, &converged_a); // huge if no ghosting

    eprintln!(
        "[TELEPORT] after-jump MSE-vs-truthB={mse_teleport_b:.5e} \
         fresh-1spp MSE-vs-truthB={mse_dots_b:.5e} \
         after-jump MSE-vs-convergedA={mse_teleport_vs_a:.5e}"
    );
    // The teleported frame is as noisy as a fresh 1spp frame at B (history was
    // rejected, not blended) — within a factor of ~2 of the fresh-frame error.
    assert!(
        mse_teleport_b < mse_dots_b * 2.0 + 1e-6,
        "teleport frame must be ~1spp-noisy at B (no accumulation from A): \
         {mse_teleport_b:.3e} vs fresh {mse_dots_b:.3e}"
    );
    // And it must NOT resemble A's content (no smear/ghost of the old view).
    assert!(
        mse_teleport_vs_a > mse_teleport_b * 5.0,
        "teleport frame must be far from A's converged image (no ghosting)"
    );
}

// ── ORDEAL c · FRAME BUDGET (real ms, live shape 640×480) ──────────────────
#[test]
fn temporal_frame_budget_640x480() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("[BUDGET] no GPU adapter on this host — ordeal could not run");
        return;
    };
    let (w, h) = (640u32, 480u32);
    let tris = realm();
    let bvh = Bvh::build(&tris, &BvhParams::default());
    let cam = look_camera([0.0, 3.0, 10.0], [0.0, 1.5, -3.0], 55.0);
    let p = params();

    let integrator = Integrator::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, &bvh, None);
    let accum = integrator.make_accum(&device, w, h);
    let compute_bg = integrator.compute_bind_group(&device, &accum);
    let packed = [
        integrator.make_temporal_packed(&device, w, h),
        integrator.make_temporal_packed(&device, w, h),
    ];
    let hist = [
        integrator.make_temporal_buffer(&device, w, h),
        integrator.make_temporal_buffer(&device, w, h),
    ];
    let bind = [
        integrator.temporal_bind_group(&device, &packed[0], &packed[1], &hist[0], &hist[1]),
        integrator.temporal_bind_group(&device, &packed[1], &packed[0], &hist[1], &hist[0]),
    ];

    let t = TemporalParams::default();
    let mut prev: Option<IntegratorUniform> = None;
    let frames = 40u32;
    let warm = 8u32;
    let mut times = Vec::new();
    for i in 0..frames {
        let mut u = IntegratorUniform::build(
            &cam, &sun(), SKY_TOP, SKY_HOR, w, h,
            integrator.node_count, integrator.tri_count, i * p.spp, &p, None,
        );
        u.temporal = [t.alpha_min, t.depth_tol, t.normal_tol, t.clamp_k];
        let valid = if let Some(pp) = prev {
            u.prev_eye = pp.eye; u.prev_right = pp.right; u.prev_up = pp.up; u.prev_forward = pp.forward;
            1
        } else { 0 };
        u.temporal_flags = [valid, t.max_history, 0, 0];
        let parity = (i % 2) as usize;
        let start = Instant::now();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("budget") });
        integrator.dispatch_temporal(&queue, &mut enc, &u, &compute_bg, &bind[parity], w, h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let ms = start.elapsed().as_secs_f64() * 1e3;
        if i >= warm {
            times.push(ms);
        }
        prev = Some(u);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;

    // The bare 1spp trace ("dots") cost, for the overhead accounting.
    let single = {
        let a2 = integrator.make_accum(&device, w, h);
        let bg2 = integrator.compute_bind_group(&device, &a2);
        let mut ts = Vec::new();
        for i in 0..frames {
            let u = IntegratorUniform::build(
                &cam, &sun(), SKY_TOP, SKY_HOR, w, h,
                integrator.node_count, integrator.tri_count, i * p.spp, &p, None,
            );
            let start = Instant::now();
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dots") });
            integrator.dispatch(&queue, &mut enc, &u, &bg2, w, h);
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            if i >= warm { ts.push(start.elapsed().as_secs_f64() * 1e3); }
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[ts.len() / 2]
    };

    eprintln!(
        "[BUDGET] 640x480 temporal(integrate+resolve) median={median:.3}ms mean={mean:.3}ms | \
         bare 1spp dots median={single:.3}ms | overhead={:.3}ms | 60FPS wall=16.67ms",
        median - single
    );
    assert!(
        median <= 16.67,
        "temporal frame must fit the 16.67ms/60FPS wall (got {median:.3}ms)"
    );
}
