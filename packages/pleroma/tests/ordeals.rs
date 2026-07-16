//! ORDEALS — the four trials that make L0 a REFERENCE, not just code.
//! Trial by fire; green = survived. Each prints its verbatim numbers.
//!
//!   1. FURNACE     — energy conservation / global equilibrium
//!   2. DIRECT      — analytic form factor (sphere→plane irradiance)
//!   3. DETERMINISM — same seed ⇒ byte-identical buffer
//!   4. ENERGY      — albedo 1 loses nothing, albedo 0.5 halves per bounce
//!
//! Every tolerance below is DERIVED, not chosen for convenience.

use pleroma::{estimate, radiance, vec3, Film, Material, Params, Ray, Scene, Shape, Vec3};
use std::f64::consts::PI;

fn approx(a: f64, b: f64, tol: f64, what: &str) {
    let d = (a - b).abs();
    assert!(
        d <= tol,
        "{what}: got {a}, expected {b}, |Δ|={d} > tol {tol}"
    );
}

// ── ORDEAL 1 · THE FURNACE TEST ─────────────────────────────────────────
// Setup: an inner lambertian sphere (albedo = 1, ALBEDO_FURNACE) fully
// enclosed by a large purely-emissive sphere of radiance Le. The canonical
// integrator correctness test.
//
// MATH: a surface point sees radiance Le from (essentially) every direction.
// Reflected radiance of a lambertian with BRDF a/π under uniform incident Le:
//   L_out = ∫_Ω (a/π) Le cosθ dω = (a/π) Le · π = a · Le.
// With a = 1 → L_out = Le. Moreover, because throughput is CONSERVED at
// albedo 1, EVERY path (however many inner bounces it takes) delivers
// exactly one Le when it finally meets the emitter — so the estimator has
// ~zero variance and the only error is the truncation bias of paths that
// exceed max_bounces without escaping, bounded by
//   |bias| ≤ P(inner-bounces > max_bounces) · Le  ≪ 1e-6 here.
// Derived tolerance: 1e-3 (safely above that bias).
#[test]
fn ordeal_furnace() {
    let le = 0.5_f64;
    let mut scene = Scene::new();
    scene.add(
        Shape::Sphere {
            center: Vec3::ZERO,
            radius: 100.0,
        },
        Material::emissive(Vec3::splat(le)),
    );
    scene.add(
        Shape::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        },
        Material::lambertian(Vec3::ONE), // albedo 1
    );
    let p = Params {
        spp: 256,
        ..Params::default()
    };
    // primary ray straight at the inner sphere
    let ray = Ray::new(vec3(0.0, 0.0, 5.0), vec3(0.0, 0.0, -1.0));
    let measured = estimate(&scene, ray, 0, &p);

    println!(
        "[FURNACE] Le={le}  measured=({:.6},{:.6},{:.6})  tol=1e-3",
        measured.x, measured.y, measured.z
    );
    approx(measured.x, le, 1e-3, "furnace R");
    approx(measured.y, le, 1e-3, "furnace G");
    approx(measured.z, le, 1e-3, "furnace B");
}

// ── ORDEAL 2 · ANALYTIC DIRECT LIGHT ────────────────────────────────────
// Setup: a lambertian plane point at the origin (normal +y, albedo a) with a
// small purely-emissive sphere (radiance Le, radius R) centered on the
// normal at height h. Nothing else emits; escaped rays are black.
//
// MATH (exact, on-axis uniform sphere): irradiance at the point
//   E = π Le sin²α,   sinα = R/h        (α = half-angle subtended)
// so E = π Le (R/h)². Reflected radiance (view-independent lambertian):
//   L = (a/π) E = a Le (R/h)².
// The unidirectional estimator draws cosine-weighted directions; each
// sample returns a·Le iff it strikes the emitter (prob P = projected solid
// angle / π = (R/h)²), else 0. So per-sample value ~ a·Le · Bernoulli(P):
//   mean  = a Le (R/h)²          (matches analytic)
//   var   = (a Le)² P(1-P),  σ_mean = a Le √(P(1-P)/N)
// With a=0.8, Le=1, R=1, h=2 → P=0.25, expected L = 0.2. At N=200000:
//   σ_mean = 0.8·√(0.25·0.75/200000) ≈ 7.75e-4  ⇒  5σ ≈ 3.9e-3.
// Derived tolerance: 4e-3 (a ~5σ two-sided CI).
#[test]
fn ordeal_direct_analytic() {
    let a = 0.8_f64;
    let le = 1.0_f64;
    let r = 1.0_f64;
    let h = 2.0_f64;
    let n = 200_000u32;

    let mut scene = Scene::new();
    scene.add(
        Shape::Plane {
            point: Vec3::ZERO,
            normal: vec3(0.0, 1.0, 0.0),
        },
        Material::lambertian(Vec3::splat(a)),
    );
    scene.add(
        Shape::Sphere {
            center: vec3(0.0, h, 0.0),
            radius: r,
        },
        Material::emissive(Vec3::splat(le)),
    );

    let p = Params {
        spp: n,
        ..Params::default()
    };
    // Primary ray hitting the plane exactly at the origin from a SHALLOW
    // side angle (radiance is view-independent) — it must NOT graze the
    // emitter sphere on the way in, so we stay below it (y ≤ 0.3 < h-R).
    let ray = Ray::new(
        vec3(4.0, 0.3, 0.0),
        (vec3(0.0, 0.0, 0.0) - vec3(4.0, 0.3, 0.0)).normalize(),
    );
    let measured = estimate(&scene, ray, 0, &p);

    let pp = (r / h) * (r / h); // projected-solid-angle / π
    let expected = a * le * pp;
    let e_analytic = PI * le * (r / h) * (r / h); // irradiance, for the log
    let sigma = a * le * ((pp * (1.0 - pp)) / n as f64).sqrt();
    let tol = 4e-3;

    println!(
        "[DIRECT] E=πLe(R/h)²={:.6}  L_expected=aLe(R/h)²={:.6}  measured={:.6}  σ_mean={:.6}  5σ≈{:.6}  tol={:.1e}",
        e_analytic, expected, measured.x, sigma, 5.0 * sigma, tol
    );
    approx(measured.x, expected, tol, "direct light");
    // grey scene ⇒ channels agree
    approx(measured.y, expected, tol, "direct light G");
    approx(measured.z, expected, tol, "direct light B");
}

// ── ORDEAL 3 · DETERMINISM ──────────────────────────────────────────────
// ENTROPY law: there is no randomness. Two renders with the same seed must
// be BYTE-IDENTICAL. The sampler is a pure function of (seed,pixel,sample,
// bounce,dim), so this holds by construction, not by luck.
#[test]
fn ordeal_determinism() {
    use pleroma::Camera;
    let mut scene = Scene::new();
    scene.add(
        Shape::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        },
        Material::lambertian(vec3(0.7, 0.3, 0.2)),
    );
    scene.add(
        Shape::Sphere {
            center: vec3(1.5, 1.5, 1.0),
            radius: 0.5,
        },
        Material::emissive(Vec3::splat(8.0)),
    );
    scene.add(
        Shape::Plane {
            point: vec3(0.0, -1.0, 0.0),
            normal: vec3(0.0, 1.0, 0.0),
        },
        Material::lambertian(Vec3::splat(0.5)),
    );
    let cam = Camera::new(
        vec3(0.0, 0.5, 5.0),
        Vec3::ZERO,
        vec3(0.0, 1.0, 0.0),
        40.0,
        1.0,
    );
    let p = Params {
        spp: 8,
        seed: 0xABCDEF,
        ..Params::default()
    };
    let a = Film::render(&scene, &cam, 32, 32, &p);
    let b = Film::render(&scene, &cam, 32, 32, &p);

    // byte-identical f32 buffer
    let ba = bytes_of(&a.data);
    let bb = bytes_of(&b.data);
    let identical = ba == bb;
    // and a DIFFERENT seed must differ (proves the seed actually drives it)
    let p2 = Params {
        seed: 0x123456,
        ..p
    };
    let c = Film::render(&scene, &cam, 32, 32, &p2);
    let differs = bytes_of(&c.data) != ba;

    println!(
        "[DETERMINISM] {} f32 bytes  same-seed identical={identical}  diff-seed differs={differs}",
        ba.len()
    );
    assert!(identical, "same seed produced different buffers");
    assert!(differs, "different seed produced identical buffer");
}

fn bytes_of(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

// ── ORDEAL 4 · ENERGY ───────────────────────────────────────────────────
// A lambertian plane point (normal +y, albedo a) sitting INSIDE a large
// emissive sphere (radiance Le): every cosine-sampled up-direction meets the
// emitter, so incident radiance over the hemisphere is uniformly Le and
//   L_out = a · Le   (exact — see furnace math; here zero self-view).
// Because every path returns exactly a·Le, the estimator variance is 0, so
// the derived 95% CI half-width σ_mean·1.96 = 0 and the tolerance is pure
// float slack (1e-6). We test a ∈ {1.0, 0.5, 0.25}:
//   a=1.0  → Le            (loses nothing)
//   a=0.5  → Le/2          (halves per bounce)
//   a=0.25 → Le/4          (two halvings)
#[test]
fn ordeal_energy() {
    let le = 1.0_f64;
    let measure = |a: f64| -> f64 {
        let mut scene = Scene::new();
        scene.add(
            Shape::Plane {
                point: Vec3::ZERO,
                normal: vec3(0.0, 1.0, 0.0),
            },
            Material::lambertian(Vec3::splat(a)),
        );
        scene.add(
            Shape::Sphere {
                center: Vec3::ZERO,
                radius: 100.0,
            },
            Material::emissive(Vec3::splat(le)),
        );
        let p = Params {
            spp: 64,
            ..Params::default()
        };
        let ray = Ray::new(vec3(0.0, 5.0, 0.0), vec3(0.0, -1.0, 0.0));
        estimate(&scene, ray, 0, &p).x
    };

    let m1 = measure(1.0);
    let m_half = measure(0.5);
    let m_quarter = measure(0.25);
    println!(
        "[ENERGY] Le={le}  a=1.0→{:.6} (exp {:.6})  a=0.5→{:.6} (exp {:.6})  a=0.25→{:.6} (exp {:.6})  tol=1e-6 (var=0)",
        m1, le, m_half, le * 0.5, m_quarter, le * 0.25
    );
    approx(m1, le, 1e-6, "energy a=1 (loses nothing)");
    approx(m_half, le * 0.5, 1e-6, "energy a=0.5 (halves)");
    approx(m_quarter, le * 0.25, 1e-6, "energy a=0.25 (two halvings)");
    // per-bounce halving relation, stated as a ratio
    approx(m_half / m1, 0.5, 1e-6, "half ratio");
    approx(m_quarter / m_half, 0.5, 1e-6, "quarter/half ratio");
}

// ── ORDEAL 5 · METALLIC FURNACE ────────────────────────────────────
// The furnace test, but the inner sphere is a PERFECT MIRROR (metallic 1,
// roughness 0, reflectance 1) instead of a lambertian. A mirror at reflectance
// 1 reflects EVERY incident direction losing nothing, so a point on it still
// sees Le from all directions after one reflection: L_out = Le exactly. This
// pins the specular lobe's energy conservation — the white furnace STILL closes
// with a conductor. Same truncation-bias bound as the lambertian furnace
// (every path carries throughput 1 until it meets the emitter) → tol 1e-3.
#[test]
fn ordeal_metallic_furnace() {
    let le = 0.5_f64;
    let mut scene = Scene::new();
    scene.add(
        Shape::Sphere {
            center: Vec3::ZERO,
            radius: 100.0,
        },
        Material::emissive(Vec3::splat(le)),
    );
    scene.add(
        Shape::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        },
        Material::metal(Vec3::ONE, 0.0), // perfect mirror, reflectance 1
    );
    let p = Params {
        spp: 256,
        ..Params::default()
    };
    let ray = Ray::new(vec3(0.0, 0.0, 5.0), vec3(0.0, 0.0, -1.0));
    let measured = estimate(&scene, ray, 0, &p);
    println!(
        "[METALLIC FURNACE] Le={le}  mirror measured=({:.6},{:.6},{:.6})  tol=1e-3",
        measured.x, measured.y, measured.z
    );
    approx(measured.x, le, 1e-3, "metallic furnace R");
    approx(measured.y, le, 1e-3, "metallic furnace G");
    approx(measured.z, le, 1e-3, "metallic furnace B");
}

// ── ORDEAL 6 · METALLIC ENERGY ─────────────────────────────────────
// A MIRROR plane point (metallic 1, roughness 0, reflectance a) inside a large
// emissive sphere. The single reflected direction meets the emitter (uniform
// Le over the hemisphere), so L_out = a·Le exactly — the mirror tint multiplies
// like an albedo. Variance is 0 (every path returns a·Le), so the tolerance is
// float slack (1e-6). Reflectance 1 → Le (loses nothing); 0.5 → Le/2; 0.25 → Le/4.
// Plus a ROUGH-metal energy BOUND: a roughness-0.3 mirror must not GAIN energy
// (L_out ≤ a·Le); GGX single-scatter conserves-or-loses (never > a·Le).
#[test]
fn ordeal_metallic_energy() {
    let le = 1.0_f64;
    let measure = |a: f64, rough: f64| -> f64 {
        let mut scene = Scene::new();
        scene.add(
            Shape::Plane {
                point: Vec3::ZERO,
                normal: vec3(0.0, 1.0, 0.0),
            },
            Material::metal(Vec3::splat(a), rough),
        );
        scene.add(
            Shape::Sphere {
                center: Vec3::ZERO,
                radius: 100.0,
            },
            Material::emissive(Vec3::splat(le)),
        );
        let p = Params {
            spp: 256,
            ..Params::default()
        };
        let ray = Ray::new(vec3(0.0, 5.0, 0.0), vec3(0.0, -1.0, 0.0));
        estimate(&scene, ray, 0, &p).x
    };
    let m1 = measure(1.0, 0.0);
    let m_half = measure(0.5, 0.0);
    let m_quarter = measure(0.25, 0.0);
    let rough = measure(1.0, 0.3);
    println!(
        "[METALLIC ENERGY] mirror a=1.0→{:.6} (exp {:.6})  a=0.5→{:.6} (exp {:.6})  a=0.25→{:.6} (exp {:.6})  rough(0.3,a=1)→{:.6} (≤{:.6})",
        m1, le, m_half, le * 0.5, m_quarter, le * 0.25, rough, le
    );
    approx(m1, le, 1e-6, "mirror energy a=1 (loses nothing)");
    approx(m_half, le * 0.5, 1e-6, "mirror energy a=0.5 (halves)");
    approx(
        m_quarter,
        le * 0.25,
        1e-6,
        "mirror energy a=0.25 (two halvings)",
    );
    // Rough metal: energy conserved or lost, NEVER gained (derived one-sided
    // bound; GGX single-scatter has no multiple-scatter energy return).
    assert!(
        rough <= le + 1e-6,
        "rough metal GAINED energy: {rough} > {le}"
    );
    assert!(
        rough > 0.5 * le,
        "rough metal lost implausibly much: {rough}"
    );
}

// ── ORDEAL 7 · MIRROR IMAGE (specular geometry) ────────────────────────
// A perfect mirror floor (y=0) reflects a small emitter directly overhead. A
// ray hitting the mirror straight down reflects straight UP and must strike the
// emitter — delivering reflectance·Le. Flip the emitter off the reflection axis
// and the same ray reflects into empty space → black. This proves the mirror
// reflects along the correct DIRECTION (a broken/diffuse mirror would scatter
// and read a dim non-zero average both on- AND off-axis).
#[test]
fn ordeal_mirror_image() {
    let le = 4.0_f64;
    let refl = 0.9_f64;
    let build = |emitter_x: f64| -> f64 {
        let mut scene = Scene::new();
        scene.add(
            Shape::Plane {
                point: Vec3::ZERO,
                normal: vec3(0.0, 1.0, 0.0),
            },
            Material::metal(Vec3::splat(refl), 0.0),
        );
        scene.add(
            Shape::Sphere {
                center: vec3(emitter_x, 6.0, 0.0),
                radius: 1.0,
            },
            Material::emissive(Vec3::splat(le)),
        );
        let p = Params {
            spp: 16,
            ..Params::default()
        };
        // straight down onto the mirror at the origin
        let ray = Ray::new(vec3(0.0, 5.0, 0.0), vec3(0.0, -1.0, 0.0));
        estimate(&scene, ray, 0, &p).x
    };
    let on_axis = build(0.0); // emitter overhead → reflection hits it
    let off_axis = build(20.0); // emitter far aside → reflection misses
    println!(
        "[MIRROR IMAGE] on-axis={on_axis:.6} (exp {:.6})  off-axis={off_axis:.6} (exp 0)",
        refl * le
    );
    approx(on_axis, refl * le, 1e-6, "mirror on-axis = refl·Le");
    approx(off_axis, 0.0, 1e-6, "mirror off-axis = black");
}

// A stray-use guard so `radiance` is exercised as public API too.
#[test]
fn radiance_is_public() {
    let mut scene = Scene::new();
    scene.add(
        Shape::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        },
        Material::emissive(Vec3::ONE),
    );
    let ray = Ray::new(vec3(0.0, 0.0, 5.0), vec3(0.0, 0.0, -1.0));
    let l = radiance(&scene, ray, 0, 0, &Params::default());
    approx(l.x, 1.0, 1e-9, "single emitter radiance");
}
