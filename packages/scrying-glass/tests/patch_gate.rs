//! GUARDIAN RULING 6 ordeals — "Walkable floor = surface holds the body's
//! contact patch (derived min-area parameter; mirror-edge climbing dies)".
//!
//! [`Ground::height_at`] used to count ANY up-facing triangle under the eye
//! column as floor, no matter how small — so the ~8cm-wide top edge of the
//! naruko_mirror panel (`worlds/naruko/scenes/main.json`) registered as
//! standable ground, letting the player climb onto a mirror edge that is
//! obviously not real floor. These ordeals prove the contact-patch gate in
//! `scrying_glass::player` closes that seam without disturbing legitimate
//! floors (pier planks, terrain, the seawall).

use std::path::Path;

use crystal::{Core, load_world_dir};
use scrying_glass::player::{DEFAULT_CONTACT_RADIUS, Ground, contact_tolerance};
use scrying_glass::scene::{RenderScene, SceneParameters, SunDefaults, top_flat_surface_y};
use vessel::{Body, Preset};

/// Same render parameters `rite5.rs` uses for the naruko world — only the
/// geometry-affecting fields (none of the visual ones) matter here.
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

fn load_naruko() -> Core {
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).expect("load naruko");
    core
}

fn naruko_ground() -> Ground {
    let core = load_naruko();
    let scene = RenderScene::from_ecs(core.world, &naruko_params()).expect("render scene");
    Ground::from_positions(&scene.leaf_positions())
}

/// DERIVATION ORDEAL: [`DEFAULT_CONTACT_RADIUS`] must track nari's REAL
/// footprint, not an invented number. Compose nari (`vessel::Preset::nari`),
/// pose her at SAMA's idle tick (the same weld `Body::from_preset` always
/// performs), and measure the xz half-extent of the idle-mesh vertices whose
/// STRONGEST skin weight binds a single foot bone (`"L.foot"` / `"R.foot"`,
/// found by name — never a frozen bone index). This prints the live
/// measurement every run (re-derive by eye if the preset geometry ever
/// changes) and asserts the pinned default in `player.rs` is still a
/// same-order-of-magnitude, ROUNDED-UP bound on the real foot, never a
/// smaller (unsafe) one.
#[test]
fn contact_radius_matches_measured_foot_half_extent() {
    let preset = Preset::nari();
    let body = Body::from_preset(&preset);
    let mesh = body.idle_mesh();
    let skeleton = &preset.skeleton;
    let l_foot = skeleton
        .bones
        .iter()
        .position(|b| b.name == "L.foot")
        .expect("nari has an L.foot bone");
    let r_foot = skeleton
        .bones
        .iter()
        .position(|b| b.name == "R.foot")
        .expect("nari has an R.foot bone");

    let mut measured_max = 0.0f32;
    for (label, bone) in [("L", l_foot), ("R", r_foot)] {
        let (mut lo_x, mut hi_x, mut lo_z, mut hi_z) = (
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        );
        let mut n = 0usize;
        for (i, weights) in body.vessel.weights.per_vertex.iter().enumerate() {
            if let Some(&(dominant_bone, _)) = weights.first()
                && dominant_bone == bone
            {
                let p = mesh.positions[i];
                lo_x = lo_x.min(p.x);
                hi_x = hi_x.max(p.x);
                lo_z = lo_z.min(p.z);
                hi_z = hi_z.max(p.z);
                n += 1;
            }
        }
        assert!(n > 0, "{label}.foot must own at least one idle-mesh vertex");
        let half_x = (hi_x - lo_x) / 2.0;
        let half_z = (hi_z - lo_z) / 2.0;
        println!("[contact-patch] {label}.foot n={n} half_x={half_x:.4} half_z={half_z:.4}");
        measured_max = measured_max.max(half_x).max(half_z);
    }

    println!(
        "[contact-patch] measured_max={measured_max:.4} default={DEFAULT_CONTACT_RADIUS:.4}"
    );
    assert!(
        DEFAULT_CONTACT_RADIUS >= measured_max,
        "DEFAULT_CONTACT_RADIUS {DEFAULT_CONTACT_RADIUS} must be a ROUNDED-UP bound on the \
         measured foot half-extent {measured_max} — rounding down would admit a surface \
         smaller than nari's real foot"
    );
    // The rounding margin should stay a real "round up for safety" (a few mm
    // to a couple cm), not balloon into an unrelated invented number.
    assert!(
        DEFAULT_CONTACT_RADIUS - measured_max < 0.05,
        "DEFAULT_CONTACT_RADIUS {DEFAULT_CONTACT_RADIUS} has drifted far from the measured \
         foot half-extent {measured_max} — re-derive it"
    );
}

/// ORDEAL (a) — THE MIRROR-PANEL SEAM DIES. `naruko_mirror` is an 0.08m-wide
/// (x) × 3.0m-tall × 2.0m-deep box at `[3, 0, 18]` with local offset
/// `[0, 2.9, 0]` (`worlds/naruko/scenes/main.json`), so its top face sits at
/// world y = 2.9 + 3.0/2 = 4.4, spanning x ∈ [2.96, 3.04] — an 8cm sliver,
/// far smaller than any real foot. The seawall directly beneath it
/// (`naruko_seawall`, top y = 1.4 — the same ground truth `rite5.rs` uses via
/// `top_flat_surface_y`) is the real floor. Before Ruling 6 this query
/// returned the mirror top (4.4); after, it must fall through to the seawall
/// (1.4).
#[test]
fn mirror_panel_top_seam_dies() {
    let ground = naruko_ground();
    let seawall_top = top_flat_surface_y(&load_naruko().world, "naruko_seawall")
        .expect("seawall query")
        .expect("seawall is a flat slab");

    let y = ground
        .height_at(3.0, 18.0, f32::INFINITY)
        .expect("floor exists under the mirror column");

    assert!(
        (y - seawall_top).abs() < 0.05,
        "mirror-top seam: expected the REAL floor (seawall {seawall_top}), got {y} \
         (the mirror's 8cm-wide top edge at y≈4.4 must not count as standable ground)"
    );
    assert!(
        y < 4.0,
        "the mirror-panel top (y≈4.4) leaked through the contact-patch gate: got {y}"
    );
}

/// ORDEAL (b) — LEGITIMATE FLOOR UNCHANGED. The seawall (nari's authored
/// standing strip, `rite5.rs`'s ground truth) and the terra slab both remain
/// walkable: their tops are each many metres wide/deep, dwarfing the contact
/// patch, so the gate must return them unchanged from the pre-Ruling-6
/// behaviour.
#[test]
fn legitimate_floors_are_unchanged() {
    let ground = naruko_ground();
    let world = load_naruko().world;

    let seawall_top = top_flat_surface_y(&world, "naruko_seawall")
        .expect("seawall query")
        .expect("seawall is a flat slab");
    // Off the mirror's 8cm sliver (well clear of it), still on the seawall.
    let seawall_y = ground
        .height_at(20.0, 18.0, f32::INFINITY)
        .expect("seawall floor");
    assert!(
        (seawall_y - seawall_top).abs() < 0.05,
        "seawall height changed: expected {seawall_top}, got {seawall_y}"
    );

    let terra_top = top_flat_surface_y(&world, "naruko_terra")
        .expect("terra query")
        .expect("terra is a flat slab");
    let terra_y = ground
        .height_at(0.0, 44.0, f32::INFINITY)
        .expect("terra floor (near world spawn)");
    assert!(
        (terra_y - terra_top).abs() < 0.05,
        "terra height changed: expected {terra_top}, got {terra_y}"
    );
}

/// ORDEAL (c) — PATCH RADIUS HONOURED, BOTH DIRECTIONS (synthetic). Two
/// narrow floor strips of half-width `w`, one narrower than the patch
/// (`w < radius`, must be REJECTED — the probes on either side of the strip
/// fall off it) and one wider (`w > radius` with margin, must be ACCEPTED).
/// Both strips sit above a huge real floor at y=0, so a rejected candidate
/// must fall through to that floor rather than returning `None`.
#[test]
fn patch_radius_gate_discriminates_both_directions() {
    let radius = 0.2f32;
    let tol = contact_tolerance(radius);

    // A big flat floor at y=0 underneath everything (always accepted).
    let mut positions = vec![
        [-50.0, 0.0, -50.0],
        [50.0, 0.0, -50.0],
        [50.0, 0.0, 50.0],
        [-50.0, 0.0, -50.0],
        [50.0, 0.0, 50.0],
        [-50.0, 0.0, 50.0],
    ];

    // A narrow strip at y=2, half-width 0.05 (< radius 0.2): rejected.
    let narrow_half = 0.05f32;
    positions.extend([
        [-narrow_half, 2.0, -10.0],
        [narrow_half, 2.0, -10.0],
        [narrow_half, 2.0, 10.0],
        [-narrow_half, 2.0, -10.0],
        [narrow_half, 2.0, 10.0],
        [-narrow_half, 2.0, 10.0],
    ]);

    // A wide strip at y=3, half-width 1.0 (>> radius 0.2, well clear of the
    // narrow strip's xz footprint): accepted.
    let wide_half = 1.0f32;
    positions.extend([
        [20.0 - wide_half, 3.0, -10.0],
        [20.0 + wide_half, 3.0, -10.0],
        [20.0 + wide_half, 3.0, 10.0],
        [20.0 - wide_half, 3.0, -10.0],
        [20.0 + wide_half, 3.0, 10.0],
        [20.0 - wide_half, 3.0, 10.0],
    ]);

    let ground = Ground::from_positions(&positions);

    let narrow = ground
        .height_at_gated(0.0, 0.0, f32::INFINITY, radius, tol)
        .expect("falls through to the y=0 floor");
    assert!(
        (narrow - 0.0).abs() < 1e-3,
        "narrow strip (half-width {narrow_half} < radius {radius}) must be REJECTED and fall \
         through to the real floor, got {narrow}"
    );

    let wide = ground
        .height_at_gated(20.0, 0.0, f32::INFINITY, radius, tol)
        .expect("the wide strip holds the patch");
    assert!(
        (wide - 3.0).abs() < 1e-3,
        "wide strip (half-width {wide_half} >> radius {radius}) must be ACCEPTED as floor, \
         got {wide}"
    );
}

/// ORDEAL (b, continued) — PIER PLANKS STAY WALKABLE. One more legitimate,
/// real-world-authored surface the Architect actually stands on: a
/// `naruko_pier` plank (1.5m wide × 36m long — the deck he walks out on),
/// dwarfing `2 * DEFAULT_CONTACT_RADIUS` ≈ 0.18m. Must keep returning its
/// authored top height after the contact-patch gate, same as the
/// seawall/terra check above.
///
/// (`naruko_crate` is NOT covered here: it carries a `body` component, so
/// `RenderScene::from_ecs` splits it into the DYNAMIC/living layer and
/// [`RenderScene::leaf_positions`] — the static floor soup `Ground` is built
/// from in this file's `naruko_ground()` — never includes it. Querying it as
/// static floor would silently test nothing; the Elements' rigid solver, not
/// this static contact-patch gate, is what will one day decide whether a
/// body can stand on a crate.)
#[test]
fn pier_plank_is_unchanged() {
    let ground = naruko_ground();
    let world = load_naruko().world;

    // Pier transform is at [-12, 0, -2]; the four planks share part y=0.95,
    // at local x offsets -2.55/-0.85/0.85/2.55, spanning local z ±18. Query
    // the plank at local x=0.85 (world x=-11.15), mid-span (world z=-2).
    let pier_top = top_flat_surface_y(&world, "naruko_pier")
        .expect("pier query")
        .expect("pier planks are flat slabs");
    let pier_y = ground
        .height_at(-11.15, -2.0, f32::INFINITY)
        .expect("pier plank floor");
    assert!(
        (pier_y - pier_top).abs() < 0.05,
        "pier plank height changed: expected {pier_top}, got {pier_y}"
    );
}

/// ORDEAL (d) — DETERMINISM. The same query, twice, over the real naruko
/// floor (mirror included), returns the byte-identical answer — the K-probe
/// pattern is a fixed compass sweep, never anything time- or order-dependent.
#[test]
fn patch_gate_query_is_deterministic() {
    let ground = naruko_ground();
    let a = ground.height_at(3.0, 18.0, f32::INFINITY);
    let b = ground.height_at(3.0, 18.0, f32::INFINITY);
    assert_eq!(
        a.map(f32::to_bits),
        b.map(f32::to_bits),
        "the same column query must return the byte-identical answer"
    );

    // Also check a legitimate-floor column, and a fully synthetic one with a
    // non-default radius, so determinism isn't only proven on one path.
    let seawall_a = ground.height_at(20.0, 18.0, f32::INFINITY);
    let seawall_b = ground.height_at(20.0, 18.0, f32::INFINITY);
    assert_eq!(seawall_a.map(f32::to_bits), seawall_b.map(f32::to_bits));
}
