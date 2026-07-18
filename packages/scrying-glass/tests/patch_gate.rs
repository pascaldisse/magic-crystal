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

/// The static-floor triangle soup itself, kept separate from [`Ground`]
/// construction so a determinism ordeal can build TWO independent [`Ground`]
/// instances from the exact same positions (see
/// `patch_gate_query_is_deterministic`) — the actual nondeterminism risk is
/// in construction (iteration order, float reduction order), not the query.
fn naruko_leaf_positions() -> Vec<[f32; 3]> {
    let core = load_naruko();
    let scene = RenderScene::from_ecs(core.world, &naruko_params()).expect("render scene");
    scene.leaf_positions()
}

fn naruko_ground() -> Ground {
    Ground::from_positions(&naruko_leaf_positions())
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

    println!("[contact-patch] measured_max={measured_max:.4} default={DEFAULT_CONTACT_RADIUS:.4}");
    // The ROUNDING RULE, computed from the live measurement (not the pinned
    // constant): round the measured max UP to the next whole centimetre.
    // Asserting equality (not just "some rounding-up margin") means this
    // guard breaks LOUDLY the day the preset's feet move by >= 1 cm, instead
    // of silently tolerating an ever-widening, unexplained gap.
    let expected_default = (measured_max * 100.0).ceil() / 100.0;
    assert!(
        (DEFAULT_CONTACT_RADIUS - expected_default).abs() < 1e-6,
        "DEFAULT_CONTACT_RADIUS {DEFAULT_CONTACT_RADIUS} must equal the measured foot \
         half-extent {measured_max} rounded UP to the next whole centimetre \
         ({expected_default}) — re-derive it if nari's preset geometry changed"
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

/// Bracket tightness for [`patch_radius_gate_discriminates_both_directions`]:
/// the E/W probes at angle 0 and π land at exactly `x = ±radius`, so the
/// EXACT geometric threshold for "does the probe still land on a strip of
/// half-width `w`" is `w == radius`. A `5%` bracket around that point —
/// reject at `radius * (1 - BRACKET_TIGHTNESS)`, accept at
/// `radius * (1 + BRACKET_TIGHTNESS)` — is tight enough that an
/// implementation pinned to any OTHER scale factor (say, testing at
/// `0.5 * radius` or `2 * radius` instead of the probe ring's true
/// `radius`) would fail one side or the other, while staying well clear of
/// float/geometric noise (the barycentric edge epsilon is `1e-4`, the height
/// acceptance epsilon `player::COLUMN_EPSILON` is `1e-3`; a `5%` shift on a
/// `0.2` m radius is `0.01` m, two orders of magnitude coarser than either).
const BRACKET_TIGHTNESS: f32 = 0.05;

/// ORDEAL (c) — PATCH RADIUS HONOURED, BOTH DIRECTIONS (synthetic). Two
/// narrow floor strips of half-width `w`, one narrower than the patch
/// (`w < radius`, must be REJECTED — the probes on either side of the strip
/// fall off it) and one wider (`w > radius`, must be ACCEPTED), both
/// bracketing the EXACT threshold `w == radius` at [`BRACKET_TIGHTNESS`] —
/// see that constant's doc for why the bracket is drawn there and not looser.
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

    // A narrow strip at y=2, half-width radius*(1 - BRACKET_TIGHTNESS):
    // just inside the reject side of the exact w==radius threshold.
    let narrow_half = radius * (1.0 - BRACKET_TIGHTNESS);
    positions.extend([
        [-narrow_half, 2.0, -10.0],
        [narrow_half, 2.0, -10.0],
        [narrow_half, 2.0, 10.0],
        [-narrow_half, 2.0, -10.0],
        [narrow_half, 2.0, 10.0],
        [-narrow_half, 2.0, 10.0],
    ]);

    // A wide strip at y=3, half-width radius*(1 + BRACKET_TIGHTNESS): just
    // past the accept side of the same threshold, well clear of the narrow
    // strip's xz footprint.
    let wide_half = radius * (1.0 + BRACKET_TIGHTNESS);
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

/// EXPECTED-ADMIT DOCUMENTATION ORDEAL — the honest boundary of Ruling 6
/// tonight. `patch_supported` tests only "does the floor SET have raw floor
/// within `tol` of `y` under all [`CONTACT_PROBE_COUNT`] ring probes" — it is
/// a ring-of-points existence check, NOT proof of one contiguous surface
/// spanning the whole disc (see the doc comment on `Ground::patch_supported`
/// in `src/player.rs`). One concrete consequence: an 8cm-wide (half-width
/// 0.04m, narrower than [`DEFAULT_CONTACT_RADIUS`]) sliver hovering low
/// enough above a real surrounding floor that ALL 8 ring probes still land
/// on that lower floor within `tol` gets ADMITTED as standable, even though
/// the sliver itself is far too narrow to hold a foot. With
/// `DEFAULT_CONTACT_RADIUS` (0.09m), `contact_tolerance` is ≈0.29m, so a
/// sliver 0.2m above the real floor sits inside that budget. This is NOT a
/// bug fix target for tonight's Ruling 6 pass — it is the accepted seam,
/// machine-recorded here so the day it matters (someone builds a low narrow
/// ledge over a big floor and expects it to reject) this test is already a
/// failing-test flip away from proving the regression. No behaviour change:
/// this test only documents existing, intentional behaviour.
#[test]
fn sliver_low_enough_above_floor_is_admitted_by_design() {
    let radius = DEFAULT_CONTACT_RADIUS;
    let tol = contact_tolerance(radius);
    let sliver_height = 0.2f32;
    assert!(
        sliver_height < tol,
        "test setup: the sliver height {sliver_height} must sit inside the {tol} tolerance \
         budget for this to demonstrate the documented admit"
    );

    // A big flat floor at y=0 underneath everything.
    let mut positions = vec![
        [-50.0, 0.0, -50.0],
        [50.0, 0.0, -50.0],
        [50.0, 0.0, 50.0],
        [-50.0, 0.0, -50.0],
        [50.0, 0.0, 50.0],
        [-50.0, 0.0, 50.0],
    ];

    // An 8cm-wide sliver (half-width 0.04, narrower than `radius`) hovering
    // `sliver_height` above the floor, centred at the origin. All 8 ring
    // probes (at distance `radius` > 0.04 from the centre) fall OFF this
    // sliver's xz footprint and land on the y=0 floor instead — within `tol`
    // of `sliver_height`, so every probe reports "floor found nearby" and
    // the sliver is admitted.
    let sliver_half = 0.04f32;
    positions.extend([
        [-sliver_half, sliver_height, -sliver_half],
        [sliver_half, sliver_height, -sliver_half],
        [sliver_half, sliver_height, sliver_half],
        [-sliver_half, sliver_height, -sliver_half],
        [sliver_half, sliver_height, sliver_half],
        [-sliver_half, sliver_height, sliver_half],
    ]);

    let ground = Ground::from_positions(&positions);
    let y = ground
        .height_at_gated(0.0, 0.0, f32::INFINITY, radius, tol)
        .expect("floor exists under the sliver column");
    assert!(
        (y - sliver_height).abs() < 1e-3,
        "documented admit: expected the low sliver at {sliver_height} to be ADMITTED (ring \
         probes all find the surrounding floor within tol), got {y} instead — if this now \
         rejects, the gate's semantics changed and this test's doc comment needs revisiting, \
         not silent deletion"
    );
}

/// ORDEAL (d) — DETERMINISM, earning its place: this exercises the FULL
/// gated query (fallthrough included, not a single-shot lookup) at a
/// probe-straddling coordinate (the mirror-panel column, `(3.0, 18.0)` —
/// its 8cm sliver rejects on the first fallthrough step and falls through to
/// the seawall, so this genuinely walks the loop, not just one raw scan)
/// over the real naruko soup, run TWICE against the SAME `Ground`, and once
/// more against a FRESH `Ground` rebuilt from the SAME leaf positions. The
/// same-`Ground`-twice checks the query path (the K-probe compass sweep is
/// fixed, never time- or order-dependent); the fresh-rebuild check guards
/// the actual risk determinism has going forward — nondeterministic
/// CONSTRUCTION (triangle iteration order, float reduction order) producing
/// a `Ground` whose floor SET differs run to run even from identical input
/// positions.
#[test]
fn patch_gate_query_is_deterministic() {
    let positions = naruko_leaf_positions();
    let ground_a = Ground::from_positions(&positions);
    let ground_b = Ground::from_positions(&positions);

    let straddle_a1 = ground_a.height_at(3.0, 18.0, f32::INFINITY);
    let straddle_a2 = ground_a.height_at(3.0, 18.0, f32::INFINITY);
    assert_eq!(
        straddle_a1.map(f32::to_bits),
        straddle_a2.map(f32::to_bits),
        "the same fallthrough-exercising query on the same Ground must return the \
         byte-identical answer"
    );
    assert!(
        straddle_a1.is_some_and(|y| y < 4.0),
        "sanity: the mirror column must actually be exercising the fallthrough (rejecting the \
         sliver), got {straddle_a1:?}"
    );

    let straddle_b = ground_b.height_at(3.0, 18.0, f32::INFINITY);
    assert_eq!(
        straddle_a1.map(f32::to_bits),
        straddle_b.map(f32::to_bits),
        "a Ground rebuilt from the identical leaf positions must answer the identical \
         fallthrough query identically — construction, not just the query, must be \
         deterministic"
    );

    // Also check a legitimate-floor column (no fallthrough needed) on both
    // instances, so determinism isn't proven only on the fallthrough path.
    let seawall_a = ground_a.height_at(20.0, 18.0, f32::INFINITY);
    let seawall_b = ground_b.height_at(20.0, 18.0, f32::INFINITY);
    assert_eq!(seawall_a.map(f32::to_bits), seawall_b.map(f32::to_bits));
}
