//! Char-editor ordeals — the substrate's proofs.
//!
//! 1. PARITY (the crown): nari + pink_cat re-expressed as params build
//!    byte-identical to the hand canon.
//! 2. DETERMINISM: same params -> byte-identical preset + identical notes.
//! 3. WATERTIGHT: every sampled param point across the ranges skins a closed
//!    mesh (every edge shared by exactly two triangles).
//! 4. NO PANIC: out-of-range params clamp + surface notes, never panic.

use std::collections::HashMap;

use char_editor::params::{
    CreatureParams, Morphology, PaletteParams, Proportions, RegionScheme, MAX_SCALE, MIN_SCALE,
};
use char_editor::{canon, BuildNote};
use homunculus::Skeleton;
use vessel::{Blend, Body, BodyRegions, Mesh, Palette, Preset, VesselParams};

/// The canonical byte image of a composed body: the idle-posed geometry plus the
/// per-vertex colours + region assignment. Two presets with identical bytes here
/// render pixel-identical.
fn body_bytes(preset: &Preset) -> Vec<u8> {
    let body = Body::from_preset(preset);
    let mesh: Mesh = body.idle_mesh();
    body.colored.to_bytes(&mesh)
}

/// Structural field-by-field equality of two presets (Preset has no `PartialEq`;
/// each of its parts does).
fn presets_equal(a: &Preset, b: &Preset) -> bool {
    a.name == b.name
        && a.skeleton == b.skeleton
        && a.vessel == b.vessel
        && a.regions == b.regions
        && a.palette == b.palette
}

// ── 1. PARITY ───────────────────────────────────────────────────────────────

#[test]
fn parity_nari_byte_identical() {
    let outcome = canon::nari_params().build("nari");
    assert!(
        outcome.is_clean(),
        "nari params should need no repair, got {:?}",
        outcome.notes
    );
    let hand = Preset::nari();

    // Structural parity, part by part.
    assert_eq!(
        outcome.preset.skeleton, hand.skeleton,
        "nari skeleton drift"
    );
    assert_eq!(outcome.preset.regions, hand.regions, "nari regions drift");
    assert_eq!(outcome.preset.palette, hand.palette, "nari palette drift");
    assert_eq!(outcome.preset.vessel, hand.vessel, "nari mesh params drift");
    assert!(presets_equal(&outcome.preset, &hand), "nari preset drift");

    // The true output: the composed body's bytes.
    assert_eq!(
        body_bytes(&outcome.preset),
        body_bytes(&hand),
        "nari composed body bytes drift"
    );
    println!("[parity] nari: byte-identical to vessel::Preset::nari (skeleton/regions/palette/mesh + composed body bytes)");
}

/// The pink_cat canon never shipped as an assembled `Preset` (only its parts:
/// `Skeleton::quadruped` + `BodyRegions::quadruped` + `Palette::pink_cat` +
/// default mesh). Assemble that hand preset here from the existing pieces and
/// prove the param build is byte-identical to it.
fn hand_pink_cat() -> Preset {
    let skeleton = Skeleton::quadruped();
    let regions = BodyRegions::quadruped(&skeleton);
    Preset {
        name: "pink_cat",
        skeleton,
        vessel: VesselParams::default(),
        regions,
        palette: Palette::pink_cat(),
    }
}

#[test]
fn parity_pink_cat_byte_identical() {
    let outcome = canon::pink_cat_params().build("pink_cat");
    assert!(
        outcome.is_clean(),
        "pink_cat params should need no repair, got {:?}",
        outcome.notes
    );
    let hand = hand_pink_cat();

    assert_eq!(
        outcome.preset.skeleton, hand.skeleton,
        "pink_cat skeleton drift vs Skeleton::quadruped"
    );
    assert_eq!(
        outcome.preset.regions, hand.regions,
        "pink_cat regions drift vs BodyRegions::quadruped"
    );
    assert_eq!(
        outcome.preset.palette, hand.palette,
        "pink_cat palette drift vs Palette::pink_cat"
    );
    assert_eq!(
        outcome.preset.vessel, hand.vessel,
        "pink_cat mesh params drift"
    );
    assert!(
        presets_equal(&outcome.preset, &hand),
        "pink_cat preset drift"
    );

    assert_eq!(
        body_bytes(&outcome.preset),
        body_bytes(&hand),
        "pink_cat composed body bytes drift"
    );
    println!("[parity] pink_cat: byte-identical to hand-assembled quadruped + Palette::pink_cat (skeleton/regions/palette/mesh + composed body bytes)");
}

// ── 2. DETERMINISM ───────────────────────────────────────────────────────────

#[test]
fn determinism_byte_identical() {
    for (label, params) in [
        ("nari", canon::nari_params()),
        ("pink_cat", canon::pink_cat_params()),
        ("default", CreatureParams::default()),
    ] {
        let a = params.build(label_static(label));
        let b = params.build(label_static(label));
        assert!(
            presets_equal(&a.preset, &b.preset),
            "{label} preset differs"
        );
        assert_eq!(a.notes, b.notes, "{label} notes differ");
        assert_eq!(
            body_bytes(&a.preset),
            body_bytes(&b.preset),
            "{label} composed body bytes differ across builds"
        );
    }
    println!("[determinism] nari + pink_cat + default: byte-identical across repeated builds");
}

fn label_static(label: &str) -> &'static str {
    match label {
        "nari" => "nari",
        "pink_cat" => "pink_cat",
        _ => "default",
    }
}

// ── 3. WATERTIGHT ─────────────────────────────────────────────────────────────

/// Count edges shared by other than two triangles — zero means watertight.
fn boundary_edges(mesh: &Mesh) -> usize {
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            *edges.entry(key).or_insert(0) += 1;
        }
    }
    edges.values().filter(|&&c| c != 2).count()
}

/// Set one proportion scalar on an otherwise-neutral set, by index. The order
/// mirrors [`Proportions`]' 15 fields.
fn set_scalar(p: &mut Proportions, index: usize, value: f32) {
    match index {
        0 => p.height = value,
        1 => p.pelvis = value,
        2 => p.torso = value,
        3 => p.neck = value,
        4 => p.head = value,
        5 => p.tail = value,
        6 => p.upper_arm = value,
        7 => p.forearm = value,
        8 => p.hand = value,
        9 => p.thigh = value,
        10 => p.shank = value,
        11 => p.foot = value,
        12 => p.shoulder_width = value,
        13 => p.hip_width = value,
        14 => p.girth = value,
        _ => unreachable!(),
    }
}

const PROPORTION_DIMS: usize = 15;

#[test]
fn watertight_across_sampled_params() {
    // N is DERIVED from the parameter dimensionality: 15 proportion scalars ×
    // 2 range extremes (MIN_SCALE, MAX_SCALE) × 2 morphologies, plus the 2
    // neutral base points (one per morphology).
    let morphologies = [Morphology::Biped, Morphology::Quadruped];
    let extremes = [MIN_SCALE, MAX_SCALE];
    let n = PROPORTION_DIMS * extremes.len() * morphologies.len() + morphologies.len();
    println!(
        "[watertight] N = {PROPORTION_DIMS} scalars × {} extremes × {} morphologies + {} base = {n} sampled param points",
        extremes.len(),
        morphologies.len(),
        morphologies.len()
    );

    // The sweep meshes at the PRODUCTION resolution (VesselParams::default) —
    // the same cell count the rendered bodies use, so the closedness proven
    // here is the closedness that ships. (Marching cubes never lets the surface
    // touch the grid boundary, so a body is closed by construction; the sweep
    // proves it holds under the full parameter span, not just the base pose.)
    let sweep_mesh = VesselParams::default();

    let mut checked = 0usize;
    let mut min_verts = usize::MAX;
    for morphology in morphologies {
        // Neutral base point.
        let base = CreatureParams {
            morphology,
            region_scheme: RegionScheme::Plain,
            proportions: Proportions::default(),
            palette: PaletteParams::default(),
            mesh: sweep_mesh,
        };
        let bv = check_watertight(&base, &format!("{morphology:?}/base"));
        min_verts = min_verts.min(bv);
        checked += 1;

        for index in 0..PROPORTION_DIMS {
            for &value in &extremes {
                let mut props = Proportions::default();
                set_scalar(&mut props, index, value);
                let params = CreatureParams {
                    morphology,
                    proportions: props,
                    ..base.clone()
                };
                let v = check_watertight(&params, &format!("{morphology:?}/scalar{index}={value}"));
                min_verts = min_verts.min(v);
                checked += 1;
            }
        }
    }
    assert_eq!(checked, n, "sampled point count mismatch");
    println!("[watertight] all {checked} sampled bodies closed (0 boundary edges) at production resolution; smallest full body {min_verts} verts (the MIN_SCALE floor is set so even the leanest girth resolves into a real, closed body)");
}

/// Assert the built body is non-empty and closed; return its vertex count.
fn check_watertight(params: &CreatureParams, label: &str) -> usize {
    let outcome = params.build("sample");
    let body = Body::from_preset(&outcome.preset);
    let mesh = body.idle_mesh();
    assert!(!mesh.positions.is_empty(), "{label}: empty mesh");
    let boundary = boundary_edges(&mesh);
    assert_eq!(
        boundary, 0,
        "{label}: not watertight ({boundary} boundary edges)"
    );
    mesh.positions.len()
}

// ── 4. NO PANIC ───────────────────────────────────────────────────────────────

#[test]
fn out_of_range_params_clamp_and_surface_never_panic() {
    let wild = CreatureParams {
        morphology: Morphology::Biped,
        region_scheme: RegionScheme::Plain,
        proportions: Proportions {
            height: f32::NAN,
            torso: f32::INFINITY,
            neck: -5.0,
            head: MAX_SCALE * 100.0,
            girth: 0.0,
            ..Proportions::default()
        },
        palette: PaletteParams {
            slots: vec![
                ("head".into(), "not-a-colour".into()),
                ("torso".into(), "#ffffff".into()),
            ],
            default: "also-bad".into(),
            blend: Blend::Hard,
        },
        mesh: VesselParams {
            resolution: 1,
            ..VesselParams::default()
        },
    };

    // Must not panic — and must produce a valid, closed body.
    let outcome = wild.build("wild");
    assert!(
        !outcome.notes.is_empty(),
        "wild input should surface repair notes"
    );

    // Every scalar landed back in range.
    let b = &outcome.preset.skeleton;
    assert!(b.validate().is_ok(), "clamped skeleton invalid");

    // The specific repairs are all surfaced.
    let has = |pred: &dyn Fn(&BuildNote) -> bool| outcome.notes.iter().any(pred);
    assert!(
        has(&|n| matches!(
            n,
            BuildNote::ScalarNotFinite {
                field: "height",
                ..
            }
        )),
        "NaN height not surfaced"
    );
    assert!(
        has(&|n| matches!(n, BuildNote::ScalarNotFinite { field: "torso", .. })),
        "inf torso not surfaced"
    );
    assert!(
        has(&|n| matches!(n, BuildNote::ScalarClamped { field: "neck", .. })),
        "negative neck not clamped"
    );
    assert!(
        has(&|n| matches!(n, BuildNote::ScalarClamped { field: "head", .. })),
        "huge head not clamped"
    );
    assert!(
        has(&|n| matches!(n, BuildNote::ScalarClamped { field: "girth", .. })),
        "zero girth not clamped"
    );
    assert!(
        has(&|n| matches!(n, BuildNote::ColorInvalid { .. })),
        "invalid slot colour not surfaced"
    );
    assert!(
        has(&|n| matches!(n, BuildNote::DefaultColorInvalid { .. })),
        "invalid default colour not surfaced"
    );
    assert!(
        has(&|n| matches!(n, BuildNote::MeshFloored { .. })),
        "degenerate resolution not floored"
    );

    // The repaired palette is entirely colour-legal.
    assert!(
        outcome.preset.palette.is_valid(),
        "repaired palette still has invalid colours"
    );

    // And it still skins a closed body.
    let body = Body::from_preset(&outcome.preset);
    assert_eq!(
        boundary_edges(&body.idle_mesh()),
        0,
        "repaired body not watertight"
    );
    println!(
        "[no-panic] wild input repaired into a valid closed body; {} notes surfaced",
        outcome.notes.len()
    );
}
