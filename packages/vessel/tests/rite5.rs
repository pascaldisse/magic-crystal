//! RITE V · WAVE V0 ordeals — THE BODY STANDS.
//!
//! The weld's trials, all pure (no GPU): the nari preset composes into a
//! standing body deterministically, the skinned triangles track the bones'
//! forward kinematics (bone→vertex transform parity, tolerance DERIVED), and
//! the vessel stays watertight after skinning at sama's idle pose.

use glam::{Affine3A, Vec3};
use homunculus::skin::capsules as bone_capsules;
use homunculus::Pose;
use std::collections::HashMap;
use vessel::{bind_world, Body, Preset};

// ---------------------------------------------------------------------------
// Ordeal V0-1 — skinning determinism: composing nari twice yields a
// BYTE-IDENTICAL idle mesh + colour set (the ENTROPY law on the whole weld).
// ---------------------------------------------------------------------------
#[test]
fn v0_skinning_is_byte_identical() {
    let a = Body::from_preset(&Preset::nari());
    let b = Body::from_preset(&Preset::nari());

    let mesh_a = a.idle_mesh();
    let mesh_b = b.idle_mesh();
    let bytes_a = mesh_a.to_le_bytes();
    let bytes_b = mesh_b.to_le_bytes();
    assert_eq!(
        bytes_a, bytes_b,
        "nari idle mesh must be byte-identical across composes"
    );

    // Colours are pose-invariant but part of the weld's determinism.
    let col_a = a.colored.to_bytes(&mesh_a);
    let col_b = b.colored.to_bytes(&mesh_b);
    assert_eq!(col_a, col_b, "nari colours must be byte-identical");

    println!(
        "[v0-determinism] nari tris={} bytes={} identical=true",
        mesh_a.indices.len() / 3,
        bytes_a.len()
    );
}

// ---------------------------------------------------------------------------
// Ordeal V0-2 — bone→vertex transform parity vs the homunculus pose.
//
// The vessel's skinning (`Vessel::posed`, i.e. `bind::deform`) must reproduce
// linear-blend skinning driven by the homunculus forward kinematics: each
// vertex is Σ wᵢ · (posed_worldᵢ · bind_worldᵢ⁻¹) · rest. We recompute that
// reference INDEPENDENTLY here (straight from `Pose::forward_kinematics` +
// `bind_world`, never through `deform`) and compare, under sama's idle pose
// (identity) AND a non-trivial probe pose (bent elbows/knees). The residual is
// pure f32 round-off; the tolerance is DERIVED from the measured worst and
// gated ~10× above it. Perturbing ONE bone's transform in the reference is
// proven to break the gate by orders — the parity actually pins the skin to
// the bones.
// ---------------------------------------------------------------------------
#[test]
fn v0_bone_vertex_transform_parity() {
    let preset = Preset::nari();
    let body = Body::from_preset(&preset);
    let skeleton = &preset.skeleton;

    // Bind-pose bone transforms (the vessel's own bind).
    let bind = bind_world(skeleton);
    let _capsules = bone_capsules(skeleton, &bind); // (the bind capsules the skin used)

    // A non-trivial probe pose: bend every elbow and knee 0.7 rad about X.
    let mut probe = Pose::bind(skeleton);
    for (i, b) in skeleton.bones.iter().enumerate() {
        if b.name.ends_with(".forearm") || b.name.ends_with(".shank") {
            probe.local_rotations[i] = glam::Quat::from_rotation_x(0.7);
        }
    }

    // Independent LBS reference from the homunculus pose (NOT via `deform`).
    let reference_lbs = |posed: &[Affine3A]| -> Vec<Vec3> {
        let skin: Vec<Affine3A> = (0..skeleton.bones.len())
            .map(|i| posed[i] * bind[i].inverse())
            .collect();
        body.vessel
            .mesh
            .positions
            .iter()
            .zip(body.vessel.weights.per_vertex.iter())
            .map(|(&rest, w)| {
                let mut acc = Vec3::ZERO;
                for &(bone, weight) in w {
                    acc += skin[bone].transform_point3(rest) * weight;
                }
                acc
            })
            .collect()
    };

    let posed_bind = Pose::bind(skeleton).forward_kinematics(skeleton);
    let posed_probe = probe.forward_kinematics(skeleton);

    let idle_mesh = body.vessel.posed(skeleton, &Pose::bind(skeleton));
    let probe_mesh = body.vessel.posed(skeleton, &probe);
    let ref_idle = reference_lbs(&posed_bind);
    let ref_probe = reference_lbs(&posed_probe);

    let mut worst_idle = 0.0f32;
    let mut worst_probe = 0.0f32;
    for vi in 0..body.vessel.mesh.positions.len() {
        worst_idle = worst_idle.max((idle_mesh.positions[vi] - ref_idle[vi]).length());
        worst_probe = worst_probe.max((probe_mesh.positions[vi] - ref_probe[vi]).length());
    }

    println!(
        "[v0-parity] verts={}  worst |Δ| idle={worst_idle:e}  probe={worst_probe:e}",
        body.vessel.mesh.positions.len()
    );
    // DERIVED tolerance: residual is pure f32 round-off in the matrix
    // compose/inverse/transform/blend. Measured worst across both poses is
    // ~2.4e-7 m; gate at 1e-5 m (~42×). The perturbed reference below (a 1 cm
    // pelvis shift) blows past this by ~1e-2 m — orders over the gate.
    const PARITY_TOL: f32 = 1.0e-5;
    assert!(
        worst_idle < PARITY_TOL,
        "idle parity {worst_idle:e} >= {PARITY_TOL:e}"
    );
    assert!(
        worst_probe < PARITY_TOL,
        "probe parity {worst_probe:e} >= {PARITY_TOL:e}"
    );

    // PROVE THE GATE BITES: perturb one bone's posed transform in the reference
    // (translate the pelvis 1 cm) and show the parity residual explodes — the
    // skin is genuinely pinned to the per-bone transforms.
    let mut broken_posed = posed_probe.clone();
    broken_posed[0].translation += glam::Vec3A::new(0.01, 0.0, 0.0);
    let ref_broken = reference_lbs(&broken_posed);
    let mut broke = 0.0f32;
    for (posed, reference) in probe_mesh.positions.iter().zip(ref_broken.iter()) {
        broke = broke.max((*posed - *reference).length());
    }
    assert!(
        broke > PARITY_TOL * 100.0,
        "a perturbed bone must break the gate hard, got {broke:e}"
    );
    println!("[v0-parity] perturbed-bone residual={broke:e} (gate proven to bite)");
}

// ---------------------------------------------------------------------------
// Ordeal V0-3 — watertight preserved after skinning: the idle-posed nari mesh
// has ZERO boundary edges (LBS is topology-invariant, so a watertight bind
// stays watertight — asserted on the POSED mesh, not just the bind).
// ---------------------------------------------------------------------------
#[test]
fn v0_watertight_after_skinning() {
    let body = Body::from_preset(&Preset::nari());
    let mesh = body.idle_mesh();

    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            *edges.entry(key).or_insert(0) += 1;
        }
    }
    let boundary = edges.values().filter(|&&c| c != 2).count();
    println!(
        "[v0-watertight] idle nari edges={} boundary(non-2)={}",
        edges.len(),
        boundary
    );
    assert_eq!(
        boundary, 0,
        "posed nari not watertight: {boundary} boundary edges"
    );
}

// ---------------------------------------------------------------------------
// Ordeal V0-4 — the preset colours are all valid schema strings and the
// avatar-canon garment hexes are present on their regions (the palette is the
// avatar, not a placeholder).
// ---------------------------------------------------------------------------
#[test]
fn v0_nari_palette_is_avatar_canon() {
    let preset = Preset::nari();
    assert!(
        preset.palette.is_valid(),
        "every nari colour must be a schema string"
    );
    // Enumerated canon garment hexes on their regions.
    assert_eq!(
        preset.palette.color_of("neck"),
        "#7c3aed",
        "neckerchief violet"
    );
    assert_eq!(preset.palette.color_of("torso"), "#16121e", "seifuku body");
    assert_eq!(
        preset.palette.color_of("skirt"),
        "#0d0a12",
        "black pleated skirt"
    );
    assert_eq!(
        preset.palette.color_of("boots"),
        "#0d0a12",
        "platform boots"
    );
    assert_eq!(preset.palette.color_of("hair"), "#16121e", "obsidian hair");

    // Every region resolves to a colour used by at least one vertex (the palette
    // actually paints the body — no dead region).
    let body = Body::from_preset(&preset);
    let used = preset.regions.counts(&body.colored.regions);
    for (ri, region) in preset.regions.regions.iter().enumerate() {
        assert!(used[ri] > 0, "region {} paints no vertex", region.name);
    }
    println!(
        "[v0-palette] nari regions={} all painted",
        preset.regions.len()
    );
}
