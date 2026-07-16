//! RITE V · WAVE V2 ordeals — THE PINK CAT.
//!
//! The QUADRUPED weld's trials, all pure (no GPU): the `pink_cat` preset
//! composes into a standing cat deterministically (byte-identical), stays
//! watertight after skinning at sama's idle pose, and its palette paints every
//! quadruped region. Same generic weld as nari's V0 ordeals — only the
//! morphology (quadruped skeleton + quadruped regions) differs.

use std::collections::HashMap;
use vessel::{Body, Preset};

// ---------------------------------------------------------------------------
// Ordeal V2-1 — skinning determinism: composing the pink_cat twice yields a
// BYTE-IDENTICAL idle mesh + colour set (the ENTROPY law on the quadruped weld).
// ---------------------------------------------------------------------------
#[test]
fn v2_cat_skinning_is_byte_identical() {
    let a = Body::from_preset(&Preset::pink_cat());
    let b = Body::from_preset(&Preset::pink_cat());

    let mesh_a = a.idle_mesh();
    let mesh_b = b.idle_mesh();
    let bytes_a = mesh_a.to_le_bytes();
    let bytes_b = mesh_b.to_le_bytes();
    assert_eq!(
        bytes_a, bytes_b,
        "pink_cat idle mesh must be byte-identical across composes"
    );

    let col_a = a.colored.to_bytes(&mesh_a);
    let col_b = b.colored.to_bytes(&mesh_b);
    assert_eq!(col_a, col_b, "pink_cat colours must be byte-identical");

    println!(
        "[v2-determinism] pink_cat tris={} bytes={} identical=true",
        mesh_a.indices.len() / 3,
        bytes_a.len()
    );
}

// ---------------------------------------------------------------------------
// Ordeal V2-2 — watertight preserved after skinning: the idle-posed cat mesh
// has ZERO boundary edges (LBS is topology-invariant → a watertight bind stays
// watertight; asserted on the POSED mesh, not just the bind).
// ---------------------------------------------------------------------------
#[test]
fn v2_cat_watertight_after_skinning() {
    let body = Body::from_preset(&Preset::pink_cat());
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
        "[v2-watertight] idle pink_cat edges={} boundary(non-2)={}",
        edges.len(),
        boundary
    );
    assert_eq!(
        boundary, 0,
        "posed pink_cat not watertight: {boundary} boundary edges"
    );
}

// ---------------------------------------------------------------------------
// Ordeal V2-3 — the quadruped palette paints every region and its colours are
// valid schema strings (the palette is the pink cat, not a placeholder).
// ---------------------------------------------------------------------------
#[test]
fn v2_cat_palette_paints_every_region() {
    let preset = Preset::pink_cat();
    assert!(
        preset.palette.is_valid(),
        "every pink_cat colour must be a schema string"
    );
    // The quadruped regions: head · body · legs · tail.
    assert_eq!(preset.palette.color_of("head"), "#ffd0dc", "soft pink face");
    assert_eq!(preset.palette.color_of("body"), "#ffc0cb", "pink coat");
    assert_eq!(preset.palette.color_of("legs"), "#ffb0c0", "pink legs");
    assert_eq!(preset.palette.color_of("tail"), "#ff9fb6", "deeper pink tail");

    // Every region resolves to a colour used by at least one vertex (no dead
    // region — the cat's tail chain IS honest geometry, so `tail` must paint).
    let body = Body::from_preset(&preset);
    let used = preset.regions.counts(&body.colored.regions);
    for (ri, region) in preset.regions.regions.iter().enumerate() {
        assert!(used[ri] > 0, "region {} paints no vertex", region.name);
    }
    println!(
        "[v2-palette] pink_cat regions={} all painted",
        preset.regions.len()
    );
}

// ---------------------------------------------------------------------------
// Ordeal V2-4 — the grounding derivation: the cat's LOWEST vertex is a `.foot`
// (paw) vertex, so grounding the whole AABB grounds the paws (the same
// derivation V1 uses, four paws instead of two feet). Proves the realm's
// authored y = -foot_min honestly rests the paws on the floor.
// ---------------------------------------------------------------------------
#[test]
fn v2_cat_lowest_vertex_is_a_paw() {
    let body = Body::from_preset(&Preset::pink_cat());
    let mesh = body.idle_mesh();

    let mut mesh_min = f32::INFINITY;
    let mut paw_min = f32::INFINITY;
    for (vi, p) in mesh.positions.iter().enumerate() {
        mesh_min = mesh_min.min(p.y);
        if let Some((bone, _)) = body.vessel.weights.per_vertex[vi].first() {
            if body.skeleton.bones[*bone].name.ends_with(".foot") {
                paw_min = paw_min.min(p.y);
            }
        }
    }
    // The lowest paw vertex IS the mesh floor — no non-paw vertex dips below it.
    assert!(
        (paw_min - mesh_min).abs() < 1e-6,
        "cat's lowest vertex must be a paw: paw_min={paw_min} mesh_min={mesh_min}"
    );
    println!("[v2-grounding] pink_cat paw_min == mesh_min == {mesh_min:.6} (paws ground the body)");
}
