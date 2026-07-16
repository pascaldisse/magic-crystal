//! The V0 ordeals. Each test is one trial from the summons.

use glam::Quat;
use homunculus::{Pose, Skeleton};
use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;
use vessel::{Mesh, Vessel, VesselParams};

fn morphologies() -> Vec<(&'static str, Skeleton)> {
    vec![
        ("humanoid", Skeleton::humanoid()),
        ("quadruped", Skeleton::quadruped()),
    ]
}

/// The cubic-cell edge the builder uses for a skeleton at given params — the
/// derived scale for the non-degeneracy epsilon.
fn cell_edge(vessel: &Vessel, params: &VesselParams) -> f32 {
    let base_radius = vessel
        .capsules
        .iter()
        .map(|c| c.radius)
        .fold(0.0f32, f32::max);
    let margin = params.smooth_k + params.padding * base_radius;
    let field = vessel::BodySdf::new(vessel.capsules.clone(), params.smooth_k);
    let (lo, hi) = field.bounds(margin);
    (hi - lo).max_element() / params.resolution as f32
}

// ---------------------------------------------------------------------------
// Ordeal 1 — mesh sanity: finite, non-degenerate triangles; unit normals.
// ---------------------------------------------------------------------------
#[test]
fn ordeal_mesh_sanity() {
    let params = VesselParams::default();
    for (label, skel) in morphologies() {
        let v = Vessel::build(&skel, &params);
        let m = &v.mesh;

        // Non-degeneracy epsilon derived from the grid cell size.
        let cell = cell_edge(&v, &params);
        let area_eps = (cell * 1.0e-3).powi(2);

        // Positions/normals finite; normals unit within 1e-4.
        let mut worst_norm_err = 0.0f32;
        for (p, n) in m.positions.iter().zip(m.normals.iter()) {
            assert!(p.is_finite(), "{label} non-finite position {p:?}");
            assert!(n.is_finite(), "{label} non-finite normal {n:?}");
            let err = (n.length() - 1.0).abs();
            worst_norm_err = worst_norm_err.max(err);
        }
        assert!(
            worst_norm_err < 1.0e-4,
            "{label} normal unit error {worst_norm_err:e} >= 1e-4"
        );

        // Every triangle finite and non-degenerate.
        let mut min_area = f32::INFINITY;
        for tri in m.indices.chunks_exact(3) {
            let (a, b, c) = (
                m.positions[tri[0] as usize],
                m.positions[tri[1] as usize],
                m.positions[tri[2] as usize],
            );
            let area = 0.5 * (b - a).cross(c - a).length();
            assert!(area.is_finite(), "{label} non-finite triangle area");
            min_area = min_area.min(area);
        }
        assert!(
            min_area > area_eps,
            "{label} degenerate triangle: min_area {min_area:e} <= eps {area_eps:e}"
        );
        println!(
            "[sanity] {label} verts={} tris={} min_area={:e} area_eps={:e} max_normal_err={:e}",
            m.vertex_count(),
            m.triangle_count(),
            min_area,
            area_eps,
            worst_norm_err
        );
    }
}

// ---------------------------------------------------------------------------
// Ordeal 2 — closedness: every edge shared by exactly two triangles.
// ---------------------------------------------------------------------------
#[test]
fn ordeal_closedness() {
    let params = VesselParams::default();
    for (label, skel) in morphologies() {
        let v = Vessel::build(&skel, &params);
        let m = &v.mesh;

        let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
        for tri in m.indices.chunks_exact(3) {
            for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                *edges.entry(key).or_insert(0) += 1;
            }
        }
        let boundary = edges.values().filter(|&&c| c != 2).count();
        println!(
            "[closedness] {label} edges={} boundary(non-2)={}",
            edges.len(),
            boundary
        );
        assert_eq!(
            boundary, 0,
            "{label} not watertight: {boundary} boundary edges"
        );
    }
}

// ---------------------------------------------------------------------------
// Ordeal 3 — weights: every vertex's weights sum to 1 within 1e-6.
// ---------------------------------------------------------------------------
#[test]
fn ordeal_weights_normalized() {
    let params = VesselParams::default();
    for (label, skel) in morphologies() {
        let v = Vessel::build(&skel, &params);
        let err = v.weights.max_sum_error();
        println!(
            "[weights] {label} verts={} max_sum_error={:e}",
            v.weights.per_vertex.len(),
            err
        );
        assert!(err < 1.0e-6, "{label} weight sum error {err:e} >= 1e-6");
    }
}

// ---------------------------------------------------------------------------
// Ordeal 4 — deformation: bending an elbow moves the forearm, not the torso.
// Regions are derived from capsule ownership (each vertex's dominant bone).
// ---------------------------------------------------------------------------
#[test]
fn ordeal_deformation_regions() {
    let params = VesselParams::default();
    let skel = Skeleton::humanoid();
    let v = Vessel::build(&skel, &params);

    let forearm = skel.index_of("L.forearm").unwrap();
    let hand = skel.index_of("L.hand").unwrap();
    let pelvis = skel.index_of("pelvis").unwrap();
    let spine: Vec<usize> = (0..)
        .map_while(|s| skel.index_of(&format!("spine.{s}")))
        .collect();

    // Region = capsule ownership. The joint that bends is the elbow, so the
    // moved sub-chain is exactly {forearm, hand}. A vertex is OWNED by that
    // sub-chain iff it carries any weight to those bones.
    let dominant = |vi: usize| v.weights.per_vertex[vi][0].0;
    let armchain_weight = |vi: usize| {
        v.weights.per_vertex[vi]
            .iter()
            .filter(|(b, _)| *b == forearm || *b == hand)
            .map(|(_, w)| *w)
            .sum::<f32>()
    };
    // Forearm region: the sub-chain owns it (dominant bone is forearm/hand).
    let forearm_region: Vec<usize> = (0..v.mesh.vertex_count())
        .filter(|&i| {
            let d = dominant(i);
            d == forearm || d == hand
        })
        .collect();
    // Torso region: a torso bone owns it AND the moved sub-chain does not
    // (zero forearm/hand weight) — the arm capsules never reach it.
    let torso_region: Vec<usize> = (0..v.mesh.vertex_count())
        .filter(|&i| {
            let d = dominant(i);
            (d == pelvis || spine.contains(&d)) && armchain_weight(i) == 0.0
        })
        .collect();
    assert!(!forearm_region.is_empty(), "no forearm-owned vertices");
    assert!(!torso_region.is_empty(), "no torso-owned vertices");

    // Bend the left elbow 90 degrees.
    let mut pose = Pose::bind(&skel);
    pose.local_rotations[forearm] = Quat::from_rotation_x(FRAC_PI_2);
    let posed = v.posed(&skel, &pose);

    let disp = |i: usize| (posed.positions[i] - v.mesh.positions[i]).length();

    let forearm_max = forearm_region
        .iter()
        .map(|&i| disp(i))
        .fold(0.0f32, f32::max);
    let forearm_mean =
        forearm_region.iter().map(|&i| disp(i)).sum::<f32>() / forearm_region.len() as f32;
    let torso_max = torso_region.iter().map(|&i| disp(i)).fold(0.0f32, f32::max);

    println!(
        "[deform] forearm verts={} mean_disp={:e} max_disp={:e} | torso verts={} max_disp={:e}",
        forearm_region.len(),
        forearm_mean,
        forearm_max,
        torso_region.len(),
        torso_max
    );

    assert!(
        forearm_max > 0.05,
        "forearm did not move (max {forearm_max:e})"
    );
    assert!(
        forearm_mean > 0.01,
        "forearm barely moved (mean {forearm_mean:e})"
    );
    assert!(torso_max < 1.0e-6, "torso moved (max {torso_max:e})");
}

// ---------------------------------------------------------------------------
// Ordeal 5 — determinism: same params -> byte-identical mesh (human + cat).
// ---------------------------------------------------------------------------
#[test]
fn ordeal_determinism() {
    let params = VesselParams::default();
    for (label, skel) in morphologies() {
        let a = Vessel::build(&skel, &params).mesh;
        let b = Vessel::build(&skel, &params).mesh;
        assert_eq!(
            a.to_le_bytes(),
            b.to_le_bytes(),
            "{label} mesh not byte-identical across builds"
        );
        println!(
            "[determinism] {label} tris={} bytes={} identical=true",
            a.triangle_count(),
            a.to_le_bytes().len()
        );
    }
}

// ---------------------------------------------------------------------------
// Ordeal 6 — both morphologies mesh without NaN at default resolution.
// ---------------------------------------------------------------------------
#[test]
fn ordeal_both_morphologies_no_nan() {
    let params = VesselParams::default();
    for (label, skel) in morphologies() {
        let m: Mesh = Vessel::build(&skel, &params).mesh;
        assert!(m.triangle_count() > 0, "{label} produced no triangles");
        for p in &m.positions {
            assert!(p.is_finite(), "{label} NaN/inf position");
        }
        for n in &m.normals {
            assert!(n.is_finite(), "{label} NaN/inf normal");
        }
        let max_idx = *m.indices.iter().max().unwrap();
        assert!(
            (max_idx as usize) < m.vertex_count(),
            "{label} index out of range"
        );
        println!(
            "[no-nan] {label} verts={} tris={} ok",
            m.vertex_count(),
            m.triangle_count()
        );
    }
}
