//! The H0 ordeals. Each test is one trial from the summons.

use glam::{Affine3A, Vec3};
use homunculus::{
    skin::bind_weights, walk_pose, walk_pose_stream, BodyParams, Pose, Skeleton, WalkParams,
};

/// Directly fold local bind transforms into world space, independent of the FK
/// pose path — the reference the roundtrip is checked against.
fn direct_bind_world(skeleton: &Skeleton) -> Vec<Affine3A> {
    let mut world = Vec::with_capacity(skeleton.bones.len());
    for (i, b) in skeleton.bones.iter().enumerate() {
        let local = b.local_bind.to_affine();
        let w = match b.parent {
            Some(p) => {
                assert!(p < i);
                world[p] * local
            }
            None => local,
        };
        world.push(w);
    }
    world
}

fn max_affine_diff(a: &[Affine3A], b: &[Affine3A]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        for (cx, cy) in x.to_cols_array().iter().zip(y.to_cols_array().iter()) {
            worst = worst.max((cx - cy).abs());
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// Ordeal 1 — bind-pose roundtrip: FK of the identity pose == bind.
// ---------------------------------------------------------------------------
#[test]
fn ordeal_bind_pose_roundtrip() {
    for (label, skel) in [
        ("humanoid", Skeleton::humanoid()),
        ("quadruped", Skeleton::quadruped()),
    ] {
        let bind = direct_bind_world(&skel);
        let fk = Pose::bind(&skel).forward_kinematics(&skel);
        let err = max_affine_diff(&bind, &fk);
        println!("[roundtrip] {label}: max |FK(identity) - bind| = {err:e}");
        assert!(err == 0.0, "{label} roundtrip drifted by {err}");
    }
}

// ---------------------------------------------------------------------------
// Ordeal 2 — every vertex's weights sum to 1 within 1e-6.
// ---------------------------------------------------------------------------
fn test_mesh() -> Vec<Vec3> {
    // A lattice of vertices enclosing the body volume.
    let mut v = Vec::new();
    let mut x = -0.6f32;
    while x <= 0.6 {
        let mut y = -1.0f32;
        while y <= 2.0 {
            let mut z = -0.4f32;
            while z <= 0.4 {
                v.push(Vec3::new(x, y, z));
                z += 0.2;
            }
            y += 0.2;
        }
        x += 0.2;
    }
    v
}

#[test]
fn ordeal_weights_normalized() {
    let verts = test_mesh();
    for (label, skel) in [
        ("humanoid", Skeleton::humanoid()),
        ("quadruped", Skeleton::quadruped()),
    ] {
        for max_inf in [None, Some(4usize)] {
            let w = bind_weights(&skel, &verts, max_inf);
            let err = w.max_sum_error();
            println!(
                "[weights] {label} verts={} max_influences={:?} max_sum_error={:e}",
                verts.len(),
                max_inf,
                err
            );
            assert!(err < 1.0e-6, "{label} sum error {err} >= 1e-6");
        }
    }
}

// ---------------------------------------------------------------------------
// Ordeal 3 — generators produce expected bone counts / lengths from params.
// ---------------------------------------------------------------------------
#[test]
fn ordeal_generator_counts_and_lengths() {
    for (label, params) in [
        ("humanoid", BodyParams::humanoid()),
        ("quadruped", BodyParams::quadruped()),
    ] {
        let skel = Skeleton::from_params(&params);

        // Count derived from formula, not a frozen constant.
        let expected = 1
            + params.spine_count as usize
            + params.neck_count as usize
            + 1
            + params.tail_segments as usize
            + 12;
        assert_eq!(expected, params.bone_count());
        assert_eq!(skel.len(), expected, "{label} bone count");
        println!(
            "[generator] {label} bones={} (expected {expected})",
            skel.len()
        );

        // Lengths derived from params.
        let checks: &[(&str, f32)] = &[
            ("spine.0", params.spine_segment_len()),
            ("head", params.head * params.height),
            ("L.upperarm", params.upper_arm * params.height),
            ("R.forearm", params.forearm * params.height),
            ("L.hand", params.hand * params.height),
            ("L.thigh", params.thigh * params.height),
            ("R.shank", params.shank * params.height),
            ("L.foot", params.foot * params.height),
        ];
        for (name, expect) in checks {
            let idx = skel
                .index_of(name)
                .unwrap_or_else(|| panic!("{name} missing"));
            let got = skel.bones[idx].length;
            println!("[generator] {label} {name}.length={got} (expected {expect})");
            assert!((got - expect).abs() < 1e-6, "{label} {name} length");
        }

        if params.tail_segments > 0 {
            let idx = skel.index_of("tail.0").unwrap();
            let got = skel.bones[idx].length;
            let expect = params.tail_segment_len();
            println!("[generator] {label} tail.0.length={got} (expected {expect})");
            assert!((got - expect).abs() < 1e-6);
        }

        assert!(
            skel.validate().is_ok(),
            "{label} invalid: {:?}",
            skel.validate()
        );
    }

    // Non-frozen: perturb a param, count/length track it.
    let mut p = BodyParams::humanoid();
    p.spine_count = 5;
    p.height = 2.0;
    let s = Skeleton::from_params(&p);
    assert_eq!(s.len(), BodyParams::humanoid().bone_count() + 2);
    let spine0 = s.bones[s.index_of("spine.0").unwrap()].length;
    assert!((spine0 - p.spine_segment_len()).abs() < 1e-6);
    println!(
        "[generator] perturbed spine_count=5 height=2.0 -> bones={} spine.0={spine0}",
        s.len()
    );
}

// ---------------------------------------------------------------------------
// Ordeal 4 — walk determinism: same params/seed -> byte-identical stream.
// ---------------------------------------------------------------------------
#[test]
fn ordeal_walk_determinism() {
    let skel = Skeleton::humanoid();
    let params = WalkParams::default();

    let a = walk_pose_stream(&skel, &params, 128);
    let b = walk_pose_stream(&skel, &params, 128);

    let bytes_a: Vec<u8> = a.iter().flat_map(|p| p.to_le_bytes()).collect();
    let bytes_b: Vec<u8> = b.iter().flat_map(|p| p.to_le_bytes()).collect();
    assert_eq!(bytes_a, bytes_b, "walk stream not byte-identical");

    // Tick 100 specifically.
    let t100_x = walk_pose(&skel, &params, 100).to_le_bytes();
    let t100_y = walk_pose(&skel, &params, 100).to_le_bytes();
    assert_eq!(t100_x, t100_y, "tick 100 not byte-identical");

    // The gait is actually moving (not a frozen bind pose) at tick 100.
    let bind = Pose::bind(&skel).to_le_bytes();
    assert_ne!(t100_x, bind, "walk pose equals bind — no motion");

    // Seed changes the stream (seed is honored).
    let seeded = WalkParams { seed: 42, ..params };
    let t100_seeded = walk_pose(&skel, &seeded, 100).to_le_bytes();
    println!(
        "[walk] stream bytes={} tick100 deterministic=true seed-sensitive={}",
        bytes_a.len(),
        t100_seeded != t100_x
    );
    assert_ne!(t100_seeded, t100_x, "seed had no effect");
}

// ---------------------------------------------------------------------------
// Ordeal 5 — morphology continuity: human -> cat stays valid at 10 samples.
// ---------------------------------------------------------------------------
#[test]
fn ordeal_morphology_continuity() {
    let human = BodyParams::humanoid();
    let cat = BodyParams::quadruped();

    for i in 0..10 {
        let t = i as f32 / 9.0;
        let params = BodyParams::lerp(&human, &cat, t);
        let skel = Skeleton::from_params(&params);

        // Structurally valid: parents precede children, finite lengths.
        skel.validate()
            .unwrap_or_else(|e| panic!("t={t} invalid skeleton: {e}"));

        // FK yields only finite transforms (no NaN anywhere).
        let world = Pose::bind(&skel).forward_kinematics(&skel);
        for (b, w) in skel.bones.iter().zip(world.iter()) {
            for c in w.to_cols_array() {
                assert!(c.is_finite(), "t={t} bone {} non-finite FK", b.name);
            }
        }

        // Parent indices intact.
        for (bi, b) in skel.bones.iter().enumerate() {
            if let Some(p) = b.parent {
                assert!(p < bi, "t={t} bad parent order");
            }
        }

        println!(
            "[morphology] t={t:.3} bones={} spine={} neck={} tail={}",
            skel.len(),
            params.spine_count,
            params.neck_count,
            params.tail_segments
        );
    }
}
