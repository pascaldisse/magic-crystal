//! The H0 ordeals. Each test is one trial from the summons.

use glam::{Affine3A, Quat, Vec3};
use homunculus::{skin::bind_weights, BodyParams, Pose, Skeleton, SocketSet};

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
// Ordeal 4 — the walk-cycle determinism ordeal moved with its generator: the
// cyclic-locomotion path now lives in `sama::gait` (canonical forward path),
// and its byte-identity / motion / seed guarantees are proven by
// `sama`'s `ordeal_gait_walk_determinism` and `ordeal_walk_parity_with_canon`.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Serialize an affine to raw bytes for exact-equality checks.
// ---------------------------------------------------------------------------
fn affine_bytes(a: &Affine3A) -> Vec<u8> {
    a.to_cols_array()
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

// ---------------------------------------------------------------------------
// Ordeal 6 — socket world under identity == bind derivation, both morphologies.
// ---------------------------------------------------------------------------
#[test]
fn ordeal_socket_bind_derivation() {
    for (label, params) in [
        ("humanoid", BodyParams::humanoid()),
        ("quadruped", BodyParams::quadruped()),
    ] {
        let skel = Skeleton::from_params(&params);
        let set = SocketSet::standard(&skel, &params);
        assert!(!set.is_empty(), "{label} standard set empty");

        // Reference bone world transforms via the direct bind fold.
        let bind = direct_bind_world(&skel);
        // Socket world transforms under the identity pose.
        let ident = Pose::bind(&skel);

        let mut worst = 0.0f32;
        for s in &set.sockets {
            let via_bind = bind[s.bone] * s.local.to_affine();
            let via_fk = set.world_of(&s.name, &skel, &ident).unwrap();
            let d = max_affine_diff(&[via_bind], &[via_fk]);
            worst = worst.max(d);
            assert!(
                d == 0.0,
                "{label} socket {} identity != bind (drift {d})",
                s.name
            );
        }
        println!(
            "[socket] {label} sockets={} max|identity - bind|={worst:e}",
            set.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Ordeal 7 — under a 90° elbow bend, right_hand_grip follows the forearm/hand
// exactly (derived), and it actually moved off its bind position.
// ---------------------------------------------------------------------------
#[test]
fn ordeal_socket_follows_elbow() {
    let params = BodyParams::humanoid();
    let skel = Skeleton::from_params(&params);
    let set = SocketSet::humanoid(&skel);

    let elbow = skel.index_of("R.forearm").expect("R.forearm");
    let mut pose = Pose::bind(&skel);
    pose.local_rotations[elbow] = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);

    // Grip world under the bent pose, from the set's shared FK.
    let bent = set.world_of("right_hand_grip", &skel, &pose).unwrap();
    // Independent reference: FK the same pose, compose the socket's own local.
    let world = pose.forward_kinematics(&skel);
    let grip = set.get("right_hand_grip").unwrap();
    let reference = world[grip.bone] * grip.local.to_affine();
    let drift = max_affine_diff(&[bent], &[reference]);
    assert!(
        drift == 0.0,
        "grip does not follow the derived hand ({drift})"
    );

    // And it truly moved relative to bind (the elbow did something).
    let at_bind = set
        .world_of("right_hand_grip", &skel, &Pose::bind(&skel))
        .unwrap();
    let moved = max_affine_diff(&[at_bind], &[bent]);
    println!("[socket] elbow 90°: grip drift-from-derived={drift:e} moved-from-bind={moved:e}");
    assert!(moved > 0.1, "grip did not follow the elbow ({moved})");
}

// ---------------------------------------------------------------------------
// Ordeal 8 — defaults preserve V0 bone counts/lengths byte-exact; the
// neck/head split is opt-in and inserts exactly one zero-length pivot without
// moving any bind-pose world position.
// ---------------------------------------------------------------------------
#[test]
fn ordeal_defaults_preserve_v0() {
    // Frozen V0 counts (pelvis + spine + neck + head + tail + 12 limb bones).
    assert!(!BodyParams::humanoid().neck_head_split);
    assert!(!BodyParams::quadruped().neck_head_split);
    assert_eq!(Skeleton::humanoid().len(), 18, "V0 humanoid bone count");
    assert_eq!(Skeleton::quadruped().len(), 30, "V0 quadruped bone count");
    assert!(Skeleton::humanoid().head_yaw_bone().is_none());

    // Turning on the split: +1 bone, a head.yaw pivot appears, and the head
    // bone's bind-pose world position is byte-identical to the unsplit body.
    let mut split = BodyParams::humanoid();
    split.neck_head_split = true;
    let base = Skeleton::humanoid();
    let refined = Skeleton::from_params(&split);
    assert_eq!(refined.len(), base.len() + 1, "split adds exactly one bone");
    assert!(refined.head_yaw_bone().is_some(), "head.yaw pivot present");
    assert_eq!(refined.bones[refined.head_yaw_bone().unwrap()].length, 0.0);

    let base_head = direct_bind_world(&base)[base.head_bone().unwrap()];
    let refined_head = direct_bind_world(&refined)[refined.head_bone().unwrap()];
    assert_eq!(
        affine_bytes(&base_head),
        affine_bytes(&refined_head),
        "split moved the head bind position"
    );
    // Head length unchanged too.
    assert_eq!(
        base.bones[base.head_bone().unwrap()].length,
        refined.bones[refined.head_bone().unwrap()].length
    );
    println!(
        "[refine] V0 counts 18/30 preserved; split humanoid bones={} head bind byte-identical",
        refined.len()
    );
}

// ---------------------------------------------------------------------------
// Ordeal 9 — socket sets are a pure function of the skeleton: byte-identical
// across rebuilds (determinism), both morphologies.
// ---------------------------------------------------------------------------
#[test]
fn ordeal_socket_determinism() {
    for (label, params) in [
        ("humanoid", BodyParams::humanoid()),
        ("quadruped", BodyParams::quadruped()),
    ] {
        let skel = Skeleton::from_params(&params);
        let a = SocketSet::standard(&skel, &params);
        let b = SocketSet::standard(&skel, &params);
        assert_eq!(a, b, "{label} socket set not deterministic");

        let ident = Pose::bind(&skel);
        let ta: Vec<u8> = a
            .world_transforms(&skel, &ident)
            .iter()
            .flat_map(affine_bytes)
            .collect();
        let tb: Vec<u8> = b
            .world_transforms(&skel, &ident)
            .iter()
            .flat_map(affine_bytes)
            .collect();
        assert_eq!(ta, tb, "{label} socket transforms not byte-identical");
        let names: Vec<&str> = a.sockets.iter().map(|s| s.name.as_str()).collect();
        println!("[socket] {label} deterministic set {names:?}");
    }
}
