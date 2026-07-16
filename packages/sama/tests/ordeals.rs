//! The S0 ordeals — determinism, walk parity, state-machine timing, look-at
//! clamps, and both-morphology validity. Run with `--nocapture` to read the
//! per-ordeal numbers.

use glam::Quat;
use homunculus::{walk_pose, Pose, Skeleton, WalkParams};
use sama::{
    gait_pose, look_at, Gait, GaitParams, Gesture, Locomotion, LocomotionParams, LookAt,
    LookAtParams,
};

/// Max per-bone quaternion angle (radians) between two poses.
fn pose_max_angle(a: &Pose, b: &Pose) -> f32 {
    a.local_rotations
        .iter()
        .zip(b.local_rotations.iter())
        .map(|(x, y)| 2.0 * x.dot(*y).abs().min(1.0).acos())
        .fold(0.0_f32, f32::max)
}

/// Max per-bone component difference between two poses.
fn pose_max_component_diff(a: &Pose, b: &Pose) -> f32 {
    a.local_rotations
        .iter()
        .zip(b.local_rotations.iter())
        .map(|(x, y)| {
            (x.x - y.x)
                .abs()
                .max((x.y - y.y).abs())
                .max((x.z - y.z).abs())
                .max((x.w - y.w).abs())
        })
        .fold(0.0_f32, f32::max)
}

fn pose_is_finite(p: &Pose) -> bool {
    p.local_rotations
        .iter()
        .all(|q| q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite())
}

/// Build a deterministic command stream and run the locomotion machine over it,
/// returning the concatenated byte stream and the poses.
fn run_commands(skeleton: &Skeleton, commands: &[f32]) -> (Vec<u8>, Vec<Pose>) {
    let mut loco = Locomotion::new(LocomotionParams::default());
    let mut bytes = Vec::new();
    let mut poses = Vec::new();
    for &c in commands {
        let p = loco.step(skeleton, c);
        bytes.extend_from_slice(&p.to_le_bytes());
        poses.push(p);
    }
    (bytes, poses)
}

/// A representative command stream: idle -> walk -> run -> walk -> idle.
fn command_stream(ticks: usize) -> Vec<f32> {
    (0..ticks)
        .map(|t| match t {
            _ if t < 40 => 0.0,  // idle
            _ if t < 90 => 1.5,  // walk
            _ if t < 150 => 5.0, // run
            _ if t < 190 => 1.5, // walk
            _ => 0.0,            // idle
        })
        .collect()
}

#[test]
fn ordeal_determinism_byte_identical() {
    let skeleton = Skeleton::humanoid();
    let commands = command_stream(200);

    let (bytes_a, poses_a) = run_commands(&skeleton, &commands);
    let (bytes_b, _poses_b) = run_commands(&skeleton, &commands);

    // Byte-identity of the whole stream, and specifically at tick 200.
    assert_eq!(
        bytes_a, bytes_b,
        "full command->pose byte stream must match"
    );
    let tick200_a = &poses_a[199];
    let (_, poses_b2) = run_commands(&skeleton, &commands);
    let tick200_b = &poses_b2[199];
    assert_eq!(
        tick200_a.to_le_bytes(),
        tick200_b.to_le_bytes(),
        "pose at tick 200 must be byte-identical across runs"
    );

    println!(
        "ORDEAL determinism: 200-tick stream = {} bytes, run A == run B == run C (byte-identical); tick-200 diff = 0",
        bytes_a.len()
    );
}

#[test]
fn ordeal_walk_parity_with_homunculus() {
    let skeleton = Skeleton::humanoid();
    let sama_params = GaitParams::walk();
    let homu_params = WalkParams::default();

    let mut max_diff = 0.0_f32;
    let mut sample_bone = 0usize;
    let mut sample_sama = Quat::IDENTITY;
    let mut sample_homu = Quat::IDENTITY;
    for tick in 0..300u64 {
        let a = gait_pose(&skeleton, &sama_params, tick);
        let b = walk_pose(&skeleton, &homu_params, tick);
        let d = pose_max_component_diff(&a, &b);
        if d >= max_diff {
            max_diff = d;
            // capture the worst bone's two values for the report
            for (i, (x, y)) in a
                .local_rotations
                .iter()
                .zip(b.local_rotations.iter())
                .enumerate()
            {
                let bd = (x.x - y.x)
                    .abs()
                    .max((x.y - y.y).abs())
                    .max((x.z - y.z).abs())
                    .max((x.w - y.w).abs());
                if (bd - d).abs() < 1e-12 {
                    sample_bone = i;
                    sample_sama = *x;
                    sample_homu = *y;
                    break;
                }
            }
        }
    }

    // Derived tolerance: f32 epsilon scale. The two paths are the same math in
    // the same order, so the difference is expected to be exactly 0.
    let tolerance = f32::EPSILON; // 1.1920929e-7
    assert!(
        max_diff <= tolerance,
        "walk parity: max component diff {max_diff:e} exceeds tolerance {tolerance:e}"
    );

    // A representative moving bone (a thigh) mid-cycle, to show real values.
    let thigh = skeleton.index_of("L.thigh").unwrap();
    let sa = gait_pose(&skeleton, &sama_params, 15).local_rotations[thigh];
    let hb = walk_pose(&skeleton, &homu_params, 15).local_rotations[thigh];

    println!(
        "ORDEAL walk parity: max |diff| over 300 ticks = {max_diff:e} (tolerance {tolerance:e}); \
         worst bone #{sample_bone}: sama={:?} homunculus={:?}; \
         L.thigh@tick15: sama={:?} homunculus={:?}",
        sample_sama.to_array(),
        sample_homu.to_array(),
        sa.to_array(),
        hb.to_array()
    );
}

#[test]
fn ordeal_state_machine_timing_and_continuity() {
    let skeleton = Skeleton::humanoid();
    let params = LocomotionParams::default();
    let blend_ticks = params.blend_ticks as u64;
    let mut loco = Locomotion::new(params);

    // Command schedule with known transition ticks.
    let idle_to_walk = 40u64;
    let walk_to_run = 90u64;
    let run_to_idle = 150u64;
    let total = 200u64;

    let mut prev: Option<Pose> = None;
    let mut max_step_angle = 0.0_f32;

    // Recorded state entry ticks and blend-complete ticks.
    let mut entered_walk = None;
    let mut entered_run = None;
    let mut entered_idle_again = None;
    let mut blend_done_after_walk = None;

    for t in 0..total {
        let cmd = if t < idle_to_walk {
            0.0
        } else if t < walk_to_run {
            1.5
        } else if t < run_to_idle {
            5.0
        } else {
            0.0
        };
        let state_before = loco.state();
        let pose = loco.step(&skeleton, cmd);

        if state_before != loco.state() {
            match loco.state() {
                Gait::Walk if entered_walk.is_none() => entered_walk = Some(t),
                Gait::Run if entered_run.is_none() => entered_run = Some(t),
                Gait::Idle if entered_idle_again.is_none() && t > walk_to_run => {
                    entered_idle_again = Some(t)
                }
                _ => {}
            }
        }
        // First tick after entering walk where the blend has completed.
        if let Some(ew) = entered_walk {
            if blend_done_after_walk.is_none() && !loco.blending() && t >= ew {
                blend_done_after_walk = Some(t);
            }
        }

        if let Some(prev_pose) = &prev {
            max_step_angle = max_step_angle.max(pose_max_angle(prev_pose, &pose));
        }
        prev = Some(pose);
    }

    // Transition timing: the state flips on exactly the tick the command crosses
    // the threshold.
    assert_eq!(
        entered_walk,
        Some(idle_to_walk),
        "walk must begin at tick 40"
    );
    assert_eq!(entered_run, Some(walk_to_run), "run must begin at tick 90");
    assert_eq!(
        entered_idle_again,
        Some(run_to_idle),
        "idle must resume at tick 150"
    );
    // Blend completes exactly `blend_ticks` after it began.
    assert_eq!(
        blend_done_after_walk,
        Some(idle_to_walk + blend_ticks),
        "walk blend must complete blend_ticks after it began"
    );

    // Continuity bound (derived): per tick the pose can move by at most the
    // fastest gait's angular step plus the blend chord spread over blend_ticks.
    // Fastest gait angular step over a tick, sampled from the run gait.
    let run = GaitParams::run();
    let mut gait_step = 0.0_f32;
    let mut pr: Option<Pose> = None;
    for t in 0..120u64 {
        let p = gait_pose(&skeleton, &run, t);
        if let Some(prev) = &pr {
            gait_step = gait_step.max(pose_max_angle(prev, &p));
        }
        pr = Some(p);
    }
    // Largest chord a blend can span: idle(bind) vs run pose.
    let mut blend_chord = 0.0_f32;
    for t in 0..120u64 {
        let bind = Pose::bind(&skeleton);
        let p = gait_pose(&skeleton, &run, t);
        blend_chord = blend_chord.max(pose_max_angle(&bind, &p));
    }
    let derived_bound = gait_step + blend_chord / blend_ticks as f32;

    assert!(
        max_step_angle <= derived_bound,
        "continuity: max per-tick angle {max_step_angle} exceeds derived bound {derived_bound}"
    );

    println!(
        "ORDEAL state machine: walk@40 run@90 idle@150 (exact); blend completes @ {} (=40+{blend_ticks}); \
         max per-tick angle = {max_step_angle:.6} rad <= derived bound {derived_bound:.6} \
         (gait_step {gait_step:.6} + chord {blend_chord:.6}/{blend_ticks})",
        blend_done_after_walk.unwrap()
    );
}

#[test]
fn ordeal_look_at_clamps_and_error() {
    let skeleton = Skeleton::humanoid();
    let head = skeleton.index_of("head").expect("head bone");
    let params = LookAtParams::default();

    // Within limits: error -> 0. Requested equals resolved.
    let req_yaw = 0.5_f32;
    let req_pitch = 0.3_f32;
    let within = LookAt::resolve(req_yaw, req_pitch, &params);
    let yaw_err = (within.yaw - req_yaw).abs();
    let pitch_err = (within.pitch - req_pitch).abs();
    assert!(yaw_err <= f32::EPSILON, "in-range yaw error {yaw_err:e}");
    assert!(
        pitch_err <= f32::EPSILON,
        "in-range pitch error {pitch_err:e}"
    );

    // Applied on identity base, the head delta reproduces the requested angles.
    let base = Pose::bind(&skeleton);
    let g = look_at(head, req_yaw, req_pitch, &params);
    let posed = g.apply(&base);
    let (axis_y, ang_y) = posed.local_rotations[head].to_axis_angle();
    // Decompose is approximate for a compound rotation; check the yaw magnitude
    // via the expected quaternion directly.
    let expected = within.rotation();
    let dot = posed.local_rotations[head].dot(expected).abs().min(1.0);
    let compose_err = 2.0 * dot.acos();
    assert!(
        compose_err <= 1e-6,
        "compose error {compose_err:e} (axis {axis_y:?} ang {ang_y})"
    );

    // Beyond limits: clamped exactly to the cone.
    let clamped = LookAt::resolve(10.0, -10.0, &params);
    assert_eq!(clamped.yaw, params.max_yaw, "yaw clamps to +max");
    assert_eq!(clamped.pitch, -params.max_pitch, "pitch clamps to -max");

    println!(
        "ORDEAL look-at: in-range req (yaw {req_yaw}, pitch {req_pitch}) -> err (yaw {yaw_err:e}, \
         pitch {pitch_err:e}); compose err {compose_err:e}; \
         over-range (10,-10) -> clamped ({}, {})",
        clamped.yaw, clamped.pitch
    );
}

#[test]
fn ordeal_both_morphologies_valid() {
    for (label, skeleton) in [
        ("human", Skeleton::humanoid()),
        ("cat", Skeleton::quadruped()),
    ] {
        skeleton.validate().expect("skeleton valid");
        let head = skeleton.index_of("head").expect("head bone");

        // Cat gait: digitigrade paws + a wagging tail.
        let mut walk = GaitParams::walk();
        let mut run = GaitParams::run();
        if label == "cat" {
            walk.digitigrade = 0.4;
            run.digitigrade = 0.5;
        }
        let params = LocomotionParams {
            walk,
            run,
            ..LocomotionParams::default()
        };
        let mut loco = Locomotion::new(params);
        let commands = command_stream(200);

        let mut checked = 0usize;
        for (t, &c) in commands.iter().enumerate() {
            let base = loco.step(&skeleton, c);
            assert!(pose_is_finite(&base), "{label} base pose NaN at tick {t}");

            // Compose a head look-at every tick.
            let yaw = (t as f32 * 0.05).sin() * 2.0; // exceeds clamp on purpose
            let pitch = (t as f32 * 0.03).cos();
            let g: Gesture = look_at(head, yaw, pitch, &LookAtParams::default());
            let posed = g.apply(&base);
            assert!(pose_is_finite(&posed), "{label} posed NaN at tick {t}");

            // FK must also stay finite.
            let world = posed.forward_kinematics(&skeleton);
            assert!(
                world
                    .iter()
                    .all(|m| m.to_scale_rotation_translation().2.is_finite()),
                "{label} FK NaN at tick {t}"
            );
            checked += 1;
        }

        println!(
            "ORDEAL morphology [{label}]: {} bones, {checked} ticks of locomotion+gesture+FK, \
             zero NaN",
            skeleton.len()
        );
    }
}
