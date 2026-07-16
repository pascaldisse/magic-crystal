//! KAMI K0 ordeals: determinism, patrol adherence, wander bounds, follow
//! convergence, look-at correctness. Each drives the REAL tick over a real
//! crystal ECS, applying the emitted ops between ticks (Flow of Data).

use crystal::{EcsWorld, Op};
use kami::{is_color_string, tick, Behavior, BindPose, Decorative, Registry, TickContext};
use serde_json::{json, Value};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// A live world plus the KAMI registration handles.
struct Harness {
    world: EcsWorld,
    reg: Registry,
}

impl Harness {
    fn new() -> Self {
        let mut world = EcsWorld::default();
        let reg = Registry::register(&mut world);
        Self { world, reg }
    }

    /// Spawn a behavior-carrying entity bound to `id` at `position`.
    fn spawn(&mut self, id: &str, position: [f64; 3], behavior: &Behavior) {
        let entity = self
            .world
            .create_entity(vec![
                (
                    self.reg.transform,
                    json!({ "position": position, "rotation": [0.0, 0.0, 0.0] }),
                ),
                (self.reg.behavior, serde_json::to_value(behavior).unwrap()),
            ])
            .unwrap();
        self.world.bind_gaia_id(id, entity).unwrap();
    }

    /// A plain transform-only entity (a follow/look-at target).
    fn spawn_target(&mut self, id: &str, position: [f64; 3]) {
        let entity = self
            .world
            .create_entity(vec![(
                self.reg.transform,
                json!({ "position": position, "rotation": [0.0, 0.0, 0.0] }),
            )])
            .unwrap();
        self.world.bind_gaia_id(id, entity).unwrap();
    }

    fn set_pos(&mut self, id: &str, position: [f64; 3]) {
        let entity = self.world.entity_for_gaia(id).unwrap();
        self.world
            .set_component_field(entity, self.reg.transform, "position", json!(position))
            .unwrap();
    }

    fn pos(&self, id: &str) -> [f64; 3] {
        let entity = self.world.entity_for_gaia(id).unwrap();
        let value = self
            .world
            .get_component_field(entity, self.reg.transform, "position")
            .unwrap();
        vec3(&value)
    }

    fn yaw(&self, id: &str) -> f64 {
        let entity = self.world.entity_for_gaia(id).unwrap();
        let value = self
            .world
            .get_component_field(entity, self.reg.transform, "rotation")
            .unwrap();
        vec3(&value)[1]
    }

    /// One tick: capture ops, then apply them (transform sets) to the world.
    fn step(&mut self, ctx: &TickContext) -> Vec<Op> {
        let ops = tick(&self.world, self.reg, ctx);
        for op in &ops {
            let Op::Set(set) = op else {
                continue;
            };
            let entity = self.world.entity_for_gaia(&set.id).unwrap();
            self.world
                .set_component(entity, self.reg.transform, set.value.clone())
                .unwrap();
        }
        ops
    }
}

fn vec3(value: &Value) -> [f64; 3] {
    let a = value.as_array().unwrap();
    [
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    ]
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Serialize a whole run's op stream to bytes for byte-identity checks.
fn run_bytes(build: impl Fn() -> Harness, seed: u64, ticks: u64) -> Vec<u8> {
    let mut h = build();
    let mut ctx = TickContext::new(seed, 0);
    let mut out = Vec::new();
    for t in 0..ticks {
        ctx = ctx.at(t);
        let ops = h.step(&ctx);
        out.extend_from_slice(serde_json::to_vec(&ops).unwrap().as_slice());
        out.push(b'\n');
    }
    out
}

// --- ORDEAL 1: determinism ------------------------------------------------

#[test]
fn ordeal_determinism_byte_identical_over_1000_ticks() {
    let build = || {
        let mut h = Harness::new();
        h.spawn(
            "patroller",
            [0.0, 0.0, 0.0],
            &Behavior::Patrol {
                waypoints: vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 8.0]],
                speed: 3.0,
            },
        );
        h.spawn(
            "roamer",
            [0.0, 0.0, 0.0],
            &Behavior::Wander {
                center: [0.0, 0.0, 0.0],
                radius: 6.0,
                speed: 2.5,
                retarget: 0.4,
            },
        );
        h.spawn_target("prey", [20.0, 0.0, 0.0]);
        h.spawn(
            "hunter",
            [-5.0, 0.0, 0.0],
            &Behavior::Follow {
                target: "prey".into(),
                speed: 4.0,
                stop_range: 1.5,
            },
        );
        h.spawn(
            "watcher",
            [3.0, 0.0, 3.0],
            &Behavior::LookAt {
                target: "prey".into(),
            },
        );
        h
    };
    let a = run_bytes(build, 0xC0FFEE, 1000);
    let b = run_bytes(build, 0xC0FFEE, 1000);
    assert_eq!(a.len(), b.len(), "op-stream length");
    assert_eq!(a, b, "op streams must be byte-identical");
    eprintln!(
        "ORDEAL determinism: 1000 ticks, {} bytes, byte-identical across 2 runs",
        a.len()
    );
}

// --- ORDEAL 2: patrol adherence -------------------------------------------

#[test]
fn ordeal_patrol_visits_every_waypoint_and_loop_closes() {
    let waypoints = [
        [0.0, 0.0, 0.0],
        [12.0, 0.0, 0.0],
        [12.0, 0.0, 9.0],
        [0.0, 0.0, 9.0],
    ];
    let speed = 3.0;
    let mut h = Harness::new();
    h.spawn(
        "p",
        waypoints[0],
        &Behavior::Patrol {
            waypoints: waypoints.to_vec(),
            speed,
        },
    );
    // Loop perimeter and derived tick bounds.
    let mut perim = 0.0;
    for i in 0..waypoints.len() {
        perim += dist(waypoints[i], waypoints[(i + 1) % waypoints.len()]);
    }
    let dt = kami::DEFAULT_DT;
    let ticks_per_loop = (perim / (speed * dt)).ceil() as u64;
    let step_len = speed * dt;

    let mut ctx = TickContext::new(1, 0);
    let mut nearest = vec![f64::INFINITY; waypoints.len()];
    let mut positions = Vec::new();
    for t in 0..(ticks_per_loop + 2) {
        ctx = ctx.at(t);
        h.step(&ctx);
        let p = h.pos("p");
        positions.push(p);
        for (i, w) in waypoints.iter().enumerate() {
            nearest[i] = nearest[i].min(dist(p, *w));
        }
    }
    // Every waypoint approached within one step over a single loop.
    for (i, n) in nearest.iter().enumerate() {
        assert!(
            *n <= step_len + 1e-6,
            "waypoint {i} nearest approach {n} > step {step_len}"
        );
    }
    // Loop closes: after a full period, position returns near the start.
    let closed = dist(positions[ticks_per_loop as usize], waypoints[0]);
    assert!(closed <= step_len + 1e-6, "loop did not close: {closed}");
    eprintln!(
        "ORDEAL patrol: perim={perim:.3}m, {ticks_per_loop} ticks/loop, step={step_len:.4}m, \
         nearest-approach per waypoint={nearest:?}, loop-close error={closed:.4}m"
    );
}

// --- ORDEAL 3: wander stays in bounds forever -----------------------------

#[test]
fn ordeal_wander_stays_in_bounds_10k_ticks() {
    let center = [4.0, 0.0, -3.0];
    let radius = 6.0;
    let mut h = Harness::new();
    h.spawn(
        "w",
        center,
        &Behavior::Wander {
            center,
            radius,
            speed: 5.0,
            retarget: 0.3,
        },
    );
    let mut ctx = TickContext::new(777, 0);
    let mut max_r: f64 = 0.0;
    for t in 0..10_000 {
        ctx = ctx.at(t);
        h.step(&ctx);
        let p = h.pos("w");
        let r = ((p[0] - center[0]).powi(2) + (p[2] - center[2]).powi(2)).sqrt();
        max_r = max_r.max(r);
        assert!(
            r <= radius + 1e-9,
            "escaped disc at tick {t}: r={r} > radius={radius}"
        );
    }
    eprintln!(
        "ORDEAL wander: 10000 ticks, radius={radius}, max observed r={max_r:.6} (never exceeded)"
    );
}

// --- ORDEAL 4: follow convergence, no overshoot ---------------------------

#[test]
fn ordeal_follow_converges_without_overshoot() {
    let speed = 4.0;
    let stop = 2.0;
    let mut h = Harness::new();
    h.spawn_target("prey", [0.0, 0.0, 0.0]);
    h.spawn(
        "hunter",
        [-30.0, 0.0, 0.0],
        &Behavior::Follow {
            target: "prey".into(),
            speed,
            stop_range: stop,
        },
    );
    let dt = kami::DEFAULT_DT;
    let prey_speed = 1.5; // slower than the hunter → catchable
    let mut ctx = TickContext::new(9, 0);
    let mut min_gap = f64::INFINITY;
    let mut converged_at = None;
    for t in 0..4000 {
        // Move the moving target first (it drifts +x).
        let prey = h.pos("prey");
        h.set_pos("prey", [prey[0] + prey_speed * dt, prey[1], prey[2]]);
        ctx = ctx.at(t);
        h.step(&ctx);
        let gap = dist(h.pos("hunter"), h.pos("prey"));
        // NEVER overshoots inside stop range.
        assert!(
            gap >= stop - 1e-9,
            "overshot into stop range at tick {t}: gap={gap}"
        );
        min_gap = min_gap.min(gap);
        if converged_at.is_none() && gap <= stop + 1e-6 {
            converged_at = Some(t);
        }
    }
    let converged_at = converged_at.expect("hunter never reached stop range");
    eprintln!(
        "ORDEAL follow: converged at tick {converged_at}, held gap≈stop={stop} thereafter, \
         min gap={min_gap:.6} (≥ stop, no overshoot)"
    );
}

// --- ORDEAL 5: look-at correctness ----------------------------------------

#[test]
fn ordeal_look_at_yaw_error_goes_to_zero() {
    let mut h = Harness::new();
    h.spawn_target("target", [10.0, 0.0, 10.0]);
    h.spawn(
        "watcher",
        [0.0, 0.0, 0.0],
        &Behavior::LookAt {
            target: "target".into(),
        },
    );
    // Transform vec3 columns are f32 in the ECS, so the round-tripped yaw and
    // target position carry f32 precision (~1e-6). Tolerance reflects storage.
    let tol = 1e-5;
    let mut ctx = TickContext::new(5, 0);
    let mut worst: f64 = 0.0;
    // Sweep the target around the watcher; yaw must track exactly each tick.
    for t in 0..360u64 {
        let ang = (t as f64).to_radians();
        let tx = 10.0 * ang.sin();
        let tz = 10.0 * ang.cos();
        h.set_pos("target", [tx, 0.0, tz]);
        ctx = ctx.at(t);
        h.step(&ctx);
        let want = (tx - 0.0f64).atan2(tz - 0.0);
        let got = h.yaw("watcher");
        let err = (got - want).abs();
        worst = worst.max(err);
        assert!(err <= tol, "yaw error {err} > tol {tol} at tick {t}");
    }
    eprintln!("ORDEAL look-at: 360 headings swept, worst yaw error={worst:.3e} (tol={tol:.0e})");
}

// ==========================================================================
// KAMI K1 — THE DECORATIVE SIX (reference-parity ordeals)
//
// Oracle values are computed FROM the exact reference JS formulas
//   shared/motion.js         (orbit, bob, path)
//   client/kernel/behaviors.js (spin, pulse, flicker)
// Each oracle constant below was produced by evaluating the reference JS
// formula (cited per case) in a f64 harness, then annotated with its hand
// check. The Rust eval transcribes the SAME formula; parity = agreement.
//
// f32-DERIVED TOLERANCE: decorative outputs ultimately land in the ECS's f32
// transform/light columns. f32 has a 24-bit mantissa → ulp(M) ≈ M·2⁻²³. The
// largest sampled magnitude is spin yaw ≈ 20.25 → ulp ≈ 20.25·2⁻²³ ≈ 2.4e-6.
// TOL = 1e-5 covers that storage floor (and dwarfs the ~1e-15 last-bit gap
// between Rust's f64 transcendentals and the JS f64 oracle). The pure eval is
// f64, so the OBSERVED error is ~1e-13; the gate is the f32 floor.
const TOL: f64 = 1e-5;

fn close(got: f64, want: f64, what: &str) {
    let err = (got - want).abs();
    assert!(
        err <= TOL,
        "{what}: got {got}, want {want}, err {err:e} > TOL {TOL:e}"
    );
}

// --- ORDEAL 6: spin parity — yaw(t) = bindYaw + speed*t --------------------
#[test]
fn ordeal_spin_formula_parity() {
    // speed=2, bindYaw=0.25. Reference behaviors.js is incremental; closed
    // form on the world clock is bindYaw + speed*t (see decorative module note).
    let b = Decorative::Spin { speed: 2.0 };
    let bind = BindPose {
        rotation: [0.0, 0.25, 0.0],
        ..BindPose::default()
    };
    // (t, oracle yaw): 0.25 + 2t
    let cases = [
        (0.0_f64, 0.250000000000000), // 0.25 + 0
        (0.5, 1.25),                  // 0.25 + 1.0
        (1.0, 2.25),                  // 0.25 + 2.0
        (PI, 6.533_185_307_179_586),  // 0.25 + 2π (phase-like boundary)
        (10.0, 20.25),                // 0.25 + 20
    ];
    let mut worst = 0.0f64;
    for (t, want) in cases {
        let got = b.eval(t, bind).rotation[1];
        worst = worst.max((got - want).abs());
        close(got, want, &format!("spin yaw t={t}"));
    }
    eprintln!("ORDEAL spin: 5 samples vs reference yaw=bindYaw+speed*t, worst err={worst:.3e}");
}

// --- ORDEAL 7: bob parity — y += sin(t*speed+phase)*amplitude --------------
#[test]
fn ordeal_bob_formula_parity() {
    let b = Decorative::Bob {
        speed: 2.0,
        phase: 0.5,
        amplitude: 1.5,
    };
    let bind = BindPose {
        position: [0.0, 3.0, 0.0],
        ..BindPose::default()
    };
    let cases = [
        (0.0_f64, 3.719138307906305),   // 3 + sin(0.5)*1.5
        (0.5, 4.496242479906082),       // 3 + sin(1.5)*1.5
        (FRAC_PI_2, 2.280861692093696), // t=π/2: 3 + sin(π+0.5)*1.5
        (3.0, 3.322679982131723),       // 3 + sin(6.5)*1.5
        (7.3, 3.856795304489983),
    ];
    let mut worst = 0.0f64;
    for (t, want) in cases {
        let s = b.eval(t, bind);
        worst = worst.max((s.position[1] - want).abs());
        close(s.position[1], want, &format!("bob y t={t}"));
        // untouched channels stay bind
        close(s.position[0], 0.0, "bob x");
        close(s.position[2], 0.0, "bob z");
    }
    eprintln!("ORDEAL bob: 5 samples vs reference y=baseY+sin(t·s+φ)·A, worst err={worst:.3e}");
}

// --- ORDEAL 8: orbit parity — circle in xz, y=cy+height --------------------
#[test]
fn ordeal_orbit_formula_parity() {
    let b = Decorative::Orbit {
        center: [1.0, 2.0, -3.0],
        radius: 8.0,
        speed: 0.5,
        phase: 0.2,
        height: 4.0,
    };
    let bind = BindPose::default();
    // (t, [x, y, z]) — y is constant cy+height = 2+4 = 6.
    let cases = [
        (0.0_f64, [8.840532622729933, 6.0, -1.410_645_353_639_51]), // angle=0.2
        (1.0, [7.118737498275908, 6.0, 2.153741497901528]),         // angle=0.7
        (PI, [-0.589354646360489, 6.0, 4.840532622729933]),         // t=π
        (TAU, [-6.840532622729933, 6.0, -4.589354646360491]),       // t=2π
        (12.5, [8.888_949_310_340_1, 6.0, -1.671663153080342]),
    ];
    let mut worst = 0.0f64;
    for (t, want) in cases {
        let p = b.eval(t, bind).position;
        for (k, &wv) in want.iter().enumerate() {
            worst = worst.max((p[k] - wv).abs());
            close(p[k], wv, &format!("orbit axis {k} t={t}"));
        }
    }
    eprintln!(
        "ORDEAL orbit: 5 samples (incl t=0,π,2π) vs cx+cos·r / cz+sin·r, worst err={worst:.3e}"
    );
}

// --- ORDEAL 9: path parity — dwell/travel walk + facing --------------------
#[test]
fn ordeal_path_formula_parity() {
    // points=[[0,0,0,1],[10,0,0],[10,0,8,0.5],[0,0,8]] speed=4 loop → total=8.5s.
    // legs (dwell,travel): (1,2.5)(0,2)(0.5,2.5) from motion.js pathLegs.
    let b = Decorative::Path {
        points: vec![
            vec![0.0, 0.0, 0.0, 1.0],
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 8.0, 0.5],
            vec![0.0, 0.0, 8.0],
        ],
        speed: 4.0,
        start: 0.0,
        phase: 0.0,
        loop_: true,
    };
    let bind = BindPose::default();
    // (t, pos, yaw)
    let cases = [
        (0.0_f64, [0.0, 0.0, 0.0], 0.0), // parked (dwell), no move → bind yaw 0
        (0.5, [0.0, 0.0, 0.0], 0.0),     // still in the 1s dwell
        (1.0, [0.0, 0.0, 0.0], FRAC_PI_2), // dwell end (boundary): about to head +x → yaw π/2
        (3.5, [10.0, 0.0, 0.0], 0.0),    // reached wp1, now heading +z
        (6.5, [8.0, 0.0, 8.0], -FRAC_PI_2), // on leg2 heading -x → yaw -π/2
        (10.0, [2.0, 0.0, 0.0], FRAC_PI_2), // wrapped (t%8.5=1.5), on leg0 heading +x
    ];
    let mut worst = 0.0f64;
    for (t, wp, wyaw) in cases {
        let s = b.eval(t, bind);
        for (k, &wv) in wp.iter().enumerate() {
            worst = worst.max((s.position[k] - wv).abs());
            close(s.position[k], wv, &format!("path axis {k} t={t}"));
        }
        close(s.rotation[1], wyaw, &format!("path yaw t={t}"));
    }
    eprintln!("ORDEAL path: 6 samples (dwell, boundary t=1, wrap) vs motion.js walk, worst err={worst:.3e}");
}

// --- ORDEAL 10: pulse parity — scale = base.scale * (1+sin(t·s)·A) ---------
#[test]
fn ordeal_pulse_formula_parity() {
    let b = Decorative::Pulse {
        speed: 3.0,
        amount: 0.2,
    };
    let bind = BindPose {
        scale: [1.0, 1.0, 1.0],
        ..BindPose::default()
    };
    // (t, k) — scale is k on every axis (base.scale = 1).
    let cases = [
        (0.0_f64, 1.000000000000000), // 1 + sin(0)*0.2
        (0.5235987755982988, 1.2),    // t=π/6: sin(π/2)=1 → 1.2 (peak boundary)
        (1.0, 1.028224001611973),     // 1 + sin(3)*0.2
        (2.0, 0.944116900360215),     // 1 + sin(6)*0.2
        (5.5, 0.857642931526175),
    ];
    let mut worst = 0.0f64;
    for (t, k) in cases {
        let s = b.eval(t, bind);
        for a in 0..3 {
            worst = worst.max((s.scale[a] - k).abs());
            close(s.scale[a], k, &format!("pulse scale axis {a} t={t}"));
        }
        // REFERENCE FINDING: pulse drives SCALE (a float), never emissive.
        assert!(
            s.emissive.is_none(),
            "pulse must not write emissive (reference drives scale)"
        );
    }
    eprintln!("ORDEAL pulse: 5 samples vs 1+sin(t·s)·A (incl peak t=π/6), worst err={worst:.3e}");
}

// --- ORDEAL 11: flicker parity — intensity = baseI*(1+noise·A) -------------
#[test]
fn ordeal_flicker_formula_parity() {
    let b = Decorative::Flicker { amount: 0.3 };
    let bind = BindPose {
        intensity: 1.0,
        ..BindPose::default()
    };
    // noise = sin(t*31)*0.5 + sin(t*47+1.3)*0.5 ; intensity = 1*(1+noise*0.3)
    let cases = [
        (0.0_f64, 1.144533727812579), // 1 + (sin0*.5 + sin(1.3)*.5)*0.3
        (0.1, 0.964324774635155),
        (0.5, 0.981974853374961),
        (1.0, 0.800926758442087),
        (3.33, 1.164309301247115),
    ];
    let mut worst = 0.0f64;
    for (t, want) in cases {
        let s = b.eval(t, bind);
        worst = worst.max((s.intensity - want).abs());
        close(s.intensity, want, &format!("flicker intensity t={t}"));
        // REFERENCE FINDING: flicker drives LIGHT INTENSITY (a float), never emissive.
        assert!(
            s.emissive.is_none(),
            "flicker must not write emissive (reference drives intensity)"
        );
    }
    eprintln!(
        "ORDEAL flicker: 5 samples vs 1+noise·A (two incommensurate sines), worst err={worst:.3e}"
    );
}

// --- ORDEAL 12: determinism — same t → byte-identical ops across runs ------
#[test]
fn ordeal_decorative_determinism_byte_identical() {
    let six: Vec<(&str, Decorative)> = vec![
        ("spinner", Decorative::Spin { speed: 2.0 }),
        (
            "bobber",
            Decorative::Bob {
                speed: 2.0,
                phase: 0.5,
                amplitude: 1.5,
            },
        ),
        (
            "orbiter",
            Decorative::Orbit {
                center: [1.0, 2.0, -3.0],
                radius: 8.0,
                speed: 0.5,
                phase: 0.2,
                height: 4.0,
            },
        ),
        (
            "walker",
            Decorative::Path {
                points: vec![
                    vec![0.0, 0.0, 0.0, 1.0],
                    vec![10.0, 0.0, 0.0],
                    vec![10.0, 0.0, 8.0, 0.5],
                    vec![0.0, 0.0, 8.0],
                ],
                speed: 4.0,
                start: 0.0,
                phase: 0.0,
                loop_: true,
            },
        ),
        (
            "pulser",
            Decorative::Pulse {
                speed: 3.0,
                amount: 0.2,
            },
        ),
        ("flame", Decorative::Flicker { amount: 0.25 }),
    ];
    let bind = BindPose::default();
    let run = || {
        let mut out = Vec::new();
        for k in 0..500u64 {
            let t = k as f64 * 0.02;
            for (id, b) in &six {
                let ops = b.ops(id, t, bind);
                out.extend_from_slice(serde_json::to_vec(&ops).unwrap().as_slice());
                out.push(b'\n');
            }
        }
        out
    };
    let a = run();
    let c = run();
    assert_eq!(a.len(), c.len(), "op-stream length");
    assert_eq!(a, c, "decorative op streams must be byte-identical");
    eprintln!(
        "ORDEAL decorative determinism: 6 kinds × 500 t, {} bytes, byte-identical",
        a.len()
    );
}

// --- ORDEAL 13: EMISSIVE = COLOR STRING law (never bool/float) -------------
#[test]
fn ordeal_emissive_is_always_a_color_string() {
    // The six never write emissive (pulse→scale, flicker→intensity), so the
    // law holds vacuously: to_ops must NEVER emit an emissive/material op with
    // a non-string value. We scan every op of every kind across many t and
    // assert any material.emissive present is a valid color string.
    let six: Vec<Decorative> = vec![
        Decorative::Spin { speed: 1.5 },
        Decorative::Bob {
            speed: 1.0,
            phase: 0.0,
            amplitude: 0.5,
        },
        Decorative::Orbit {
            center: [0.0, 0.0, 0.0],
            radius: 10.0,
            speed: 0.3,
            phase: 0.0,
            height: 0.0,
        },
        Decorative::Path {
            points: vec![vec![0.0, 0.0, 0.0], vec![5.0, 0.0, 5.0]],
            speed: 2.0,
            start: 0.0,
            phase: 0.0,
            loop_: false,
        },
        Decorative::Pulse {
            speed: 2.0,
            amount: 0.08,
        },
        Decorative::Flicker { amount: 0.25 },
    ];
    let bind = BindPose::default();
    let mut emissive_ops = 0usize;
    for k in 0..300u64 {
        let t = k as f64 * 0.03;
        for b in &six {
            let s = b.eval(t, bind);
            // Sample-level: emissive is None for all six.
            assert!(s.emissive.is_none(), "no decorative kind may emit emissive");
            for op in s.to_ops("e") {
                let crystal::Op::Set(set) = op else {
                    continue;
                };
                if set.component == "material" {
                    if let Some(em) = set.value.get("emissive") {
                        emissive_ops += 1;
                        assert!(
                            is_color_string(em),
                            "emissive must be a color string, got {em}"
                        );
                    }
                }
            }
        }
    }
    // Sanity of the guard itself: bools/floats are rejected, colors accepted.
    assert!(!is_color_string(&json!(true)));
    assert!(!is_color_string(&json!(0.5)));
    assert!(is_color_string(&json!("#ffc46b")));
    assert!(is_color_string(&json!("red")));
    eprintln!(
        "ORDEAL emissive-law: 6 kinds × 300 t, {emissive_ops} emissive ops emitted (0 expected: \
         pulse→scale, flicker→intensity); guard rejects bool/float, accepts #hex/named"
    );
}
