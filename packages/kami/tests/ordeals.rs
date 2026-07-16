//! KAMI K0 ordeals: determinism, patrol adherence, wander bounds, follow
//! convergence, look-at correctness. Each drives the REAL tick over a real
//! crystal ECS, applying the emitted ops between ticks (Flow of Data).

use crystal::{EcsWorld, Op};
use kami::{tick, Behavior, Registry, TickContext};
use serde_json::{json, Value};

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
