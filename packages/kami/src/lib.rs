//! # KAMI — the animating presence of a vessel
//!
//! Data-driven behavior ticks over the [`crystal`] ECS. A `behavior`
//! component names a motion policy; the tick is a pure `(world, ctx) -> ops`
//! function that emits transform `Set` ops. Nothing mutates the world in
//! place — the caller applies the ops (Flow of Data). Same
//! `(seed, entropy, world)` → byte-identical op stream (Ultradeterminism,
//! `gaia-dreamforge/ENTROPY.md`).
//!
//! ## Reference-engine findings (telegraphic)
//!
//! Grepped the worktree's `server/` + `client/` + `shared/`:
//!
//! - `shared/schema.js` §`behavior` → reference `behavior` = DECORATIVE
//!   deterministic kinematics on the WORLD CLOCK, NOT AI. kinds:
//!   `spin|bob|orbit|path|pulse|flicker`. fields: `speed amplitude phase
//!   center radius height points start loop amount`. default
//!   `{type:'spin',speed:1}`. array-or-single (`OneOrMany`).
//! - `shared/motion.js` → `animatedPosition(comps,time,heightFn)`: pure fn of
//!   world time. `orbit` = circle(center,radius,speed,phase). `path` =
//!   constant-speed waypoint walk, 4th number per point = dwell seconds, `loop`
//!   closes vs parks, `start` anchors ($now in trigger ops), time-parameterized
//!   so dwell+travel share one clock. `bob` = sin on y. legs cached per
//!   behavior object (WeakMap; ops replace the object wholesale → invalidates).
//! - `server/sense.js` → senses READ `behaviorList(comps)`, never tick it;
//!   motion is client-computed from the shared clock. No server behavior
//!   daemon exists — "behavior" is not an AI system in the reference engine.
//! - No `daemon` component in schema; daemons = external processes over
//!   HTTP/WS ops (AGENTS.md), not world data.
//!
//! CONCLUSION: the reference `behavior` is decorative motion, not the
//! patrol/wander/follow/look-at AI this crate needs. So KAMI defines a CLEAN
//! MINIMAL schema (documented as such, below) for data-driven AI ticks.
//! Overlap: KAMI [`Behavior::Patrol`] ≡ reference `path` (waypoint loop +
//! speed) but stateless closed-form on the entropy clock.
//!
//! ## KAMI schema (this crate — clean minimal, NOT the reference `behavior`)
//!
//! A `behavior` component (buffer = raw JSON) is one tagged object, `kind` ∈:
//! - `patrol`  — waypoint loop at constant speed (closed-form on the clock).
//! - `wander`  — bounded disc, hash-driven headings (`hash(seed,entropy,id)`).
//! - `follow`  — chase a target id at speed, stop at `stop_range` (never
//!   overshoots inside it).
//! - `look_at` — face a target id (yaw set exactly → error 0).
//!
//! `transform` = `position: vec3`, `rotation: vec3` (euler; facing = `rotation[1]`
//! yaw radians — matches the reference transform's euler rotation array).

use crystal::{ComponentId, EcsWorld, Entity, Op, QuerySpec, SetOp};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

mod cat;
pub use cat::{CatDrive, CatMind, CatState};

mod decorative;
pub use decorative::{is_color_string, BindPose, Decorative, Sample};

mod entropy;
pub use entropy::hash;

/// Fixed tick delta (seconds). Timing is `entropy * dt` — the entropy value is
/// the temporal x-axis (`ENTROPY.md`), never a wall clock.
pub const DEFAULT_DT: f64 = 0.02;

/// Component-registration handles for a KAMI-ready ECS world.
#[derive(Clone, Copy, Debug)]
pub struct Registry {
    pub transform: ComponentId,
    pub behavior: ComponentId,
}

impl Registry {
    /// Register `transform` (position/rotation vec3) and `behavior` (buffer).
    /// Idempotent-friendly: reuses ids if the names already exist.
    pub fn register(world: &mut EcsWorld) -> Self {
        let transform = world.component_id("transform").unwrap_or_else(|| {
            world
                .register_component_json(
                    r#"{"name":"transform","fields":{"position":"vec3","rotation":"vec3"}}"#,
                )
                .expect("register transform")
        });
        let behavior = world.component_id("behavior").unwrap_or_else(|| {
            world
                .register_component_json(r#"{"name":"behavior","buffer":true}"#)
                .expect("register behavior")
        });
        Self {
            transform,
            behavior,
        }
    }
}

/// The tick's read-only clock inputs. `entropy` is the tick index (x-axis).
#[derive(Clone, Copy, Debug)]
pub struct TickContext {
    pub seed: u64,
    pub entropy: u64,
    pub dt: f64,
}

impl TickContext {
    pub fn new(seed: u64, entropy: u64) -> Self {
        Self {
            seed,
            entropy,
            dt: DEFAULT_DT,
        }
    }
    pub fn at(self, entropy: u64) -> Self {
        Self { entropy, ..self }
    }
}

/// A KAMI behavior policy. Tagged by `kind`; every param has a default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Behavior {
    /// Loop through `waypoints` at `speed` m/s (closed-form on the clock).
    Patrol {
        #[serde(default)]
        waypoints: Vec<[f64; 3]>,
        #[serde(default = "default_speed")]
        speed: f64,
    },
    /// Roam a disc `center`/`radius`; heading rerolls every `retarget` seconds
    /// from `hash(seed, bucket, entity)`. Clamped to the disc → bounded forever.
    Wander {
        #[serde(default)]
        center: [f64; 3],
        #[serde(default = "default_radius")]
        radius: f64,
        #[serde(default = "default_speed")]
        speed: f64,
        #[serde(default = "default_retarget")]
        retarget: f64,
    },
    /// Chase entity `target` (gaia id) at `speed`, halting at `stop_range`.
    Follow {
        target: String,
        #[serde(default = "default_speed")]
        speed: f64,
        #[serde(default = "default_stop_range")]
        stop_range: f64,
    },
    /// Face entity `target` (gaia id): set yaw exactly toward it.
    LookAt { target: String },
}

fn default_speed() -> f64 {
    2.0
}
fn default_radius() -> f64 {
    10.0
}
fn default_retarget() -> f64 {
    1.5
}
fn default_stop_range() -> f64 {
    1.5
}

/// Tick every entity carrying a DECORATIVE `behavior` (`{"type": ...}` — the
/// [`Decorative`] six) + `transform`. Pure: reads the ECS, evaluates each kind
/// at world time `entropy * dt` against its cached rest pose in `binds`, and
/// returns ABSOLUTE `transform` `Set` ops (position/rotation/scale) in a
/// deterministic (entity-sorted) order. Never mutates `world` — the caller
/// applies the ops (Flow of Data), so re-reading the transform next tick never
/// compounds (the bind pose is the fixed origin, not the live transform).
///
/// `binds` is keyed by gaia id; a missing entry falls back to
/// [`BindPose::default`]. Entities whose `behavior` is not a decorative kind
/// (e.g. the AI [`Behavior`] schema, tag `kind`) are skipped here — [`tick`]
/// drives those.
pub fn tick_decorative(
    world: &EcsWorld,
    reg: Registry,
    binds: &std::collections::BTreeMap<String, BindPose>,
    ctx: &TickContext,
) -> Vec<Op> {
    let mut entities = world.query(&QuerySpec {
        all: vec![reg.behavior, reg.transform],
        ..Default::default()
    });
    entities.sort_by_key(|e| (e.index, e.generation));
    let t = ctx.entropy as f64 * ctx.dt;

    let mut ops = Vec::new();
    for entity in entities {
        let Some(id) = world.gaia_id_for(entity) else {
            continue;
        };
        let id = id.to_string();
        let Ok(raw) = world.get_component(entity, reg.behavior) else {
            continue;
        };
        let Ok(behavior) = serde_json::from_value::<Decorative>(raw) else {
            continue; // not a decorative kind — the AI `tick` handles it
        };
        let bind = binds.get(&id).copied().unwrap_or_default();
        let sample = behavior.eval(t, bind);
        ops.push(Op::Set(SetOp {
            id,
            component: "transform".into(),
            value: json!({
                "position": sample.position,
                "rotation": sample.rotation,
                "scale": sample.scale,
            }),
            extra: Default::default(),
        }));
    }
    ops
}

/// Tick every entity carrying a `behavior` + `transform`. Pure: reads the ECS,
/// returns transform `Set` ops in a deterministic (entity-sorted) order.
/// Never mutates `world`.
pub fn tick(world: &EcsWorld, reg: Registry, ctx: &TickContext) -> Vec<Op> {
    let mut entities = world.query(&QuerySpec {
        all: vec![reg.behavior, reg.transform],
        ..Default::default()
    });
    // Stable order: query walks archetypes in HashMap order (nondeterministic);
    // sort by (index, generation) so the op stream is byte-identical per run.
    entities.sort_by_key(|e| (e.index, e.generation));

    let mut ops = Vec::new();
    for entity in entities {
        let Ok(raw) = world.get_component(entity, reg.behavior) else {
            continue;
        };
        let Ok(behavior) = serde_json::from_value::<Behavior>(raw) else {
            continue;
        };
        if let Some(op) = tick_one(world, reg, ctx, entity, &behavior) {
            ops.push(op);
        }
    }
    ops
}

fn tick_one(
    world: &EcsWorld,
    reg: Registry,
    ctx: &TickContext,
    entity: Entity,
    behavior: &Behavior,
) -> Option<Op> {
    let id = world.gaia_id_for(entity)?.to_string();
    let pos = read_vec3(world, entity, reg.transform, "position")?;

    let (new_pos, yaw) = match behavior {
        Behavior::Patrol { waypoints, speed } => patrol(waypoints, *speed, ctx, pos)?,
        Behavior::Wander {
            center,
            radius,
            speed,
            retarget,
        } => wander(*center, *radius, *speed, *retarget, ctx, entity, pos),
        Behavior::Follow {
            target,
            speed,
            stop_range,
        } => {
            let tpos = target_pos(world, reg, target)?;
            follow(pos, tpos, *speed, *stop_range, ctx)
        }
        Behavior::LookAt { target } => {
            let tpos = target_pos(world, reg, target)?;
            (pos, yaw_to(pos, tpos))
        }
    };

    Some(Op::Set(SetOp {
        id,
        component: "transform".into(),
        value: json!({ "position": new_pos, "rotation": [0.0, yaw, 0.0] }),
        extra: Default::default(),
    }))
}

// --- policies -------------------------------------------------------------

/// Closed-form arc-length walk of the closed waypoint loop at `entropy*dt`.
fn patrol(
    waypoints: &[[f64; 3]],
    speed: f64,
    ctx: &TickContext,
    pos: [f64; 3],
) -> Option<([f64; 3], f64)> {
    if waypoints.is_empty() {
        return None;
    }
    if waypoints.len() == 1 {
        return Some((waypoints[0], 0.0));
    }
    // Segment lengths around the closed loop W[i] -> W[(i+1)%n].
    let n = waypoints.len();
    let mut lengths = Vec::with_capacity(n);
    let mut total = 0.0;
    for i in 0..n {
        let l = dist(waypoints[i], waypoints[(i + 1) % n]);
        lengths.push(l);
        total += l;
    }
    if total <= 0.0 {
        return Some((waypoints[0], 0.0));
    }
    let elapsed = ctx.entropy as f64 * ctx.dt;
    let mut d = (speed * elapsed) % total;
    if d < 0.0 {
        d += total;
    }
    let mut i = 0;
    while i < n && d > lengths[i] {
        d -= lengths[i];
        i += 1;
    }
    let a = waypoints[i % n];
    let b = waypoints[(i + 1) % n];
    let seg = lengths[i % n];
    let t = if seg > 0.0 { d / seg } else { 0.0 };
    let p = [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ];
    let _ = pos;
    Some((p, yaw_to(a, b)))
}

/// Bounded roam: heading from `hash(seed, bucket, entity)`, clamped to the disc.
fn wander(
    center: [f64; 3],
    radius: f64,
    speed: f64,
    retarget: f64,
    ctx: &TickContext,
    entity: Entity,
    pos: [f64; 3],
) -> ([f64; 3], f64) {
    let dt_secs = retarget.max(ctx.dt);
    let bucket = ((ctx.entropy as f64 * ctx.dt) / dt_secs) as u64;
    let h = hash(ctx.seed, bucket, entity.index as u64);
    // Map the low 53 bits to a heading in [0, 2π).
    let unit = (h >> 11) as f64 / (1u64 << 53) as f64;
    let heading = unit * std::f64::consts::TAU;
    let step = speed * ctx.dt;
    let mut p = [
        pos[0] + heading.cos() * step,
        pos[1],
        pos[2] + heading.sin() * step,
    ];
    // Clamp into the disc (xz) — provably bounded for all time.
    let dx = p[0] - center[0];
    let dz = p[2] - center[2];
    let r = (dx * dx + dz * dz).sqrt();
    if r > radius {
        // Margin absorbs the ECS's f32 storage rounding so the value that lands
        // back in the world is provably ≤ radius (never escapes on round-trip).
        let k = radius * (1.0 - 1e-6) / r;
        p[0] = center[0] + dx * k;
        p[2] = center[2] + dz * k;
    }
    (p, heading)
}

/// Move toward `target`, stopping so distance stays ≥ `stop_range` (no overshoot).
fn follow(
    pos: [f64; 3],
    target: [f64; 3],
    speed: f64,
    stop_range: f64,
    ctx: &TickContext,
) -> ([f64; 3], f64) {
    let d = dist(pos, target);
    let yaw = yaw_to(pos, target);
    if d <= stop_range || d <= 0.0 {
        return (pos, yaw);
    }
    let travel = (speed * ctx.dt).min(d - stop_range);
    let k = travel / d;
    let p = [
        pos[0] + (target[0] - pos[0]) * k,
        pos[1] + (target[1] - pos[1]) * k,
        pos[2] + (target[2] - pos[2]) * k,
    ];
    (p, yaw)
}

// --- helpers --------------------------------------------------------------

fn target_pos(world: &EcsWorld, reg: Registry, target: &str) -> Option<[f64; 3]> {
    let entity = world.entity_for_gaia(target)?;
    read_vec3(world, entity, reg.transform, "position")
}

fn read_vec3(world: &EcsWorld, entity: Entity, comp: ComponentId, field: &str) -> Option<[f64; 3]> {
    let value = world.get_component_field(entity, comp, field).ok()?;
    let array = value.as_array()?;
    Some([
        array.first().and_then(Value::as_f64).unwrap_or(0.0),
        array.get(1).and_then(Value::as_f64).unwrap_or(0.0),
        array.get(2).and_then(Value::as_f64).unwrap_or(0.0),
    ])
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Yaw (radians) facing from `a` toward `b` in the xz plane.
fn yaw_to(a: [f64; 3], b: [f64; 3]) -> f64 {
    (b[0] - a[0]).atan2(b[2] - a[2])
}
