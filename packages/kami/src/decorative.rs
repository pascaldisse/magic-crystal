//! # The Decorative Six — reference-faithful deterministic kinematics
//!
//! DreamForge parity atom (KAMI K1). The reference engine's `behavior`
//! component is DECORATIVE motion on the world clock, NOT the patrol/wander AI
//! of [`crate::Behavior`] (K0). This module transcribes the reference's six
//! kinds — `spin · bob · orbit · path · pulse · flicker` — as PURE functions
//! of `(params, world-time t, bind pose)`, formula-for-formula.
//!
//! ## Where the reference formulas actually live (grepped in the worktree)
//!
//! - `shared/motion.js` → `animatedPosition(comps, time, heightFn)` holds
//!   ONLY `orbit`, `bob`, `path` (the position kinds).
//! - `client/kernel/behaviors.js` → `Behaviors.run(b, id, group, dt, t)` holds
//!   `spin` (rotation), `pulse` (scale), `flicker` (light intensity).
//! - `shared/schema.js` §`behavior` → fields/enum/defaults (cited per param).
//!
//! ## ⚠ Honest reference findings (float-semantics / channel differences)
//!
//! - **spin is frame-INCREMENTAL in the reference**:
//!   `group.rotation.y += (b.speed ?? 1) * dt`. That is an accumulator, not a
//!   closed form of world time — it depends on frame cadence and the group's
//!   current rotation. Integrating the reference's own rate over `[0, t]` from
//!   the bind yaw gives the deterministic closed form on the world clock —
//!   the SAME shape every other decorative kind already uses:
//!   `yaw(t) = bind_yaw + speed * t`. This module emits that closed form and
//!   documents the divergence here.
//! - **pulse drives SCALE, flicker drives LIGHT INTENSITY** — NOT emissive.
//!   The reference `pulse` sets `group.scale = base.scale * k`; `flicker` sets
//!   `light.intensity = baseIntensity * mul`. Neither ever writes an
//!   `emissive` field. So the `emissive` channel is never touched by any of
//!   the six, and the GAIA "EMISSIVE = COLOR STRING" law holds vacuously
//!   (see [`Sample::emissive`], always `None`). Documented, not fabricated:
//!   inventing an emissive color mapping would break formula parity.
//! - **path facing** matches `behaviors.js`: yaw from the xz displacement
//!   between `pos(t)` and `pos(t + 0.4)`, applied only when
//!   `dx² + dz² > 0.0004`, else the bind yaw is kept.

use crystal::{Op, SetOp};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// The entity's authored rest transform — `group.userData.base` in
/// `client/kernel/view.js:453` (`{ position, rotation, scale }`) plus the base
/// light intensity (`light.userData.baseIntensity`, `view.js:550`). Every
/// decorative kind is evaluated RELATIVE to this pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BindPose {
    pub position: [f64; 3],
    pub rotation: [f64; 3],
    pub scale: [f64; 3],
    pub intensity: f64,
}

impl Default for BindPose {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            intensity: 1.0,
        }
    }
}

/// The evaluated result of one decorative behavior at a world time. A full
/// pose (bind on every channel a given kind does not drive) plus the light
/// intensity and the (always-`None`) emissive color — see the module note.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub position: [f64; 3],
    pub rotation: [f64; 3],
    pub scale: [f64; 3],
    pub intensity: f64,
    /// Reference `pulse`/`flicker` never write emissive → always `None`.
    /// If a future kind DID drive emissive it MUST be a color string
    /// (`is_color_string`), never a bool/float (EMISSIVE = COLOR STRING law).
    pub emissive: Option<String>,
}

impl Sample {
    fn from_bind(bind: BindPose) -> Self {
        Self {
            position: bind.position,
            rotation: bind.rotation,
            scale: bind.scale,
            intensity: bind.intensity,
            emissive: None,
        }
    }

    /// Deterministic op stream for this sample: a `transform` set (position,
    /// rotation, scale), a `light` set (intensity), and — only if a kind ever
    /// drives it — an `emissive` set (color string). Pure, entity-agnostic;
    /// same `Sample` → byte-identical ops.
    pub fn to_ops(&self, id: &str) -> Vec<Op> {
        let mut ops = vec![
            Op::Set(SetOp {
                id: id.to_string(),
                component: "transform".into(),
                value: json!({
                    "position": self.position,
                    "rotation": self.rotation,
                    "scale": self.scale,
                }),
                extra: Default::default(),
            }),
            Op::Set(SetOp {
                id: id.to_string(),
                component: "light".into(),
                value: json!({ "intensity": self.intensity }),
                extra: Default::default(),
            }),
        ];
        if let Some(color) = &self.emissive {
            ops.push(Op::Set(SetOp {
                id: id.to_string(),
                component: "material".into(),
                value: json!({ "emissive": color }),
                extra: Default::default(),
            }));
        }
        ops
    }
}

/// One decorative behavior. Tagged by `type` (matching the reference JSON:
/// `{ type: 'spin', ... }`); every param defaults to the reference default,
/// cited per field below.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Decorative {
    /// Rotate about local +y. Reference: `behaviors.js` `rotation.y += speed*dt`
    /// (incremental); closed form here `yaw = bind_yaw + speed*t`.
    Spin {
        /// `b.speed ?? 1` (behaviors.js).
        #[serde(default = "spin_speed")]
        speed: f64,
    },
    /// Sinusoidal y offset. Reference `motion.js`:
    /// `y += sin(t*speed + phase) * amplitude`.
    Bob {
        /// `b.speed ?? 1` (motion.js).
        #[serde(default = "bob_speed")]
        speed: f64,
        /// `b.phase ?? 0` (motion.js).
        #[serde(default)]
        phase: f64,
        /// `b.amplitude ?? 0.5` (motion.js).
        #[serde(default = "bob_amplitude")]
        amplitude: f64,
    },
    /// Circle in xz about `center`. Reference `motion.js`:
    /// `angle = t*speed + phase; x = cx + cos·r; z = cz + sin·r; y = cy + height`
    /// (the no-ground branch — this atom carries no terrain height function).
    Orbit {
        /// `b.center ?? [0,0,0]` (motion.js).
        #[serde(default)]
        center: [f64; 3],
        /// `b.radius ?? 10` (motion.js).
        #[serde(default = "orbit_radius")]
        radius: f64,
        /// `b.speed ?? 0.3` (motion.js — note: orbit's default differs from bob).
        #[serde(default = "orbit_speed")]
        speed: f64,
        /// `b.phase ?? 0` (motion.js).
        #[serde(default)]
        phase: f64,
        /// `b.height ?? 0`, added to `center.y` (motion.js).
        #[serde(default)]
        height: f64,
    },
    /// Constant-speed waypoint walk. Reference `motion.js` `pathLegs` + walk:
    /// each point is `[x, y, z, dwell?]`; `dwell` = seconds parked at the
    /// point's START; `loop` closes vs parks at the end; `start` anchors the
    /// world-time origin; `phase` is seconds along the cycle. Facing = yaw of
    /// `pos(t+0.4) - pos(t)` when moved > 0.02 m, else bind yaw (behaviors.js).
    Path {
        /// `b.points ?? []` — `[[x,y,z,dwell?], …]`. < 2 points ⇒ bind pose.
        #[serde(default)]
        points: Vec<Vec<f64>>,
        /// `b.speed ?? 2` (motion.js `pathLegs`), floored at 0.01.
        #[serde(default = "path_speed")]
        speed: f64,
        /// `b.start ?? 0` (motion.js).
        #[serde(default)]
        start: f64,
        /// `b.phase ?? 0` (motion.js).
        #[serde(default)]
        phase: f64,
        /// `b.loop` (falsy ⇒ park at last point).
        #[serde(default, rename = "loop")]
        loop_: bool,
    },
    /// Breathe the bind scale. Reference `behaviors.js`:
    /// `k = 1 + sin(t*speed) * amount; scale = base.scale * k`.
    Pulse {
        /// `b.speed ?? 2` (behaviors.js).
        #[serde(default = "pulse_speed")]
        speed: f64,
        /// `b.amount ?? 0.08` (behaviors.js).
        #[serde(default = "pulse_amount")]
        amount: f64,
    },
    /// Candle-flicker the bind light intensity. Reference `behaviors.js`:
    /// `noise = sin(t*31)*0.5 + sin(t*47 + 1.3)*0.5;`
    /// `intensity = baseIntensity * (1 + noise*amount)`.
    Flicker {
        /// `b.amount ?? 0.25` (behaviors.js).
        #[serde(default = "flicker_amount")]
        amount: f64,
    },
}

fn spin_speed() -> f64 {
    1.0
}
fn bob_speed() -> f64 {
    1.0
}
fn bob_amplitude() -> f64 {
    0.5
}
fn orbit_radius() -> f64 {
    10.0
}
fn orbit_speed() -> f64 {
    0.3
}
fn path_speed() -> f64 {
    2.0
}
fn pulse_speed() -> f64 {
    2.0
}
fn pulse_amount() -> f64 {
    0.08
}
fn flicker_amount() -> f64 {
    0.25
}

impl Decorative {
    /// Evaluate this behavior at world time `t` against `bind`. PURE:
    /// no world state, no RNG, no wall clock — `(self, t, bind)` fully
    /// determines the [`Sample`]. Same inputs → identical output, always.
    pub fn eval(&self, t: f64, bind: BindPose) -> Sample {
        let mut s = Sample::from_bind(bind);
        match self {
            // yaw(t) = bind_yaw + speed*t  (closed form; see module note)
            Decorative::Spin { speed } => {
                s.rotation[1] = bind.rotation[1] + speed * t;
            }
            // y += sin(t*speed + phase) * amplitude
            Decorative::Bob {
                speed,
                phase,
                amplitude,
            } => {
                s.position[1] = bind.position[1] + (t * speed + phase).sin() * amplitude;
            }
            // angle = t*speed + phase; x = cx + cos·r; z = cz + sin·r; y = cy + height
            Decorative::Orbit {
                center,
                radius,
                speed,
                phase,
                height,
            } => {
                let angle = t * speed + phase;
                s.position[0] = center[0] + angle.cos() * radius;
                s.position[2] = center[2] + angle.sin() * radius;
                s.position[1] = center[1] + height;
            }
            Decorative::Path {
                points,
                speed,
                start,
                phase,
                loop_,
            } => {
                if let Some(p) = path_pos(points, *speed, *start, *phase, *loop_, t) {
                    s.position = p;
                    // facing: displacement over the next 0.4 s (behaviors.js)
                    if let Some(n) = path_pos(points, *speed, *start, *phase, *loop_, t + 0.4) {
                        let dx = n[0] - p[0];
                        let dz = n[2] - p[2];
                        if dx * dx + dz * dz > 0.0004 {
                            s.rotation[1] = dx.atan2(dz);
                        }
                    }
                }
                // < 2 points ⇒ Sample stays the bind pose (reference: no-op).
            }
            // k = 1 + sin(t*speed)*amount; scale = base.scale * k
            Decorative::Pulse { speed, amount } => {
                let k = 1.0 + (t * speed).sin() * amount;
                s.scale = [bind.scale[0] * k, bind.scale[1] * k, bind.scale[2] * k];
            }
            // noise = sin(t*31)*0.5 + sin(t*47+1.3)*0.5; I = baseI*(1 + noise*amount)
            Decorative::Flicker { amount } => {
                let noise = (t * 31.0).sin() * 0.5 + (t * 47.0 + 1.3).sin() * 0.5;
                s.intensity = bind.intensity * (1.0 + noise * amount);
            }
        }
        s
    }

    /// Evaluate and lower to ops in one call (bound to gaia id `id`).
    pub fn ops(&self, id: &str, t: f64, bind: BindPose) -> Vec<Op> {
        self.eval(t, bind).to_ops(id)
    }
}

/// Precomputed per-leg schedule of a path: `(dwell, travel)` seconds for each
/// segment `points[i] -> points[i+1]`, plus the loop `total`. Transcribes
/// `motion.js` `pathLegs`.
fn path_legs(points: &[Vec<f64>], speed: f64) -> (Vec<(f64, f64)>, f64) {
    let mut legs = Vec::with_capacity(points.len().saturating_sub(1));
    let mut total = 0.0;
    for i in 1..points.len() {
        let dwell = points[i - 1].get(3).copied().unwrap_or(0.0);
        let d = {
            let a = &points[i - 1];
            let b = &points[i];
            ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
        };
        let travel = d / speed.max(0.01);
        legs.push((dwell, travel));
        total += dwell + travel;
    }
    if let Some(last) = points.last() {
        total += last.get(3).copied().unwrap_or(0.0); // looping pause at the end
    }
    (legs, total)
}

/// Position along a path at world time `time`. `None` when < 2 points (the
/// reference path branch is a no-op then). Transcribes `motion.js` exactly.
fn path_pos(
    points: &[Vec<f64>],
    speed: f64,
    start: f64,
    phase: f64,
    loop_: bool,
    time: f64,
) -> Option<[f64; 3]> {
    if points.len() < 2 {
        return None;
    }
    let (legs, total) = path_legs(points, speed);
    let mut t = (time - start).max(0.0) + phase;
    if loop_ {
        t = if total != 0.0 {
            ((t % total) + total) % total
        } else {
            0.0
        };
    } else {
        t = t.min(total);
    }
    let mut seg = legs.len() - 1;
    let mut k = 1.0;
    for (i, &(dwell, travel)) in legs.iter().enumerate() {
        if t < dwell {
            seg = i;
            k = 0.0;
            break;
        }
        t -= dwell;
        if t < travel {
            seg = i;
            k = if travel != 0.0 {
                (t / travel).min(1.0)
            } else {
                1.0
            };
            break;
        }
        t -= travel;
    }
    let a = &points[seg];
    let c = &points[seg + 1];
    Some([
        a[0] + (c[0] - a[0]) * k,
        a[1] + (c[1] - a[1]) * k,
        a[2] + (c[2] - a[2]) * k,
    ])
}

/// EMISSIVE = COLOR STRING law: a value bound for an `emissive` field must be a
/// color string (`#rgb`/`#rrggbb`/named), never a bool or float. Used by the
/// ordeals to prove no decorative kind ever emits a non-string emissive.
pub fn is_color_string(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::String(s) => {
            let s = s.trim();
            if let Some(hex) = s.strip_prefix('#') {
                matches!(hex.len(), 3 | 6 | 8) && hex.chars().all(|c| c.is_ascii_hexdigit())
            } else {
                // a named color: non-empty, alphabetic
                !s.is_empty() && s.chars().all(|c| c.is_ascii_alphabetic())
            }
        }
        _ => false,
    }
}
