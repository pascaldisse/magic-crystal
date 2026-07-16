//! # The pink cat's mind — a deterministic idle loop (Rite V · V2)
//!
//! A [`CatMind`] is the behavior spirit of the pink_cat vessel by the ramen
//! stall. Like the decorative six, it is a PURE function of the world clock
//! (`t = tick · dt`) — no RNG, no wall clock — so the same tick stream yields a
//! byte-identical drive stream (Ultradeterminism, `ENTROPY.md`). It emits a
//! [`CatDrive`] per tick: where the cat's body sits, which way it faces, and how
//! fast it moves (the speed the composed vessel's SAMA locomotion consumes —
//! zero holds the idle pose, positive walks the legs).
//!
//! ## The loop (one closed cycle, then it repeats from home)
//!
//! `Sit → TailFlick → Walk (a small circuit) → Sit …`
//!
//! - **Sit** — parked at `home`, speed 0 (SAMA idle pose).
//! - **TailFlick** — still parked, speed 0, but `tail` swings a flick angle.
//!   HONEST DEFERRAL: the composed vessel's SAMA locomotion drives the leg
//!   gait, not the tail chain, so the tail angle is COMPUTED and carried here
//!   (the state is real, the number is real) but the low-poly capsule tail is
//!   not yet bent by it — deferred, never faked. When a tail-pose seam exists
//!   this value drives it with no change to the loop.
//! - **Walk** — a closed diamond circuit of radius `radius` around `home`,
//!   walked at `walk_speed`; it STARTS and ENDS at `home`, so the return into
//!   Sit is continuous (no teleport). Speed is positive here → the vessel's
//!   legs animate.
//!
//! The circuit is centred on `home`, so the cat never wanders off — bounded for
//! all time by construction (the four diamond corners sit exactly `radius` from
//! home).

use serde::{Deserialize, Serialize};

/// Which phase of the idle loop the cat is in this tick. Tagged with an
/// explicit byte so the determinism ordeal can serialize the full drive stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CatState {
    /// Parked at home, idle pose.
    Sit,
    /// Parked at home, tail flicking (idle pose; tail angle carried).
    TailFlick,
    /// Walking the closed circuit around home.
    Walk,
}

impl CatState {
    /// A stable byte tag (drive-stream serialization / hashing).
    pub fn tag(self) -> u8 {
        match self {
            CatState::Sit => 0,
            CatState::TailFlick => 1,
            CatState::Walk => 2,
        }
    }
}

/// The cat's drive for one tick — the pose command the composed vessel consumes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CatDrive {
    /// World-space body position (xz on the circuit; y held at `home.y`).
    pub position: [f64; 3],
    /// Facing yaw (radians) — the walk heading, or the last heading while sitting.
    pub yaw: f64,
    /// Commanded speed (m/s): 0 sitting/flicking, `walk_speed` on the circuit.
    pub speed: f64,
    /// The loop phase this tick.
    pub state: CatState,
    /// Tail flick angle (radians) — nonzero only during [`CatState::TailFlick`]
    /// (carried for a future tail-pose seam; see the module note).
    pub tail: f64,
}

impl CatDrive {
    /// Canonical little-endian byte serialization of the drive — the form the
    /// determinism ordeal compares tick-for-tick (position, yaw, speed, tail as
    /// f64 LE, then the state tag). Every field is covered, so byte-equality of
    /// two streams is total drive equality.
    pub fn to_le_bytes(&self) -> [u8; 8 * 6 + 1] {
        let mut out = [0u8; 8 * 6 + 1];
        out[0..8].copy_from_slice(&self.position[0].to_le_bytes());
        out[8..16].copy_from_slice(&self.position[1].to_le_bytes());
        out[16..24].copy_from_slice(&self.position[2].to_le_bytes());
        out[24..32].copy_from_slice(&self.yaw.to_le_bytes());
        out[32..40].copy_from_slice(&self.speed.to_le_bytes());
        out[40..48].copy_from_slice(&self.tail.to_le_bytes());
        out[48] = self.state.tag();
        out
    }
}

/// The pink cat's idle-loop behavior, parsed from its `behavior` component
/// (`{"kind":"cat", "home":[x,y,z], "radius":r, ...}`). Pure and deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CatMind {
    /// The rest point — the cat sits here and every circuit is centred on it.
    #[serde(default)]
    pub home: [f64; 3],
    /// Circuit radius (m): the diamond corners sit this far from `home`.
    #[serde(default = "default_radius")]
    pub radius: f64,
    /// Seconds parked in [`CatState::Sit`].
    #[serde(default = "default_sit")]
    pub sit: f64,
    /// Seconds in [`CatState::TailFlick`].
    #[serde(default = "default_flick")]
    pub flick: f64,
    /// Walk speed on the circuit (m/s).
    #[serde(default = "default_walk_speed")]
    pub walk_speed: f64,
    /// Tail flick angular amplitude (radians).
    #[serde(default = "default_tail_amp")]
    pub tail_amp: f64,
}

fn default_radius() -> f64 {
    1.4
}
fn default_sit() -> f64 {
    3.0
}
fn default_flick() -> f64 {
    1.5
}
fn default_walk_speed() -> f64 {
    0.8
}
fn default_tail_amp() -> f64 {
    0.5
}

impl Default for CatMind {
    fn default() -> Self {
        Self {
            home: [0.0, 0.0, 0.0],
            radius: default_radius(),
            sit: default_sit(),
            flick: default_flick(),
            walk_speed: default_walk_speed(),
            tail_amp: default_tail_amp(),
        }
    }
}

impl CatMind {
    /// The closed diamond circuit around `home`: five waypoints starting AND
    /// ending at home (E → N → W → S → home), so walking it is a loop that
    /// returns exactly to the rest point. Corners sit `radius` from home.
    fn circuit(&self) -> [[f64; 3]; 6] {
        let [hx, hy, hz] = self.home;
        let r = self.radius;
        [
            [hx, hy, hz],
            [hx + r, hy, hz],
            [hx, hy, hz + r],
            [hx - r, hy, hz],
            [hx, hy, hz - r],
            [hx, hy, hz],
        ]
    }

    /// Perimeter of the closed circuit (5 equal diamond sides of `r·√2`).
    fn circuit_perimeter(&self) -> f64 {
        // home→E, E→N, N→W, W→S are each r·√2; S→home is r·√2 too (5 sides,
        // but home→E and S→home are r each — recompute honestly per segment).
        let c = self.circuit();
        let mut total = 0.0;
        for i in 1..c.len() {
            total += seg_len(c[i - 1], c[i]);
        }
        total
    }

    /// Seconds the Walk phase lasts (perimeter / walk_speed).
    pub fn walk_duration(&self) -> f64 {
        self.circuit_perimeter() / self.walk_speed.max(1e-6)
    }

    /// Total seconds of one full loop (Sit + TailFlick + Walk).
    pub fn loop_duration(&self) -> f64 {
        self.sit + self.flick + self.walk_duration()
    }

    /// The drive at world time `t = tick · dt`. PURE: `(self, t)` fully
    /// determines the [`CatDrive`] — same inputs, byte-identical output forever.
    pub fn drive(&self, t: f64) -> CatDrive {
        let loop_dur = self.loop_duration();
        // Fold time into one loop period (t ≥ 0 always in the tick stream).
        let phase = if loop_dur > 0.0 {
            ((t % loop_dur) + loop_dur) % loop_dur
        } else {
            0.0
        };

        if phase < self.sit {
            // SIT — parked at home, facing +z (the stall), idle.
            CatDrive {
                position: self.home,
                yaw: 0.0,
                speed: 0.0,
                state: CatState::Sit,
                tail: 0.0,
            }
        } else if phase < self.sit + self.flick {
            // TAIL FLICK — parked, tail swings a full flick over the phase.
            let u = (phase - self.sit) / self.flick.max(1e-6); // 0..1
            let tail = (u * std::f64::consts::TAU).sin() * self.tail_amp;
            CatDrive {
                position: self.home,
                yaw: 0.0,
                speed: 0.0,
                state: CatState::TailFlick,
                tail,
            }
        } else {
            // WALK — constant-speed march along the closed circuit.
            let walked = (phase - self.sit - self.flick) * self.walk_speed;
            let (position, yaw) = self.walk_at(walked);
            CatDrive {
                position,
                yaw,
                speed: self.walk_speed,
                state: CatState::Walk,
                tail: 0.0,
            }
        }
    }

    /// Position + heading a distance `d` metres along the closed circuit from
    /// its start (home). Clamped to the perimeter (the caller only ever passes
    /// `d ∈ [0, perimeter]` within the Walk phase).
    fn walk_at(&self, d: f64) -> ([f64; 3], f64) {
        let c = self.circuit();
        let mut remaining = d.max(0.0);
        for i in 1..c.len() {
            let a = c[i - 1];
            let b = c[i];
            let len = seg_len(a, b);
            if remaining <= len || i == c.len() - 1 {
                let k = if len > 0.0 {
                    (remaining / len).min(1.0)
                } else {
                    0.0
                };
                let pos = [
                    a[0] + (b[0] - a[0]) * k,
                    a[1] + (b[1] - a[1]) * k,
                    a[2] + (b[2] - a[2]) * k,
                ];
                let yaw = (b[0] - a[0]).atan2(b[2] - a[2]);
                return (pos, yaw);
            }
            remaining -= len;
        }
        (c[0], 0.0)
    }
}

fn seg_len(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
}
