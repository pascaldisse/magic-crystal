//! Locomotion state machine: idle / walk / run.
//!
//! Driven by a commanded speed per tick, it classifies a target gait against
//! speed thresholds and cross-fades between gaits over `blend_ticks` using
//! homunculus nlerp blending. Everything is tick-indexed and pure: the same
//! command stream produces a byte-identical pose stream (`ENTROPY.md`).
//!
//! Continuity is structural: when a transition begins, the blend source is the
//! pose emitted on the previous tick, so the output never jumps at a transition
//! edge. The blend factor rises linearly from `0` to `1` over exactly
//! `blend_ticks`, so it is monotonic and the transition timing matches the
//! params exactly.

use crate::gait::{gait_pose, GaitParams};
use homunculus::{Pose, Skeleton};

/// The three locomotion states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gait {
    /// Standing at bind pose.
    Idle,
    /// The walk gait.
    Walk,
    /// The run gait.
    Run,
}

/// Tuning of the locomotion state machine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocomotionParams {
    /// Commanded speed at or above which the state is at least [`Gait::Walk`].
    pub walk_threshold: f32,
    /// Commanded speed at or above which the state is [`Gait::Run`].
    pub run_threshold: f32,
    /// Ticks a cross-fade between states takes (`0` = instant).
    pub blend_ticks: u32,
    /// Gait parameters used in the [`Gait::Walk`] state.
    pub walk: GaitParams,
    /// Gait parameters used in the [`Gait::Run`] state.
    pub run: GaitParams,
}

impl Default for LocomotionParams {
    fn default() -> Self {
        Self {
            walk_threshold: 0.2,
            run_threshold: 3.0,
            blend_ticks: 12,
            walk: GaitParams::walk(),
            run: GaitParams::run(),
        }
    }
}

/// A running locomotion state machine. Advance it one tick at a time with
/// [`Locomotion::step`].
#[derive(Clone, Debug)]
pub struct Locomotion {
    params: LocomotionParams,
    tick: u64,
    state: Gait,
    blend_start: u64,
    blend_from: Option<Pose>,
    last: Option<Pose>,
    last_blend: f32,
}

impl Locomotion {
    /// A fresh machine starting in [`Gait::Idle`] at tick `0`.
    pub fn new(params: LocomotionParams) -> Self {
        Self {
            params,
            tick: 0,
            state: Gait::Idle,
            blend_start: 0,
            blend_from: None,
            last: None,
            last_blend: 1.0,
        }
    }

    /// Classify a commanded speed into a target gait.
    pub fn classify(&self, commanded_speed: f32) -> Gait {
        if commanded_speed >= self.params.run_threshold {
            Gait::Run
        } else if commanded_speed >= self.params.walk_threshold {
            Gait::Walk
        } else {
            Gait::Idle
        }
    }

    /// The pose for a given state at the current tick.
    fn state_pose(&self, skeleton: &Skeleton, state: Gait) -> Pose {
        match state {
            Gait::Idle => Pose::bind(skeleton),
            Gait::Walk => gait_pose(skeleton, &self.params.walk, self.tick),
            Gait::Run => gait_pose(skeleton, &self.params.run, self.tick),
        }
    }

    /// Advance one tick against `commanded_speed`, returning the blended pose.
    pub fn step(&mut self, skeleton: &Skeleton, commanded_speed: f32) -> Pose {
        let desired = self.classify(commanded_speed);
        if desired != self.state {
            // Begin a new cross-fade. The source is the last emitted pose, so
            // the output is continuous across the transition edge.
            let from = self
                .last
                .clone()
                .unwrap_or_else(|| self.state_pose(skeleton, self.state));
            self.blend_from = Some(from);
            self.blend_start = self.tick;
            self.state = desired;
        }

        let target = self.state_pose(skeleton, self.state);
        let out = match &self.blend_from {
            Some(from) => {
                let t = if self.params.blend_ticks == 0 {
                    1.0
                } else {
                    ((self.tick - self.blend_start) as f32 / self.params.blend_ticks as f32)
                        .clamp(0.0, 1.0)
                };
                let blended = from.blend(&target, t);
                self.last_blend = t;
                if t >= 1.0 {
                    self.blend_from = None;
                }
                blended
            }
            None => {
                self.last_blend = 1.0;
                target
            }
        };

        self.last = Some(out.clone());
        self.tick += 1;
        out
    }

    /// The current settled/target state.
    pub fn state(&self) -> Gait {
        self.state
    }

    /// The next tick index this machine will emit.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Whether a cross-fade is in progress.
    pub fn blending(&self) -> bool {
        self.blend_from.is_some()
    }

    /// The blend factor `[0,1]` used on the most recent [`step`].
    ///
    /// [`step`]: Locomotion::step
    pub fn blend_factor(&self) -> f32 {
        self.last_blend
    }
}
