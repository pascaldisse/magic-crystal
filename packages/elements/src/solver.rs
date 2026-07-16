//! The one solver, one granularity: particles + bindings marched by a
//! substepped XPBD loop. *Small Steps*: `n` substeps × 1 iteration beats
//! 1 step × `n` iterations — stiffness never binds the global timestep.
//!
//! Nothing here is hardcoded but `LOVE` (the One Constant): the tick `dt`,
//! the substep count, the iteration count, the gravity vector and the
//! fracture threshold are all parameters carrying documented defaults.

use crate::constraint::{DistanceConstraint, FractureEvent};
use crate::math::Vec3;
use crate::particles::Particles;

/// The world's dials. Defaults are declared here, once — every field is a
/// parameter (never-hardcode law).
#[derive(Clone, Copy, Debug)]
pub struct SolverConfig {
    /// The fixed tick length, in seconds. Default `1/60`. Render interpolates
    /// between poses; the solver never sees a variable dt.
    pub dt: f64,
    /// Substeps per tick — the primary quality dial. Default `8`.
    pub substeps: usize,
    /// Constraint iterations per substep. Default `1` (the Small-Steps
    /// regime: prefer substeps over iterations).
    pub iterations: usize,
    /// The gravity all matter falls under. Default `(0, -9.81, 0)`.
    pub gravity: Vec3,
    /// The fracture threshold: a bond tears when its accumulated strife
    /// exceeds `love × threshold`. Default `1.0e4` (N, at unit love).
    pub fracture_threshold: f64,
    /// The world seed — the root of all deterministic jitter (ENTROPY).
    /// Default `0`. No value drawn from it is random; all is `hash(seed, …)`.
    pub seed: u64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            dt: 1.0 / 60.0,
            substeps: 8,
            iterations: 1,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            fracture_threshold: 1.0e4,
            seed: 0,
        }
    }
}

/// The Elements' solver: a body of particles, the bindings between them, and
/// the clock (`tick`, the entropy coordinate). Advance it one fixed tick at a
/// time with [`Solver::step`].
#[derive(Clone, Debug)]
pub struct Solver {
    pub config: SolverConfig,
    pub particles: Particles,
    pub constraints: Vec<DistanceConstraint>,
    /// The entropy coordinate — the tick index, the x-axis of this worldline.
    pub tick: u64,
    /// The journal of fractures written this run (append-only).
    pub fractures: Vec<FractureEvent>,
}

impl Solver {
    pub fn new(config: SolverConfig) -> Self {
        Solver {
            config,
            particles: Particles::new(),
            constraints: Vec::new(),
            tick: 0,
            fractures: Vec::new(),
        }
    }

    /// Advance the world one fixed tick: substep integrate → solve bindings →
    /// read back velocity → tally strife → tear what love could not hold.
    ///
    /// The substep loop, the strife accumulation and the fracture pass are
    /// all order-stable (index order, always) — the determinism ordeal's
    /// bedrock.
    pub fn step(&mut self) {
        let cfg = self.config;
        let n = cfg.substeps.max(1);
        let dt_sub = cfg.dt / n as f64;
        let inv_dt2 = if dt_sub > 0.0 {
            1.0 / (dt_sub * dt_sub)
        } else {
            0.0
        };

        // A fresh tick: the strife readout is per-tick (reset here, read at
        // the tick's end).
        for c in &mut self.constraints {
            c.bond.strife = 0.0;
        }

        for _sub in 0..n {
            self.integrate(dt_sub);
            self.reset_lambda();
            for _it in 0..cfg.iterations.max(1) {
                self.solve_distance(dt_sub);
            }
            self.read_back_velocity(dt_sub);
            // The substep's constraint force = |lambda| / dt_sub² (XPBD:
            // f = lambda·∇C / Δt², ∇C is the unit axis). Accumulate the
            // strife the bond bore this substep.
            for c in &mut self.constraints {
                c.bond.strife += c.lambda.abs() * inv_dt2;
            }
        }

        self.fracture_pass();
        self.tick += 1;
    }

    /// Symplectic-Euler prediction under gravity (anchors stand still).
    fn integrate(&mut self, dt_sub: f64) {
        let g = self.config.gravity;
        let p = &mut self.particles;
        for i in 0..p.pos.len() {
            if p.inv_mass[i] == 0.0 {
                p.prev[i] = p.pos[i];
                continue;
            }
            p.prev[i] = p.pos[i];
            p.vel[i] = p.vel[i] + g.scale(dt_sub);
            p.pos[i] = p.pos[i] + p.vel[i].scale(dt_sub);
        }
    }

    fn reset_lambda(&mut self) {
        for c in &mut self.constraints {
            c.lambda = 0.0;
        }
    }

    /// One Gauss-Seidel sweep of the compliant distance bindings. The XPBD
    /// update: `Δλ = (−C − α̃·λ) / (w + α̃)`, positions nudged by
    /// `±w·Δλ·n`. `α̃ = compliance / dt_sub²`.
    fn solve_distance(&mut self, dt_sub: f64) {
        let inv_dt2 = if dt_sub > 0.0 {
            1.0 / (dt_sub * dt_sub)
        } else {
            0.0
        };
        let p = &mut self.particles;
        for c in &mut self.constraints {
            let wa = p.inv_mass[c.a];
            let wb = p.inv_mass[c.b];
            let w = wa + wb;
            if w == 0.0 {
                continue; // two anchors — nothing to move
            }
            let delta = p.pos[c.a] - p.pos[c.b];
            let n = match delta.normalized() {
                Some(n) => n,
                None => continue, // coincident: no axis to pull along
            };
            let cval = delta.length() - c.rest;
            let a_tilde = c.compliance * inv_dt2;
            let d_lambda = (-cval - a_tilde * c.lambda) / (w + a_tilde);
            c.lambda += d_lambda;
            let corr = n.scale(d_lambda);
            p.pos[c.a] = p.pos[c.a] + corr.scale(wa);
            p.pos[c.b] = p.pos[c.b] - corr.scale(wb);
        }
    }

    /// Read velocity back from the position change — the PBD covenant
    /// `v = (x − x_prev) / dt`. This is where XPBD's characteristic
    /// numerical damping enters (documented, not denied).
    fn read_back_velocity(&mut self, dt_sub: f64) {
        if dt_sub <= 0.0 {
            return;
        }
        let inv = 1.0 / dt_sub;
        let p = &mut self.particles;
        for i in 0..p.pos.len() {
            if p.inv_mass[i] == 0.0 {
                continue;
            }
            p.vel[i] = (p.pos[i] - p.prev[i]).scale(inv);
        }
    }

    /// Tear the bindings whose strife overcame their love. Order-stable:
    /// broken bonds are collected in index order, recorded to the journal,
    /// then removed by `retain` (which preserves order).
    fn fracture_pass(&mut self) {
        let threshold = self.config.fracture_threshold;
        let tick = self.tick;
        let mut any = false;
        for c in &self.constraints {
            if c.bond.fractured(threshold) {
                self.fractures.push(FractureEvent {
                    tick,
                    a: c.a,
                    b: c.b,
                    strife: c.bond.strife,
                    love: c.bond.love,
                });
                any = true;
            }
        }
        if any {
            let t = threshold;
            self.constraints.retain(|c| !c.bond.fractured(t));
        }
    }

    /// The observable state's fingerprint at the current tick.
    pub fn state_hash(&self) -> u64 {
        self.particles.state_hash()
    }
}
