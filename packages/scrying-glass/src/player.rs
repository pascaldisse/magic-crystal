//! The Embodiment (Rite E0): a first-person walk controller for the Scrying
//! Glass window.
//!
//! The Architect wants to WALK his world, so the native window's camera becomes
//! a body: [`Player`] holds a pose + velocity, integrates gravity every tick,
//! and clamps its feet to the floor by ray-marching the render scene's own
//! triangles ([`Ground`]). No terrain oracle, no collider components — the mesh
//! you see IS the floor you stand on.
//!
//! Every feel constant lives in [`PlayerParams`] and is read from the
//! environment ([`PlayerParams::from_env`]); nothing here is hardcoded. Defaults
//! track the reference GAIA-World-Engine player controller
//! (`client/kernel/player.js`): eye height 1.7 m standing / 1.0 crouched, walk 6
//! / run 14 / crouch 3 m·s⁻¹, and a Half-Life-sized jump — with the notable
//! exception of gravity, which the Rite spec pins to a real 9.81 m·s⁻² so the
//! jump arc ordeal reads in physical units.
//!
//! The integrator is deterministic: a fixed `dt` and a fixed tick sequence
//! produce a byte-identical pose stream (the first ordeal), which is what lets
//! the `/walk` debug organ drive reproducible play-tests without a keyboard.

use glam::Vec3;
use std::collections::HashSet;

/// Cosine cutoff between floor and wall in [`Ground::from_positions`]: a
/// triangle whose face-normal y-component magnitude is at or below this is
/// steeper than `acos(0.3) ≈ 72.54°` from horizontal and is dropped as a
/// wall, never floor. Named (not a bare `0.3` in two call sites) so the
/// GUARDIAN RULING 6 contact-patch tolerance derivation below can cite the
/// EXACT cutoff the floor set was already built with.
const WALL_NORMAL_Y_COS_CUTOFF: f32 = 0.3;

/// Deterministic compass-point probe count for the GUARDIAN RULING 6
/// contact-patch test: eight points (N/NE/E/SE/S/SW/W/NW) give an even
/// angular spread around the candidate without any randomness. This cost is
/// paid EVERY tick on the winning candidate (never skipped), plus again on
/// every fallthrough rejection — see [`Ground::height_at_gated`]'s doc
/// comment for the honest per-tick accounting (it is NOT a per-triangle
/// cost, but it is not a rare one either).
const CONTACT_PROBE_COUNT: usize = 8;

/// Acceptance epsilon [`Ground::raw_height_at`] uses to decide whether a
/// candidate height `y` still qualifies under a given `ceiling`: `y <=
/// ceiling + COLUMN_EPSILON`. Named so [`Ground::height_at_gated`]'s retry
/// step can be DERIVED from it (see [`GATED_EXCLUSION_STEP`]) instead of
/// living as a second, independently-chosen literal that could silently
/// drift smaller than this one.
const COLUMN_EPSILON: f32 = 1e-3;

/// Retry-exclusion step [`Ground::height_at_gated`] subtracts from a
/// rejected candidate's height before re-scanning. MUST strictly exceed
/// [`COLUMN_EPSILON`]: the next scan accepts any `y' <= new_ceiling +
/// COLUMN_EPSILON`, i.e. any `y' <= y - GATED_EXCLUSION_STEP +
/// COLUMN_EPSILON`. If the step were `<= COLUMN_EPSILON`, that bound would
/// be `>= y`, so the just-rejected candidate `y' == y` would re-qualify on
/// the very next iteration and the fallthrough loop would never terminate
/// (this was the exact bug: a 1e-4 step against a 1e-3 acceptance epsilon).
/// Doubling the epsilon keeps the derivation in one place, expressed in
/// code, while staying well inside float noise for any realistic scene
/// scale.
const GATED_EXCLUSION_STEP: f32 = 2.0 * COLUMN_EPSILON;

/// GUARDIAN RULING 6 contact-patch radius default, m
/// (`GAIA_PLAYER_CONTACT_RADIUS`). MEASURED, not invented: nari (the "nari"
/// vessel preset, `vessel::Preset::nari()`) is composed and posed at SAMA's
/// idle tick, and the xz half-extent of the idle-mesh vertices whose
/// STRONGEST skin weight binds to a single foot bone (`"L.foot"` /
/// `"R.foot"`) is measured directly — this is her real footprint, not a
/// stance width (which would span both feet). See the ordeal
/// `contact_radius_matches_measured_foot_half_extent` in
/// `tests/patch_gate.rs`, which prints the live measurement every run:
/// `L.foot half_x=0.0762 half_z=0.0762`, `R.foot half_x=0.0807 half_z=0.0768`
/// (captured on the current preset geometry). The default is the largest of
/// those four half-extents (0.0807 m). ROUNDING RULE (so the constant is
/// reproducible, not aesthetic): round the measured max UP to the next whole
/// centimetre — `(0.0807 → 0.09)`, i.e. `(measured_max * 100).ceil() / 100`.
/// This rounding is SAFE but not FREE, and both halves matter: rounding up
/// means the gate never ADMITS a surface smaller than her actual foot (the
/// bug Ruling 6 exists to close), but it also means the gate will
/// false-REJECT any legitimate ledge whose half-width falls in
/// `[0.0807, 0.09)` — a real, if narrow, correctness cost paid for the
/// safety margin.
pub const DEFAULT_CONTACT_RADIUS: f32 = 0.09;

/// Derive the GUARDIAN RULING 6 contact-patch height tolerance from a patch
/// `radius`, using the SAME wall cutoff [`Ground::from_positions`] already
/// uses to decide floor vs wall. DERIVATION: `from_positions` keeps any
/// triangle with `|normal.y| > WALL_NORMAL_Y_COS_CUTOFF`, i.e. it already
/// calls anything up to `acos(WALL_NORMAL_Y_COS_CUTOFF)` from horizontal a
/// walkable slope. Walking `radius` metres straight up that steepest
/// walkable slope changes height by `radius * tan(acos(WALL_NORMAL_Y_COS_CUTOFF))`
/// — that is the largest height step a probe can legitimately see while
/// still standing on floor the code already calls walkable, so it is exactly
/// the tolerance band: any bigger jump means the probe left real floor.
pub fn contact_tolerance(radius: f32) -> f32 {
    radius * WALL_NORMAL_Y_COS_CUTOFF.acos().tan()
}

/// A held control. The window's keyboard and the `/walk` organ both speak in
/// these intents, never raw key codes, so the controller stays input-agnostic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Key {
    /// Walk along the look direction (`KeyW`).
    Forward,
    /// Walk against the look direction (`KeyS`).
    Back,
    /// Strafe toward the look-relative left (`KeyA`).
    Left,
    /// Strafe toward the look-relative right (`KeyD`).
    Right,
    /// Leave the ground when grounded (`Space`).
    Jump,
    /// Run instead of walk while held (`Shift`).
    Run,
    /// Lower the eye and slow to a crouch (`Ctrl`/`KeyC`).
    Crouch,
}

impl Key {
    /// Map a `/walk` key token to an intent. Accepts the reference engine's
    /// code names and short aliases; unknown tokens are ignored by the caller.
    pub fn from_token(token: &str) -> Option<Self> {
        match token.trim().to_ascii_lowercase().as_str() {
            "w" | "keyw" | "forward" | "up" => Some(Self::Forward),
            "s" | "keys" | "back" | "backward" | "down" => Some(Self::Back),
            "a" | "keya" | "left" => Some(Self::Left),
            "d" | "keyd" | "right" => Some(Self::Right),
            "space" | "jump" => Some(Self::Jump),
            "shift" | "shiftleft" | "shiftright" | "run" => Some(Self::Run),
            "c" | "keyc" | "ctrl" | "control" | "controlleft" | "controlright" | "crouch" => {
                Some(Self::Crouch)
            }
            _ => None,
        }
    }
}

/// Tunable feel constants. Read once from the environment; the controller reads
/// them every tick, so nothing about the body's motion is a magic number.
#[derive(Clone, Copy, Debug)]
pub struct PlayerParams {
    /// Downward acceleration, m·s⁻² (`GAIA_PLAYER_GRAVITY`).
    pub gravity: f32,
    /// Terminal fall speed, m·s⁻¹ (`GAIA_PLAYER_TERMINAL`).
    pub terminal: f32,
    /// Upward launch speed on jump, m·s⁻¹ (`GAIA_PLAYER_JUMP`). Apex ≈ v²/2g.
    pub jump_speed: f32,
    /// Walk speed, m·s⁻¹ (`GAIA_PLAYER_WALK`).
    pub walk_speed: f32,
    /// Run speed while [`Key::Run`] is held, m·s⁻¹ (`GAIA_PLAYER_RUN`).
    pub run_speed: f32,
    /// Crouch speed, m·s⁻¹ (`GAIA_PLAYER_CROUCH`).
    pub crouch_speed: f32,
    /// Standing eye height above the feet, m (`GAIA_PLAYER_EYE_STAND`).
    pub eye_stand: f32,
    /// Crouched eye height above the feet, m (`GAIA_PLAYER_EYE_CROUCH`).
    pub eye_crouch: f32,
    /// Mouse-look radians per pixel of motion (`GAIA_PLAYER_SENSITIVITY`).
    pub mouse_sensitivity: f32,
    /// Horizontal velocity smoothing rate, s⁻¹ (`GAIA_PLAYER_MOVE_DAMP`).
    pub move_damp: f32,
    /// Eye-height crouch/stand smoothing rate, s⁻¹ (`GAIA_PLAYER_EYE_DAMP`).
    pub eye_damp: f32,
    /// Ground-follow snap band + smoothing rate, s⁻¹ (`GAIA_PLAYER_GROUND_FOLLOW`).
    pub ground_follow: f32,
    /// Feet-above-floor slack still counted as grounded, m (`GAIA_PLAYER_GROUND_SNAP`).
    pub ground_snap: f32,
    /// Fall below this world Y returns to the last safe ground (`GAIA_PLAYER_VOID_Y`).
    pub void_y: f32,
    /// Max pitch magnitude in radians (`GAIA_PLAYER_PITCH_LIMIT`).
    pub pitch_limit: f32,
    /// GUARDIAN RULING 6 contact-patch radius, m — how big a footprint the
    /// floor under the body's column must actually hold before it counts as
    /// standable (`GAIA_PLAYER_CONTACT_RADIUS`). See [`DEFAULT_CONTACT_RADIUS`]
    /// for the measured derivation.
    pub contact_radius: f32,
}

impl PlayerParams {
    /// Read every constant from the environment, falling back to the
    /// reference-engine defaults (gravity excepted — the Rite spec pins it to
    /// 9.81 m·s⁻²).
    pub fn from_env() -> Result<Self, String> {
        fn number(name: &str, default: f32) -> Result<f32, String> {
            match std::env::var(name) {
                Ok(value) => value
                    .parse::<f32>()
                    .map_err(|_| format!("{name} must be a number, got {value:?}"))
                    .and_then(|parsed| {
                        if parsed.is_finite() {
                            Ok(parsed)
                        } else {
                            Err(format!("{name} must be finite, got {value:?}"))
                        }
                    }),
                Err(_) => Ok(default),
            }
        }
        let params = Self {
            gravity: number("GAIA_PLAYER_GRAVITY", 9.81)?,
            terminal: number("GAIA_PLAYER_TERMINAL", 55.0)?,
            jump_speed: number("GAIA_PLAYER_JUMP", 5.0)?,
            walk_speed: number("GAIA_PLAYER_WALK", 6.0)?,
            run_speed: number("GAIA_PLAYER_RUN", 14.0)?,
            crouch_speed: number("GAIA_PLAYER_CROUCH", 3.0)?,
            eye_stand: number("GAIA_PLAYER_EYE_STAND", 1.7)?,
            eye_crouch: number("GAIA_PLAYER_EYE_CROUCH", 1.0)?,
            mouse_sensitivity: number("GAIA_PLAYER_SENSITIVITY", 0.0022)?,
            move_damp: number("GAIA_PLAYER_MOVE_DAMP", 10.0)?,
            eye_damp: number("GAIA_PLAYER_EYE_DAMP", 12.0)?,
            ground_follow: number("GAIA_PLAYER_GROUND_FOLLOW", 12.0)?,
            ground_snap: number("GAIA_PLAYER_GROUND_SNAP", 0.35)?,
            void_y: number("GAIA_PLAYER_VOID_Y", -120.0)?,
            pitch_limit: number("GAIA_PLAYER_PITCH_LIMIT", 1.45)?,
            contact_radius: number("GAIA_PLAYER_CONTACT_RADIUS", DEFAULT_CONTACT_RADIUS)?,
        };
        if params.gravity <= 0.0
            || params.terminal <= 0.0
            || params.walk_speed <= 0.0
            || params.eye_stand <= 0.0
            || params.eye_crouch <= 0.0
            || params.contact_radius <= 0.0
        {
            return Err(
                "player gravity, terminal, walk speed, eye heights and contact radius must be positive"
                    .into(),
            );
        }
        Ok(params)
    }
}

/// A single walkable triangle of the render scene, kept in world space with its
/// face normal so the floor query can reject walls and interpolate height.
#[derive(Clone, Copy)]
struct Triangle {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    normal_y: f32,
}

impl Triangle {
    /// The floor height at column `(x, z)`, or `None` when the column misses
    /// this triangle's 2-D (xz) projection.
    fn height_at(&self, x: f32, z: f32) -> Option<f32> {
        // Barycentric weights in the xz plane; y is interpolated from them.
        let denom = (self.b.z - self.c.z) * (self.a.x - self.c.x)
            + (self.c.x - self.b.x) * (self.a.z - self.c.z);
        if denom.abs() < 1e-9 {
            return None;
        }
        let u = ((self.b.z - self.c.z) * (x - self.c.x) + (self.c.x - self.b.x) * (z - self.c.z))
            / denom;
        let v = ((self.c.z - self.a.z) * (x - self.c.x) + (self.a.x - self.c.x) * (z - self.c.z))
            / denom;
        let w = 1.0 - u - v;
        const EDGE: f32 = 1e-4;
        if u >= -EDGE && v >= -EDGE && w >= -EDGE {
            Some(u * self.a.y + v * self.b.y + w * self.c.y)
        } else {
            None
        }
    }
}

/// The render scene reduced to a set of up-facing triangles — everything the
/// body can stand on. A downward column query returns the highest floor under a
/// ceiling, which is exactly the ground-snap the controller needs.
pub struct Ground {
    triangles: Vec<Triangle>,
}

impl Ground {
    /// Build the floor set from the render scene's world-space triangle soup
    /// (each consecutive three positions is one triangle). Triangles whose face
    /// tilts past ~72° from vertical are walls and are dropped.
    pub fn from_positions(positions: &[[f32; 3]]) -> Self {
        let mut triangles = Vec::with_capacity(positions.len() / 3);
        for chunk in positions.chunks_exact(3) {
            let a = Vec3::from_array(chunk[0]);
            let b = Vec3::from_array(chunk[1]);
            let c = Vec3::from_array(chunk[2]);
            let normal = (b - a).cross(c - a).normalize_or_zero();
            if normal.y.abs() <= WALL_NORMAL_Y_COS_CUTOFF {
                continue; // near-vertical: a wall, never a floor
            }
            triangles.push(Triangle {
                a,
                b,
                c,
                normal_y: normal.y,
            });
        }
        Self { triangles }
    }

    /// Number of retained walkable triangles (diagnostics / tests).
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// RAW column scan (pre-Ruling-6): the highest up-facing triangle whose
    /// interpolated height sits at or below `ceiling`, with no regard for how
    /// big that triangle's surface actually is. Kept private — every public
    /// query above this runs the GUARDIAN RULING 6 contact-patch gate on top
    /// of it, so a mirror-panel edge (or any other sliver) never counts as
    /// floor on its own.
    fn raw_height_at(&self, x: f32, z: f32, ceiling: f32) -> Option<f32> {
        let mut best: Option<f32> = None;
        for triangle in &self.triangles {
            if triangle.normal_y.abs() <= WALL_NORMAL_Y_COS_CUTOFF {
                continue;
            }
            if let Some(y) = triangle.height_at(x, z)
                && y <= ceiling + COLUMN_EPSILON
            {
                best = Some(best.map_or(y, |current| current.max(y)));
            }
        }
        best
    }

    /// GUARDIAN RULING 6: whether the FLOOR SET (not necessarily one
    /// connected surface) has SOME raw floor within `tol` of `y` under EVERY
    /// one of [`CONTACT_PROBE_COUNT`] deterministic compass points on the
    /// circle of `radius` around `(x, z)`. This is exactly what it tests and
    /// no more: it is a ring-of-points existence check on the floor set, NOT
    /// a proof that a single contiguous surface spans the whole disc. A
    /// surface smaller than the patch (the mirror-panel top edge) has probes
    /// that either miss the surface entirely (raw floor far below, at
    /// whatever real ground sits under the mirror) or land on no floor at
    /// all — either way the probe fails and the whole candidate is rejected,
    /// which is what closes the mirror-edge seam. But the same looseness
    /// means the gate can be fooled two ways it does not claim to guard
    /// against: (1) a sliver low enough above the surrounding real floor
    /// that ALL 8 probes still land on that lower floor within `tol` gets
    /// ADMITTED even though it is itself no wider than the sliver (see the
    /// `sliver_low_enough_above_floor_is_admitted_by_design` documentation
    /// ordeal in `tests/patch_gate.rs` for the concrete boundary); (2)
    /// disjoint patches of floor scattered around the ring, none of them
    /// spanning the centre, could in principle each individually satisfy one
    /// probe and add up to a false ADMIT. Neither case is exercised by
    /// naruko's authored geometry tonight, so this is documented risk, not a
    /// bug — closing it would need per-probe surface identity, not just a
    /// height check, and is left to a future ruling.
    fn patch_supported(&self, x: f32, z: f32, y: f32, radius: f32, tol: f32) -> bool {
        if radius <= 0.0 {
            return true; // patch test disabled — degrade to the raw query
        }
        for i in 0..CONTACT_PROBE_COUNT {
            let angle = i as f32 * std::f32::consts::TAU / CONTACT_PROBE_COUNT as f32;
            let probe_x = x + radius * angle.cos();
            let probe_z = z + radius * angle.sin();
            // Search a column generously above and below `y` (not bounded by
            // the caller's original ceiling — the patch test cares only
            // whether floor exists NEAR y, not whether it's the tallest
            // thing under the original ceiling).
            match self.raw_height_at(probe_x, probe_z, y + tol) {
                Some(probe_y) if (probe_y - y).abs() <= tol => {}
                _ => return false,
            }
        }
        true
    }

    /// Highest floor height in column `(x, z)` at or below `ceiling` that
    /// ALSO passes the GUARDIAN RULING 6 contact-patch gate for the given
    /// `radius`/`tol` (see [`contact_tolerance`] for deriving `tol` from
    /// `radius`). Falls through to the next-highest raw candidate when the
    /// top one is too small a surface to hold a foot (a mirror-panel edge, a
    /// rail cap, …), all the way down to `None` if nothing under the column
    /// can hold the patch.
    ///
    /// PERFORMANCE: this is called every tick, and `patch_supported` runs on
    /// the SUCCESS path before every `Some` return — so it is paid EVERY
    /// tick, always, not conditionally: `CONTACT_PROBE_COUNT` (8) full column
    /// scans just to confirm the winning candidate, plus another 8 for each
    /// fallthrough rejection (a mirror edge costs 16 total: 8 to reject the
    /// sliver, 8 to confirm the seawall beneath it). That is roughly 9× the
    /// cost of the pre-gate single scan per fallthrough step — never every
    /// triangle in the scene individually re-probed, but still a real,
    /// always-paid multiplier. Measured harmless at naruko scale (~63k
    /// tri-tests/tick for one walker on the current floor set); revisit this
    /// budget if the floor set or walker count grows by an order of
    /// magnitude.
    pub fn height_at_gated(
        &self,
        x: f32,
        z: f32,
        ceiling: f32,
        radius: f32,
        tol: f32,
    ) -> Option<f32> {
        let mut ceiling = ceiling;
        // Belt-and-braces bound, PROVEN unreachable, not just hoped safe:
        // each retry lowers the ceiling by GATED_EXCLUSION_STEP (2 *
        // COLUMN_EPSILON), which strictly exceeds COLUMN_EPSILON — the only
        // epsilon `raw_height_at` uses to accept a candidate. So every
        // accepted `y` this loop sees is strictly less than every `y` it saw
        // before (`y_next <= ceiling_next + COLUMN_EPSILON == y_prev -
        // GATED_EXCLUSION_STEP + COLUMN_EPSILON < y_prev`), i.e. candidates
        // form a strictly decreasing sequence. `raw_height_at` only ever
        // returns heights that come from this floor's own triangle set, so
        // there are at most `self.triangles.len()` distinct candidate
        // heights to strictly decrease through before the column runs out of
        // floor and `raw_height_at` returns `None` (caught by the `?` above,
        // outside this loop). Therefore this loop CANNOT run more than
        // `triangles.len()` iterations while the exclusion-step invariant
        // holds — if it ever does, the invariant itself broke (someone
        // changed GATED_EXCLUSION_STEP or COLUMN_EPSILON without keeping the
        // former strictly larger, or `raw_height_at`'s acceptance rule
        // changed underneath it). That is a logic bug in code the proof
        // above says can never legitimately execute, so it gets a loud crash
        // in EVERY profile — not a quiet `None` that would silently drop the
        // player through the floor in the field — so it is caught in CI
        // instead of hidden behind a "safe" fallback.
        let max_iterations = self.triangles.len();
        for _ in 0..=max_iterations {
            let y = self.raw_height_at(x, z, ceiling)?;
            if self.patch_supported(x, z, y, radius, tol) {
                return Some(y);
            }
            // Exclude this candidate (and anything within float noise of it)
            // and retry with the next-highest raw floor below it.
            ceiling = y - GATED_EXCLUSION_STEP;
        }
        panic!(
            "height_at_gated exceeded {max_iterations} fallthrough iterations (bounded by the \
             triangle count) — the GATED_EXCLUSION_STEP > COLUMN_EPSILON exclusion invariant \
             this bound is proven from (see this loop's doc comment) must have broken; this is \
             unreachable-by-proof code firing, a logic bug that deserves a crash in every \
             profile, not a quiet fall-through-world"
        );
    }

    /// Highest floor height in column `(x, z)` that sits at or below
    /// `ceiling`, or `None` when nothing walkable lies under the column
    /// there. Patch-gated (GUARDIAN RULING 6) with [`DEFAULT_CONTACT_RADIUS`]
    /// so every existing caller inherits the mirror-edge fix without
    /// threading a radius through; the live [`Player`] instead calls
    /// [`Ground::height_at_gated`] directly with its own [`PlayerParams`]
    /// (so `GAIA_PLAYER_CONTACT_RADIUS` actually takes effect at runtime).
    pub fn height_at(&self, x: f32, z: f32, ceiling: f32) -> Option<f32> {
        self.height_at_gated(
            x,
            z,
            ceiling,
            DEFAULT_CONTACT_RADIUS,
            contact_tolerance(DEFAULT_CONTACT_RADIUS),
        )
    }
}

/// A first-person pose snapshot — what `/pose` reports and what the `/walk`
/// stream is made of.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Eye position in world space (feet are `position.y - eye_height`).
    pub position: Vec3,
    /// Look yaw, radians (0 faces −Z, matching the render camera).
    pub yaw: f32,
    /// Look pitch, radians (negative looks down).
    pub pitch: f32,
    /// Current eye height above the feet, m.
    pub eye_height: f32,
    /// Whether the feet are resting on the floor this tick.
    pub grounded: bool,
    /// Vertical velocity, m·s⁻¹ (gravity/jump axis).
    pub vy: f32,
}

/// The embodied camera: a body that walks, falls, and stands on the render
/// scene's own geometry.
pub struct Player {
    /// Feel constants (read from the environment at construction).
    pub params: PlayerParams,
    /// Eye position in world space.
    pub position: Vec3,
    /// Smoothed horizontal velocity (y unused; gravity lives in [`Self::vy`]).
    pub velocity: Vec3,
    /// Vertical velocity, m·s⁻¹.
    pub vy: f32,
    /// Look yaw, radians.
    pub yaw: f32,
    /// Look pitch, radians.
    pub pitch: f32,
    /// Current (smoothed) eye height, m.
    pub eye_height: f32,
    /// Held controls this tick.
    pub keys: HashSet<Key>,
    /// True while the feet rest on the floor.
    pub grounded: bool,
    /// Held Space is one jump until release (edge-triggered launch).
    jump_locked: bool,
    /// Last static ground pose — a void fall returns here.
    last_safe: Option<Vec3>,
    /// Spawn eye pose `(position, yaw)` for respawns.
    spawn: (Vec3, f32),
}

impl Player {
    /// Spawn a body at the world's spawn eye pose. `spawn_eye`/`spawn_yaw` come
    /// straight from the loaded world (the scene camera), never hardcoded.
    pub fn new(params: PlayerParams, spawn_eye: Vec3, spawn_yaw: f32) -> Self {
        Self {
            params,
            position: spawn_eye,
            velocity: Vec3::ZERO,
            vy: 0.0,
            yaw: spawn_yaw,
            pitch: 0.0,
            eye_height: params.eye_stand,
            keys: HashSet::new(),
            grounded: false,
            jump_locked: false,
            last_safe: None,
            spawn: (spawn_eye, spawn_yaw),
        }
    }

    /// Reset motion state back to the spawn eye pose.
    pub fn respawn(&mut self) {
        self.position = self.spawn.0;
        self.yaw = self.spawn.1;
        self.pitch = 0.0;
        self.velocity = Vec3::ZERO;
        self.vy = 0.0;
        self.eye_height = self.params.eye_stand;
        self.grounded = false;
        self.jump_locked = false;
    }

    /// Apply a mouse-look delta in pixels (window input). Yaw turns with −Δx,
    /// pitch with −Δy, clamped to the pitch limit — identical to the reference
    /// controller's frame.
    pub fn look(&mut self, dx: f32, dy: f32) {
        let s = self.params.mouse_sensitivity;
        self.yaw -= dx * s;
        self.pitch = (self.pitch - dy * s).clamp(-self.params.pitch_limit, self.params.pitch_limit);
    }

    /// The current pose snapshot.
    pub fn pose(&self) -> Pose {
        Pose {
            position: self.position,
            yaw: self.yaw,
            pitch: self.pitch,
            eye_height: self.eye_height,
            grounded: self.grounded,
            vy: self.vy,
        }
    }

    /// Advance the body one fixed tick against the floor. Deterministic in
    /// `dt` and the held-key set: identical inputs yield an identical pose.
    pub fn step(&mut self, dt: f32, ground: &Ground) {
        // Crouch: the eye sinks toward crouch height; grounded follow lowers the
        // camera with it.
        let crouch = self.keys.contains(&Key::Crouch);
        let target_eye = if crouch {
            self.params.eye_crouch
        } else {
            self.params.eye_stand
        };
        self.eye_height += (target_eye - self.eye_height) * (dt * self.params.eye_damp).min(1.0);
        if !self.keys.contains(&Key::Jump) {
            self.jump_locked = false;
        }

        let speed = if crouch {
            self.params.crouch_speed
        } else if self.keys.contains(&Key::Run) {
            self.params.run_speed
        } else {
            self.params.walk_speed
        };

        // Look-relative move frame on the ground plane (pitch does not tilt
        // walking — this is a body, not a fly-cam).
        let forward = Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos());
        let right = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());
        let mut wish = Vec3::ZERO;
        if self.keys.contains(&Key::Forward) {
            wish += forward;
        }
        if self.keys.contains(&Key::Back) {
            wish -= forward;
        }
        if self.keys.contains(&Key::Right) {
            wish += right;
        }
        if self.keys.contains(&Key::Left) {
            wish -= right;
        }
        if wish.length_squared() > 0.0 {
            wish = wish.normalize() * speed;
        }
        let blend = (dt * self.params.move_damp).min(1.0);
        self.velocity.x += (wish.x - self.velocity.x) * blend;
        self.velocity.z += (wish.z - self.velocity.z) * blend;
        self.position.x += self.velocity.x * dt;
        self.position.z += self.velocity.z * dt;

        // Fell out of the world → return to the last safe ground.
        if self.position.y < self.params.void_y {
            match self.last_safe {
                Some(safe) => self.position = safe,
                None => self.respawn(),
            }
            self.velocity = Vec3::ZERO;
            self.vy = 0.0;
            self.grounded = false;
            return;
        }

        let x = self.position.x;
        let z = self.position.z;
        let feet = self.position.y - self.eye_height;
        let ground_y = ground.height_at_gated(
            x,
            z,
            self.position.y + 1e-3,
            self.params.contact_radius,
            contact_tolerance(self.params.contact_radius),
        );

        match ground_y {
            Some(g) if feet <= g + self.params.ground_snap && self.vy <= 0.0 => {
                // Grounded — Space launches, otherwise snap to the floor.
                if self.keys.contains(&Key::Jump) && !self.jump_locked {
                    self.jump_locked = true;
                    self.vy = self.params.jump_speed;
                    self.position.y += self.vy * dt;
                    self.grounded = false;
                } else {
                    self.grounded = true;
                    self.vy = 0.0;
                    let follow = (dt * self.params.ground_follow).min(1.0);
                    self.position.y += (g + self.eye_height - self.position.y) * follow;
                    self.last_safe = Some(Vec3::new(
                        self.position.x,
                        g + self.eye_height,
                        self.position.z,
                    ));
                }
            }
            _ => {
                // Airborne: gravity, clamped to terminal, caught by the floor.
                self.grounded = false;
                self.vy = (self.vy - self.params.gravity * dt).max(-self.params.terminal);
                self.position.y += self.vy * dt;
                if let Some(g) = ground_y
                    && self.position.y - self.eye_height <= g
                {
                    self.position.y = g + self.eye_height;
                    self.vy = 0.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat 40×40 floor at y=0, centred on the origin — two big triangles.
    fn flat_floor() -> Ground {
        Ground::from_positions(&[
            [-20.0, 0.0, -20.0],
            [20.0, 0.0, -20.0],
            [20.0, 0.0, 20.0],
            [-20.0, 0.0, -20.0],
            [20.0, 0.0, 20.0],
            [-20.0, 0.0, 20.0],
        ])
    }

    fn test_params() -> PlayerParams {
        PlayerParams {
            gravity: 9.81,
            terminal: 55.0,
            jump_speed: 5.0,
            walk_speed: 6.0,
            run_speed: 14.0,
            crouch_speed: 3.0,
            eye_stand: 1.7,
            eye_crouch: 1.0,
            mouse_sensitivity: 0.0022,
            move_damp: 10.0,
            eye_damp: 12.0,
            ground_follow: 12.0,
            ground_snap: 0.35,
            void_y: -120.0,
            pitch_limit: 1.45,
            contact_radius: DEFAULT_CONTACT_RADIUS,
        }
    }

    const DT: f32 = 1.0 / 60.0;

    #[test]
    fn floor_query_ignores_walls_and_finds_top() {
        // A flat floor triangle plus a vertical wall quad — only the floor counts.
        let ground = Ground::from_positions(&[
            [-5.0, 0.0, -5.0],
            [5.0, 0.0, -5.0],
            [5.0, 0.0, 5.0],
            // vertical wall in the xz column of the origin
            [-1.0, 0.0, 0.0],
            [-1.0, 5.0, 0.0],
            [1.0, 5.0, 0.0],
        ]);
        assert_eq!(ground.triangle_count(), 1);
        let y = ground
            .height_at(0.0, -1.0, 10.0)
            .expect("floor under origin");
        assert!((y - 0.0).abs() < 1e-5, "floor y {y}");
    }

    #[test]
    fn gravity_integration_is_deterministic() {
        let render = |seed_keys: &[Key]| -> Vec<Pose> {
            let mut player = Player::new(test_params(), Vec3::new(0.0, 7.0, 0.0), 0.0);
            for &k in seed_keys {
                player.keys.insert(k);
            }
            let ground = flat_floor();
            (0..180)
                .map(|_| {
                    player.step(DT, &ground);
                    player.pose()
                })
                .collect()
        };
        let a = render(&[Key::Forward]);
        let b = render(&[Key::Forward]);
        // Byte-identical: compare the raw bit patterns of every pose field.
        for (pa, pb) in a.iter().zip(&b) {
            assert_eq!(
                pa.position.to_array().map(f32::to_bits),
                pb.position.to_array().map(f32::to_bits)
            );
            assert_eq!(pa.yaw.to_bits(), pb.yaw.to_bits());
            assert_eq!(pa.vy.to_bits(), pb.vy.to_bits());
            assert_eq!(pa.eye_height.to_bits(), pb.eye_height.to_bits());
        }
    }

    #[test]
    fn ground_clamp_settles_at_eye_height_and_never_sinks() {
        let mut player = Player::new(test_params(), Vec3::new(0.0, 7.0, 0.0), 0.0);
        let ground = flat_floor();
        let mut min_feet = f32::INFINITY;
        for _ in 0..180 {
            player.step(DT, &ground);
            min_feet = min_feet.min(player.position.y - player.eye_height);
        }
        // Rest at floor + eye height, feet at 0, never punched through.
        assert!(player.grounded, "should be grounded after settling");
        assert!(
            (player.position.y - 1.7).abs() < 1e-2,
            "eye {}",
            player.position.y
        );
        assert!(min_feet > -1e-2, "feet sank to {min_feet}");
    }

    #[test]
    fn jump_arc_apex_matches_v_squared_over_two_g() {
        let params = test_params();
        let mut player = Player::new(params, Vec3::new(0.0, 7.0, 0.0), 0.0);
        let ground = flat_floor();
        // Settle to the floor first.
        for _ in 0..180 {
            player.step(DT, &ground);
        }
        let rest = player.position.y;
        // Hold jump for one launch, then coast.
        player.keys.insert(Key::Jump);
        player.step(DT, &ground);
        player.keys.remove(&Key::Jump);
        let mut apex = player.position.y;
        for _ in 0..240 {
            player.step(DT, &ground);
            apex = apex.max(player.position.y);
        }
        let expected = params.jump_speed * params.jump_speed / (2.0 * params.gravity);
        let rise = apex - rest;
        assert!(
            (rise - expected).abs() < expected * 0.12,
            "jump rise {rise} vs expected {expected}"
        );
    }

    #[test]
    fn walk_covers_speed_times_time_forward() {
        let mut player = Player::new(test_params(), Vec3::new(0.0, 7.0, 0.0), 0.0);
        let ground = flat_floor();
        for _ in 0..180 {
            player.step(DT, &ground);
        }
        let start = player.position;
        player.keys.insert(Key::Forward);
        let ticks = 60;
        for _ in 0..ticks {
            player.step(DT, &ground);
        }
        let moved = player.position - start;
        // yaw 0 → forward is −Z; nothing sideways.
        assert!(moved.x.abs() < 1e-2, "drifted x {}", moved.x);
        assert!(moved.z < 0.0, "walked the wrong way: {}", moved.z);
        let ideal = test_params().walk_speed * DT * ticks as f32; // 6 m
        let distance = -moved.z;
        // The velocity ramp costs ~0.5 m; within 25% of the ideal is the bar.
        assert!(
            distance > ideal * 0.75 && distance < ideal * 1.05,
            "walked {distance} m vs ideal {ideal} m"
        );
    }
}
