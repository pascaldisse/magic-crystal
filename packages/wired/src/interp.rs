//! Client-side presence interpolation — the cure for the reference engine's
//! teleport-stutter, wired into OUR client from day one.
//!
//! # The defect it kills
//! Remote presences and server-controlled entities arrive as discrete
//! server-tick state slams: the transform jumps to the newest value the
//! instant a frame lands, so at a 60Hz render against a 10Hz feed the body
//! freezes for five frames then teleports. Client-authored decoratives look
//! smooth only because THEY move locally every frame.
//!
//! # The fix (Valve/Quake-style entity interpolation)
//! Every remote transform is buffered with its arrival time. The consumer
//! samples a RENDER CLOCK that lags the newest arrival by [`InterpConfig::delay`]
//! (default two server ticks), and reads a position *between* the two buffered
//! snapshots that bracket it — continuous motion at any render rate. Gaps
//! (a dropped frame) extrapolate from the last segment's velocity up to
//! [`InterpConfig::extrapolation_cap`], then hold.
//!
//! The own presence is NEVER interpolated: the local client owns its own body
//! (local authority), so it reads the raw [`crate::WorldView`], not this buffer.
//!
//! Pure and deterministic: every time is an explicit `f64` seconds supplied by
//! the caller (no wall clock inside), so the same snapshot stream sampled at the
//! same query times yields byte-identical output.

use crystal::{EntityDoc, EntityMap};
use serde_json::Number;
use std::collections::{BTreeMap, VecDeque};
use std::f64::consts::TAU;

/// A sampled transform for one entity: world-space position and facing yaw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// World-space position `[x, y, z]`.
    pub position: [f64; 3],
    /// Facing angle in radians (interpolated on the shortest arc).
    pub yaw: f64,
}

/// Interpolation tuning. All durations are seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpConfig {
    /// How far the render clock lags the newest arrival. The sampler always
    /// reads `now - delay`, so it interpolates between two arrived snapshots
    /// instead of chasing the newest. Default = `2 * tick`.
    pub delay: f64,
    /// Max seconds past the newest snapshot the sampler will extrapolate
    /// (from the last segment's velocity) before holding the last position.
    pub extrapolation_cap: f64,
    /// Server tick period (seconds) — the cadence remote state arrives at.
    /// Only used to derive the default `delay`.
    pub tick: f64,
}

impl InterpConfig {
    /// Config for a server ticking at `hz` snapshots/second: `delay = 2/hz`,
    /// `extrapolation_cap = 3/hz` (three ticks of coast before freezing).
    pub fn for_tick_hz(hz: f64) -> Self {
        let tick = if hz > 0.0 { 1.0 / hz } else { 0.1 };
        Self {
            delay: 2.0 * tick,
            extrapolation_cap: 3.0 * tick,
            tick,
        }
    }
}

impl Default for InterpConfig {
    /// 10Hz assumption: 200ms delay, 300ms extrapolation cap.
    fn default() -> Self {
        Self::for_tick_hz(10.0)
    }
}

/// One entity's arrival-ordered snapshot buffer.
#[derive(Clone, Debug, Default)]
struct Track {
    /// Ascending by time: `(arrival_seconds, sample)`.
    samples: VecDeque<(f64, Sample)>,
}

/// A cap on retained snapshots per track, so a long-lived link can't grow the
/// buffer without bound even if the time-window prune keeps everything.
const MAX_SAMPLES: usize = 4096;

/// Buffers remote transforms and samples an interpolated view of them.
#[derive(Clone, Debug)]
pub struct Interpolator {
    config: InterpConfig,
    /// The own presence id — never buffered, never interpolated.
    own: Option<String>,
    tracks: BTreeMap<String, Track>,
}

impl Interpolator {
    /// New interpolator. `own` is the local presence id to exclude (local
    /// authority); pass `None` to interpolate every entity.
    pub fn new(config: InterpConfig, own: Option<String>) -> Self {
        Self {
            config,
            own,
            tracks: BTreeMap::new(),
        }
    }

    /// The active tuning.
    pub fn config(&self) -> InterpConfig {
        self.config
    }

    /// Set the own presence id (excluded from buffering).
    pub fn set_own(&mut self, own: Option<String>) {
        self.own = own;
    }

    /// Drop every buffer — used on an atomic resync so stale history from the
    /// previous session can never bleed into the new one.
    pub fn clear(&mut self) {
        self.tracks.clear();
    }

    /// Drop one entity's buffer (it despawned).
    pub fn forget(&mut self, id: &str) {
        self.tracks.remove(id);
    }

    /// Number of tracked (buffered) entities.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// True when no entity is buffered.
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Buffer one transform for `id` arriving at `time` (seconds). The own
    /// presence is ignored. Out-of-order or duplicate `time`s are dropped so
    /// the buffer stays strictly ascending (server ticks are monotonic).
    pub fn record(&mut self, id: &str, sample: Sample, time: f64) {
        if self.own.as_deref() == Some(id) {
            return;
        }
        let window = self.retain_window();
        let track = self.tracks.entry(id.to_string()).or_default();
        if let Some(&(last_t, _)) = track.samples.back() {
            if time <= last_t {
                return; // not newer — ignore (monotonic ticks only)
            }
        }
        track.samples.push_back((time, sample));
        // Prune history older than the sampler could ever need.
        while track.samples.len() > 2 {
            let (front_t, _) = track.samples[0];
            if time - front_t > window {
                track.samples.pop_front();
            } else {
                break;
            }
        }
        while track.samples.len() > MAX_SAMPLES {
            track.samples.pop_front();
        }
    }

    /// Buffer every entity in `entities` that carries a position, at `time`.
    /// Convenience for feeding a whole [`EntityMap`] each frame.
    pub fn observe(&mut self, entities: &EntityMap, time: f64) {
        for (id, doc) in entities {
            if let Some(sample) = sample_of(doc) {
                self.record(id, sample, time);
            }
        }
    }

    /// Sampled position/facing for `id` at render time `now` (seconds) —
    /// interpolated on the render clock `now - delay`. `None` if `id` is not
    /// buffered. This is the stutter-free view the consumer renders.
    pub fn sample(&self, id: &str, now: f64) -> Option<Sample> {
        let track = self.tracks.get(id)?;
        let s = &track.samples;
        if s.is_empty() {
            return None;
        }
        let target = now - self.config.delay;
        let (first_t, first_s) = s[0];
        if target <= first_t {
            return Some(first_s); // before the buffer — hold oldest
        }
        let (last_t, last_s) = *s.back().unwrap();
        if target >= last_t {
            return Some(self.extrapolate(s, last_t, last_s, target));
        }
        // Bracket the target between two arrived snapshots and lerp.
        for i in 0..s.len() - 1 {
            let (t0, s0) = s[i];
            let (t1, s1) = s[i + 1];
            if target >= t0 && target <= t1 {
                let span = t1 - t0;
                let f = if span > 0.0 {
                    (target - t0) / span
                } else {
                    0.0
                };
                return Some(lerp_sample(s0, s1, f));
            }
        }
        Some(last_s)
    }

    /// Every buffered entity, interpolated at `now`.
    pub fn sampled_view(&self, now: f64) -> BTreeMap<String, Sample> {
        self.tracks
            .keys()
            .filter_map(|id| self.sample(id, now).map(|s| (id.clone(), s)))
            .collect()
    }

    /// The RAW slam value the reference engine would show: the newest snapshot
    /// with arrival `<= now`, with no delay and no interpolation. Exposed so a
    /// test can measure smooth-vs-slam and prove the buffer discriminates.
    pub fn slam(&self, id: &str, now: f64) -> Option<Sample> {
        let track = self.tracks.get(id)?;
        let mut out = None;
        for (t, s) in &track.samples {
            if *t <= now {
                out = Some(*s);
            } else {
                break;
            }
        }
        out.or_else(|| track.samples.front().map(|(_, s)| *s))
    }

    /// How much history a track keeps: enough to bracket the oldest render
    /// clock the sampler will ask for, plus a coast/extrapolation margin.
    fn retain_window(&self) -> f64 {
        self.config.delay + self.config.extrapolation_cap + 8.0 * self.config.tick + 0.5
    }

    /// Coast past the newest snapshot using the last segment's velocity,
    /// clamped to `extrapolation_cap`, then hold.
    fn extrapolate(
        &self,
        s: &VecDeque<(f64, Sample)>,
        last_t: f64,
        last_s: Sample,
        target: f64,
    ) -> Sample {
        let over = (target - last_t).clamp(0.0, self.config.extrapolation_cap);
        if over <= 0.0 || s.len() < 2 {
            return last_s;
        }
        let (prev_t, prev_s) = s[s.len() - 2];
        let dt = last_t - prev_t;
        if dt <= 0.0 {
            return last_s;
        }
        let mut out = last_s;
        for k in 0..3 {
            let v = (last_s.position[k] - prev_s.position[k]) / dt;
            out.position[k] = last_s.position[k] + v * over;
        }
        let dyaw = shortest_arc(last_s.yaw - prev_s.yaw) / dt;
        out.yaw = last_s.yaw + dyaw * over;
        out
    }
}

/// Extract a [`Sample`] from an entity doc: `transform.position` and
/// `presence.yaw`. `None` when the entity has no position (nothing to place).
pub fn sample_of(doc: &EntityDoc) -> Option<Sample> {
    let position = doc.transform.as_ref()?.position.as_ref()?;
    if position.len() < 3 {
        return None;
    }
    let position = [num(&position[0]), num(&position[1]), num(&position[2])];
    let yaw = doc
        .presence
        .as_ref()
        .and_then(|p| p.extra.get("yaw"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    Some(Sample { position, yaw })
}

fn num(n: &Number) -> f64 {
    n.as_f64().unwrap_or(0.0)
}

/// Wrap an angle delta into `[-PI, PI]` — the shortest signed rotation.
fn shortest_arc(delta: f64) -> f64 {
    let mut d = delta % TAU;
    if d > TAU / 2.0 {
        d -= TAU;
    } else if d < -TAU / 2.0 {
        d += TAU;
    }
    d
}

/// Interpolated yaw along the shortest arc (the "slerp" for a single angle).
fn lerp_yaw(a: f64, b: f64, t: f64) -> f64 {
    a + shortest_arc(b - a) * t
}

fn lerp_sample(a: Sample, b: Sample, t: f64) -> Sample {
    Sample {
        position: [
            a.position[0] + (b.position[0] - a.position[0]) * t,
            a.position[1] + (b.position[1] - a.position[1]) * t,
            a.position[2] + (b.position[2] - a.position[2]) * t,
        ],
        yaw: lerp_yaw(a.yaw, b.yaw, t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: f64, yaw: f64) -> Sample {
        Sample {
            position: [x, 0.0, 0.0],
            yaw,
        }
    }

    /// A moving entity streamed at 10Hz, sampled at 60Hz, is CONTINUOUS: no
    /// render step exceeds the derived smooth bound — and the raw slam it
    /// replaces takes steps an order of magnitude larger. The gate is the gap
    /// between the two numbers.
    #[test]
    fn smooth_vs_slam_bound() {
        let cfg = InterpConfig::for_tick_hz(10.0);
        let mut interp = Interpolator::new(cfg, None);

        // 5 m/s along x, snapshots every 0.1s for 2s.
        let speed = 5.0;
        let feed_dt = 0.1;
        for k in 0..=20 {
            let t = k as f64 * feed_dt;
            interp.record("mob", s(speed * t, 0.0), t);
        }

        // Sample at 60Hz across the render-valid window (delay .. newest).
        let render_dt = 1.0 / 60.0;
        let start = cfg.delay + render_dt; // first frame with a full bracket
        let end = 2.0; // newest arrival
        let mut prev_smooth: Option<f64> = None;
        let mut prev_slam: Option<f64> = None;
        let mut max_smooth_step = 0.0_f64;
        let mut max_slam_step = 0.0_f64;
        let mut now = start;
        while now <= end {
            let smooth = interp.sample("mob", now).unwrap().position[0];
            let slam = interp.slam("mob", now - cfg.delay).unwrap().position[0];
            if let Some(p) = prev_smooth {
                max_smooth_step = max_smooth_step.max((smooth - p).abs());
            }
            if let Some(p) = prev_slam {
                max_slam_step = max_slam_step.max((slam - p).abs());
            }
            prev_smooth = Some(smooth);
            prev_slam = Some(slam);
            now += render_dt;
        }

        // Derived bound: one render frame of motion at the feed speed, plus a
        // hair for float error. Smooth motion cannot step further than this.
        let bound = speed * render_dt + 1e-9;
        println!("smooth_max_step={max_smooth_step:.6} bound={bound:.6} slam_max_step={max_slam_step:.6}");
        assert!(
            max_smooth_step <= bound,
            "interpolated step {max_smooth_step} exceeded smooth bound {bound}"
        );
        // The slam baseline jumps a whole feed interval at each 10Hz boundary.
        assert!(
            max_slam_step > bound * 3.0,
            "slam step {max_slam_step} should dwarf the smooth bound {bound}"
        );
    }

    /// The render clock lags the newest arrival by exactly `delay`, so at a
    /// query time the sampler reads the position the entity held `delay` ago.
    #[test]
    fn delay_param_honored_exactly() {
        let cfg = InterpConfig {
            delay: 0.2,
            extrapolation_cap: 0.3,
            tick: 0.1,
        };
        let mut interp = Interpolator::new(cfg, None);
        // x == t exactly (1 m/s).
        for k in 0..=20 {
            let t = k as f64 * 0.1;
            interp.record("m", s(t, 0.0), t);
        }
        // At now=1.0 the render clock is 0.8 → x should be 0.8.
        let x = interp.sample("m", 1.0).unwrap().position[0];
        assert!((x - 0.8).abs() < 1e-9, "delay not honored: x={x}");
    }

    /// Past the newest snapshot the sampler coasts on velocity up to the cap,
    /// then freezes — it never chases the newest arrival past the cap.
    #[test]
    fn extrapolation_cap_honored() {
        let cfg = InterpConfig {
            delay: 0.0,
            extrapolation_cap: 0.05,
            tick: 0.1,
        };
        let mut interp = Interpolator::new(cfg, None);
        interp.record("m", s(0.0, 0.0), 0.0);
        interp.record("m", s(1.0, 0.0), 0.1); // 10 m/s
                                              // 0.2s past the last snapshot, but cap is 0.05s → coast 0.05*10 = 0.5.
        let x = interp.sample("m", 0.3).unwrap().position[0];
        assert!((x - 1.5).abs() < 1e-9, "cap not honored: x={x}");
        // Right at the cap edge, same value (frozen beyond).
        let x2 = interp.sample("m", 1.0).unwrap().position[0];
        assert!((x2 - 1.5).abs() < 1e-9, "beyond cap should hold: x={x2}");
    }

    /// Same snapshot stream + same query times ⇒ byte-identical sampled stream.
    #[test]
    fn determinism_byte_identical() {
        let cfg = InterpConfig::for_tick_hz(10.0);
        let feed: Vec<(f64, Sample)> = (0..=30)
            .map(|k| {
                let t = k as f64 * 0.1;
                (t, s((t * 3.0).sin() * 4.0, (t * 1.3).cos()))
            })
            .collect();
        let run = || {
            let mut interp = Interpolator::new(cfg, None);
            for (t, sample) in &feed {
                interp.record("m", *sample, *t);
            }
            let mut out = String::new();
            let mut now = cfg.delay;
            while now <= 3.0 {
                let v = interp.sample("m", now).unwrap();
                out.push_str(&format!(
                    "{:.17},{:.17},{:.17},{:.17}\n",
                    v.position[0], v.position[1], v.position[2], v.yaw
                ));
                now += 1.0 / 60.0;
            }
            out
        };
        assert_eq!(run(), run(), "sampled stream is not deterministic");
    }

    /// Own presence is never buffered — local authority.
    #[test]
    fn own_presence_never_buffered() {
        let mut interp = Interpolator::new(InterpConfig::default(), Some("me".to_string()));
        interp.record("me", s(1.0, 0.0), 0.0);
        interp.record("other", s(2.0, 0.0), 0.0);
        assert!(interp.sample("me", 1.0).is_none());
        assert!(interp.sample("other", 1.0).is_some());
    }

    /// Yaw crosses the +/-PI seam on the short arc, never the long way round.
    #[test]
    fn yaw_slerps_shortest_arc() {
        let mut interp = Interpolator::new(
            InterpConfig {
                delay: 0.0,
                extrapolation_cap: 0.0,
                tick: 0.1,
            },
            None,
        );
        // 170° → -170° is a 20° short hop across the seam, not 340°.
        let a = 170.0_f64.to_radians();
        let b = (-170.0_f64).to_radians();
        interp.record("m", s(0.0, a), 0.0);
        interp.record("m", s(0.0, b), 1.0);
        let mid = interp.sample("m", 0.5).unwrap().yaw;
        // Midpoint should sit at +/-180°, i.e. |yaw| ~ PI, NOT near 0.
        assert!(
            mid.abs() > 175.0_f64.to_radians(),
            "yaw took the long way: mid={mid}"
        );
    }

    /// Out-of-order / duplicate arrival times are dropped (monotonic ticks).
    #[test]
    fn rejects_non_monotonic_arrivals() {
        let mut interp = Interpolator::new(InterpConfig::default(), None);
        interp.record("m", s(0.0, 0.0), 1.0);
        interp.record("m", s(5.0, 0.0), 0.5); // older — dropped
        interp.record("m", s(9.0, 0.0), 1.0); // duplicate time — dropped
                                              // Only the first survived; sampling far ahead holds it.
        let v = interp.sample("m", 100.0).unwrap();
        assert_eq!(v.position[0], 0.0);
    }

    /// `observe` folds a whole entity map, extracting position + yaw.
    #[test]
    fn observe_extracts_from_entity_map() {
        use serde_json::json;
        let mut entities = EntityMap::new();
        let doc: EntityDoc = serde_json::from_value(json!({
            "transform": { "position": [3.0, 1.0, 2.0] },
            "presence": { "kind": "player", "yaw": 0.75 },
        }))
        .unwrap();
        entities.insert("p".into(), doc);
        let mut interp = Interpolator::new(InterpConfig::default(), None);
        interp.observe(&entities, 0.0);
        let v = interp.sample("p", 0.0).unwrap();
        assert_eq!(v.position, [3.0, 1.0, 2.0]);
        assert!((v.yaw - 0.75).abs() < 1e-12);
    }
}
