//! Presence bodies — remote wired presences rendered as skinned vessel bodies
//! (WIRED N2: THE WIRED BECOMES VISIBLE).
//!
//! A live wired session streams remote presences (id + position/yaw) through the
//! interpolation buffer. This layer turns each such presence into a real BODY in
//! the traced world: every presence drives a [`BodyInstance`] composed from a
//! DEFAULT preset (a parameter — e.g. the pale biped), placed at the presence's
//! interpolated pose and skinned per tick. The bodies' triangles are spliced
//! into the SAME dynamic BVH the realm's living layer and physics feed
//! ([`crate::scene::RenderScene::dynamic_leaf_triangles`] extends them in), so a
//! remote body is real to the light and the camera.
//!
//! # Deterministic add / remove
//! [`PresenceBodies::sync`] is the sole entry: a presence in the new pose set
//! but not yet embodied gets a fresh body (join); an embodied id absent from the
//! set is dropped (reap); survivors are driven to their new pose. Bodies are
//! keyed in a [`BTreeMap`], so [`PresenceBodies::leaf_triangles`] emits them in a
//! fixed id order — identical pose streams yield byte-identical triangle soup.
//!
//! # Gait from motion
//! The commanded gait speed is DERIVED from the presence's horizontal position
//! delta over the drive `dt` — a moving presence walks, a still one idles. The
//! own presence is never here (the local client owns its body); this layer only
//! embodies the REMOTE presences the interp buffer holds.

use std::collections::BTreeMap;

use crate::player::Ground;
use crate::scene::{BodyInstance, LeafTriangle};

/// One presence's world pose this tick: interpolated position and facing yaw —
/// exactly what the wired interpolation buffer samples (`interp::Sample`), lifted
/// here so this crate stays free of the wired dependency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PresencePose {
    /// World-space position `[x, y, z]` (the presence transform).
    pub position: [f64; 3],
    /// Facing angle in radians (yaw about +Y).
    pub yaw: f64,
}

impl PresencePose {
    /// A pose from an explicit position and yaw.
    pub fn new(position: [f64; 3], yaw: f64) -> Self {
        Self { position, yaw }
    }
}

/// The set of remote presences currently embodied, each a [`BodyInstance`] the
/// traced BVH splices in. Driven per tick by [`PresenceBodies::sync`].
#[derive(Debug)]
pub struct PresenceBodies {
    /// The preset every presence body is composed from (the parameter).
    default_preset: String,
    /// Drive delta (seconds) — the span between two `sync` calls, used to derive
    /// the gait speed from the position delta.
    dt: f32,
    /// id → embodied body, ordered for deterministic triangle emission.
    bodies: BTreeMap<String, BodyInstance>,
    /// id → last synced position, for the per-tick speed derivation.
    last: BTreeMap<String, [f64; 3]>,
}

impl PresenceBodies {
    /// A fresh, empty embodiment layer. `default_preset` names the vessel preset
    /// every presence body is composed from; `dt` is the seconds between `sync`
    /// calls (the render/feed cadence), used to turn position deltas into gait
    /// speed. A `dt <= 0` disables the derived gait (bodies always idle).
    pub fn new(default_preset: impl Into<String>, dt: f32) -> Self {
        Self {
            default_preset: default_preset.into(),
            dt,
            bodies: BTreeMap::new(),
            last: BTreeMap::new(),
        }
    }

    /// Reconcile the embodied set to `poses` (id → interpolated pose) and drive
    /// every survivor. DETERMINISTIC: joins add a fresh default-preset body,
    /// reaps remove the gone ids, survivors are re-placed at their new pose with
    /// a gait speed derived from the horizontal position delta over `dt`. `floor`
    /// grounds the bodies (their feet rest on the surface); `None` keeps authored
    /// y. An unknown default preset is surfaced (never invented). Call once per
    /// render tick before splicing the dynamic BVH.
    pub fn sync(
        &mut self,
        poses: &BTreeMap<String, PresencePose>,
        floor: Option<&Ground>,
    ) -> Result<(), String> {
        // Reap: drop every embodied id no longer present.
        self.bodies.retain(|id, _| poses.contains_key(id));
        self.last.retain(|id, _| poses.contains_key(id));

        for (id, pose) in poses {
            match self.bodies.get_mut(id) {
                Some(body) => {
                    // Survivor: derive the gait speed from the position delta.
                    let speed = self
                        .last
                        .get(id)
                        .map(|prev| ground_speed(*prev, pose.position, self.dt))
                        .unwrap_or(0.0);
                    body.drive(pose.position, pose.yaw, speed, floor);
                }
                None => {
                    // Join: a fresh default-preset body at the presence pose.
                    let body = BodyInstance::from_preset(
                        id.clone(),
                        &self.default_preset,
                        pose.position,
                        pose.yaw,
                        floor,
                    )?;
                    self.bodies.insert(id.clone(), body);
                }
            }
            self.last.insert(id.clone(), pose.position);
        }
        Ok(())
    }

    /// Every embodied presence's skinned world-space triangles, in id order — the
    /// DYNAMIC partition this layer splices into the traced BVH. Byte-identical
    /// for identical pose streams (deterministic id ordering + pure skinning).
    pub fn leaf_triangles(&self) -> Vec<LeafTriangle> {
        let mut out = Vec::new();
        for body in self.bodies.values() {
            out.extend_from_slice(&body.world_tris);
        }
        out
    }

    /// The embodied body for `id`, if any (its `world_tris`, model, pose, gait).
    pub fn body(&self, id: &str) -> Option<&BodyInstance> {
        self.bodies.get(id)
    }

    /// The embodied presence ids, in order.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.bodies.keys().map(String::as_str)
    }

    /// Number of embodied presences.
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// True when no presence is embodied.
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// The default preset every presence body is composed from.
    pub fn default_preset(&self) -> &str {
        &self.default_preset
    }
}

/// Horizontal ground speed (m/s) between two positions over `dt` seconds — the
/// gait command. Vertical motion is ignored (feet-on-floor gait); `dt <= 0`
/// yields 0 (idle).
fn ground_speed(prev: [f64; 3], now: [f64; 3], dt: f32) -> f32 {
    if dt <= 0.0 {
        return 0.0;
    }
    let dx = (now[0] - prev[0]) as f32;
    let dz = (now[2] - prev[2]) as f32;
    (dx * dx + dz * dz).sqrt() / dt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poses(entries: &[(&str, [f64; 3], f64)]) -> BTreeMap<String, PresencePose> {
        entries
            .iter()
            .map(|(id, p, y)| (id.to_string(), PresencePose::new(*p, *y)))
            .collect()
    }

    /// A join adds a body; a reap removes it — the embodied set tracks the pose
    /// set exactly, and the triangle soup follows (non-empty ⇒ empty).
    #[test]
    fn join_and_reap_track_the_pose_set() {
        let mut bodies = PresenceBodies::new("nari", 1.0 / 60.0);
        assert!(bodies.is_empty());

        bodies
            .sync(&poses(&[("a", [0.0, 0.0, 0.0], 0.0)]), None)
            .unwrap();
        assert_eq!(bodies.len(), 1);
        assert!(!bodies.leaf_triangles().is_empty(), "joined body has tris");

        // Second presence joins.
        bodies
            .sync(
                &poses(&[("a", [0.0, 0.0, 0.0], 0.0), ("b", [2.0, 0.0, 0.0], 0.0)]),
                None,
            )
            .unwrap();
        assert_eq!(bodies.len(), 2);
        assert_eq!(bodies.ids().collect::<Vec<_>>(), vec!["a", "b"]);

        // 'a' reaped, 'b' survives.
        bodies
            .sync(&poses(&[("b", [2.0, 0.0, 0.0], 0.0)]), None)
            .unwrap();
        assert_eq!(bodies.len(), 1);
        assert!(bodies.body("a").is_none(), "reaped body is gone");
        assert!(bodies.body("b").is_some());

        // All reaped → empty splice.
        bodies.sync(&poses(&[]), None).unwrap();
        assert!(bodies.is_empty());
        assert!(bodies.leaf_triangles().is_empty(), "empty splice");
    }

    /// The splice is byte-deterministic: two independent layers fed the SAME
    /// pose stream emit byte-identical triangle soup (add/remove + skinning are
    /// deterministic, id-ordered).
    #[test]
    fn splice_byte_deterministic() {
        let stream = [
            poses(&[("p1", [0.0, 0.0, 0.0], 0.0)]),
            poses(&[("p1", [0.1, 0.0, 0.0], 0.1), ("p2", [3.0, 0.0, 1.0], 1.0)]),
            poses(&[("p1", [0.2, 0.0, 0.0], 0.2), ("p2", [3.1, 0.0, 1.0], 1.1)]),
            poses(&[("p2", [3.2, 0.0, 1.0], 1.2)]),
        ];
        let run = || {
            let mut bodies = PresenceBodies::new("nari", 1.0 / 60.0);
            let mut bytes = Vec::new();
            for frame in &stream {
                bodies.sync(frame, None).unwrap();
                for tri in bodies.leaf_triangles() {
                    for corner in &tri.positions {
                        for v in corner {
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                }
            }
            bytes
        };
        assert_eq!(run(), run(), "presence splice is not byte-deterministic");
    }

    /// A moving presence tracks the interpolated position: the body's world
    /// transform follows the pose it was driven to (position tracks the buffer).
    #[test]
    fn body_transform_tracks_the_pose() {
        let mut bodies = PresenceBodies::new("nari", 1.0 / 60.0);
        bodies
            .sync(&poses(&[("m", [0.0, 0.0, 0.0], 0.0)]), None)
            .unwrap();
        bodies
            .sync(&poses(&[("m", [5.0, 0.0, -2.0], 0.0)]), None)
            .unwrap();
        let model = bodies.body("m").unwrap().model();
        let translation = model.w_axis;
        assert!(
            (translation.x - 5.0).abs() < 1e-4,
            "x tracks: {}",
            translation.x
        );
        assert!(
            (translation.z + 2.0).abs() < 1e-4,
            "z tracks: {}",
            translation.z
        );
    }

    /// A moving presence walks (non-zero gait), a still one idles — the gait is
    /// derived from motion, not authored.
    #[test]
    fn motion_drives_the_gait() {
        use sama::Gait;
        let mut bodies = PresenceBodies::new("nari", 1.0 / 60.0);
        bodies
            .sync(&poses(&[("w", [0.0, 0.0, 0.0], 0.0)]), None)
            .unwrap();
        // Drive it a long way each tick for many ticks → the walk state engages.
        for k in 1..=60 {
            let x = k as f64 * 0.1; // 6 m/s at 60Hz
            bodies
                .sync(&poses(&[("w", [x, 0.0, 0.0], 0.0)]), None)
                .unwrap();
        }
        assert!(
            bodies.body("w").unwrap().commanded_speed() > 0.0,
            "moving ⇒ speed"
        );
        assert_ne!(
            bodies.body("w").unwrap().gait(),
            Gait::Idle,
            "moving ⇒ walks"
        );
    }
}
