//! Hierarchical skeleton model and the parametric body generator.
//!
//! One generator, [`Skeleton::from_params`], spans the whole morphological
//! range: [`BodyParams::humanoid`] gives an upright biped, [`BodyParams::quadruped`]
//! gives a tailed cat, and any [`BodyParams::lerp`] between them is a valid
//! skeleton. Nothing is hand-authored — proportions are parameters, bone
//! counts are derived (see [`BodyParams::bone_count`]).

use crate::pose::Transform;
use glam::{Quat, Vec3};
use std::f32::consts::{FRAC_PI_2, PI};

/// One bone in the hierarchy.
///
/// `local_bind` is the bone's transform relative to its parent in the rest
/// (bind) pose; `length` is measured along the bone's local +Y axis, so the
/// bone's tip in its own frame is `(0, length, 0)`. `radius` drives the
/// implicit capsule used for skinning.
#[derive(Clone, Debug, PartialEq)]
pub struct Bone {
    /// Human-readable, stable name (e.g. `"L.thigh"`, `"spine.2"`).
    pub name: String,
    /// Index of the parent bone, or `None` for the root.
    pub parent: Option<usize>,
    /// Rest transform relative to the parent frame.
    pub local_bind: Transform,
    /// Bone length along local +Y.
    pub length: f32,
    /// Capsule radius for skinning.
    pub radius: f32,
}

/// Parametric description of a body. All proportions are fractions of `height`
/// (in metres) unless noted; every field has a default (the human value via
/// [`Default`]). Discrete topology is set by `spine_count`, `neck_count` and
/// `tail_segments`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyParams {
    /// Reference height in metres (overall scale).
    pub height: f32,
    /// Number of spine segments between pelvis and chest (>= 1).
    pub spine_count: u32,
    /// Number of neck segments between chest and head (>= 1).
    pub neck_count: u32,
    /// Number of tail segments (0 = no tail).
    pub tail_segments: u32,
    /// Posture from 0 (upright biped) to 1 (horizontal quadruped).
    pub stance: f32,

    /// Pelvis (root) length, fraction of height.
    pub pelvis: f32,
    /// Total spine length, fraction of height (split across `spine_count`).
    pub spine: f32,
    /// Total neck length, fraction of height (split across `neck_count`).
    pub neck: f32,
    /// Head length, fraction of height.
    pub head: f32,
    /// Total tail length, fraction of height (split across `tail_segments`).
    pub tail: f32,

    /// Upper arm / front-upper-leg length, fraction of height.
    pub upper_arm: f32,
    /// Forearm / front-lower-leg length, fraction of height.
    pub forearm: f32,
    /// Hand / front paw length, fraction of height.
    pub hand: f32,
    /// Thigh length, fraction of height.
    pub thigh: f32,
    /// Shank length, fraction of height.
    pub shank: f32,
    /// Foot / hind paw length, fraction of height.
    pub foot: f32,

    /// Full shoulder width, fraction of height.
    pub shoulder_width: f32,
    /// Full hip width, fraction of height.
    pub hip_width: f32,
    /// Base capsule radius, fraction of height.
    pub bone_radius: f32,
}

impl Default for BodyParams {
    /// The default is a human.
    fn default() -> Self {
        Self::humanoid()
    }
}

impl BodyParams {
    /// Preset: an upright human biped (height 1.8 m).
    pub fn humanoid() -> Self {
        Self {
            height: 1.8,
            spine_count: 3,
            neck_count: 1,
            tail_segments: 0,
            stance: 0.0,
            pelvis: 0.06,
            spine: 0.30,
            neck: 0.05,
            head: 0.13,
            tail: 0.0,
            upper_arm: 0.16,
            forearm: 0.14,
            hand: 0.10,
            thigh: 0.245,
            shank: 0.23,
            foot: 0.10,
            shoulder_width: 0.20,
            hip_width: 0.11,
            bone_radius: 0.04,
        }
    }

    /// Preset: a tailed quadruped cat (shoulder height 0.5 m).
    pub fn quadruped() -> Self {
        Self {
            height: 0.5,
            spine_count: 6,
            neck_count: 2,
            tail_segments: 8,
            stance: 1.0,
            pelvis: 0.10,
            spine: 0.55,
            neck: 0.15,
            head: 0.20,
            tail: 0.60,
            upper_arm: 0.28,
            forearm: 0.26,
            hand: 0.06,
            thigh: 0.30,
            shank: 0.28,
            foot: 0.08,
            shoulder_width: 0.16,
            hip_width: 0.16,
            bone_radius: 0.06,
        }
    }

    /// Number of bones a skeleton with these params will have.
    ///
    /// Derived, not frozen: pelvis + spine + neck + head + tail + two 3-bone
    /// arms + two 3-bone legs.
    pub fn bone_count(&self) -> usize {
        1 + self.spine_count as usize
            + self.neck_count as usize
            + 1
            + self.tail_segments as usize
            + 6
            + 6
    }

    /// Length of a single spine segment (0 if `spine_count == 0`).
    pub fn spine_segment_len(&self) -> f32 {
        div_or_zero(self.spine * self.height, self.spine_count)
    }
    /// Length of a single neck segment (0 if `neck_count == 0`).
    pub fn neck_segment_len(&self) -> f32 {
        div_or_zero(self.neck * self.height, self.neck_count)
    }
    /// Length of a single tail segment (0 if `tail_segments == 0`).
    pub fn tail_segment_len(&self) -> f32 {
        div_or_zero(self.tail * self.height, self.tail_segments)
    }
    /// Base capsule radius in metres.
    pub fn radius(&self) -> f32 {
        self.bone_radius * self.height
    }

    /// Linear blend between two parameter sets. Continuous fields lerp; the
    /// discrete counts lerp then round to the nearest integer, so morphology
    /// transitions (human -> cat) stay valid at every sample.
    pub fn lerp(a: &BodyParams, b: &BodyParams, t: f32) -> BodyParams {
        let l = |x: f32, y: f32| x + (y - x) * t;
        let li = |x: u32, y: u32| (x as f32 + (y as f32 - x as f32) * t).round() as u32;
        BodyParams {
            height: l(a.height, b.height),
            spine_count: li(a.spine_count, b.spine_count),
            neck_count: li(a.neck_count, b.neck_count),
            tail_segments: li(a.tail_segments, b.tail_segments),
            stance: l(a.stance, b.stance),
            pelvis: l(a.pelvis, b.pelvis),
            spine: l(a.spine, b.spine),
            neck: l(a.neck, b.neck),
            head: l(a.head, b.head),
            tail: l(a.tail, b.tail),
            upper_arm: l(a.upper_arm, b.upper_arm),
            forearm: l(a.forearm, b.forearm),
            hand: l(a.hand, b.hand),
            thigh: l(a.thigh, b.thigh),
            shank: l(a.shank, b.shank),
            foot: l(a.foot, b.foot),
            shoulder_width: l(a.shoulder_width, b.shoulder_width),
            hip_width: l(a.hip_width, b.hip_width),
            bone_radius: l(a.bone_radius, b.bone_radius),
        }
    }
}

fn div_or_zero(total: f32, count: u32) -> f32 {
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

/// A hierarchical skeleton: bones stored parents-before-children.
#[derive(Clone, Debug, PartialEq)]
pub struct Skeleton {
    /// Bones in topological order (every parent index precedes its children).
    pub bones: Vec<Bone>,
}

impl Skeleton {
    /// Generate the humanoid preset.
    pub fn humanoid() -> Self {
        Self::from_params(&BodyParams::humanoid())
    }

    /// Generate the quadruped preset.
    pub fn quadruped() -> Self {
        Self::from_params(&BodyParams::quadruped())
    }

    /// Number of bones.
    pub fn len(&self) -> usize {
        self.bones.len()
    }
    /// Whether the skeleton has no bones.
    pub fn is_empty(&self) -> bool {
        self.bones.is_empty()
    }

    /// Find a bone index by name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.bones.iter().position(|b| b.name == name)
    }

    /// Validate structural invariants: every parent index precedes its child,
    /// all transforms/lengths/radii are finite, and lengths are non-negative.
    pub fn validate(&self) -> Result<(), String> {
        for (i, b) in self.bones.iter().enumerate() {
            if let Some(p) = b.parent {
                if p >= i {
                    return Err(format!("bone {i} ({}) parent {p} not before it", b.name));
                }
            }
            if !b.length.is_finite() || b.length < 0.0 {
                return Err(format!("bone {i} ({}) bad length {}", b.name, b.length));
            }
            if !b.radius.is_finite() || b.radius < 0.0 {
                return Err(format!("bone {i} ({}) bad radius {}", b.name, b.radius));
            }
            let t = b.local_bind;
            if !t.translation.is_finite() || !quat_is_finite(t.rotation) {
                return Err(format!("bone {i} ({}) non-finite transform", b.name));
            }
        }
        Ok(())
    }

    /// Build a skeleton from parameters. This is the whole generator: one code
    /// path from human to cat, driven entirely by `params`.
    pub fn from_params(params: &BodyParams) -> Skeleton {
        let mut builder = Builder::default();
        let radius = params.radius();

        // Spine bends from vertical (biped) toward horizontal (quadruped) about
        // the local X axis; because we only rotate about X, every bone's local
        // X axis stays aligned with world X, so sideways limb offsets are stable
        // for any stance.
        let spine_tilt = Quat::from_rotation_x(params.stance * FRAC_PI_2);
        let down = Quat::from_rotation_x(PI); // local +Y -> points down
        let tail_dir = Quat::from_rotation_x(PI - params.stance * FRAC_PI_2);

        // Root: pelvis at origin.
        let pelvis_len = params.pelvis * params.height;
        let pelvis = builder.push(Bone {
            name: "pelvis".into(),
            parent: None,
            local_bind: Transform::IDENTITY,
            length: pelvis_len,
            radius,
        });

        // Spine chain up from the pelvis tip.
        let spine_seg = params.spine_segment_len();
        let mut prev = pelvis;
        let mut prev_len = pelvis_len;
        for s in 0..params.spine_count {
            let rot = if s == 0 { spine_tilt } else { Quat::IDENTITY };
            let bone = builder.push(Bone {
                name: format!("spine.{s}"),
                parent: Some(prev),
                local_bind: Transform::new(Vec3::new(0.0, prev_len, 0.0), rot),
                length: spine_seg,
                radius,
            });
            prev = bone;
            prev_len = spine_seg;
        }
        let chest = prev;
        let chest_len = prev_len;

        // Neck chain + head from the chest tip.
        let neck_seg = params.neck_segment_len();
        let mut nprev = chest;
        let mut nprev_len = chest_len;
        for n in 0..params.neck_count {
            let bone = builder.push(Bone {
                name: format!("neck.{n}"),
                parent: Some(nprev),
                local_bind: Transform::from_translation(Vec3::new(0.0, nprev_len, 0.0)),
                length: neck_seg,
                radius,
            });
            nprev = bone;
            nprev_len = neck_seg;
        }
        builder.push(Bone {
            name: "head".into(),
            parent: Some(nprev),
            local_bind: Transform::from_translation(Vec3::new(0.0, nprev_len, 0.0)),
            length: params.head * params.height,
            radius,
        });

        // Tail from the pelvis, pointing backward/down.
        let tail_seg = params.tail_segment_len();
        let mut tprev = pelvis;
        for k in 0..params.tail_segments {
            let (trans, rot) = if k == 0 {
                (Vec3::new(0.0, 0.0, 0.0), tail_dir)
            } else {
                (Vec3::new(0.0, tail_seg, 0.0), Quat::IDENTITY)
            };
            tprev = builder.push(Bone {
                name: format!("tail.{k}"),
                parent: Some(tprev),
                local_bind: Transform::new(trans, rot),
                length: tail_seg,
                radius,
            });
        }

        // Arms / front legs branch sideways from the chest, then hang down.
        let half_shoulder = params.shoulder_width * params.height * 0.5;
        for (side, sx) in [("L", 1.0f32), ("R", -1.0f32)] {
            let upper = builder.push(Bone {
                name: format!("{side}.upperarm"),
                parent: Some(chest),
                local_bind: Transform::new(Vec3::new(sx * half_shoulder, chest_len, 0.0), down),
                length: params.upper_arm * params.height,
                radius,
            });
            let fore = builder.push(Bone {
                name: format!("{side}.forearm"),
                parent: Some(upper),
                local_bind: Transform::from_translation(Vec3::new(
                    0.0,
                    params.upper_arm * params.height,
                    0.0,
                )),
                length: params.forearm * params.height,
                radius,
            });
            builder.push(Bone {
                name: format!("{side}.hand"),
                parent: Some(fore),
                local_bind: Transform::from_translation(Vec3::new(
                    0.0,
                    params.forearm * params.height,
                    0.0,
                )),
                length: params.hand * params.height,
                radius,
            });
        }

        // Legs / hind legs branch sideways from the pelvis, then hang down.
        let half_hip = params.hip_width * params.height * 0.5;
        for (side, sx) in [("L", 1.0f32), ("R", -1.0f32)] {
            let thigh = builder.push(Bone {
                name: format!("{side}.thigh"),
                parent: Some(pelvis),
                local_bind: Transform::new(Vec3::new(sx * half_hip, 0.0, 0.0), down),
                length: params.thigh * params.height,
                radius,
            });
            let shank = builder.push(Bone {
                name: format!("{side}.shank"),
                parent: Some(thigh),
                local_bind: Transform::from_translation(Vec3::new(
                    0.0,
                    params.thigh * params.height,
                    0.0,
                )),
                length: params.shank * params.height,
                radius,
            });
            builder.push(Bone {
                name: format!("{side}.foot"),
                parent: Some(shank),
                local_bind: Transform::from_translation(Vec3::new(
                    0.0,
                    params.shank * params.height,
                    0.0,
                )),
                length: params.foot * params.height,
                radius,
            });
        }

        Skeleton {
            bones: builder.bones,
        }
    }
}

#[derive(Default)]
struct Builder {
    bones: Vec<Bone>,
}
impl Builder {
    fn push(&mut self, bone: Bone) -> usize {
        let i = self.bones.len();
        self.bones.push(bone);
        i
    }
}

fn quat_is_finite(q: Quat) -> bool {
    q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite()
}
