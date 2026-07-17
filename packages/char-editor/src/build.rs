//! The build weld — [`CreatureParams`] into a [`vessel::Preset`], plus notes.
//!
//! Pure data-in → data-out: no rendering, no I/O, no RNG. The proportions scale
//! the base [`homunculus::BodyParams`], the region scheme partitions the
//! resulting skeleton, and the palette paints the partition. Every out-of-range
//! parameter is repaired and recorded (see [`BuildNote`]); the builder NEVER
//! panics on caller input. Determinism is total: the same params yield a
//! byte-identical preset and the same note list on every run.

use homunculus::{BodyParams, Skeleton};
use vessel::region::Classifier;
use vessel::{BodyRegions, Palette, Preset};

use crate::note::BuildNote;
use crate::params::{
    CreatureParams, Morphology, PaletteParams, RegionScheme, LOVE, MAX_SCALE, MIN_RESOLUTION,
    MIN_SCALE,
};

/// The result of building a creature: the composed preset plus every repair the
/// builder made to the caller's parameters. An empty `notes` means the input
/// was fully in-range and colour-legal.
#[derive(Clone, Debug)]
pub struct BuildOutcome {
    /// The composed vessel preset (skeleton + mesh params + regions + palette).
    pub preset: Preset,
    /// Every clamp / substitution the builder applied, in a deterministic order.
    pub notes: Vec<BuildNote>,
}

impl BuildOutcome {
    /// Whether the input was accepted verbatim (no repairs).
    pub fn is_clean(&self) -> bool {
        self.notes.is_empty()
    }
}

impl CreatureParams {
    /// Build this parameter set into a named [`vessel::Preset`], repairing any
    /// out-of-range value and recording it in the returned notes. Never panics.
    pub fn build(&self, name: &'static str) -> BuildOutcome {
        let mut notes = Vec::new();

        let body = self.derive_body(&mut notes);
        let skeleton = Skeleton::from_params(&body);

        let regions = build_regions(self.morphology, self.region_scheme, &skeleton);
        let palette = build_palette(&self.palette, &mut notes);
        let mesh = self.derive_mesh(&mut notes);

        BuildOutcome {
            preset: Preset {
                name,
                skeleton,
                vessel: mesh,
                regions,
                palette,
            },
            notes,
        }
    }

    /// Scale the base morphology by the (clamped) proportions.
    fn derive_body(&self, notes: &mut Vec<BuildNote>) -> BodyParams {
        let base = self.morphology.base_body();
        let p = &self.proportions;
        // Each proportion is clamped, then multiplies its base field. A scalar
        // of LOVE (1.0) leaves the base field byte-identical (`x * 1.0 == x`).
        // Clamp sequentially so each repair is recorded in field order.
        let height = clamp_scalar("height", p.height, notes);
        let pelvis = clamp_scalar("pelvis", p.pelvis, notes);
        let torso = clamp_scalar("torso", p.torso, notes);
        let neck = clamp_scalar("neck", p.neck, notes);
        let head = clamp_scalar("head", p.head, notes);
        let tail = clamp_scalar("tail", p.tail, notes);
        let upper_arm = clamp_scalar("upper_arm", p.upper_arm, notes);
        let forearm = clamp_scalar("forearm", p.forearm, notes);
        let hand = clamp_scalar("hand", p.hand, notes);
        let thigh = clamp_scalar("thigh", p.thigh, notes);
        let shank = clamp_scalar("shank", p.shank, notes);
        let foot = clamp_scalar("foot", p.foot, notes);
        let shoulder_width = clamp_scalar("shoulder_width", p.shoulder_width, notes);
        let hip_width = clamp_scalar("hip_width", p.hip_width, notes);
        let girth = clamp_scalar("girth", p.girth, notes);
        BodyParams {
            height: base.height * height,
            pelvis: base.pelvis * pelvis,
            spine: base.spine * torso,
            neck: base.neck * neck,
            head: base.head * head,
            tail: base.tail * tail,
            upper_arm: base.upper_arm * upper_arm,
            forearm: base.forearm * forearm,
            hand: base.hand * hand,
            thigh: base.thigh * thigh,
            shank: base.shank * shank,
            foot: base.foot * foot,
            shoulder_width: base.shoulder_width * shoulder_width,
            hip_width: base.hip_width * hip_width,
            bone_radius: base.bone_radius * girth,
            // Discrete topology is inherited from the base morphology.
            spine_count: base.spine_count,
            neck_count: base.neck_count,
            tail_segments: base.tail_segments,
            neck_head_split: base.neck_head_split,
            stance: base.stance,
        }
    }

    /// The mesh knobs, with the resolution raised to its floor if too low.
    fn derive_mesh(&self, notes: &mut Vec<BuildNote>) -> vessel::VesselParams {
        let mut mesh = self.mesh;
        if mesh.resolution < MIN_RESOLUTION {
            notes.push(BuildNote::MeshFloored {
                field: "resolution",
                supplied: mesh.resolution,
                floored_to: MIN_RESOLUTION,
            });
            mesh.resolution = MIN_RESOLUTION;
        }
        mesh
    }
}

/// Clamp one proportion scalar into the contract range, recording any repair.
/// Non-finite → LOVE; out of `[MIN_SCALE, MAX_SCALE]` → the nearest bound.
fn clamp_scalar(field: &'static str, value: f32, notes: &mut Vec<BuildNote>) -> f32 {
    if !value.is_finite() {
        notes.push(BuildNote::ScalarNotFinite {
            field,
            replaced_with: LOVE,
        });
        return LOVE;
    }
    if !(MIN_SCALE..=MAX_SCALE).contains(&value) {
        let clamped = value.clamp(MIN_SCALE, MAX_SCALE);
        notes.push(BuildNote::ScalarClamped {
            field,
            supplied: value,
            clamped_to: clamped,
        });
        return clamped;
    }
    value
}

/// The bone-name classifiers for a morphology + region scheme. These are pure,
/// generic bone-name rules (never character-specific), and their ORDER is the
/// region order the palette slots align to.
fn classifiers(morphology: Morphology, scheme: RegionScheme) -> Vec<(&'static str, Classifier)> {
    match (morphology, scheme) {
        // The dressed-biped partition: garments read as their own bands. Purely
        // bone-name derived, so it fits any clothed humanoid.
        (Morphology::Biped, RegionScheme::Clothed) => vec![
            ("hair", (|n| n == "head") as Classifier),
            ("neck", |n| n.starts_with("neck.")),
            ("torso", |n| n == "pelvis" || n.starts_with("spine.")),
            ("arms", |n| {
                n.ends_with(".upperarm") || n.ends_with(".forearm")
            }),
            ("hands", |n| n.ends_with(".hand")),
            ("skirt", |n| n.ends_with(".thigh")),
            ("shins", |n| n.ends_with(".shank")),
            ("boots", |n| n.ends_with(".foot")),
        ],
        // Plain biped and any quadruped scheme fall back to the vessel default
        // partitions (built below); this arm is unused for them.
        _ => Vec::new(),
    }
}

/// Partition the skeleton per the morphology + scheme.
fn build_regions(morphology: Morphology, scheme: RegionScheme, skeleton: &Skeleton) -> BodyRegions {
    match (morphology, scheme) {
        (Morphology::Biped, RegionScheme::Plain) => BodyRegions::humanoid(skeleton),
        (Morphology::Biped, RegionScheme::Clothed) => {
            BodyRegions::classify(skeleton, &classifiers(morphology, scheme))
        }
        // A quadruped has no dressed variant; both schemes give the cat partition.
        (Morphology::Quadruped, _) => BodyRegions::quadruped(skeleton),
    }
}

/// Turn palette params into a [`Palette`], substituting the default for any
/// invalid slot colour and a neutral grey for an invalid default.
fn build_palette(params: &PaletteParams, notes: &mut Vec<BuildNote>) -> Palette {
    // Repair the default first: everything else falls back to it.
    let default = if vessel::color::is_valid(&params.default) {
        params.default.clone()
    } else {
        let grey = crate::params::neutral_grey();
        notes.push(BuildNote::DefaultColorInvalid {
            supplied: params.default.clone(),
            replaced_with: grey.clone(),
        });
        grey
    };

    let colors = params
        .slots
        .iter()
        .map(|(slot, color)| {
            if vessel::color::is_valid(color) {
                (slot.clone(), color.clone())
            } else {
                notes.push(BuildNote::ColorInvalid {
                    slot: slot.clone(),
                    supplied: color.clone(),
                    replaced_with: default.clone(),
                });
                (slot.clone(), default.clone())
            }
        })
        .collect();

    Palette {
        colors,
        default,
        blend: params.blend,
    }
}
