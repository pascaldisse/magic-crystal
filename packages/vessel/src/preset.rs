//! Body PRESETS and the compose weld (Rite V — the Embodied Ones).
//!
//! A preset is a named, pure parameter bundle: the [`homunculus`] skeleton, the
//! [`VesselParams`] that mesh it, the [`BodyRegions`] partition, and the
//! [`Palette`] that paints it. Presets are DATA — the engine never special-cases
//! a creature; a realm entity names a preset (`body = { preset: "nari" }`) and
//! the compose path works for any creature. This is the same convention as a
//! prefab or a named essence: the library is generic vessel data, the realm
//! only references it by name.
//!
//! [`Body::from_preset`] is the WELD: it builds the vessel (skin the skeleton),
//! colours it, and poses it at SAMA's canonical idle (tick 0) — sama is the sole
//! pose source, so the standing body is exactly what the motion spirit emits at
//! rest. The idle-posed, coloured triangles are what the light then sees.
//!
//! ## nari — the avatar (avatar canon, `NARUKO.md`)
//!
//! The enumerated canon hexes (strings) map onto the regions the humanoid
//! skeleton HONESTLY carries. What exceeds a capsule skeleton's V0 fidelity is
//! DEFERRED, never faked — see [`Preset::nari`] for the deferred list.

use crate::color;
use crate::region::{BodyRegions, ColoredMesh, Palette};
use crate::{Vessel, VesselParams};
use glam::Vec3;
use homunculus::{Pose, Skeleton};
use sama::{Locomotion, LocomotionParams};

/// A named body preset: everything needed to compose a standing body, as pure
/// parameters. Two morphologies ship (`nari` humanoid; `pink_cat` quadruped is
/// the V2 lane) but the mechanism is generic — a new creature is a new preset,
/// never engine code.
#[derive(Clone, Debug)]
pub struct Preset {
    /// Stable preset name (the realm's `body.preset` string).
    pub name: &'static str,
    /// The skeleton to skin.
    pub skeleton: Skeleton,
    /// Meshing parameters (SDF smooth-union + marching-cubes resolution).
    pub vessel: VesselParams,
    /// The bone→region partition the palette paints over.
    pub regions: BodyRegions,
    /// The per-region colour strings.
    pub palette: Palette,
}

impl Preset {
    /// Resolve a preset by its `body.preset` name. `None` for an unknown name —
    /// the caller decides what an unknown body means (nothing is invented here).
    pub fn by_name(name: &str) -> Option<Preset> {
        match name {
            "nari" => Some(Self::nari()),
            "pink_cat" => Some(Self::pink_cat()),
            _ => None,
        }
    }

    /// pink_cat — the pink cat by the ramen stall (Rite V · V2). A QUADRUPED
    /// morphology: the tailed cat skeleton ([`Skeleton::quadruped`]) skinned and
    /// painted with the [`Palette::pink_cat`] preset over the quadruped region
    /// partition (head · body · legs · tail). Same generic weld as nari, only
    /// the morphology differs — the engine special-cases nothing.
    ///
    /// DEFERRED — features the low-poly capsule quadruped honestly cannot carry
    /// (represented as nothing, listed, never faked):
    /// - EARS — no ear bones on the preset cat skeleton, so the head is one
    ///   smooth capsule; the ears fold into `head` (region note in
    ///   [`BodyRegions::quadruped`]). No pointed silhouette.
    /// - WHISKERS / facial features — no face geometry on a capsule skull.
    /// - PAW toes / claws — the `.foot` is a single small capsule tip.
    /// - fur texture — flat per-region albedo, no strand/shading detail.
    ///
    /// The `tail` region IS carried (the skeleton has an 8-segment tail chain),
    /// so the tail silhouette is honest geometry.
    pub fn pink_cat() -> Preset {
        let skeleton = Skeleton::quadruped();
        let regions = BodyRegions::quadruped(&skeleton);
        let palette = Palette::pink_cat();
        Preset {
            name: "pink_cat",
            skeleton,
            vessel: VesselParams::default(),
            regions,
            palette,
        }
    }

    /// nari — the avatar on the seawall (avatar canon, `NARUKO.md §Avatar`).
    ///
    /// The enumerated canon colours (STRINGS) are mapped onto the regions the
    /// default humanoid skeleton carries, split finer than the stock humanoid
    /// partition so the seifuku, the neckerchief, the skirt and the boots each
    /// read as their own band:
    /// - `hair`  (the head bone / crown)  → obsidian `#16121e`
    /// - `neck`  (the neck chain)         → neckerchief violet `#7c3aed`
    /// - `torso` (pelvis + spine)         → seifuku body `#16121e`
    /// - `arms`  (upper arm + forearm)    → seifuku sleeves `#16121e`
    /// - `hands` (the hand tips)          → skin
    /// - `skirt` (the thighs)             → black pleated skirt `#0d0a12`
    /// - `shins` (the shanks)             → skin (bare leg above the boot)
    /// - `boots` (the feet)               → platform boots `#0d0a12`
    ///
    /// SKIN NOTE: the enumerated canon lists the garments, not a skin hex; the
    /// avatar IMAGE (`reference/naruko/nari-seifuku-red.png`, itself canon) reads
    /// as pale skin, so `hands`/`shins`/the fallback take a pale skin string —
    /// faithful to the image, flagged here as image-derived (not enumerated),
    /// never invented detail.
    ///
    /// DEFERRED — features that exceed a V0 capsule skeleton + flat-region
    /// palette (represented HONESTLY as nothing, listed here, not faked):
    /// - iris `#c1121f` — no eye/face geometry on a capsule skull
    /// - single fang — no mouth geometry
    /// - hair violet ends — the head is one region, no per-strand gradient
    /// - platform boot purple laces `#7c3aed` — no lace geometry
    /// - black pleated skirt CHAIN — no chain accessory
    /// - thigh-strap heart — no strap/heart accessory
    /// - bandaid, left knee — no decal system
    /// - bag with cat charm — no held-prop / socket accessory in V0
    pub fn nari() -> Preset {
        let skeleton = Skeleton::humanoid();
        // Region partition finer than the stock humanoid, derived from bone
        // NAMES (generic — tracks whatever the humanoid generator yields).
        let regions = BodyRegions::classify(
            &skeleton,
            &[
                ("hair", |n| n == "head"),
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
        );
        // Skin string: image-derived pale skin (see SKIN NOTE), not enumerated.
        let skin = "#e8c9ac";
        let palette = Palette {
            colors: vec![
                ("hair".into(), "#16121e".into()),
                ("neck".into(), "#7c3aed".into()),
                ("torso".into(), "#16121e".into()),
                ("arms".into(), "#16121e".into()),
                ("hands".into(), skin.into()),
                ("skirt".into(), "#0d0a12".into()),
                ("shins".into(), skin.into()),
                ("boots".into(), "#0d0a12".into()),
            ],
            default: skin.into(),
            blend: crate::region::Blend::Hard,
        };
        Preset {
            name: "nari",
            skeleton,
            vessel: VesselParams::default(),
            regions,
            palette,
        }
    }
}

/// SAMA's canonical idle pose at tick 0 — the sole pose source for a standing
/// body. A fresh [`Locomotion`] begins in [`sama::Gait::Idle`]; commanding zero
/// speed keeps it there, and its idle pose is the skeleton's bind pose. Routing
/// through sama (rather than calling `Pose::bind` directly) makes the motion
/// spirit the single origin of every body pose — V1 will drive the same seam.
pub fn idle_pose(skeleton: &Skeleton) -> Pose {
    let mut locomotion = Locomotion::new(LocomotionParams::default());
    locomotion.step(skeleton, 0.0)
}

/// A composed, standing body — the Rite V weld made concrete. Built once from a
/// [`Preset`]: the skinned vessel, its per-vertex colours (pose-invariant), the
/// skeleton, and sama's idle pose. [`Body::idle_mesh`] is the geometry the light
/// sees; determinism is inherited from every stage (same preset → identical
/// bytes).
#[derive(Clone, Debug)]
pub struct Body {
    /// The preset name this body was composed from.
    pub name: &'static str,
    /// The skeleton (bind topology).
    pub skeleton: Skeleton,
    /// The built vessel (bind mesh + weights + bind transforms).
    pub vessel: Vessel,
    /// Per-vertex colour strings + region assignment (pose-invariant).
    pub colored: ColoredMesh,
    /// SAMA's idle pose (tick 0) — the standing pose.
    pub idle_pose: Pose,
}

impl Body {
    /// Compose a body from a preset: build the vessel, colour it, and take the
    /// idle pose from sama. Pure and deterministic.
    pub fn from_preset(preset: &Preset) -> Body {
        let vessel = Vessel::build(&preset.skeleton, &preset.vessel);
        let colored = vessel.colored(&preset.regions, &preset.palette);
        let idle_pose = idle_pose(&preset.skeleton);
        Body {
            name: preset.name,
            skeleton: preset.skeleton.clone(),
            vessel,
            colored,
            idle_pose,
        }
    }

    /// The idle-posed body mesh in skeleton-local space (pelvis at the origin) —
    /// the exact triangles the light sees, deformed by sama's idle pose.
    pub fn idle_mesh(&self) -> crate::Mesh {
        self.vessel.posed(&self.skeleton, &self.idle_pose)
    }

    /// Axis-aligned local bounds `(min, max)` of the idle-posed mesh — the body's
    /// footprint before its realm transform. `None` for an empty mesh. Callers
    /// (the Oracle, the placement) derive world bounds by transforming this.
    pub fn idle_local_bounds(&self) -> Option<(Vec3, Vec3)> {
        let mesh = self.idle_mesh();
        let mut iter = mesh.positions.iter();
        let first = *iter.next()?;
        let mut lo = first;
        let mut hi = first;
        for &p in iter {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        Some((lo, hi))
    }

    /// Per-vertex LINEAR-RGB colour, parsed from the colour strings — the albedo
    /// the renderer skins onto each vertex. Parallel to the mesh vertex arrays.
    pub fn vertex_albedo(&self) -> Vec<Vec3> {
        self.colored
            .colors
            .iter()
            .map(|c| color::parse(c).unwrap_or(Vec3::ZERO))
            .collect()
    }
}
