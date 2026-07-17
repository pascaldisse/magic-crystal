//! The plain-English parameter surface — everything a creature is, as data.
//!
//! A [`CreatureParams`] is the whole input: a [`Morphology`] (biped or
//! quadruped), a [`RegionScheme`] (how the body partitions into coloured
//! bands), a set of [`Proportions`] (scalar multipliers on the base
//! morphology), a [`PaletteParams`] (a colour STRING per band), and the mesh
//! resolution knobs. Every field has a default; the default of every
//! proportion scalar is [`LOVE`] (1.0) — the neutral multiplier that leaves the
//! base morphology untouched. This is the sole numeric literal the parameter
//! logic leans on: a proportion is either LOVE (unchanged) or a caller's own
//! number, and the base magnitudes all come from [`homunculus::BodyParams`],
//! never hardcoded here.

use homunculus::BodyParams;
use vessel::{Blend, VesselParams};

/// **LOVE = 1.0** — the One Constant (mirrors `elements::LOVE`). The neutral
/// proportion multiplier: a scalar of LOVE scales a base proportion by itself,
/// i.e. leaves it exactly unchanged. It is the default of every scalar and the
/// only bare literal the proportion logic uses.
pub const LOVE: f32 = 1.0;

/// The proportion-scalar contract range. A scalar outside `[MIN_SCALE,
/// MAX_SCALE]` is clamped (never rejected, never a panic) — the declared bounds
/// of the parameter, not an algorithm magic number.
// The floor is deliberately above zero: it is the leanest scale at which even
// the girth (limb radius) still resolves into a full, closed body at the
// production mesh resolution, so every in-range parameter yields a real body
// (below it, thin limbs under-resolve to a fragment — see the watertight
// ordeal). Extreme values clamp UP to this floor; they never panic.
pub const MIN_SCALE: f32 = 0.25;
/// Upper bound of the proportion-scalar contract range (see [`MIN_SCALE`]).
pub const MAX_SCALE: f32 = 6.0;

/// The smallest marching-cubes resolution the builder will mesh at — below this
/// the surface degenerates. The declared floor of the mesh parameter.
pub const MIN_RESOLUTION: usize = 8;

/// The body plan: which base skeleton the proportions scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Morphology {
    /// An upright two-legged body (the human base, [`BodyParams::humanoid`]).
    Biped,
    /// A four-legged, tailed body (the cat base, [`BodyParams::quadruped`]).
    Quadruped,
}

impl Morphology {
    /// The base parametric body this morphology scales. All magnitudes live in
    /// homunculus, so nothing is hardcoded in this crate.
    pub fn base_body(self) -> BodyParams {
        match self {
            Morphology::Biped => BodyParams::humanoid(),
            Morphology::Quadruped => BodyParams::quadruped(),
        }
    }
}

/// How the body partitions into coloured regions the palette paints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionScheme {
    /// The grounded default partition for the morphology
    /// ([`vessel::BodyRegions::humanoid`] / [`vessel::BodyRegions::quadruped`]):
    /// head · torso · arms · hands · legs · feet (biped) or head · body · legs ·
    /// tail (quadruped).
    Plain,
    /// A dressed biped's finer partition, split so garments read as their own
    /// bands: hair · neck · torso · arms · hands · skirt · shins · boots. Purely
    /// bone-name derived (thigh→skirt, shank→shins, foot→boots), so it is
    /// generic to any clothed humanoid — not one character. Only meaningful for
    /// [`Morphology::Biped`].
    Clothed,
}

/// Scalar multipliers on the base morphology's proportions. Every field is a
/// pure multiplier whose default is [`LOVE`] (1.0 = unchanged). A field of LOVE
/// reproduces the base morphology's value byte-for-byte (`x * 1.0 == x`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Proportions {
    /// Overall height (multiplies the base metre height).
    pub height: f32,
    /// Pelvis (root) length.
    pub pelvis: f32,
    /// Torso / spine length.
    pub torso: f32,
    /// Neck length.
    pub neck: f32,
    /// Head length.
    pub head: f32,
    /// Tail length (no effect on a tailless base).
    pub tail: f32,
    /// Upper arm / front-upper-leg length.
    pub upper_arm: f32,
    /// Forearm / front-lower-leg length.
    pub forearm: f32,
    /// Hand / front-paw length.
    pub hand: f32,
    /// Thigh length.
    pub thigh: f32,
    /// Shank length.
    pub shank: f32,
    /// Foot / hind-paw length.
    pub foot: f32,
    /// Shoulder width.
    pub shoulder_width: f32,
    /// Hip width.
    pub hip_width: f32,
    /// Limb girth (base capsule radius).
    pub girth: f32,
}

impl Default for Proportions {
    /// Every scalar is [`LOVE`] — the base morphology, untouched.
    fn default() -> Self {
        Self {
            height: LOVE,
            pelvis: LOVE,
            torso: LOVE,
            neck: LOVE,
            head: LOVE,
            tail: LOVE,
            upper_arm: LOVE,
            forearm: LOVE,
            hand: LOVE,
            thigh: LOVE,
            shank: LOVE,
            foot: LOVE,
            shoulder_width: LOVE,
            hip_width: LOVE,
            girth: LOVE,
        }
    }
}

/// The per-region palette, as plain colour STRINGS (the DreamForge colour =
/// string law). Each slot names a region and its colour; regions the palette
/// does not name fall back to `default`. Unknown/invalid colour strings are
/// repaired at build time, never panicked on.
#[derive(Clone, Debug, PartialEq)]
pub struct PaletteParams {
    /// `(region_name, colour_string)` pairs, in the order the regions appear.
    pub slots: Vec<(String, String)>,
    /// Fallback colour string for any region without a slot.
    pub default: String,
    /// How adjacent region colours meet at a boundary.
    pub blend: Blend,
}

impl Default for PaletteParams {
    /// A neutral grey body with hard region seams — a valid, colour-legal
    /// starting point that names no region (all fall back to the default).
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            default: neutral_grey(),
            blend: Blend::Hard,
        }
    }
}

/// The last-resort neutral colour used when even a palette default is invalid.
/// A single declared string, not scattered magic.
pub fn neutral_grey() -> String {
    "#808080".to_string()
}

/// The whole creature, as data: a morphology, its region scheme, its
/// proportions, its palette, and the mesh resolution knobs. Pure input — build
/// it into a [`vessel::Preset`] with [`CreatureParams::build`].
#[derive(Clone, Debug, PartialEq)]
pub struct CreatureParams {
    /// Biped or quadruped base plan.
    pub morphology: Morphology,
    /// How the body partitions into coloured regions.
    pub region_scheme: RegionScheme,
    /// Scalar multipliers on the base proportions.
    pub proportions: Proportions,
    /// The per-region colour strings.
    pub palette: PaletteParams,
    /// Mesh build knobs (SDF blend, marching-cubes resolution, …).
    pub mesh: VesselParams,
}

impl Default for CreatureParams {
    /// A plain grey biped at the base human proportions — the blank slate.
    fn default() -> Self {
        Self {
            morphology: Morphology::Biped,
            region_scheme: RegionScheme::Plain,
            proportions: Proportions::default(),
            palette: PaletteParams::default(),
            mesh: VesselParams::default(),
        }
    }
}
