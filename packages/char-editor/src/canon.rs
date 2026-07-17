//! The canon re-expressed as parameters.
//!
//! The proof that the parametric substrate SUBSUMES the hand-authored canon:
//! [`nari_params`] and [`pink_cat_params`] are ordinary [`CreatureParams`] whose
//! [`CreatureParams::build`] output is byte-identical to the hand presets these
//! creatures shipped as (`vessel::Preset::nari` and the quadruped +
//! `Palette::pink_cat` components). If the canon can be spoken in the parameter
//! language with zero drift, the language is complete enough to hold it.
//!
//! Both are default proportions ([`LOVE`](crate::LOVE) everywhere) — the canon lives at the
//! base morphology; only the region scheme and the colour strings distinguish
//! them. See `tests/ordeals.rs::parity_*` for the byte-parity verdicts.

use vessel::Blend;

use crate::params::{CreatureParams, Morphology, PaletteParams, Proportions, RegionScheme};

/// nari — the avatar, as parameters. Biped + the [`RegionScheme::Clothed`]
/// partition (hair · neck · torso · arms · hands · skirt · shins · boots), the
/// enumerated garment colours, pale image-derived skin as the default, hard
/// seams. Builds byte-identical to `vessel::Preset::nari`.
pub fn nari_params() -> CreatureParams {
    // The image-derived pale skin (see `vessel::Preset::nari` SKIN NOTE).
    let skin = "#e8c9ac";
    CreatureParams {
        morphology: Morphology::Biped,
        region_scheme: RegionScheme::Clothed,
        proportions: Proportions::default(),
        palette: PaletteParams {
            slots: vec![
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
            blend: Blend::Hard,
        },
        mesh: Default::default(),
    }
}

/// pink_cat — the naruko cat, as parameters. Quadruped + the plain quadruped
/// partition (head · body · legs · tail), the demonstration pink coat, soft
/// boundary blend. Builds byte-identical to the hand-assembled quadruped
/// components (`Skeleton::quadruped` + `BodyRegions::quadruped` +
/// `Palette::pink_cat` + default mesh).
pub fn pink_cat_params() -> CreatureParams {
    CreatureParams {
        morphology: Morphology::Quadruped,
        region_scheme: RegionScheme::Plain,
        proportions: Proportions::default(),
        palette: PaletteParams {
            slots: vec![
                ("head".into(), "#ffd0dc".into()),
                ("body".into(), "#ffc0cb".into()),
                ("legs".into(), "#ffb0c0".into()),
                ("tail".into(), "#ff9fb6".into()),
            ],
            default: "#ffc0cb".into(),
            blend: Blend::Smooth { width: 0.4 },
        },
        mesh: Default::default(),
    }
}
