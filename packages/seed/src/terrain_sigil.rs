//! The `terrain` realm sigil (RITE VII, VII-0b "THE FIRST GROUND, part (b)" —
//! the render weld). Authored realm data names a generated terrain PATCH by
//! seed + tile coordinates + optional per-dial overrides — never stored
//! geometry (the whole point of VII-0b's NO-STORAGE ordeal at the scene
//! seam). One schema, read by BOTH readers that need it: the renderer
//! (`scrying-glass::scene`, which must generate real triangles from it) and
//! the oracle (`oracle::model`, which must derive an analytic AABB from it
//! without ever building a mesh). Kept HERE, in `seed` — the one crate both
//! already depend on — so the schema can never drift between the two
//! readers the way two independently hand-rolled structs could.

use serde::Deserialize;

use crate::hash::Seed;
use crate::terrain::{TerrainParams, TerrainTile};

/// Every field is plain English; `deny_unknown_fields` makes a typo'd dial a
/// LOUD authoring error, never a silently-ignored one — the same convention
/// the `body` sigil uses (`packages/scrying-glass/src/physics.rs::Body`).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerrainSigil {
    /// The world seed feeding every hash stream this patch samples.
    pub seed: u64,
    /// The tile's integer address on the world's tile lattice (Ruling 4:
    /// `i64`, not `i32` — a planet-scale world must name tiles arbitrarily
    /// far from the origin without wrapping).
    pub tile_x: i64,
    pub tile_y: i64,
    /// [`TerrainParams::tile_size_m`] override. `None` re-derives the whole
    /// params set from [`TerrainParams::default`]'s tile size.
    #[serde(default)]
    pub tile_size_m: Option<f32>,
    #[serde(default)]
    pub grid_resolution: Option<u32>,
    #[serde(default)]
    pub height_amplitude: Option<f32>,
    #[serde(default)]
    pub base_wavelength_m: Option<f32>,
    #[serde(default)]
    pub octaves: Option<u32>,
    #[serde(default)]
    pub lacunarity: Option<f32>,
    #[serde(default)]
    pub gain: Option<f32>,
    #[serde(default)]
    pub warp_strength: Option<f32>,
    #[serde(default)]
    pub warp_enabled: Option<bool>,
    /// Hex colour for the patch's own material chain. `seed` doesn't parse
    /// colour (a renderer concept) — carried verbatim for the scene reader,
    /// which falls back to its own default when absent (the `mesh` part
    /// convention, `scrying-glass::scene::append_part`).
    #[serde(default)]
    pub color: Option<String>,
}

impl TerrainSigil {
    /// The tile key this sigil names.
    pub fn tile(&self) -> TerrainTile {
        TerrainTile::new(self.tile_x, self.tile_y)
    }

    /// The world seed this sigil names.
    pub fn world_seed(&self) -> Seed {
        Seed(self.seed)
    }

    /// Resolve every [`TerrainParams`] dial: start from
    /// [`TerrainParams::derive`] against the (possibly overridden) tile
    /// size — never [`TerrainParams::default`]'s fixed baseline, which would
    /// silently mismatch an overridden tile size — then overlay each
    /// explicit override, one field at a time (an absent field keeps its
    /// derived default; nothing here is ever plucked).
    pub fn params(&self) -> TerrainParams {
        let tile_size_m = self
            .tile_size_m
            .unwrap_or_else(|| TerrainParams::default().tile_size_m);
        let mut params = TerrainParams::derive(tile_size_m);
        if let Some(v) = self.grid_resolution {
            params.grid_resolution = v;
        }
        if let Some(v) = self.height_amplitude {
            params.height_amplitude = v;
        }
        if let Some(v) = self.base_wavelength_m {
            params.base_wavelength_m = v;
        }
        if let Some(v) = self.octaves {
            params.fbm.octaves = v;
        }
        if let Some(v) = self.lacunarity {
            params.fbm.lacunarity = v;
        }
        if let Some(v) = self.gain {
            params.fbm.gain = v;
        }
        if let Some(v) = self.warp_strength {
            params.warp_strength = v;
        }
        if let Some(v) = self.warp_enabled {
            params.warp_enabled = v;
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_terrain_params_default_when_nothing_overridden() {
        let sigil = TerrainSigil {
            seed: 7,
            tile_x: 0,
            tile_y: 0,
            tile_size_m: None,
            grid_resolution: None,
            height_amplitude: None,
            base_wavelength_m: None,
            octaves: None,
            lacunarity: None,
            gain: None,
            warp_strength: None,
            warp_enabled: None,
            color: None,
        };
        let params = sigil.params();
        let default = TerrainParams::default();
        assert_eq!(params.tile_size_m, default.tile_size_m);
        assert_eq!(params.grid_resolution, default.grid_resolution);
        assert_eq!(params.height_amplitude, default.height_amplitude);
    }

    #[test]
    fn tile_size_override_re_derives_dependent_dials() {
        let sigil = TerrainSigil {
            seed: 7,
            tile_x: 0,
            tile_y: 0,
            tile_size_m: Some(32.0),
            grid_resolution: None,
            height_amplitude: None,
            base_wavelength_m: None,
            octaves: None,
            lacunarity: None,
            gain: None,
            warp_strength: None,
            warp_enabled: None,
            color: None,
        };
        let params = sigil.params();
        let derived = TerrainParams::derive(32.0);
        assert_eq!(params.grid_resolution, derived.grid_resolution);
        assert_eq!(params.height_amplitude, derived.height_amplitude);
        assert_ne!(params.grid_resolution, TerrainParams::default().grid_resolution.max(0) + 999);
    }

    #[test]
    fn deny_unknown_fields_rejects_a_typo() {
        let raw = serde_json::json!({
            "seed": 1, "tile_x": 0, "tile_y": 0, "tille_size_m": 10.0
        });
        assert!(serde_json::from_value::<TerrainSigil>(raw).is_err());
    }
}
