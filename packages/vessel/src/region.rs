//! Body regions and per-region coloring.
//!
//! A [`BodyRegions`] map partitions a skeleton's bones into named regions
//! (head, torso, arms, …). The region OF A VERTEX is the region that owns its
//! **max-weight bone** — capsule ownership, the exact precedent the V0
//! deformation ordeal already leans on (a vertex belongs to whichever bone
//! capsule binds it strongest). The partition is total and disjoint: every
//! bone belongs to exactly one region, so every vertex resolves to exactly one
//! region.
//!
//! Determinism / tie-break: homunculus sorts each vertex's influences by
//! DESCENDING weight with a STABLE sort ([`slice::sort_by`] on `total_cmp`),
//! and its raw influences are enumerated in ascending bone index. So when two
//! bones bind a vertex with bit-identical weight, the LOWER bone index wins the
//! `[0]` slot and thus decides the region. The rule is fully deterministic and
//! needs no random tiebreak.
//!
//! A [`Palette`] assigns a schema color STRING (§`color`) to each region and a
//! [`Blend`] mode for region boundaries. [`Palette::apply`] turns a bound mesh
//! into per-vertex color strings — hard-edged (each vertex takes its region's
//! color) or smoothly blended across a weight-ratio boundary band.
//!
//! Regions/palettes are pure PARAMETERS (data), never code specialization: the
//! two preset region maps are derived from bone NAMES (not frozen indices), and
//! the example palettes ([`Palette::pale_skin_dark_hair`], [`Palette::pink_cat`])
//! are documented param sets that only DEMONSTRATE the mechanism — the canon
//! avatar palettes live in realm/character data, downstream of this crate.

use crate::color;
use crate::mesh::Mesh;
use glam::Vec3;
use homunculus::{Skeleton, SkinWeights};

/// A bone-name classifier: accepts the names that belong to one region.
pub type Classifier = fn(&str) -> bool;

/// One named region: the bones it owns (indices into the skeleton).
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    /// Stable region name (e.g. `"head"`, `"torso"`).
    pub name: String,
    /// Skeleton bone indices owned by this region.
    pub bones: Vec<usize>,
}

/// A total, disjoint partition of a skeleton's bones into named regions.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyRegions {
    /// Regions in a fixed, canonical order (the region INDEX is its position
    /// here — stable across builds for a given morphology).
    pub regions: Vec<Region>,
}

impl BodyRegions {
    /// Build from an ordered list of `(name, predicate)` classifiers: each bone
    /// is placed in the FIRST region whose predicate accepts its name. Panics if
    /// any bone matches no region (the partition must be total).
    ///
    /// Deriving from names (not indices) keeps this generic — it tracks the
    /// parametric skeleton generator, whatever bone counts a morphology yields.
    pub fn classify(skeleton: &Skeleton, classifiers: &[(&str, Classifier)]) -> BodyRegions {
        let mut regions: Vec<Region> = classifiers
            .iter()
            .map(|(name, _)| Region {
                name: (*name).to_string(),
                bones: Vec::new(),
            })
            .collect();
        for (bi, bone) in skeleton.bones.iter().enumerate() {
            let slot = classifiers
                .iter()
                .position(|(_, pred)| pred(&bone.name))
                .unwrap_or_else(|| panic!("bone {bi} ({}) matched no region", bone.name));
            regions[slot].bones.push(bi);
        }
        BodyRegions { regions }
    }

    /// Default humanoid partition: head · torso · arms · hands · legs · feet.
    ///
    /// (The preset humanoid skeleton carries no dedicated hair bone, so the
    /// grounded default folds the scalp into `head`; a skeleton that DID carry
    /// a `hair.*` chain would gain a `hair` region by adding one classifier —
    /// the mechanism is bone-set membership, nothing here is head-specific.)
    pub fn humanoid(skeleton: &Skeleton) -> BodyRegions {
        Self::classify(
            skeleton,
            &[
                ("head", |n| n == "head" || n.starts_with("neck.")),
                ("torso", |n| n == "pelvis" || n.starts_with("spine.")),
                ("arms", |n| {
                    n.ends_with(".upperarm") || n.ends_with(".forearm")
                }),
                ("hands", |n| n.ends_with(".hand")),
                ("legs", |n| n.ends_with(".thigh") || n.ends_with(".shank")),
                ("feet", |n| n.ends_with(".foot")),
            ],
        )
    }

    /// Default quadruped partition: head · body · legs · tail.
    ///
    /// (No dedicated ear bone exists in the preset cat, so ears fold into
    /// `head`, exactly as hair folds into the humanoid head — add an `ears`
    /// classifier the day the skeleton grows ear bones.)
    pub fn quadruped(skeleton: &Skeleton) -> BodyRegions {
        Self::classify(
            skeleton,
            &[
                ("head", |n| n == "head" || n.starts_with("neck.")),
                ("body", |n| n == "pelvis" || n.starts_with("spine.")),
                ("legs", |n| {
                    n.ends_with(".upperarm")
                        || n.ends_with(".forearm")
                        || n.ends_with(".hand")
                        || n.ends_with(".thigh")
                        || n.ends_with(".shank")
                        || n.ends_with(".foot")
                }),
                ("tail", |n| n.starts_with("tail.")),
            ],
        )
    }

    /// Number of regions.
    pub fn len(&self) -> usize {
        self.regions.len()
    }
    /// Whether there are no regions.
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Find a region index by name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.regions.iter().position(|r| r.name == name)
    }

    /// Reverse lookup: bone index → region index. Bones not owned by any region
    /// map to `None` (never happens for a partition built by [`Self::classify`]).
    pub fn bone_region(&self, bone_count: usize) -> Vec<Option<usize>> {
        let mut map = vec![None; bone_count];
        for (ri, region) in self.regions.iter().enumerate() {
            for &b in &region.bones {
                map[b] = Some(ri);
            }
        }
        map
    }

    /// The region owning each vertex, by the max-weight-bone rule. `regions[vi]`
    /// is the index into [`Self::regions`]. Panics if a vertex's dominant bone
    /// belongs to no region (the partition must cover every bone).
    pub fn assign(&self, weights: &SkinWeights, bone_count: usize) -> Vec<usize> {
        let bone_to_region = self.bone_region(bone_count);
        weights
            .per_vertex
            .iter()
            .enumerate()
            .map(|(vi, w)| {
                let dominant = w
                    .first()
                    .unwrap_or_else(|| panic!("vertex {vi} has no influences"))
                    .0;
                bone_to_region[dominant]
                    .unwrap_or_else(|| panic!("bone {dominant} (vertex {vi}) has no region"))
            })
            .collect()
    }

    /// Vertex count per region for a given assignment (parallel to
    /// [`Self::regions`]).
    pub fn counts(&self, assignment: &[usize]) -> Vec<usize> {
        let mut c = vec![0usize; self.regions.len()];
        for &r in assignment {
            c[r] += 1;
        }
        c
    }
}

/// How region colors meet at a boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Blend {
    /// Each vertex takes its owning region's color verbatim — hard seams.
    Hard,
    /// Blend region colors near boundaries. `width` ∈ `[0,1]` is the boundary
    /// band measured in per-region weight ratio: a region contributes only if
    /// its accumulated weight is within `width` of the dominant region's
    /// weight. `width = 0` collapses to [`Blend::Hard`]; `width = 1` is a fully
    /// proportional blend by region weight.
    Smooth {
        /// Boundary band as a weight-ratio fraction in `[0,1]`.
        width: f32,
    },
}

/// A palette: a schema color STRING per region name plus a boundary [`Blend`].
///
/// Colors are strings (the EMISSIVE/color = string law); [`Palette::apply`]
/// re-emits per-vertex colors as canonical `#rrggbb` strings, so the whole
/// pipeline stays inside the schema color vocabulary.
#[derive(Clone, Debug, PartialEq)]
pub struct Palette {
    /// `(region_name, color_string)` pairs. A region with no entry falls back
    /// to `default`.
    pub colors: Vec<(String, String)>,
    /// Fallback color string for regions the palette does not name.
    pub default: String,
    /// Boundary blend mode.
    pub blend: Blend,
}

impl Palette {
    /// Look up a region's color string (its entry, else `default`).
    pub fn color_of(&self, region: &str) -> &str {
        self.colors
            .iter()
            .find(|(n, _)| n == region)
            .map(|(_, c)| c.as_str())
            .unwrap_or(&self.default)
    }

    /// Whether every color string (entries + default) is a valid schema color.
    pub fn is_valid(&self) -> bool {
        color::is_valid(&self.default) && self.colors.iter().all(|(_, c)| color::is_valid(c))
    }

    /// Color a bound mesh: returns per-vertex color STRINGS (`#rrggbb`), one per
    /// vertex, and the region assignment used. `weights` must be the vessel's
    /// bind weights; `regions` the partition; `bone_count` the skeleton size.
    ///
    /// Hard mode gives each vertex its region's canonical color. Smooth mode
    /// accumulates each vertex's influence weight per region, keeps the regions
    /// within `width` of the dominant region's weight, and blends their colors
    /// in LINEAR light before re-encoding — a controllable boundary band.
    pub fn apply(
        &self,
        regions: &BodyRegions,
        weights: &SkinWeights,
        bone_count: usize,
    ) -> ColoredMesh {
        let assignment = regions.assign(weights, bone_count);
        let bone_to_region = regions.bone_region(bone_count);
        // Precompute each region's linear color (fallback → default).
        let region_lin: Vec<Vec3> = regions
            .regions
            .iter()
            .map(|r| color::parse(self.color_of(&r.name)).unwrap_or(Vec3::ZERO))
            .collect();

        let colors: Vec<String> = weights
            .per_vertex
            .iter()
            .enumerate()
            .map(|(vi, w)| {
                let dom_region = assignment[vi];
                match self.blend {
                    Blend::Hard => color::to_hex(region_lin[dom_region]),
                    Blend::Smooth { width } if width <= 0.0 => {
                        color::to_hex(region_lin[dom_region])
                    }
                    Blend::Smooth { width } => {
                        // Accumulate weight per region for this vertex.
                        let mut region_w = vec![0.0f32; regions.regions.len()];
                        for &(bone, wt) in w {
                            if let Some(r) = bone_to_region[bone] {
                                region_w[r] += wt;
                            }
                        }
                        let dom_w = region_w[dom_region];
                        let cutoff = dom_w * (1.0 - width.clamp(0.0, 1.0));
                        // Blend regions whose weight is within the band.
                        let mut acc = Vec3::ZERO;
                        let mut total = 0.0f32;
                        for (ri, &rw) in region_w.iter().enumerate() {
                            if rw > 0.0 && rw >= cutoff {
                                acc += region_lin[ri] * rw;
                                total += rw;
                            }
                        }
                        if total > 0.0 {
                            color::to_hex(acc / total)
                        } else {
                            color::to_hex(region_lin[dom_region])
                        }
                    }
                }
            })
            .collect();

        ColoredMesh {
            colors,
            regions: assignment,
        }
    }

    /// Example palette — a pale-skinned, dark-haired humanoid (nari's flesh:
    /// pale skin, dark crown, white coat over the torso/arms, dark legs/boots).
    /// A DEMONSTRATION param set, not the frozen canon (that lives in character
    /// data). Uses the humanoid region names.
    pub fn pale_skin_dark_hair() -> Palette {
        Palette {
            colors: vec![
                ("head".into(), "#f3e0d0".into()), // pale skin + dark crown folded in
                ("torso".into(), "#ffffff".into()), // white coat
                ("arms".into(), "#f2f2f4".into()), // coat sleeves
                ("hands".into(), "#f3e0d0".into()), // pale skin
                ("legs".into(), "#1a1a22".into()), // dark trousers
                ("feet".into(), "#101014".into()), // dark boots
            ],
            default: "#808080".into(),
            blend: Blend::Smooth { width: 0.35 },
        }
    }

    /// Example palette — a pink-coated quadruped (the naruko cat). DEMONSTRATION
    /// param set. Uses the quadruped region names.
    pub fn pink_cat() -> Palette {
        Palette {
            colors: vec![
                ("head".into(), "#ffd0dc".into()), // soft pink face
                ("body".into(), "#ffc0cb".into()), // pink coat
                ("legs".into(), "#ffb0c0".into()), // pink legs
                ("tail".into(), "#ff9fb6".into()), // deeper pink tail
            ],
            default: "#ffc0cb".into(),
            blend: Blend::Smooth { width: 0.4 },
        }
    }
}

/// The colored output of a vessel: per-vertex color STRINGS and the region
/// assignment they were derived from. Parallel to the mesh's vertex arrays.
#[derive(Clone, Debug, PartialEq)]
pub struct ColoredMesh {
    /// Per-vertex color, a canonical `#rrggbb` schema string.
    pub colors: Vec<String>,
    /// Per-vertex owning region index (into [`BodyRegions::regions`]).
    pub regions: Vec<usize>,
}

impl ColoredMesh {
    /// Number of colored vertices.
    pub fn len(&self) -> usize {
        self.colors.len()
    }
    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// Canonical byte serialization (colors joined by `\n`, then a `\0`, then
    /// region indices little-endian) — the form the determinism ordeal compares
    /// alongside the geometry's [`Mesh::to_le_bytes`].
    pub fn to_bytes(&self, mesh: &Mesh) -> Vec<u8> {
        let mut out = mesh.to_le_bytes();
        out.push(0);
        for c in &self.colors {
            out.extend_from_slice(c.as_bytes());
            out.push(b'\n');
        }
        out.push(0);
        for &r in &self.regions {
            out.extend_from_slice(&(r as u32).to_le_bytes());
        }
        out
    }
}
