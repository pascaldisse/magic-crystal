//! DreamForge RAIN senses (package `oracle`): PULL-ONLY, data-space vision over
//! the Crystal ECS. Zero GPU, zero rendering, no streaming, no timers.
//!
//! LOOKING IS A VERB (RAIN.md): [`look()`] fires on demand and returns a
//! [`Glance`] — captions (default) + a glance grid of dominant entityId +
//! depth. [`proprio()`] reads an entity's own pose/components. This is Matrix
//! vision: seeing the world's vector truth, never pixels.

pub mod geom;
pub mod look;
pub mod model;
pub mod proprio;

pub use geom::{Aabb, Affine, Vec3};
pub use look::{
    look, EyePose, Glance, Layers, LookError, LookParams, NearEntity, DEPTH_TIE_EPS, EMPTY_CELL,
};
pub use model::{EntityGeom, SenseEntity, World, DEFAULT_PART_HALF};
pub use proprio::{proprio, ComponentSummary, Proprio};

#[cfg(test)]
mod tests {
    use super::*;
    use crystal::Core;
    use std::path::PathBuf;

    fn naruko_dir() -> PathBuf {
        // The ordeals below are exact hand-derivations against a PINNED naruko
        // (the 7-vessel realm, eye at the derived spawn) — the fixture is the
        // gate's fixed geometric truth, never the growing canon realm. The CLI
        // (`oracle-cli`) still gazes at the live ../../worlds/naruko by default.
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/naruko")
            .canonicalize()
            .expect("naruko fixture dir")
    }

    fn naruko() -> World {
        World::load(naruko_dir()).expect("load naruko")
    }

    fn caption_ids(g: &Glance) -> Vec<String> {
        g.nearest.iter().map(|n| n.id.clone()).collect()
    }

    #[test]
    fn loads_naruko_scene_into_ecs() {
        let world = naruko();
        // Every authored entity id is bound in the ECS.
        for id in [
            "env",
            "world_spawn",
            "naruko_terra",
            "naruko_seawall",
            "naruko_sea",
            "lighthouse_rock",
            "lighthouse_tower",
        ] {
            assert!(world.get(id).is_some(), "missing entity {id}");
            assert!(
                world.core.world.entity_for_gaia(id).is_some(),
                "id {id} not bound in ECS"
            );
        }
        // Canonical naruko authors `emissive` as a COLOR STRING; the protocol
        // types it `Option<String>`, so the doc parses cleanly (no warning) and
        // the emissive color is surfaced as data, never widened to a bool.
        assert_eq!(
            world
                .geometry("lighthouse_tower")
                .unwrap()
                .emissive
                .as_deref(),
            Some("#f3e9ff")
        );
        assert!(
            world.schema_warnings.is_empty(),
            "unexpected schema warnings: {:?}",
            world.schema_warnings
        );
    }

    #[test]
    fn spawn_pose_faces_the_lighthouse() {
        let world = naruko();
        let eye = world.spawn_pose().expect("spawn pose");
        assert_eq!(eye.position, [0.0, 4.0, 38.0]);
        assert_eq!(eye.yaw, 0.0);
        let glance = look(&world, eye, LookParams::default()).unwrap();

        // GATE: captions name the lighthouse at a plausible ~160m range.
        let tower = glance
            .nearest
            .iter()
            .find(|n| n.id == "lighthouse_tower")
            .expect("lighthouse_tower in nearest captions");
        assert!(
            (150.0..175.0).contains(&tower.range),
            "lighthouse range {} not ~160m",
            tower.range
        );

        // GATE: at a resolving grid the glance holds the lighthouse above the
        // horizon row. At the 8×8 default the far ~5.5 m tower subtends ~4° —
        // smaller than a 7.5° cell — so ray-true coverage honestly leaves it out
        // of the coarse grid (it stays a caption); it resolves from grid 32 up.
        let regard = look(
            &world,
            eye,
            LookParams {
                grid: 32,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            regard.is_above_horizon("lighthouse_tower"),
            "lighthouse_tower not above horizon at grid 32; cells: {:?}",
            regard.cells_of("lighthouse_tower")
        );
    }

    /// ORDEAL #7b — the ground plane is world-support and must NOT eat a
    /// caption slot; the default gaze names structures, not the terra you stand
    /// on (its center sits ~4m straight below the eye, extent 400m ≫ range).
    #[test]
    fn default_naruko_excludes_ground_from_captions() {
        let world = naruko();
        let eye = world.spawn_pose().unwrap();
        let g = look(&world, eye, LookParams::default()).unwrap();
        let caps = caption_ids(&g);
        assert!(
            !caps.contains(&"naruko_terra".to_string()),
            "terra must be demoted from captions, got {caps:?}"
        );
        // The lighthouse, however, IS a caption.
        assert!(
            caps.contains(&"lighthouse_tower".to_string()),
            "captions {caps:?}"
        );
        // Requesting support surfaces brings terra back (param, not hardcoded).
        let with = look(
            &world,
            eye,
            LookParams {
                include_support: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(caption_ids(&with).contains(&"naruko_terra".to_string()));
    }

    /// ORDEAL #7a — looking due WEST (yaw π/2) the lighthouse sits at ~90° to
    /// the right, far outside a 60° FOV. Real frustum culling must keep it OUT
    /// of both the captions and the grid (the dead-clamp bug stamped it onto a
    /// border cell). Derivation: fwd=(-1,0,0), the tower lies along -Z from the
    /// eye → bearing 90°, well beyond the 30° half-FOV.
    #[test]
    fn yaw_west_rejects_the_lighthouse_entirely() {
        let world = naruko();
        let eye = EyePose {
            position: world.spawn_pose().unwrap().position,
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
        };
        let g = look(
            &world,
            eye,
            LookParams {
                grid: 32,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            !caption_ids(&g).contains(&"lighthouse_tower".to_string()),
            "tower must be absent from captions at yaw π/2, got {:?}",
            caption_ids(&g)
        );
        assert!(
            g.cells_of("lighthouse_tower").is_empty(),
            "tower must occupy NO grid cell at yaw π/2, got {:?}",
            g.cells_of("lighthouse_tower")
        );
    }

    /// ORDEAL #7c — independent hand derivation of the lighthouse cells at
    /// grid 32, fov 60, from the spawn eye (0,4,38) facing -Z. A cell is filled
    /// ONLY on a true ray/AABB hit; its depth is that ray-true entry distance.
    ///
    /// tower world AABB (union of its parts): x∈[-5.5,5.5], y∈[19,63],
    /// z∈[-125.5,-114.5] — the z span is set by the 5.5-radius gallery ring
    /// (part 2 at y=35: z = -120 ± 5.5), NOT the 4.5 shaft. eye=(0,4,38):
    /// fwd=(0,0,-1), right=(1,0,0), up=(0,1,0), tan_half=tan(30°)=0.5774.
    ///   z_view = 38 - z  ⇒ FRONT face (z=-114.5) at 152.5 m, BACK (z=-125.5)
    ///   at 163.5 m along the AXIS; a cell ray is tilted, so its true entry
    ///   distance is LONGER than the axial 152.5.
    ///   ndc_x = x/(z_view·th),  ndc_y = (y-4)/(z_view·th),
    ///   col = floor((ndc_x·0.5+0.5)·32),  row = floor((0.5 - ndc_y·0.5)·32).
    /// x=±5.5 at the front (z_view=152.5, th·z_view=88.05): ndc_x=±0.0625 ⇒
    ///   cols {15,16}. The gallery/shaft span rows 5..=13, but the tower OWNS
    ///   exactly cols {15,16} × rows 5..=12 (16 cells) — and row 13 does NOT
    ///   belong to it for the reason below.
    ///
    /// N4 — TRUE per-row ray-entry depths (nearest of the two columns), rows
    /// 5→12: 163.102, 161.235, 159.536, 158.010, 156.663, 155.499, 154.522,
    /// 153.736. The band is therefore [153.736, 163.102]. EVERY row enters the
    /// FRONT z face (z=-114.5, axial 152.5 m) — NONE reaches the back face; the
    /// numbers exceed 152.5 purely because a tilted cell ray travels a longer
    /// PATH to that same front plane. Row 5 (highest up the tower, steepest ray)
    /// has the largest tilt, so the longest path (163.102 m); row 12 (near the
    /// base, flattest ray) hugs the 153.7 m front. DERIVATION (the inquisitor's,
    /// verbatim): the path length to the front plane is d = 152.5·L where
    /// L = √(1 + x² + y²) and x, y are the cell's ndc scaling of the ray
    /// (x = ndc_x·tan_half, y = ndc_y·tan_half) — the ray direction is
    /// fwd + x·right + y·up, whose length is L, so reaching axial depth 152.5
    /// costs 152.5·L along the ray. NOT 152.5 — no cell ray is axial, so none
    /// attains the axial front distance; and NOT a "back region" hit — 163.102
    /// is a front-face path length, not the 163.5 m axial back-face distance.
    ///   ROW 13 is the rock's, NOT by occlusion: the tower's min y is 19.0, and
    ///   the row-13 rays cross the tower's z span at y = 17.757 (front face) …
    ///   18.749 (back face), BELOW 19.0 — the tower rays MISS entirely, so
    ///   lighthouse_rock (whose top reaches those cells) is the only hit.
    ///
    /// FLOAT TOLERANCE (DERIVED, not plucked): the reference depths above are the
    /// analytic d = 152.5·L per row; the live result is the same geometry through
    /// `camera_basis`/`normalize`/`ray_aabb` in f32. The analytic-vs-asserted
    /// discrepancy across the eight rows peaks at ≈3.25e-4 m (row 5, where the
    /// steepest ray accumulates the most f32 round-off in the direction
    /// normalize + slab test). TOL = 1e-3 m is exactly a 10× margin over that
    /// measured max — tight enough that a wrong z-span (±1 m) or a fabricated
    /// off-ray depth fails by orders of magnitude, loose enough never to flap on
    /// float order-of-operations. This REPLACES the plucked 5e-2 m tolerance
    /// (50× looser than justified) and, before it, the undocumented 0.5 m oracle.
    #[test]
    fn lighthouse_cells_and_depth_are_geometry_truth_at_grid_32() {
        let world = naruko();
        let eye = world.spawn_pose().unwrap();
        let g = look(
            &world,
            eye,
            LookParams {
                grid: 32,
                layers: Layers::BOTH,
                ..Default::default()
            },
        )
        .unwrap();

        // EXACT cell set: cols {15,16} × rows 5..=12 — no more, no less. A wrong
        // AABB derivation (e.g. the old z∈[-124.5,-115.5]) would shift these.
        let cells: std::collections::BTreeSet<(usize, usize)> =
            g.cells_of("lighthouse_tower").into_iter().collect();
        let expected: std::collections::BTreeSet<(usize, usize)> = (5..=12)
            .flat_map(|r| [(r, 15usize), (r, 16usize)])
            .collect();
        assert_eq!(
            cells, expected,
            "tower cell set is not the derived {{15,16}}×{{5..=12}}"
        );

        // The dominance rule is real: lighthouse_rock owns the base row 13
        // because the tower's rays MISS it there (ray y 17.757..18.749 < the
        // tower's min y 19.0), NOT by occlusion.
        assert_eq!(g.cell_id(13 * g.grid + 15), Some("lighthouse_rock"));
        assert_eq!(g.cell_id(13 * g.grid + 16), Some("lighthouse_rock"));

        // N4 — EXACT per-row ray-entry depth band. Reference nearest-of-columns
        // depths rows 5→12 (see the derivation above): each cell must land
        // within a DERIVED 1e-3 m float tolerance (10× the ≈3.25e-4 m measured
        // analytic-vs-live max), and the overall band is the true
        // [153.736, 163.102] — never the axial 152.5.
        const ROW_DEPTH: [(usize, f32); 8] = [
            (5, 163.102),
            (6, 161.235),
            (7, 159.536),
            (8, 158.010),
            (9, 156.663),
            (10, 155.499),
            (11, 154.522),
            (12, 153.736),
        ];
        const TOL: f32 = 1e-3;
        for (row, expect) in ROW_DEPTH {
            let d = (15..=16)
                .map(|c| g.cell_depth(row * g.grid + c))
                .fold(f32::INFINITY, f32::min);
            assert!(
                (d - expect).abs() < TOL,
                "row {row} nearest depth {d} != derived {expect} (tol {TOL})"
            );
        }
        // The band extremes are the true [153.736, 163.102].
        let depths: Vec<f32> = cells
            .iter()
            .map(|&(r, c)| g.cell_depth(r * g.grid + c))
            .collect();
        let lo = depths.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = depths.iter().copied().fold(0.0f32, f32::max);
        assert!(
            (lo - 153.736).abs() < TOL,
            "nearest tower depth {lo} != derived 153.736"
        );
        assert!(
            (hi - 163.102).abs() < TOL,
            "farthest tower depth {hi} != derived 163.102"
        );
    }

    /// ORDEAL #7d — the horizon row follows pitch (grid/2 is only the pitch-0
    /// case). Looking up 30° with a 30° half-FOV drops the horizon to the
    /// bottom edge; looking down 30° raises it to the top.
    #[test]
    fn horizon_row_tracks_pitch() {
        let world = naruko();
        let base = world.spawn_pose().unwrap();
        let grid = 8usize;
        let at = |pitch: f32| {
            look(
                &world,
                EyePose { pitch, ..base },
                LookParams {
                    grid,
                    ..Default::default()
                },
            )
            .unwrap()
            .horizon_row()
        };
        assert_eq!(at(0.0), grid / 2, "pitch 0 horizon is grid/2");
        assert_eq!(
            at(30f32.to_radians()),
            grid,
            "pitch +30° drops horizon to the bottom edge"
        );
        assert_eq!(
            at(-30f32.to_radians()),
            0,
            "pitch -30° raises horizon to the top"
        );
        assert!(at(15f32.to_radians()) > grid / 2 && at(15f32.to_radians()) < grid);
    }

    /// ORDEAL #7e — TRUE lazy layers, proven by an instrumented op counter (not
    /// just by inspecting storage). The id channel does ZERO per-cell work when
    /// ids are not requested; depth is the dominance key (always computed) but
    /// its OUTPUT is withheld unless the depth layer asked.
    #[test]
    fn lazy_layers_are_proven_by_the_op_counter() {
        use crate::look::{ID_ALLOC_OPS, ID_STAMP_OPS, ID_STRING_OPS};
        let ops = || ID_STAMP_OPS.with(|c| c.get());
        let allocs = || ID_ALLOC_OPS.with(|c| c.get());
        let strings = || ID_STRING_OPS.with(|c| c.get());
        // Reset BEFORE each full `look()` so the window spans the COLLECTION
        // phase too — the id-string counter proves depth-only births no grid id
        // string anywhere, not merely inside `rasterize`.
        let reset = || {
            ID_STAMP_OPS.with(|c| c.set(0));
            ID_ALLOC_OPS.with(|c| c.set(0));
            ID_STRING_OPS.with(|c| c.set(0));
        };

        let world = naruko();
        let eye = world.spawn_pose().unwrap();

        // depth-only: id buffers never constructed (zero id-channel ALLOCATION),
        // id per-cell work == 0, and no id-channel output/table materializes.
        reset();
        let depth = look(
            &world,
            eye,
            LookParams {
                grid: 32,
                layers: Layers::DEPTH,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(ops(), 0, "depth-only must do ZERO id-channel work");
        assert_eq!(
            allocs(),
            0,
            "depth-only must do ZERO id-channel allocation (SoA: ids channel None)"
        );
        // COLLECTION-PHASE OBSERVATION (N1): the counter spans the whole call, so
        // this catches a regression that clones id STRINGS during collection
        // (e.g. reverting `visible` to `Vec<(String, Aabb)>`) — depth-only must
        // birth zero grid id strings, in collection OR rasterize.
        assert_eq!(
            strings(),
            0,
            "depth-only must birth ZERO grid id strings (collection carries indices)"
        );
        assert!(depth.ids.is_none(), "depth-only ⇒ no id index buffer");
        assert!(depth.id_table.is_empty(), "depth-only ⇒ no interned table");
        assert!(depth.depth.is_some(), "depth-only ⇒ depth buffer present");
        assert!(
            depth.ids_layer().iter().all(|id| id.is_none()),
            "ids not requested ⇒ none stored"
        );
        assert!(
            depth.depth_layer().iter().any(|d| d.is_finite()),
            "depth requested ⇒ present"
        );

        // ids-only: id work happens; depth OUTPUT withheld (+inf everywhere).
        reset();
        let ids = look(
            &world,
            eye,
            LookParams {
                grid: 32,
                layers: Layers::IDS,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(ops() > 0, "ids requested ⇒ id-channel work runs");
        assert!(allocs() > 0, "ids requested ⇒ id-channel buffers allocated");
        assert!(
            strings() > 0,
            "ids requested ⇒ grid id strings interned (born) at least once"
        );
        assert!(ids.ids.is_some(), "ids requested ⇒ id index buffer present");
        assert!(
            ids.depth.is_none(),
            "depth not requested ⇒ no depth buffer (SoA channel None)"
        );
        assert!(
            ids.depth_layer().is_empty(),
            "depth not requested ⇒ empty depth layer"
        );
        assert!(
            ids.ids_layer().iter().any(|id| id.is_some()),
            "ids requested ⇒ present"
        );

        // no grid layer: nothing computed at all.
        reset();
        let none = look(
            &world,
            eye,
            LookParams {
                layers: Layers::NONE,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(ops(), 0, "no grid ⇒ no id work");
        assert_eq!(allocs(), 0, "no grid ⇒ no id allocation");
        assert_eq!(strings(), 0, "no grid ⇒ no id string born");
        assert_eq!(
            none.cell_count(),
            0,
            "no grid layer requested ⇒ no cells computed"
        );
        assert!(none.depth.is_none() && none.ids.is_none());
        assert!(
            none.entity_count > 0,
            "captions still work without the grid"
        );
    }

    /// ORDEAL #7e-b — NO FABRICATED DEPTH. Independent probe: reconstruct every
    /// stamped cell's ray and confirm it TRULY intersects the AABB of the entity
    /// it names, at the reported depth. Zero cells may carry a depth whose ray
    /// misses (the old off-ray closest-point fallback stamped 144/29 misses).
    #[test]
    fn stamped_cells_are_all_true_ray_hits_no_fabricated_depth() {
        use crate::geom::{add, camera_basis, normalize, ray_aabb, scale3};
        let world = naruko();
        let eye = world.spawn_pose().unwrap();
        let params = LookParams {
            grid: 32,
            layers: Layers::BOTH,
            ..Default::default()
        };
        let g = look(&world, eye, params).unwrap();

        let (fwd, right, up) = camera_basis(eye.yaw, eye.pitch);
        let tan_half = (params.fov_deg.to_radians() * 0.5).tan();
        let grid = g.grid;

        let mut stamped = 0usize;
        let mut fabricated = 0usize;
        for i in 0..g.cell_count() {
            let cell_depth = g.cell_depth(i);
            if !cell_depth.is_finite() {
                assert!(g.cell_id(i).is_none(), "empty depth cell must have no id");
                continue;
            }
            stamped += 1;
            let (row, col) = (i / grid, i % grid);
            let ndc_x = ((col as f32 + 0.5) / grid as f32) * 2.0 - 1.0;
            let ndc_y = 1.0 - ((row as f32 + 0.5) / grid as f32) * 2.0;
            let dir = normalize(add(
                fwd,
                add(
                    scale3(right, ndc_x * tan_half),
                    scale3(up, ndc_y * tan_half),
                ),
            ));
            let id = g.cell_id(i).expect("stamped cell names an entity");
            let bounds = world.geometry(id).unwrap().bounds.unwrap();
            match ray_aabb(eye.position, dir, &bounds, params.near, params.far) {
                Some(t) if (t - cell_depth).abs() < 0.5 => {}
                _ => fabricated += 1,
            }
        }
        assert!(stamped > 0, "grid must stamp some cells");
        assert_eq!(
            fabricated, 0,
            "{fabricated}/{stamped} stamped cells were fabricated (ray misses)"
        );
    }

    /// ORDEAL #7g-b — non-bypassable allocation safety. Raising `max_grid` can
    /// never make allocation unsafe: a 10^6 grid (10^12 cells) is a TYPED error
    /// (checked arithmetic / fallible reserve), never an OOM, panic, or abort.
    #[test]
    fn oversized_grid_is_a_typed_error_not_oom() {
        let world = naruko();
        let eye = world.spawn_pose().unwrap();
        let r = look(
            &world,
            eye,
            LookParams {
                grid: 1_000_000,
                max_grid: 2_000_000,
                layers: Layers::BOTH,
                ..Default::default()
            },
        );
        assert!(
            matches!(
                r,
                Err(LookError::AllocFailed { .. }) | Err(LookError::GridOverflow { .. })
            ),
            "grid 10^6 must be a typed allocation error, got {r:?}"
        );
    }

    /// ORDEAL #7f — the API reads a LIVE ECS. Gaze, mutate the tower's transform
    /// through the ECS, gaze again: the new range and cells must reflect it, no
    /// reload.
    #[test]
    fn live_ecs_mutation_is_reflected_on_next_gaze() {
        let world = naruko();
        let eye = world.spawn_pose().unwrap();
        let before = look(
            &world,
            eye,
            LookParams {
                grid: 32,
                ..Default::default()
            },
        )
        .unwrap();
        let r0 = before
            .nearest
            .iter()
            .find(|n| n.id == "lighthouse_tower")
            .unwrap()
            .range;

        // Move the tower much closer in Z via a raw ECS component write.
        let mut world = world;
        let entity = world
            .core
            .world
            .entity_for_gaia("lighthouse_tower")
            .unwrap();
        let tid = world.core.world.component_id("transform").unwrap();
        world
            .core
            .world
            .set_component(
                entity,
                tid,
                serde_json::json!({ "v": { "position": [0, 19, -40] } }),
            )
            .unwrap();

        let after = look(
            &world,
            eye,
            LookParams {
                grid: 32,
                ..Default::default()
            },
        )
        .unwrap();
        let r1 = after
            .nearest
            .iter()
            .find(|n| n.id == "lighthouse_tower")
            .expect("tower still visible after move")
            .range;
        assert!(
            r1 < r0 - 50.0,
            "range must drop after the ECS move ({r0} -> {r1})"
        );
        // The nearer tower subtends a larger span than before.
        assert!(
            after.cells_of("lighthouse_tower").len() > before.cells_of("lighthouse_tower").len(),
            "closer tower must cover more cells"
        );
    }

    /// ORDEAL #7g — input validation: typed errors, never a panic, never an
    /// unbounded allocation.
    #[test]
    fn invalid_params_are_typed_errors() {
        let world = naruko();
        let eye = world.spawn_pose().unwrap();
        let p = LookParams::default();
        assert!(matches!(
            look(&world, eye, LookParams { grid: 0, ..p }),
            Err(LookError::InvalidGrid { .. })
        ));
        assert!(matches!(
            look(&world, eye, LookParams { grid: 100_000, ..p }),
            Err(LookError::InvalidGrid { .. })
        ));
        assert!(matches!(
            look(&world, eye, LookParams { fov_deg: 0.0, ..p }),
            Err(LookError::InvalidFov { .. })
        ));
        assert!(matches!(
            look(
                &world,
                eye,
                LookParams {
                    fov_deg: 200.0,
                    ..p
                }
            ),
            Err(LookError::InvalidFov { .. })
        ));
        assert!(matches!(
            look(
                &world,
                eye,
                LookParams {
                    fov_deg: f32::NAN,
                    ..p
                }
            ),
            Err(LookError::InvalidFov { .. })
        ));
        assert!(matches!(
            look(
                &world,
                eye,
                LookParams {
                    near: 10.0,
                    far: 5.0,
                    ..p
                }
            ),
            Err(LookError::InvalidRange { .. })
        ));
        assert!(look(&world, eye, p).is_ok());
    }

    /// N6 — `tie_eps` is validated: NaN, +inf, or a negative value is a typed
    /// `InvalidTieEps`, never a silent nonsense tiebreak. Zero and any finite
    /// positive value are accepted (0 = exact-depth ties only).
    #[test]
    fn invalid_tie_eps_is_a_typed_error() {
        let world = naruko();
        let eye = world.spawn_pose().unwrap();
        let p = LookParams::default();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1e-6, -1.0] {
            assert!(
                matches!(
                    look(&world, eye, LookParams { tie_eps: bad, ..p }),
                    Err(LookError::InvalidTieEps { .. })
                ),
                "tie_eps {bad} must be InvalidTieEps"
            );
        }
        // Finite, >= 0 accepted.
        assert!(look(&world, eye, LookParams { tie_eps: 0.0, ..p }).is_ok());
        assert!(look(&world, eye, LookParams { tie_eps: 0.25, ..p }).is_ok());
    }

    /// N5 — HONEST byte budget. The ceiling ruling holds only if the non-OOM
    /// promise is REAL: the SoA reserve is checked in BYTES before any
    /// allocation. A grid at the cell ceiling with ids requested on a tiny world
    /// fits the default byte budget (derived from the 4096² both-layers worst
    /// case) and succeeds; lowering `max_grid_bytes` proves the breach path is a
    /// typed `ByteBudget` error — never an OOM, panic, or abort.
    #[test]
    fn byte_budget_is_enforced_before_allocation() {
        let world = naruko();
        let eye = world.spawn_pose().unwrap();

        // At the cell ceiling (4096² = MAX_GRID_CELLS) with ids requested on a
        // tiny world: fits the DEFAULT budget → OK.
        let ok = look(
            &world,
            eye,
            LookParams {
                grid: 4096,
                max_grid: 4096,
                layers: Layers::BOTH,
                ..Default::default()
            },
        );
        assert!(
            ok.is_ok(),
            "ceiling grid within the default byte budget must succeed, got {ok:?}"
        );

        // Lower the byte budget below what this grid needs → typed ByteBudget,
        // rejected BEFORE any allocation (no OOM).
        let breach = look(
            &world,
            eye,
            LookParams {
                grid: 4096,
                max_grid: 4096,
                layers: Layers::BOTH,
                max_grid_bytes: 1024, // 1 KiB — far below 4096² × per-cell bytes
                ..Default::default()
            },
        );
        assert!(
            matches!(breach, Err(LookError::ByteBudget { .. })),
            "under-budget grid must be a typed ByteBudget error, got {breach:?}"
        );

        // N5 REMAINDER — the ID-TABLE / MAPPING path is in the budget. A budget
        // that covers ONLY the per-cell grid buffers (excluding the O(entities)
        // id-table + mapping) must now BREACH when ids are requested, proving the
        // id-side bytes are accounted for (they were previously infallible and
        // unbudgeted). `grid_only` is exactly the per-cell reserve; the id side
        // (mapping + interned strings) pushes est over it.
        let grid = 32usize;
        let per_cell = 4 + std::mem::size_of::<Option<usize>>() + 4;
        let grid_only = grid * grid * per_cell;
        let table_breach = look(
            &world,
            eye,
            LookParams {
                grid,
                layers: Layers::IDS,
                max_grid_bytes: grid_only,
                ..Default::default()
            },
        );
        assert!(
            matches!(table_breach, Err(LookError::ByteBudget { .. })),
            "id-table + mapping bytes must be inside the budget (grid-only budget must breach), got {table_breach:?}"
        );
        // With the default budget (grid worst case + id-side headroom) the same
        // ids grid fits — the id side is a bounded O(entities) term, not O(cells).
        let table_ok = look(
            &world,
            eye,
            LookParams {
                grid,
                layers: Layers::IDS,
                ..Default::default()
            },
        );
        assert!(
            table_ok.is_ok(),
            "ids grid within the default budget must succeed, got {table_ok:?}"
        );
    }

    /// N1(c) — SERIALIZATION HONESTY. A depth-only glance's JSON must (1) OMIT
    /// the id channels entirely (`ids`, `id_table` absent — not `null`, not
    /// `[]`), and (2) carry NO `null` token in the depth channel: unhit cells
    /// serialize as the documented [`SPARSE_DEPTH_SENTINEL`] (`-1`), never
    /// `null`. Symmetrically an ids-only glance omits `depth`.
    #[test]
    fn serialized_channels_are_honest_no_null_no_unrequested_keys() {
        use crate::look::SPARSE_DEPTH_SENTINEL;
        let world = naruko();
        let eye = world.spawn_pose().unwrap();

        // depth-only
        let depth = look(
            &world,
            eye,
            LookParams {
                grid: 32,
                layers: Layers::DEPTH,
                ..Default::default()
            },
        )
        .unwrap();
        let json = serde_json::to_string(&depth).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            v.get("ids").is_none(),
            "depth-only JSON must omit the ids channel, got {json}"
        );
        assert!(
            v.get("id_table").is_none(),
            "depth-only JSON must omit id_table"
        );
        // The per-cell `ids` OUTPUT channel is absent as a top-level key (checked
        // above via `v.get`). Note `"ids"` still appears inside `layers` (the
        // requested-flags struct) — that is a bool flag, not the id data channel,
        // so only `id_table` (a data-channel-unique key) is substring-checked.
        assert!(
            !json.contains("\"id_table\""),
            "no id-table key may appear in a depth-only glance: {json}"
        );
        let arr = v
            .get("depth")
            .and_then(|d| d.as_array())
            .expect("depth channel present when requested");
        assert!(
            arr.iter().all(|x| !x.is_null()),
            "depth channel must carry NO null placeholder"
        );
        assert!(
            arr.iter()
                .any(|x| x.as_f64() == Some(SPARSE_DEPTH_SENTINEL as f64)),
            "unhit cells must serialize as the sparse sentinel (-1), not null"
        );
        // Belt-and-suspenders: the entire depth-only doc has no bare null token.
        assert!(
            !json.contains("null"),
            "depth-only glance JSON must be null-free, got {json}"
        );

        // ids-only: the depth channel is ABSENT, ids present.
        let ids = look(
            &world,
            eye,
            LookParams {
                grid: 32,
                layers: Layers::IDS,
                ..Default::default()
            },
        )
        .unwrap();
        let ijson = serde_json::to_string(&ids).unwrap();
        let iv: serde_json::Value = serde_json::from_str(&ijson).unwrap();
        // `depth` still appears inside `layers` (the flags struct); the DATA
        // channel is the top-level `depth` key, absent here (checked via `get`).
        assert!(
            iv.get("depth").is_none(),
            "ids-only JSON must omit the top-level depth data channel, got {ijson}"
        );
        assert!(iv.get("ids").is_some(), "ids-only JSON must carry ids");
    }

    #[test]
    fn deeper_grid_levels_are_just_a_param() {
        let world = naruko();
        let eye = world.spawn_pose().unwrap();
        for grid in [8usize, 32, 128] {
            let g = look(
                &world,
                eye,
                LookParams {
                    grid,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(g.cell_count(), grid * grid, "grid {grid} size");
            // The tower is always a caption (a real frustum hit) …
            assert!(
                caption_ids(&g).contains(&"lighthouse_tower".to_string()),
                "grid {grid} caption"
            );
            // … and it resolves into the grid once the cells are fine enough to
            // catch its ~4° subtense (grid-8 cells are 7.5° — honestly too coarse
            // for a ray-true hit, no fabricated coverage papers over it).
            if grid >= 32 {
                assert!(
                    g.is_above_horizon("lighthouse_tower"),
                    "grid {grid} above horizon"
                );
            }
        }
    }

    #[test]
    fn proprio_reports_pose_and_components() {
        let world = naruko();
        let p = proprio(&world, "lighthouse_tower").expect("proprio");
        assert_eq!(p.position, [0.0, 19.0, -120.0]);
        assert!(p.bounds_center.is_some());
        assert_eq!(p.emissive.as_deref(), Some("#f3e9ff"));
        assert!(p.components.iter().any(|c| c.name == "mesh"));

        let spawn = proprio(&world, "world_spawn").expect("spawn proprio");
        assert!(spawn.yaw.is_some());
    }

    #[test]
    fn look_is_pure_geometry_and_directional() {
        let world = naruko();
        let pos = world.spawn_pose().unwrap().position;
        // Facing away from the lighthouse (+Z) it must NOT be in the grid.
        let away = EyePose {
            position: pos,
            yaw: std::f32::consts::PI,
            pitch: 0.0,
        };
        let g = look(&world, away, LookParams::default()).unwrap();
        assert!(!g.is_above_horizon("lighthouse_tower"));
        assert!(g.cells_of("lighthouse_tower").is_empty());
    }

    /// ORDEAL #9 — the world is loaded into a Core ONCE; repeated gazes must not
    /// reload or rebuild it. ENV-GATED with a graceful skip ONLY — the committed
    /// test never references a path outside client-rs. Point `GAIA_BOOMTOWN_WORLD`
    /// at a large world to run it. The perf bounds are env-overridable params
    /// with documented defaults, never hardcoded magic:
    ///   `GAIA_PERF_GAZES` (default 20) — repeated gazes timed,
    ///   `GAIA_PERF_MAX_MS` (default 250) — per-gaze ceiling.
    #[test]
    fn boomtown_loads_once_and_gazes_stay_cheap() {
        let Some(root) = std::env::var_os("GAIA_BOOMTOWN_WORLD").map(PathBuf::from) else {
            eprintln!("GAIA_BOOMTOWN_WORLD unset; skipping perf ordeal");
            return;
        };
        if !root.is_dir() {
            eprintln!(
                "GAIA_BOOMTOWN_WORLD not a directory; skipping: {}",
                root.display()
            );
            return;
        }

        let env_usize = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let env_f64 = |k: &str, d: f64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let gazes = env_usize("GAIA_PERF_GAZES", 20).max(1);
        let max_per_gaze_ms = env_f64("GAIA_PERF_MAX_MS", 250.0);

        // Single load into the Core (may be slow — that's fine).
        let load_start = std::time::Instant::now();
        let world = World::load(&root).expect("load boomtown");
        let load_ms = load_start.elapsed().as_secs_f64() * 1e3;
        assert!(world.entities.len() > 1000, "boomtown should be large");

        let eye = world.spawn_pose().unwrap_or(EyePose {
            position: [0.0, 2.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
        });

        // Warm gaze, then a batch of repeated gazes over the SAME live world.
        let _ = look(&world, eye, LookParams::default()).unwrap();
        let start = std::time::Instant::now();
        for _ in 0..gazes {
            let g = look(&world, eye, LookParams::default()).unwrap();
            std::hint::black_box(g.entity_count);
        }
        let per_gaze_ms = start.elapsed().as_secs_f64() * 1e3 / gazes as f64;

        eprintln!(
            "boomtown: {} entities, load {load_ms:.0} ms, {per_gaze_ms:.1} ms/gaze",
            world.entities.len()
        );
        assert!(
            per_gaze_ms < max_per_gaze_ms,
            "default gaze {per_gaze_ms:.1} ms exceeded {max_per_gaze_ms} ms — world may be rebuilding per gaze"
        );
    }

    /// The live-Core constructor: build a Core by hand, register an entity, gaze.
    #[test]
    fn from_core_live_path_constructs_a_world() {
        let mut core = Core::default();
        let tid = core
            .world
            .register_component_json(r#"{"name":"transform","fields":{"v":"object"}}"#)
            .unwrap();
        let mid = core
            .world
            .register_component_json(r#"{"name":"mesh","fields":{"v":"object"}}"#)
            .unwrap();
        let e = core
            .world
            .create_entity(vec![
                (tid, serde_json::json!({"v":{"position":[0,0,-10]}})),
                (
                    mid,
                    serde_json::json!({"v":{"parts":[{"shape":"box","size":[2,2,2]}]}}),
                ),
            ])
            .unwrap();
        core.world.bind_gaia_id("cube", e).unwrap();

        let mut world = World::from_core(core, ".");
        world.register("cube".into(), e, vec!["transform".into(), "mesh".into()]);

        let g = world.geometry("cube").unwrap();
        assert_eq!(g.origin, [0.0, 0.0, -10.0]);
        let eye = EyePose {
            position: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
        };
        let glance = look(&world, eye, LookParams::default()).unwrap();
        assert!(
            !glance.cells_of("cube").is_empty(),
            "cube must project ahead"
        );
    }
}
