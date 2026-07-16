//! CANON ORDEALS — hand-derived against the LIVE canon realm `worlds/naruko`
//! (the 12-vessel realm the CLI gazes at by default), NOT the pinned 7-vessel
//! fixture under `tests/fixtures/naruko`. The fixture ordeals (in `src/lib.rs`)
//! stay the frozen geometric gate; THESE ordeals derive the same pure-geometry
//! truth against the canon scene and the canon spawn eye `[0,7,44]` yaw 0.
//!
//! DERIVATION DISCIPLINE (the Council's law): every asserted number is
//! hand-derived from the scene JSON's transforms/bounds with the math shown in
//! comments, and every tolerance is DERIVED from the f64-analytic-vs-live-f32
//! discrepancy actually measured on this geometry — never plucked.
//!
//! CANON GEOMETRY (no rotations/scales anywhere in `worlds/naruko`, so a world
//! AABB is the closed form `entityPos + partPos ± halfExtent`, unioned over the
//! visible parts; half = size/2 (box), [r,h/2,r] (cylinder widest ring / cone),
//! [r,r,r] (sphere)). The vessels and their derived world AABBs:
//!   env, world_spawn      — no mesh ⇒ no bounds (never renderable/captioned)
//!   naruko_terra          x[-200,200]  y[-0.5,0]     z[8,68]      (ground slab)
//!   naruko_seawall        x[-60,60]    y[0,1.4]      z[17,19]
//!   naruko_sea            x[-1000,1000] y[-1.65,-1.275] z[-1160,40] (huge slab)
//!   lighthouse_rock       x[-22,22]    y[-2,19]      z[-142,-98]
//!   lighthouse_tower      x[-5.5,5.5]  y[19,63]      z[-125.5,-114.5]
//!   naruko_pier           x[-15.3,-8.7] y[-2.7,1.025] z[-20,16]
//!   naruko_chain_posts    x[-34.14,30.14] y[1.4,2.5]  z[17.86,18.14]
//!   naruko_city_massing   x[30,78]     y[-2,56]      z[-59,-16]
//!   naruko_lantern        x[-7.58,-6.95] y[0,4.05]   z[19.45,20.55]
//!   naruko_stall_massing  x[-3.8,1.8]  y[0,2.9]      z[23,27.45]
//!   naruko_chrome_orb     x[-12.4,-11.6] y[1.02,3.22] z[11.6,12.4] (post+orb union)
//!   naruko_crate          x[-11.55,-10.75] y[4.1,4.9] z[12.6,13.4] (0.8 box, body
//!                         hung above the pier near the stall; authored/rest pose
//!                         — the physics only moves it at runtime, senses read the
//!                         static scene) center [-11.15, 4.5, 13]
//! Eye basis at yaw 0: fwd=(0,0,-1), right=(1,0,0), up=(0,1,0); FOV 60 vertical
//! (aspect 1) ⇒ tan_half = tan(30°) = 0.5773502692.

use oracle::{look, EyePose, Glance, Layers, LookParams, World};
use std::path::PathBuf;

/// The LIVE canon realm the CLI defaults to (`packages/oracle/../../worlds/naruko`).
/// NOT the pinned fixture — these ordeals derive against the growing 12-vessel
/// canon. Never mutated (read-only gaze).
fn canon_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../worlds/naruko")
        .canonicalize()
        .expect("canon naruko dir")
}

fn canon() -> World {
    World::load(canon_dir()).expect("load canon naruko")
}

fn caption_ids(g: &Glance) -> Vec<String> {
    g.nearest.iter().map(|n| n.id.clone()).collect()
}

fn range_of(g: &Glance, id: &str) -> f32 {
    g.nearest
        .iter()
        .find(|n| n.id == id)
        .unwrap_or_else(|| panic!("{id} not in captions: {:?}", caption_ids(g)))
        .range
}

/// CANON #1 — the DEFAULT glance from the canon spawn eye [0,7,44] yaw 0 sees
/// exactly the renderable vessels whose world AABB intersects the 60° frustum.
///
/// DERIVATION (frustum = 6 inward planes through the eye; a vessel is in-frustum
/// iff no plane's positive vertex falls outside). env & world_spawn carry no
/// mesh ⇒ never renderable. All ten meshed vessels project into the forward
/// (-Z) 60° cone from [0,7,44]:
///   • terra/seawall/sea straddle or sit just ahead and below (in) ;
///   • pier/lantern/chain_posts/stall_massing are the near foreground (in) ;
///   • city_massing sits to the RIGHT — its nearest corner projects to
///     ndc_x = 30/(60·tan30) = 0.867 < 1, INSIDE the right plane (in) ;
///   • lighthouse_rock/tower are straight ahead at ~164/167 m (in).
/// ⇒ entity_count = 10 (every meshed vessel), and every one of the ten ids is a
/// real frustum hit — no dead-clamp phantom, no missing vessel.
#[test]
fn canon_default_glance_frustum_set_is_the_ten_meshed_vessels() {
    let world = canon();
    let eye = world.spawn_pose().expect("canon spawn pose");
    assert_eq!(eye.position, [0.0, 7.0, 44.0], "canon spawn eye");
    assert_eq!(eye.yaw, 0.0, "canon spawn yaw");

    // Full frustum set (captions with a wide nearest_n and support included).
    let g = look(
        &world,
        eye,
        LookParams {
            nearest_n: 32,
            include_support: true,
            ..Default::default()
        },
    )
    .unwrap();
    // Realm grew at the Living World merge: lighthouse_beacon extracted from
    // the tower (center [0, 56.5, -120], range 171.3075 from spawn) — an
    // eleventh meshed vessel, in-frustum from the spawn eye. Then PLEROMA L2
    // set the chrome orb on a pier post (post cylinder + mirror sphere; union
    // AABB center [-12, 2.12, 12], range 34.5227 from spawn) — the TWELFTH
    // meshed vessel. It sits x=-12 at z_view=32 ahead: |x_off|=12 < the 60°
    // half-width tan30·32 = 18.475, INSIDE the left plane ⇒ in-frustum.
    // Then ELEMENTS P3 hung a wooden crate (a `body`) above the pier near the
    // stall (AABB center [-11.15, 4.5, 13], range 33.0390 from spawn) — the
    // THIRTEENTH meshed vessel. z_view = 44−13 = 31 ahead; |x_off| = 11.15 <
    // the half-width tan30·31 = 17.898, INSIDE the left plane ⇒ in-frustum.
    assert_eq!(
        g.entity_count, 13,
        "exactly the thirteen meshed vessels are in-frustum"
    );
    let caps = caption_ids(&g);
    for id in [
        "naruko_terra",
        "naruko_seawall",
        "naruko_sea",
        "lighthouse_rock",
        "lighthouse_tower",
        "naruko_pier",
        "naruko_chain_posts",
        "naruko_city_massing",
        "naruko_lantern",
        "naruko_stall_massing",
        "lighthouse_beacon",
        "naruko_chrome_orb",
        "naruko_crate",
    ] {
        assert!(caps.contains(&id.to_string()), "{id} must be in-frustum");
    }
    // env & world_spawn have no bounds ⇒ never captioned.
    assert!(!caps.contains(&"env".to_string()));
    assert!(!caps.contains(&"world_spawn".to_string()));
}

/// CANON #2 — the nearest-N caption ORDERING with derived ranges. Captions rank
/// by the eye→bounds-center distance; world-support surfaces (extent ≫ range,
/// here terra 400 m and sea 2000 m) are demoted out of the caption slots.
///
/// DERIVATION — range = |center − eye|, center = AABB midpoint, eye = [0,7,44]:
///   stall_massing  center [-1, 1.45, 25.225]  → √(1+30.80+352.50)   = 19.6037
///   lantern        center [-7.265, 2.025, 20] → √(52.78+24.75+576)  = 25.6320
///   chain_posts    center [-2, 1.95, 18]      → √(4+25.50+676)      = 26.5613
///   seawall        center [0, 0.7, 18]        → √(0+39.69+676)      = 26.7524
///   crate          center [-11.15, 4.5, 13]   → √(124.32+6.25+961)  = 33.0390
///   chrome_orb     center [-12, 2.12, 12]     → √(144+23.81+1024)   = 34.5227
///   pier           center [-12, -0.8375, -2]  → √(144+61.43+2116)   = 48.1812
///   city_massing   center [54, 27, -37.5]     → √(2916+400+6642.25) = 99.7910
///   lighthouse_rock center[0, 8.5, -120]      → √(0+2.25+26896)     = 164.0069
///   lighthouse_tower center[0, 41, -120]      → √(1156+26896)       = 167.4873
///   (sea 604.06, terra 9.41 are SUPPORT — demoted.)
/// So the default nearest_n=5 captions, in order (the P3 crate's 33.0390 now
/// displaces the chrome orb out of the top-5 — chrome orb falls to 6th at
/// 34.5227, pier to 7th at 48.1812):
///   [stall_massing, lantern, chain_posts, seawall, crate].
/// TOLERANCE (DERIVED): each range is the live f32 √(Σ(center−eye)²) vs the f64
/// reference above quoted to 4 decimals. The measured live-vs-reference
/// discrepancy across all eleven ranges peaks at 6.1e-5 m (at the 604 m sea
/// center) — that budget is the 4-decimal reference rounding (≤5e-5) plus the
/// f32 center/sub/sqrt round-off (≈1e-5). RANGE_TOL = 1e-3 m is ≈16× that
/// measured max — tight enough that a wrong center (±0.1 m) or a wrong AABB
/// fails by ≥100×, loose enough never to flap on the last quoted digit.
#[test]
fn canon_nearest_ordering_and_ranges_are_derived() {
    let world = canon();
    let eye = world.spawn_pose().unwrap();

    // Default nearest_n=5, support demoted.
    let g = look(&world, eye, LookParams::default()).unwrap();
    assert_eq!(
        caption_ids(&g),
        vec![
            "naruko_stall_massing",
            "naruko_lantern",
            "naruko_chain_posts",
            "naruko_seawall",
            "naruko_crate",
        ],
        "default nearest-5 caption order (P3 crate 33.0390 displaces chrome orb)"
    );

    // Support surfaces never eat a caption slot.
    assert!(!caption_ids(&g).contains(&"naruko_terra".to_string()));
    assert!(!caption_ids(&g).contains(&"naruko_sea".to_string()));

    // Derived ranges (RANGE_TOL derived above), read at a wide nearest_n so the
    // lighthouse pair is present too.
    const RANGE_TOL: f32 = 1e-3;
    let wide = look(
        &world,
        eye,
        LookParams {
            nearest_n: 32,
            include_support: true,
            ..Default::default()
        },
    )
    .unwrap();
    for (id, expect) in [
        ("naruko_stall_massing", 19.6037_f32),
        ("naruko_lantern", 25.6320),
        ("naruko_chain_posts", 26.5613),
        ("naruko_seawall", 26.7524),
        ("naruko_crate", 33.0390),
        ("naruko_chrome_orb", 34.5227),
        ("naruko_pier", 48.1812),
        ("naruko_city_massing", 99.7910),
        ("lighthouse_rock", 164.0069),
        ("lighthouse_tower", 167.4873),
        ("naruko_terra", 9.4108),
        ("naruko_sea", 604.0593),
    ] {
        let r = range_of(&wide, id);
        assert!(
            (r - expect).abs() < RANGE_TOL,
            "range({id}) live {r} != derived {expect} (tol {RANGE_TOL})"
        );
    }
}

/// CANON #3 — at the DEFAULT grid 8 the lighthouse tower resolves into ZERO
/// grid cells (it stays a caption only). This is honest coarse-grid coverage,
/// not a bug: a cell is filled ONLY on a true ray/AABB hit.
///
/// DERIVATION — the tower AABB x∈[-5.5,5.5] at the front face z=-114.5 is
/// z_view = 44-(-114.5) = 158.5 m ahead. Its horizontal subtense is
/// 11 m / 158.5 m = 0.0694 rad = 3.97°, NARROWER than a grid-8 cell (60°/8 =
/// 7.5°). The tower center is x=0 (ndc_x 0), so the two straddling columns are
/// col 3 (center ndc_x -0.125) and col 4 (+0.125). A cell-center ray at
/// ndc_x = ±0.125 crosses the tower's depth at world-x = ndc_x·tan_half·z_view =
/// 0.125·0.5774·158.5 = ±11.44 m — OUTSIDE the ±5.5 m half-width. Both rays
/// MISS, so the tower owns no grid-8 cell (matches the fixture's grid-8 finding
/// for its own eye). It resolves only from grid ≥ 32 (CANON #4).
#[test]
fn canon_tower_owns_no_grid8_cell() {
    let world = canon();
    let eye = world.spawn_pose().unwrap();
    let g = look(
        &world,
        eye,
        LookParams {
            grid: 8,
            layers: Layers::BOTH,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        g.cells_of("lighthouse_tower").is_empty(),
        "tower must own NO grid-8 cell (subtense 3.97° < 7.5° cell), got {:?}",
        g.cells_of("lighthouse_tower")
    );
    // It IS still a caption (a real frustum hit) at a wide enough nearest_n.
    let caps = look(
        &world,
        eye,
        LookParams {
            nearest_n: 32,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(caption_ids(&caps).contains(&"lighthouse_tower".to_string()));
}

/// CANON #4 — the tower's OWNED grid-32 cells and their per-cell ray-entry
/// depths, hand-derived through the 60° frustum from eye [0,7,44], with the
/// dominance boundary against lighthouse_rock proven geometrically.
///
/// COLUMNS — front face z_view=158.5, tan_half·z_view = 91.51. |ndc_x| ≤
/// 5.5/91.51 = 0.0601 to hit the ±5.5 m half-width. Grid-32 column centers
/// ndc_x = (col+0.5)/16 − 1: col15 → -0.03125, col16 → +0.03125 (both inside
/// 0.0601); col14 → -0.09375 (x=8.58 m, MISS). ⇒ cols {15,16}.
///
/// ROWS — the front-face intersection y = 7 + z_view·ndc_y·tan_half must land in
/// the tower's y∈[19,63]. ndc_y = 1 − (row+0.5)/16. Solving 19 ≤ y ≤ 63 gives
/// rows 6..=13 (row 6 highest, row 13 the base). Row 14 (ndc_y 0.09375) would
/// put the tower ray at y = 7 + 158.5·0.09375·0.5774 = 15.58 < 19 — it MISSES
/// the tower; so the tower owns rows 6..=13 only (16 cells).
///
/// DOMINANCE at the base — row 13 belongs to the TOWER, not the nearer rock,
/// because the ROCK ray MISSES there: at row 13 (ndc_y 0.15625) the rock's front
/// z=-98 is z_view=142, so the ray reaches the rock's z at y = 7 +
/// 142·0.15625·0.5774 = 19.81 > 19 (the rock's max y) — a miss; the tower (front
/// y = 7 + 158.5·0.15625·0.5774 = 21.29 ∈ [19,63]) is the only hit. Row 14
/// flips: the tower ray dips to 15.58 < 19 (miss) while the rock ray enters at
/// y = 14.69 < 19 (hit) — so row 14 is the ROCK's (nearer, 142.23 m).
///
/// DEPTH (closed form) — every owned cell enters through the FRONT z-face
/// (axial 158.5 m); a tilted cell ray travels a longer PATH d = 158.5·L to that
/// same plane, where L = √(1 + X² + Y²), X = ndc_x·tan_half, Y = ndc_y·tan_half
/// (the ray dir fwd + X·right + Y·up has length L, so reaching axial depth 158.5
/// costs 158.5·L along the unit ray). NOT the axial 158.5 (no cell ray is
/// axial), NOT the back-face 169.5. Per-row nearest-of-{15,16} depths, rows
/// 6→13:  167.5787, 165.8126, 164.2268, 162.8265, 161.6167, 160.6015,
/// 159.7847, 159.1693 — band [159.1693, 167.5787] (row 6 top = steepest ray =
/// longest path; row 13 base = flattest = shortest). These reference depths are
/// the f64 analytic 158.5·L, matched to ≤5.7e-14 m by an independent f64
/// ray/AABB slab probe.
///
/// TOLERANCE (DERIVED): the live result is the same geometry through
/// `camera_basis`/`normalize`/`ray_aabb` in f32. The measured analytic-vs-live
/// f32 discrepancy across the eight rows peaks at 8.83e-6 m; DEPTH_TOL = 1e-4 m
/// is a ≈11× margin over that measured max — tight enough that a wrong z-span
/// (±1 m) or an off-ray fabricated depth fails by orders of magnitude.
#[test]
fn canon_tower_cells_and_depth_band_at_grid_32() {
    let world = canon();
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

    // EXACT owned cell set: cols {15,16} × rows {6..=13} (16 cells).
    let cells: std::collections::BTreeSet<(usize, usize)> =
        g.cells_of("lighthouse_tower").into_iter().collect();
    let expected: std::collections::BTreeSet<(usize, usize)> = (6..=13)
        .flat_map(|r| [(r, 15usize), (r, 16usize)])
        .collect();
    assert_eq!(
        cells, expected,
        "tower owned cells are not the derived {{15,16}}×{{6..=13}}"
    );

    // DOMINANCE boundary: row 13 is the tower's (rock ray misses), row 14 is the
    // rock's (tower ray misses).
    assert_eq!(g.cell_id(13 * g.grid + 15), Some("lighthouse_tower"));
    assert_eq!(g.cell_id(13 * g.grid + 16), Some("lighthouse_tower"));
    assert_eq!(g.cell_id(14 * g.grid + 15), Some("lighthouse_rock"));
    assert_eq!(g.cell_id(14 * g.grid + 16), Some("lighthouse_rock"));

    // Per-row derived depths (f64 analytic 158.5·L; widen the live f32).
    const ROW_DEPTH: [(usize, f64); 8] = [
        (6, 167.578696),
        (7, 165.812595),
        (8, 164.226783),
        (9, 162.826529),
        (10, 161.616656),
        (11, 160.601466),
        (12, 159.784670),
        (13, 159.169322),
    ];
    const DEPTH_TOL: f64 = 1e-4;
    for (row, expect) in ROW_DEPTH {
        let d = (15..=16)
            .map(|c| g.cell_depth(row * g.grid + c))
            .fold(f32::INFINITY, f32::min) as f64;
        let delta = (d - expect).abs();
        assert!(
            delta < DEPTH_TOL,
            "row {row} nearest depth {d} != derived {expect} (Δ {delta}, tol {DEPTH_TOL})"
        );
    }

    // Band extremes are the true [159.169322, 167.578696].
    let depths: Vec<f64> = cells
        .iter()
        .map(|&(r, c)| g.cell_depth(r * g.grid + c) as f64)
        .collect();
    let lo = depths.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = depths.iter().copied().fold(0.0f64, f64::max);
    assert!((lo - 159.169322).abs() < DEPTH_TOL, "nearest depth {lo}");
    assert!((hi - 167.578696).abs() < DEPTH_TOL, "farthest depth {hi}");
}

/// CANON #5 — the DEPTH BAND of one derived cell column (col 16), front-face
/// path-length math verified per row. For each owned row the live depth must
/// equal 158.5·L (L = √(1 + X² + Y²), X = ndc_x·tan_half, Y = ndc_y·tan_half),
/// with ndc_x = (16.5/16) − 1 = 0.03125 for column 16 — an independent
/// re-derivation of the depth channel from the ray geometry (not a copy of the
/// CANON #4 constants).
#[test]
fn canon_depth_band_column_16_is_front_face_path_length() {
    let world = canon();
    let eye = world.spawn_pose().unwrap();
    let g = look(
        &world,
        eye,
        LookParams {
            grid: 32,
            layers: Layers::DEPTH,
            ..Default::default()
        },
    )
    .unwrap();

    let tan_half = 30f64.to_radians().tan(); // tan(30°)
    let zfront = 44.0 - (-114.5); // = 158.5 axial front-face depth
    let ndc_x = (16.5 / 16.0) - 1.0; // column 16 center = +0.03125
    const DEPTH_TOL: f64 = 1e-4; // ≈11× the 8.98e-6 m measured max path-length Δ
    for row in 6..=13usize {
        let ndc_y = 1.0 - (row as f64 + 0.5) / 16.0;
        let x = ndc_x * tan_half;
        let y = ndc_y * tan_half;
        let l = (1.0 + x * x + y * y).sqrt();
        let analytic = zfront * l; // d = 158.5 · L
        let live = g.cell_depth(row * g.grid + 16) as f64;
        let delta = (live - analytic).abs();
        assert!(
            delta < DEPTH_TOL,
            "row {row} col16 live {live} != front-path {analytic} (Δ {delta})"
        );
    }
}

/// CANON #6 — a MOVED-eye glance from the pier deck, eye [-13,2.7,15] yaw≈0.
/// Derive the caption set, that the lighthouse pair stays in-frustum, and the
/// pier range.
///
/// DERIVATION — eye [-13,2.7,15], basis unchanged (yaw 0). Renderable vessels
/// whose AABB still meets the forward 60° cone: pier (now under/around the eye),
/// terra (huge slab, straddles — SUPPORT), lighthouse_rock, lighthouse_tower,
/// and sea (SUPPORT). The market foreground (seawall/chain_posts/lantern/
/// stall_massing at z∈[17,27]) is now BEHIND the eye (z > 15) ⇒ culled; the
/// city sits far to the right, outside the 60° cone from x=-13 ⇒ culled. The
/// chrome orb (center [-12,2.12,12]) sits just ahead of the pier eye: − eye
/// [-13,2.7,15] = [1,-0.58,-3], within the z_view=3 cone (half-width
/// tan30·3 = 1.732 > |x_off|=1 and > |y_off|=0.58) ⇒ in-frustum. So
/// entity_count = 7 (with lighthouse_beacon); the non-support captions are
///   [chrome_orb, pier, lighthouse_rock, lighthouse_tower, beacon, sea].
/// Chrome-orb range: [1,-0.58,-3] → √(1 + 0.3364 + 9) = 3.2150 m — nearer
/// than the pier, so it leads the caption order.
/// Pier range: center [-12,-0.8375,-2] − eye [-13,2.7,15] = [1,-3.5375,-17] →
///   √(1 + 12.51 + 289) = 17.3929 m. Lighthouse still ahead: rock 135.75 m,
///   tower 140.93 m, both bearing ≈ +5.5° (well inside the 30° half-FOV).
#[test]
fn canon_moved_eye_pier_glance() {
    let world = canon();
    let eye = EyePose {
        position: [-13.0, 2.7, 15.0],
        yaw: 0.0,
        pitch: 0.0,
    };
    let g = look(
        &world,
        eye,
        LookParams {
            nearest_n: 16,
            include_support: true,
            ..Default::default()
        },
    )
    .unwrap();
    // + lighthouse_beacon (range 145.9056) since the Living World merge, and
    // + naruko_chrome_orb (range 3.2150, right beside the pier eye) since
    // PLEROMA L2 — seven meshed vessels in this frustum.
    assert_eq!(g.entity_count, 7, "moved-eye in-frustum count");

    // Non-support caption set (default demotes terra & sea).
    let plain = look(
        &world,
        eye,
        LookParams {
            nearest_n: 16,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        caption_ids(&plain),
        vec![
            "naruko_chrome_orb",
            "naruko_pier",
            "lighthouse_rock",
            "lighthouse_tower",
            "lighthouse_beacon",
            "naruko_sea",
        ],
        "moved-eye non-support caption set/order (chrome orb 3.2150 leads; beacon 145.9056 between tower 140.93 and sea)"
    );

    // Lighthouse pair still in-frustum.
    assert!(caption_ids(&g).contains(&"lighthouse_tower".to_string()));
    assert!(caption_ids(&g).contains(&"lighthouse_rock".to_string()));

    // Derived pier range (RANGE_TOL as in CANON #2).
    const RANGE_TOL: f32 = 1e-3;
    let r = range_of(&plain, "naruko_pier");
    assert!(
        (r - 17.3929).abs() < RANGE_TOL,
        "pier range live {r} != derived 17.3929"
    );
    // Chrome orb leads the order at 3.2150 m (nearer than the pier).
    let ro = range_of(&plain, "naruko_chrome_orb");
    assert!(
        (ro - 3.2150).abs() < RANGE_TOL,
        "chrome orb range live {ro} != derived 3.2150"
    );
}

/// CANON #7 — DETERMINISM. The canon glance is a pure function of world DATA:
/// serializing it twice yields BYTE-IDENTICAL output (both layers, so ids,
/// id_table and depth channels are all exercised).
#[test]
fn canon_glance_serialization_is_deterministic() {
    let world = canon();
    let eye = world.spawn_pose().unwrap();
    let params = LookParams {
        grid: 32,
        layers: Layers::BOTH,
        nearest_n: 8,
        ..Default::default()
    };
    let a = serde_json::to_string(&look(&world, eye, params).unwrap()).unwrap();
    let b = serde_json::to_string(&look(&world, eye, params).unwrap()).unwrap();
    assert_eq!(a, b, "canon glance must serialize byte-identically");

    // A fresh world load must also produce the identical serialization (the
    // gaze reads the scene, never accumulated state).
    let world2 = canon();
    let c = serde_json::to_string(&look(&world2, eye, params).unwrap()).unwrap();
    assert_eq!(a, c, "canon glance must be load-invariant");
}
