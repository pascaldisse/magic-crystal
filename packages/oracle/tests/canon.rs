//! F6 NOTE (senses read SOLVER TRUTH — NARUKO.md · GUARDIAN RULINGS · item 5):
//! most ordeals below gaze at the FRESHLY-LOADED realm — the AUTHORED
//! load-pose, physics never ticked. That is legitimate STATIC-scene truth for
//! every NON-physics vessel (nothing ever moves them, so load pose IS solver
//! truth). But for the physics `body` vessels (`naruko_crate`,
//! `naruko_stack_crate_0/1/2`) the load pose is NOT the runtime senses-truth
//! — ruling F6 makes the solver REST pose canonical for those four. This is a
//! REAL migration, not a relabel: `canon_nearest_ordering_and_ranges_are_derived`
//! below gazes a SEPARATE, solver-rested `World` (built by
//! `tests/rest_pose/mod.rs::rested_canon_world`, the same shared machinery
//! `rest_pose_canon.rs` uses) for exactly those four rows, and asserts the
//! SOLVER-MEASURED ranges (33.4042/34.3197/34.1931/34.0872) — not the
//! authored ones. `rest_pose_canon.rs` is the dedicated F6 proof (rest-tick
//! determinism + full per-vessel hand derivation with the analytic
//! cross-check shown); this file's migrated rows are the headline canon
//! consequence of that ruling. The header AABBs below stay labelled AUTHORED
//! (load-pose) since they describe the entities' authored geometry, with the
//! solver-rested numbers noted alongside for cross-reference.
//!
//! CANON ORDEALS — hand-derived against the LIVE canon realm `worlds/naruko`
//! (the 26-entity/24-in-frustum-meshed-vessel realm the CLI gazes at by
//! default — see CANON #1's derivation), NOT the pinned 7-vessel
//! fixture under `tests/fixtures/naruko`. The fixture ordeals (in `src/lib.rs`)
//! stay the frozen geometric gate; THESE ordeals derive the same pure-geometry
//! truth against the canon scene and the canon spawn eye `[0,7,44]` yaw 0.
//!
//! DERIVATION DISCIPLINE (the Council's law): every asserted number is
//! hand-derived from the scene JSON's transforms/bounds with the math shown in
//! comments, and every tolerance is DERIVED from the f64-analytic-vs-live-f32
//! discrepancy actually measured on this geometry — never plucked.
//!
//! CANON GEOMETRY (no entity rotations/scales in `worlds/naruko`; parts were
//! rotation-free until the SIGNAL RINGS, whose box chords carry z-rotations —
//! the oracle honors part Euler XYZ exactly (`model.rs`,
//! `rotation_rotates_the_world_aabb`), and the rings' union AABB has its own
//! closed form, derived below). For unrotated parts a world AABB is the closed
//! form `entityPos + partPos ± halfExtent`, unioned over the
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
//!   naruko_lantern        x[-8.05,-6.95] y[0,4.05]   z[19.45,20.55]
//!   naruko_stall_massing  x[-3.8,1.8]  y[0,2.9]      z[23,27.45]
//!   naruko_chrome_orb     x[-12.4,-11.6] y[1.02,3.22] z[11.6,12.4] (post+orb union)
//!   nari                  x[-0.2597,0.2598] y[1.4000,3.5467] z[17.918,18.082]
//!     — RITE V, the 13th vessel. NOT authored primitives: her bounds are the
//!     SKINNED vessel at sama's idle pose (`body = {preset:"nari"}`), a
//!     deterministic mesh whose skeleton-local AABB is min[-0.259727,-1.104977,
//!     -0.081836] max[0.259753,1.041707,0.081837] (byte-identical per the vessel
//!     Rite-V determinism ordeal), placed by her transform `pos=[0,2.505,18]`
//!     (feet on the seawall top y=1.4 = 2.505−1.104977). The senses compose the
//!     SAME body the renderer does — one truth. If the vessel geometry changes,
//!     these bounds change and the ordeal flags it (the body IS realm data).
//!   naruko_crate          AUTHORED (load-pose) x[-11.55,-10.75] y[4.1,4.9]
//!                         z[12.6,13.4] (0.8 box, body hung above the pier
//!                         near the stall — ELEMENTS P3, the 14th vessel).
//!                         center [-11.15, 4.5, 13]. This is NOT the canon
//!                         senses-truth for this vessel — ruling F6 makes the
//!                         SOLVER-RESTED pose canon: measured rest center
//!                         [-11.15,1.4759,13], range 33.4042 (analytic
//!                         cross-check y0=pier_top+half+radius=1.4750, range
//!                         33.4043, residual 0.0009 m — see
//!                         `rest_pose_canon.rs` for the full derivation and
//!                         `canon_nearest_ordering_and_ranges_are_derived`
//!                         below, which gazes the RESTED world for this row).
//!                         The solver drops it 3.025 m from its authored hang.
//!   naruko_stack_crate_0  AUTHORED (load-pose) x[-14.05,-13.25] y[1.075,1.875]
//!   naruko_stack_crate_1  z[12.6,13.4] (0.8 box, body,
//!   naruko_stack_crate_2  y[1.925,2.725]/[2.775,3.575] z[12.6,13.4]) RITE VI
//!                         · VI-1 — a stack of three crates authored resting on
//!                         the pier planks (chained rest-height derivation, same
//!                         convention as `naruko_crate`), the 22nd-24th vessels.
//!                         centers [-13.65, 1.475/2.325/3.175, 13]. NOT the
//!                         canon senses-truth either — ruling F6's SOLVER-
//!                         RESTED centers are [-13.65,1.4754/2.3259/3.1767,13],
//!                         ranges 34.3197/34.1931/34.0872 (analytic cross-check
//!                         34.3198/34.1931/34.0872 — see `rest_pose_canon.rs`).
//!                         The stack was authored ALREADY at its solver-rest,
//!                         so the rest pose matches this load pose to well
//!                         under REST_TOL (0.0004/0.0001/0.0003 m) — F6
//!                         confirms rather than moves the stack, but the
//!                         canon row below still gazes the RESTED world, per
//!                         the ruling.
//!   naruko_cat            x[-5.0808,-4.9192] y[0.0000,0.4555] z[22.9420,23.4789]
//!     — RITE V·V2, the 15th vessel. NOT authored primitives: the SKINNED
//!     pink_cat vessel (QUADRUPED morphology) at sama's idle pose
//!     (`body = {preset:"pink_cat"}`), skeleton-local AABB min[-0.080758,
//!     -0.359289,-0.058035] max[0.080762,0.096240,0.478921] (byte-identical per
//!     the V2 vessel determinism ordeal), placed by her transform
//!     `pos=[-5,0.359289,23]` (four paws on the terra top y=0 = 0.359289 +
//!     (-0.359289); the lowest paw vertex IS the mesh min y, so grounding the
//!     whole AABB grounds the paws). center [-4.999998, 0.227765, 23.210443]
//!   signal_ring_a         x[-6.275,6.275]   y[50.225,62.775] z[-118.175,-117.825]
//!   signal_ring_b         x[-10.275,10.275] y[46.225,66.775] z[-117.675,-117.325]
//!   signal_ring_c         x[-14.275,14.275] y[42.225,70.775] z[-117.175,-116.825]
//!     — THE SIGNAL RINGS (keyart: the lighthouse broadcasts). Each ring = 24
//!     box chords (size [2R·sin(π/24), 0.55, 0.35]) centered on a radius-R
//!     circle in the x-y plane about the beacon axis [0, 56.5], rotated
//!     [0,0,θ+π/2] (length tangent, 0.55 radial, 0.35 deep); R = 6/10/14, ring
//!     planes z = −118/−117.5/−117. UNION AABB CLOSED FORM: 24 ≡ 0 (mod 4)
//!     puts chords at θ = 0/90/180/270 whose corners reach exactly ±(R + 0.275)
//!     on the axes; every off-axis chord stays inside — worst case R=14, k=1
//!     (θ=15°): corner radius √((R+0.275)² + (chord/2 = 1.82735)²) = 14.3915 at
//!     angle 15° − atan(1.82735/14.275) = 7.706° → x = 14.3915·cos(7.706°) =
//!     14.2616 < 14.275 ⇒ the axis chords own the extents. So x,y ∈ ±(R+0.275)
//!     about [0, 56.5], z = ring plane ± 0.175; center = the entity position
//!     exactly (full symmetry). The `pulse` behavior scales only the RUNTIME
//!     pose — the senses read the STATIC scene (authored bind, crate precedent).
//!   naruko_mirror         x[2.96,3.04]  y[1.4,4.4]   z[17,19]
//!     — THE MIRROR PROOF: a polished panel (metallic 1.0, roughness 0.03)
//!     standing on the seawall top y=1.4; plane x=3, face toward −x.
//!     center [3, 2.9, 18]
//!   naruko_mirror_minor   x[-9.04,-8.96] y[1.4,4.4]  z[17,19]
//!     — the facing panel across the wall (mirror-in-mirror). center [-9, 2.9, 18]
//!   naruko_kami_orb       x[-1.35,-0.65] y[1.85,2.55] z[17.75,18.45]
//!     — cyan emissive sphere r 0.35 riding a kami orbit (center [-2.6,2.2,18.1],
//!     r 1.6, speed 0.9). Canon bounds are the AUTHORED bind pose — the orbit at
//!     angle 0 — the crate precedent: the senses read the static scene, the kami
//!     only moves it at runtime. center [-1, 2.2, 18.1]
//!   ── REALM SHINE (the Architect's spawn-sightline show, 07-17): five vessels
//!   dressing the terra 12-18 m DEAD AHEAD of the spawn eye so the ray tracer's
//!   hand is visible where his eyes open. All authored at z in [26,32] — AHEAD
//!   of the spawn eye (z 44) yet BEHIND every denoiser pose eye (front z 22,
//!   orbit_-20/+40 z 17-21) so the pinned RMSE bounds never see them, and behind
//!   the pier eye (z 15) so CANON #6 is untouched. ──
//!   naruko_show_chrome    x[2.4,6.6]  y[0,5.7]    z[27.4,31.6]
//!     — the Rite IV L2 CLOSE OBJECT: a chrome sphere (metallic 1.0, roughness
//!     0.02 = the perfect-mirror delta lobe) r 2.1 at part [0,3.6,0] on a dark
//!     pedestal cylinder (r 0.6, h 2.4) at [0,1.2,0]; transform [4.5,0,29.5].
//!     Union AABB: the sphere owns x/y-max/z, the pedestal owns y-min 0 ⇒
//!     center [4.5, 2.85, 29.5]. range = √(4.5²+4.15²+14.5²) = √247.7225 = 15.7392.
//!   naruko_show_mirror    x[-7.5858,-5.4142] y[0.15,6.65] z[26.1139,29.8861]
//!     — a polished panel (metallic 1.0, roughness 0.03) box size [0.18,6.5,4.2]
//!     at part [0,3.4,0], ROTATED [0,-0.5,0] (Euler y); transform [-6.5,0,28].
//!     The part center rides the entity Y-axis, so the Y-rotation leaves the
//!     center at [-6.5,3.4,28]; the world-AABB half-extents spread by the
//!     rotation (signal-ring rotated-AABB precedent): X_half = |hx·cosβ| +
//!     |hz·sinβ| = |0.09·0.877583| + |2.1·(-0.479426)| = 1.08578, Z_half =
//!     |hx·sinβ| + |hz·cosβ| = 0.043148 + 1.842924 = 1.88607 (hy 3.25 unaffected).
//!     range = √(6.5²+3.6²+16²) = √311.21 = 17.6411.
//!   naruko_show_light_a   x[1.05,1.95]  y[1.55,2.45]  z[28.55,29.45]
//!     — violet #b98aff emissive sphere r 0.45 on a kami orbit (center
//!     [-1.5,2.0,29], r 3.0, speed 0.5, phase 0); AUTHORED bind (angle 0) =
//!     transform [1.5,2.0,29] (kami_orb precedent: senses read the static bind,
//!     the orbit moves it only at runtime). range = √(1.5²+5²+15²) = √252.25 =
//!     15.8824.
//!   naruko_show_light_b   x[-4.45,-3.55] y[3.15,4.05] z[28.55,29.45]
//!     — cyan #37e0ff emissive sphere r 0.45, orbit (center [-1.5,3.6,29], r 2.5,
//!     speed -0.55, phase π); bind (angle π ⇒ cos -1) = transform [-4.0,3.6,29].
//!     range = √(4²+3.4²+15²) = √252.56 = 15.8921.
//!   naruko_show_light_c   x[-1.95,-1.05] y[4.75,5.65] z[31.35,32.25]
//!     — pink #ff6bb0 emissive sphere r 0.45, orbit (center [-1.5,5.2,29], r 2.8,
//!     speed 0.8, phase π/2 ⇒ sin 1) = transform [-1.5,5.2,31.8].
//!     range = √(1.5²+1.8²+12.2²) = √154.33 = 12.4230.
//!   naruko_first_ground   x[0,64]  y[-9.6,9.6]  z[128,192]
//!     — RITE VII · VII-0b (THE FIRST GROUND, the render weld), the 25th
//!     vessel: a GENERATED terrain patch, authored ONLY as a sigil
//!     `{seed:20260717, tile_x:0, tile_y:2}` — no mesh, no stored geometry.
//!     Bounds are ANALYTIC, derived from the sigil alone (never a built
//!     mesh): x/z span exactly the tile footprint (`tile_origin ..
//!     tile_origin+tile_size_m`; tile_size_m defaults to 64, and
//!     `tile_origin_m = (tile_x·grid_resolution, tile_y·grid_resolution) ×
//!     cell_size_m` = `(0·64, 2·64) × 1.0` = `(0, 128)`, `packages/seed/src/
//!     terrain.rs::tile_origin_m`); y spans `± height_amplitude`
//!     (`tile_size_m × DEFAULT_SLOPE_FRACTION` = `64 × 0.15` = `9.6`) — the
//!     fBm's normalized `[-1,1]` sum can never exceed its own peak amplitude
//!     regardless of octave count (`height_at_grid_index`'s `sum / norm`
//!     doc). center [32, 0, 160]. Placed clear of every other vessel: the
//!     realm's nearest static geometry going +z is `naruko_terra`, which
//!     ends at z=68 (see that row above) — 60 m of open ground short of the
//!     patch's z=128 near edge, so nothing overlaps. Range from the canon
//!     eye [0,7,44]: d = [32,-7,116], range = √(32²+7²+116²) = √14529 =
//!     120.5363 m — see `canon_terrain_patch_bounds_and_range_are_derived`.
//!     NOT in the default frustum: fwd=(0,0,-1) at yaw 0, and the patch's
//!     forward-axis component `dot(d,fwd) = -116` is NEGATIVE (the whole
//!     patch sits entirely BEHIND the eye's look direction, past the near
//!     plane's back side) — the 6-plane frustum test excludes it outright,
//!     so `canon_default_glance_frustum_set_is_the_ten_meshed_vessels`'s
//!     entity_count (24) and caption set are UNCHANGED by this growth (shown
//!     by that test still passing verbatim — the realm-growth law's other
//!     branch: sometimes the honest re-derivation is "no change", and this
//!     comment records why rather than silently relying on it).
//!   ── THE PHYSICS PLAYGROUND (toy-add a49bd77, 07-18): nine `body` box
//!   vessels (unrotated 0.8 cubes, half=0.4) planted on `naruko_terra`
//!   (ground top y=0 in this footprint) south of the market, z∈[32,35] —
//!   AABB = entityPos ± 0.4 on every axis (no part offset). ──
//!   playground_stack_0   pos [-3.0,0.45,35.0]  center [-3.0,0.45,35.0]
//!   playground_stack_1   pos [-3.0,1.30,35.0]  center [-3.0,1.30,35.0]
//!   playground_stack_2   pos [-3.0,2.15,35.0]  center [-3.0,2.15,35.0]
//!   playground_stack_3   pos [-3.0,3.00,35.0]  center [-3.0,3.00,35.0]
//!   playground_stack_4   pos [-3.0,3.85,35.0]  center [-3.0,3.85,35.0]
//!     — a 5-crate authored-at-rest column (a49bd77's `stack`): each level
//!     y = ground(0) + half(0.4) + contact_radius(0.05) = 0.45, then +0.85
//!     per level (2·half + contact_radius = 0.85, the SAME simplified
//!     chain-rest convention `naruko_stack_crate_0/1/2` was authored at —
//!     that stack's F6 solver-rested gaze (`rest_pose_canon.rs`) measured
//!     the true chain including `contact_margin` (1 mm) departs from this
//!     convention by ≤0.0004 m, well under RANGE_TOL — so these five are
//!     read at their AUTHORED (load) pose here, unticked, same precedent).
//!   playground_break_crate pos [-0.8,0.45,34.5] center [-0.8,0.45,34.5]
//!     — the bonded/fracturable crate (a49bd77 + b3ba864 love=0.5 +
//!     ad22ce8 settle=90), also ground-rest y=0.45, standing alone at rest
//!     (no impulse applied by a static gaze) so load pose is senses-truth
//!     unticked, same reasoning.
//!   playground_pyramid_0 pos [-2.35,0.45,32.0] center [-2.35,0.45,32.0]
//!   playground_pyramid_1 pos [-1.65,0.45,32.0] center [-1.65,0.45,32.0]
//!   playground_pyramid_2 pos [-2.00,1.30,32.0] center [-2.00,1.30,32.0]
//!     — a two-base/one-apex pyramid (a49bd77): the two base crates rest on
//!     the ground at y=0.45 (same formula), the apex centers exactly
//!     between them in x ((-2.35-1.65)/2 = -2.0) at y=1.30 = 0.45 + 0.85
//!     (one chain-rest level up, same convention as the stack).
//!   FRUSTUM (eye [0,7,44] yaw 0, planes below): the LEFT/RIGHT/TOP planes
//!   never bind for this cluster (all nine clear those three planes by wide
//!   margins — worst-case right-plane dot -13.42 vs offset -22, worst-case
//!   top-plane dot -15.84 vs offset -28.06, shown by direct computation);
//!   only the BOTTOM plane (n=(0,0.8660254,-0.5), offset
//!   0.8660254·7 − 0.5·44 = 6.0621780 − 22 = −15.9378220) is close enough to
//!   matter, since every crate sits low (y≈0.45–1.30) far out (z≈32–35).
//!   Conservative AABB test: outside iff 0.8660254·mx_y − 0.5·mn_z <
//!   −15.9378220, using the box's TOP-front corner (mx_y = center.y+0.4,
//!   mn_z = center.z−0.4 — the corner that maximizes the dot product):
//!     stack_0        mx_y=0.85 mn_z=34.6 → 0.7361216−17.3     = −16.5639 < −15.9378 ⇒ OUT (by 0.6261)
//!     stack_1        mx_y=1.70 mn_z=34.6 → 1.4722432−17.3     = −15.8278 ≥ −15.9378 ⇒ IN  (margin 0.1100)
//!     stack_2        mx_y=2.55 mn_z=34.6 → 2.2083648−17.3     = −15.0916 ≥ −15.9378 ⇒ IN
//!     stack_3        mx_y=3.40 mn_z=34.6 → 2.9444864−17.3     = −14.3555 ≥ −15.9378 ⇒ IN
//!     stack_4        mx_y=4.25 mn_z=34.6 → 3.6806080−17.3     = −13.6194 ≥ −15.9378 ⇒ IN
//!     break_crate    mx_y=0.85 mn_z=34.1 → 0.7361216−17.05    = −16.3139 < −15.9378 ⇒ OUT (by 0.3761)
//!     pyramid_0      mx_y=0.85 mn_z=31.6 → 0.7361216−15.80    = −15.0639 ≥ −15.9378 ⇒ IN
//!     pyramid_1      mx_y=0.85 mn_z=31.6 → 0.7361216−15.80    = −15.0639 ≥ −15.9378 ⇒ IN
//!     pyramid_2      mx_y=1.70 mn_z=31.6 → 1.4722432−15.80    = −14.3278 ≥ −15.9378 ⇒ IN
//!   ⇒ SEVEN of the nine are in-frustum (stack_1/2/3/4, pyramid_0/1/2);
//!   stack_0 and break_crate — the two LOWEST-and-farthest boxes (only
//!   ground level at z=34.6/34.1) — fall just below the bottom plane and
//!   are excluded. entity_count grows 29→36.
//!   Ranges (center−eye, eye [0,7,44]), the in-frustum seven:
//!     stack_1   d=[-3,-5.70,-9] → √(9+32.49+81)   = √122.49  = 11.0675
//!     stack_2   d=[-3,-4.85,-9] → √(9+23.5225+81) = √113.5225= 10.6547
//!     stack_3   d=[-3,-4.00,-9] → √(9+16+81)      = √106     = 10.2956
//!     stack_4   d=[-3,-3.15,-9] → √(9+9.9225+81)  = √99.9225 = 9.9961
//!     pyramid_0 d=[-2.35,-6.55,-12] → √(5.5225+42.9025+144) = √192.425 = 13.8717
//!     pyramid_1 d=[-1.65,-6.55,-12] → √(2.7225+42.9025+144) = √189.625 = 13.7704
//!     pyramid_2 d=[-2,-5.70,-12]    → √(4+32.49+144)        = √180.49  = 13.4347
//!   All seven undercut every prior nearest-5 entrant (`naruko_show_light_c`
//!   at 12.4230 was nearest before) — the four stack crates AND light_c beat
//!   the pyramid trio, so the new default top-5 is
//!   [stack_4, stack_3, stack_2, stack_1, show_light_c] (9.9961 < 10.2956 <
//!   10.6547 < 11.0675 < 12.4230), pushing show_chrome (15.7392) etc. down;
//!   the pyramid trio (13.43–13.87) ranks 6th–8th, still ahead of show_chrome.
//! Eye basis at yaw 0: fwd=(0,0,-1), right=(1,0,0), up=(0,1,0); FOV 60 vertical
//! (aspect 1) ⇒ tan_half = tan(30°) = 0.5773502692.

use oracle::{look, EyePose, Glance, Layers, LookParams, World};
use std::path::PathBuf;

/// F6 — shared tick-to-rest + transform-injection machinery, reused verbatim
/// from `rest_pose_canon.rs` (see `tests/rest_pose/mod.rs`).
mod rest_pose;

/// The LIVE canon realm the CLI defaults to (`packages/oracle/../../worlds/naruko`).
/// NOT the pinned fixture — these ordeals derive against the growing
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
///
/// RITE V grows the realm to THIRTEEN: `nari` (the embodied vessel) stands on
/// the seawall, world AABB center [0, 2.4734, 18] — dead ahead at z_view=26,
/// |x|≈0 ⇒ deep inside the 60° cone ⇒ in-frustum. Then ELEMENTS P3 grows it to
/// FOURTEEN: `naruko_crate` (a wooden `body`) hung above the pier, world AABB
/// center [-11.15, 4.5, 13] — z_view=31 ahead, |x_off|=11.15 < tan30·31 = 17.898
/// ⇒ inside the left plane ⇒ in-frustum. Then RITE V·V2 grows it to FIFTEEN:
/// `naruko_cat` (the skinned pink_cat vessel) by the ramen stall, world AABB
/// center [-5, 0.2278, 23.21] — z_view = 44−23.21 = 20.79 ahead, |x_off|=5 <
/// tan30·20.79 = 12.00 ⇒ inside the left plane ⇒ in-frustum.
///
/// THE SIGNAL RINGS grow it to EIGHTEEN: three meshed ring vessels dead ahead
/// on the beacon axis (centers [0, 56.5, −118/−117.5/−117], z_view = 162/161.5/
/// 161), |x| ≤ 14.275 ≪ tan30·161 = 92.95 and y_off = 49.5 ± 14.275 ≪ the same
/// half-height ⇒ all three deep inside the 60° cone ⇒ in-frustum. The 16th–18th
/// meshed vessels.
///
/// THE MIRROR PROOF grows it to TWENTY-ONE: naruko_mirror (center [3,2.9,18],
/// z_view 26, |x_off|=3 < tan30·26 = 15.01), naruko_mirror_minor (center
/// [-9,2.9,18], |x_off|=9 < 15.01) and naruko_kami_orb (bind center
/// [-1,2.2,18.1], z_view 25.9, |x_off|=1 < tan30·25.9 = 14.95) — the 19th–21st
/// meshed vessels, all dead ahead of the spawn eye ⇒ in-frustum. So
/// entity_count = 21. RITE VI · VI-1 (THE STACK TOPPLES) then adds THREE more:
/// `naruko_stack_crate_0/1/2`, a stack of `body` crates authored resting on
/// the pier planks (center [-13.65, y, 13], y = 1.475/2.325/3.175). Each sits
/// z_view = 44−13 = 31 ahead (same as `naruko_crate`), |x_off| = 13.65 <
/// tan30·31 = 17.898 ⇒ inside both side planes; y_off ∈
/// [1.475−7, 3.175−7] = [−5.525,−3.825], well inside the same half-width
/// vertically (aspect 1, same cone) ⇒ in-frustum. entity_count = 24.
///
/// REALM SHINE grows it to TWENTY-NINE: the five spawn-sightline vessels
/// (naruko_show_chrome center [4.5,2.85,29.5], _mirror [-6.5,3.4,28],
/// _light_a [1.5,2,29], _light_b [-4,3.6,29], _light_c [-1.5,5.2,31.8]), all
/// z in [27,32] DEAD AHEAD of the spawn eye. Each |x_off| and |y_off| (from
/// eye y=7) sits well inside its cone half-width tan30·z_view (chrome 4.5/4.15
/// < 8.37, mirror 6.5/3.6 < 9.24, light_a 1.5/5.0 < 8.66, light_b 4.0/3.4 <
/// 8.66, light_c 1.5/1.8 < 7.04) ⇒ all five in-frustum. entity_count = 29.
///
/// THE PHYSICS PLAYGROUND (toy-add a49bd77) grows it to THIRTY-SIX: seven of
/// the nine new box vessels clear the BOTTOM plane (the only plane close
/// enough to bind at their low y / far z — left/right/top all clear by wide
/// margins), the conservative AABB corner test derived in the header comment
/// (n=(0,0.8660254,-0.5), offset -15.9378220): stack_1 (margin +0.1100),
/// stack_2, stack_3, stack_4, pyramid_0, pyramid_1, pyramid_2 all land INSIDE;
/// stack_0 (-0.6261) and playground_break_crate (-0.3761) land OUTSIDE — the
/// two lowest boxes sitting at the shallowest z (34.6/34.1, nearest the eye's
/// downward sightline edge) dip just under the bottom plane. entity_count =
/// 29 + 7 = 36. PLAY adds `bldg_tower` + `bldg_basin` = 38.
#[test]
fn canon_default_glance_frustum_set_is_the_ten_meshed_vessels() {
    let world = canon();
    let eye = world.spawn_pose().expect("canon spawn pose");
    assert_eq!(eye.position, [0.0, 7.0, 44.0], "canon spawn eye");
    assert_eq!(eye.yaw, 0.0, "canon spawn yaw");

    // Full frustum set (captions with a wide nearest_n and support included).
    // nearest_n=40: 38 in-frustum meshed vessels (old 29→36; PLAY's
    // bldg_tower + bldg_basin →38) + terra + sea (support) = 40 exactly.
    let g = look(
        &world,
        eye,
        LookParams {
            nearest_n: 40,
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
    // RITE V stood `nari` on the seawall (skinned vessel, AABB center
    // [0, 2.4734, 18], range 26.3911 from spawn) — the THIRTEENTH meshed vessel,
    // dead ahead ⇒ in-frustum. Then ELEMENTS P3 hung a wooden crate (a `body`)
    // above the pier near the stall (AABB center [-11.15, 4.5, 13], range
    // 33.0390 from spawn) — the FOURTEENTH meshed vessel. z_view = 44−13 = 31
    // ahead; |x_off| = 11.15 < the half-width tan30·31 = 17.898, INSIDE the left
    // plane ⇒ in-frustum. RITE V·V2 skinned the pink_cat by the stall (AABB
    // center [-5, 0.2278, 23.21], range 22.4292 from spawn) — the FIFTEENTH
    // meshed vessel. z_view = 44−23.21 = 20.79 ahead; |x_off| = 5 < the
    // half-width tan30·20.79 = 12.00, INSIDE the left plane ⇒ in-frustum.
    // + the three SIGNAL RINGS (centers [0,56.5,−118/−117.5/−117], ranges
    // 169.3938/168.9157/168.4377 from spawn — derivation in CANON #2), the
    // 16th–18th meshed vessels, dead ahead on the beacon axis ⇒ in-frustum.
    // THE MIRROR PROOF stood two polished panels on the seawall (naruko_mirror
    // center [3,2.9,18] and naruko_mirror_minor center [-9,2.9,18], both
    // |x_off| < tan30·26 = 15.01) and set the cyan kami orb between them
    // (naruko_kami_orb, bind center [-1,2.2,18.1], |x_off|=1 < tan30·25.9 =
    // 14.95) — the 19th–21st meshed vessels, dead ahead. THE PHYSICS
    // PLAYGROUND adds SEVEN more (stack_1/2/3/4, pyramid_0/1/2 — bottom-plane
    // corner test derived in the header comment); stack_0 and
    // playground_break_crate are the two that fall just OUTSIDE the bottom
    // plane and stay uncaptioned.
    //
    // HAND DERIVATION — old 36, one vessel each:
    // naruko_terra, naruko_seawall, naruko_sea, lighthouse_rock,
    // lighthouse_tower, naruko_pier, naruko_chain_posts, naruko_city_massing,
    // naruko_lantern, naruko_stall_massing, lighthouse_beacon,
    // naruko_chrome_orb, nari, naruko_cat, signal_ring_a, signal_ring_b,
    // signal_ring_c, naruko_mirror, naruko_mirror_minor, naruko_kami_orb,
    // naruko_crate, naruko_stack_crate_0, naruko_stack_crate_1,
    // naruko_stack_crate_2, naruko_show_chrome, naruko_show_mirror,
    // naruko_show_light_a, naruko_show_light_b, naruko_show_light_c,
    // playground_stack_1, playground_stack_2, playground_stack_3,
    // playground_stack_4, playground_pyramid_0, playground_pyramid_1,
    // playground_pyramid_2 = 36. PLAY scene additions each carry one mesh:
    // bldg_tower (bonded body) + bldg_basin (fluid container mesh) = 2;
    // 36 + 2 = 38. The two old culled playground bodies remain excluded.
    assert_eq!(
        g.entity_count, 38,
        "exactly thirty-eight meshed vessels are in-frustum: hand-derived old 36 + bldg_tower + bldg_basin"
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
        "nari",
        "naruko_crate",
        "naruko_cat",
        "signal_ring_a",
        "signal_ring_b",
        "signal_ring_c",
        "naruko_mirror",
        "naruko_mirror_minor",
        "naruko_kami_orb",
        "naruko_stack_crate_0",
        "naruko_stack_crate_1",
        "naruko_stack_crate_2",
        "naruko_show_chrome",
        "naruko_show_mirror",
        "naruko_show_light_a",
        "naruko_show_light_b",
        "naruko_show_light_c",
        "playground_stack_1",
        "playground_stack_2",
        "playground_stack_3",
        "playground_stack_4",
        "playground_pyramid_0",
        "playground_pyramid_1",
        "playground_pyramid_2",
        "bldg_tower",
        "bldg_basin",
    ] {
        assert!(caps.contains(&id.to_string()), "{id} must be in-frustum");
    }
    // env & world_spawn have no bounds ⇒ never captioned.
    assert!(!caps.contains(&"env".to_string()));
    assert!(!caps.contains(&"world_spawn".to_string()));
    // The two culled playground boxes never enter the caption set (bottom-
    // plane corner test: stack_0 margin -0.6261, break_crate margin -0.3761 —
    // header comment derivation).
    assert!(!caps.contains(&"playground_stack_0".to_string()));
    assert!(!caps.contains(&"playground_break_crate".to_string()));
}

/// CANON #2 — the nearest-N caption ORDERING with derived ranges. Captions rank
/// by the eye→bounds-center distance; world-support surfaces (extent ≫ range,
/// here terra 400 m and sea 2000 m) are demoted out of the caption slots.
///
/// DERIVATION — range = |center − eye|, center = AABB midpoint, eye = [0,7,44]:
///   stall_massing  center [-1, 1.45, 25.225]  → √(1+30.80+352.50)   = 19.6037
///   cat            center [-5, 0.2278, 23.210] → √(25+45.863+432.206) = 22.4292
///   lantern        center [-7.5, 2.025, 20]   → √(56.25+24.75+576)  = 25.6320
///   chain_posts    center [-2, 1.95, 18]      → √(4+25.50+676)      = 26.5613
///   seawall        center [0, 0.7, 18]        → √(0+39.69+676)      = 26.7524
///   crate          center [-11.15, 4.5, 13]   → √(124.32+6.25+961)  = 33.0390
///   chrome_orb     center [-12, 2.12, 12]     → √(144+23.81+1024)   = 34.5227
///   nari           center [0, 2.4734, 18]     → √(0+20.488+676)     = 26.3911
///   pier           center [-12, -0.8375, -2]  → √(144+61.43+2116)   = 48.1812
///   city_massing   center [54, 27, -37.5]     → √(2916+400+6642.25) = 99.7910
///   lighthouse_rock center[0, 8.5, -120]      → √(0+2.25+26896)     = 164.0069
///   lighthouse_tower center[0, 41, -120]      → √(1156+26896)       = 167.4873
///   signal_ring_a  center [0, 56.5, -118]     → √(0+2450.25+26244)  = 169.3938
///   signal_ring_b  center [0, 56.5, -117.5]   → √(0+2450.25+26082.25)= 168.9157
///   signal_ring_c  center [0, 56.5, -117]     → √(0+2450.25+25921)  = 168.4377
///   mirror         center [3, 2.9, 18]        → √(9+16.81+676)      = 26.4917
///   mirror_minor   center [-9, 2.9, 18]       → √(81+16.81+676)     = 27.8175
///   kami_orb       center [-1, 2.2, 18.1]     → √(1+23.04+670.81)   = 26.3600
///   playground_stack_1   center [-3,1.30,35] → √(9+32.49+81)     = 11.0675
///   playground_stack_2   center [-3,2.15,35] → √(9+23.5225+81)  = 10.6547
///   playground_stack_3   center [-3,3.00,35] → √(9+16+81)       = 10.2956
///   playground_stack_4   center [-3,3.85,35] → √(9+9.9225+81)   = 9.9961
///   playground_pyramid_0 center [-2.35,0.45,32] → √(5.5225+42.9025+144) = 13.8717
///   playground_pyramid_1 center [-1.65,0.45,32] → √(2.7225+42.9025+144) = 13.7704
///   playground_pyramid_2 center [-2,1.30,32]    → √(4+32.49+144)        = 13.4347
///   (stack_0 11.5283, break_crate 11.5669 are OUT OF FRUSTUM — never
///   captioned, see CANON #1's bottom-plane derivation — so they never
///   compete for a slot.)
///   (sea 604.06, terra 9.41 are SUPPORT — demoted.)
/// The SIGNAL RINGS at 168.4–169.4 m sit far beyond the top-5 band ⇒ inert on
/// the caption order (they rank 16th–18th).
/// So the default nearest_n=5 captions, in order. RITE V inserts `nari` at
/// 26.3911 — between lantern (25.6320) and chain_posts (26.5613). RITE V·V2
/// inserts `naruko_cat` at 22.4292 — between stall_massing (19.6037) and lantern
/// (25.6320), so the cat takes the SECOND slot. THE MIRROR PROOF's kami orb
/// (26.3600) slots just AHEAD of nari (26.3911) — the 3.1 cm range gap is
/// 31× RANGE_TOL — taking the FOURTH slot and pushing chain_posts (26.5613)
/// out of the top-5:
///   [stall_massing, naruko_cat, lantern, naruko_kami_orb, nari]
/// (mirror 6th at 26.4917, chain_posts 7th at 26.5613, seawall 8th at 26.7524,
/// mirror_minor 9th at 27.8175, crate 10th at 33.0390, chrome orb 11th at
/// 34.5227, pier 12th at 48.1812, city 13th, rock/tower 14th/15th, rings
/// 16th–18th). THE PHYSICS PLAYGROUND then displaces the WHOLE top-5: its
/// seven in-frustum boxes range 9.9961–13.8717, all nearer than the previous
/// leader `naruko_show_light_c` (12.4230) except the pyramid trio (13.43–
/// 13.87, which slot AFTER light_c) — the four stack crates (9.9961/10.2956/
/// 10.6547/11.0675, nearest-first) fill slots 1–4 and light_c takes slot 5:
///   [stack_4, stack_3, stack_2, stack_1, show_light_c]
/// pushing show_chrome (15.7392) out of the top-5; the pyramid trio
/// (13.4347/13.7704/13.8717) ranks 6th–8th, ahead of show_chrome (9th).
/// (stack_0 and break_crate never enter this competition — CANON #1 excludes
/// them from the frustum outright.)
/// TOLERANCE (DERIVED): each range is the live f32 √(Σ(center−eye)²) vs the f64
/// reference above quoted to 4 decimals. The measured live-vs-reference
/// discrepancy across all twenty ranges peaks at 6.1e-5 m (at the 604 m sea
/// center) — that budget is the 4-decimal reference rounding (≤5e-5) plus the
/// f32 center/sub/sqrt round-off (≈1e-5). RANGE_TOL = 1e-3 m is ≈16× that
/// measured max — tight enough that a wrong center (±0.1 m) or a wrong AABB
/// fails by ≥100×, loose enough never to flap on the last quoted digit.
///
/// F6 MIGRATION (ruling: "senses read SOLVER TRUTH"): the four PHYSICS rows
/// (`naruko_crate`, `naruko_stack_crate_0/1/2`) below gaze a SEPARATE,
/// solver-RESTED `World` (`rest_pose::rested_canon_world`, the same shared
/// machinery `rest_pose_canon.rs` uses, ticked to the checked-deterministic
/// `REST_TICK`) instead of the freshly-loaded `world` every other row uses.
/// Their expected ranges are therefore the SOLVER-MEASURED numbers
/// (33.4042/34.3197/34.1931/34.0872), not the authored load-pose ones
/// (33.0390/34.3198/34.1932/34.0874) — full derivation, both numbers, and the
/// residuals in `rest_pose_canon.rs`. ORDERING CHECK: the crate's range grows
/// 33.0390→33.4042 (Δ+0.3652, the solver drops it 3.025 m from its authored
/// hang) but its neighbors are mirror_minor at 27.8175 (below) and chrome_orb
/// at 34.5227 (above) — 33.4042 still sits strictly between them, so the rank
/// is UNCHANGED (still directly before chrome_orb). The three stack crates
/// move by ≤0.0002 m (authored ALREADY at solver-rest) — no rank changes
/// there either. Static (non-physics) rows are UNCHANGED: their load pose IS
/// solver truth, nothing ever moves them.
#[test]
fn canon_nearest_ordering_and_ranges_are_derived() {
    let world = canon();
    let eye = world.spawn_pose().unwrap();

    // Default nearest_n=5, support demoted. THE PHYSICS PLAYGROUND now owns
    // the whole top-5: the four stack crates (9.9961/10.2956/10.6547/11.0675,
    // nearest-first) undercut every prior entrant, and show_light_c (12.4230,
    // the old leader) takes the fifth slot ahead of the pyramid trio
    // (13.4347-13.8717) and show_chrome (15.7392). terra (9.4108, nearer than
    // stack_4) stays SUPPORT-demoted.
    let g = look(&world, eye, LookParams::default()).unwrap();
    assert_eq!(
        caption_ids(&g),
        vec![
            "playground_stack_4",
            "playground_stack_3",
            "playground_stack_2",
            "playground_stack_1",
            "naruko_show_light_c",
        ],
        "default nearest-5 caption order (Physics Playground: the stack column 4/3/2/1 leads, nearer than every prior entrant; light_c holds the fifth slot)"
    );

    // Support surfaces never eat a caption slot.
    assert!(!caption_ids(&g).contains(&"naruko_terra".to_string()));
    assert!(!caption_ids(&g).contains(&"naruko_sea".to_string()));

    // Derived ranges (RANGE_TOL derived above), read at a wide nearest_n so the
    // lighthouse pair is present too.
    const RANGE_TOL: f32 = 1e-3;
    // nearest_n=40 (see CANON #1: 36 in-frustum + terra + sea = 38).
    let wide = look(
        &world,
        eye,
        LookParams {
            nearest_n: 40,
            include_support: true,
            ..Default::default()
        },
    )
    .unwrap();
    for (id, expect) in [
        ("naruko_stall_massing", 19.6037_f32),
        ("naruko_cat", 22.4292),
        ("naruko_lantern", 25.6320),
        ("naruko_chain_posts", 26.5613),
        ("naruko_seawall", 26.7524),
        ("nari", 26.3911),
        ("naruko_kami_orb", 26.3600),
        ("naruko_mirror", 26.4917),
        ("naruko_mirror_minor", 27.8175),
        ("naruko_chrome_orb", 34.5227),
        ("naruko_pier", 48.1812),
        ("naruko_city_massing", 99.7910),
        ("lighthouse_rock", 164.0069),
        ("lighthouse_tower", 167.4873),
        ("signal_ring_a", 169.3938),
        ("signal_ring_b", 168.9157),
        ("signal_ring_c", 168.4377),
        ("naruko_terra", 9.4108),
        ("naruko_sea", 604.0593),
        // REALM SHINE (derived above; the spawn-sightline show).
        ("naruko_show_light_c", 12.4230),
        ("naruko_show_chrome", 15.7392),
        ("naruko_show_light_a", 15.8824),
        ("naruko_show_light_b", 15.8921),
        ("naruko_show_mirror", 17.6411),
        // THE PHYSICS PLAYGROUND (derived in the header comment; only the
        // seven in-frustum boxes — stack_0/break_crate are culled, no range
        // to read).
        ("playground_stack_1", 11.0675),
        ("playground_stack_2", 10.6547),
        ("playground_stack_3", 10.2956),
        ("playground_stack_4", 9.9961),
        ("playground_pyramid_0", 13.8717),
        ("playground_pyramid_1", 13.7704),
        ("playground_pyramid_2", 13.4347),
    ] {
        let r = range_of(&wide, id);
        assert!(
            (r - expect).abs() < RANGE_TOL,
            "range({id}) live {r} != derived {expect} (tol {RANGE_TOL})"
        );
    }

    // The two culled Physics Playground boxes are OUT of frustum (CANON #1's
    // bottom-plane derivation) — they own no caption/range even at this wide
    // nearest_n=32.
    assert!(!caption_ids(&wide).contains(&"playground_stack_0".to_string()));
    assert!(!caption_ids(&wide).contains(&"playground_break_crate".to_string()));

    // F6 — the four PHYSICS rows gaze the SOLVER-RESTED world, not `wide`
    // (see the F6 MIGRATION doc comment above this test; full derivation +
    // residuals in `rest_pose_canon.rs`).
    //   naruko_crate          rested [-11.15,1.4759,13] range 33.4042
    //     (authored [-11.15,4.5,13] range 33.0390 — the solver drops it
    //     3.025 m from its authored hang; Δ +0.3652)
    //   naruko_stack_crate_0  rested [-13.65,1.4754,13] range 34.3197
    //     (authored range 34.3198, Δ -0.0001 — authored ALREADY at rest)
    //   naruko_stack_crate_1  rested [-13.65,2.3259,13] range 34.1931
    //     (authored range 34.1932, Δ -0.0001)
    //   naruko_stack_crate_2  rested [-13.65,3.1767,13] range 34.0872
    //     (authored range 34.0874, Δ -0.0002)
    let (rested_world, _dials) = rest_pose::rested_canon_world();
    let wide_rested = look(
        &rested_world,
        eye,
        LookParams {
            nearest_n: 40,
            include_support: true,
            ..Default::default()
        },
    )
    .unwrap();
    for (id, expect) in [
        ("naruko_crate", 33.4042_f32),
        ("naruko_stack_crate_0", 34.3197),
        ("naruko_stack_crate_1", 34.1931),
        ("naruko_stack_crate_2", 34.0872),
    ] {
        let r = range_of(&wide_rested, id);
        assert!(
            (r - expect).abs() < RANGE_TOL,
            "SOLVER-TRUTH range({id}) live {r} != derived {expect} (tol {RANGE_TOL})"
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
/// THE SIGNAL RINGS change NOTHING here: their planes sit BEHIND the tower's
/// front face (ring z_view = 161–162 > 158.5), so along every shared cell ray
/// the tower's front-face hit is nearer and keeps the cell; at the rock's row
/// 14 a ring-depth ray passes y = 7 + 162·0.09375·0.5774 ≈ 15.8 ≪ the rings'
/// min y 42.225 ⇒ miss. No owned cell moves.
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
/// entity_count = 7 (with lighthouse_beacon); with the three SIGNAL RINGS
/// (in-frustum: centers dead ahead, e.g. ring_a − eye = [13, 53.8, −133], all
/// inside the 60° cone; against the crate-culling right side-plane the ring
/// AABBs' max projection ≈ 70.9 ≫ offset 3.7583 ⇒ inside) it is TEN; the
/// non-support captions are
///   [chrome_orb, pier, rock, tower, ring_c, ring_b, ring_a, beacon, sea].
/// Ring ranges from the pier eye [-13, 2.7, 15]:
///   ring_a center [0,56.5,-118]   → √(169+2894.44+17689)    = 144.0571
///   ring_b center [0,56.5,-117.5] → √(169+2894.44+17556.25) = 143.5956
///   ring_c center [0,56.5,-117]   → √(169+2894.44+17424)    = 143.1343
/// — all three between tower (140.93) and beacon (145.9056), nearest last-
/// authored plane first (ring_c, the widest, has the smallest z_view).
/// Chrome-orb range: [1,-0.58,-3] → √(1 + 0.3364 + 9) = 3.2150 m — nearer
/// than the pier, so it leads the caption order.
/// The ELEMENTS-P3 crate sits near this eye too (AABB center [-11.15,4.5,13],
/// static drop pose) but is CULLED: the conservative frustum test projects its
/// AABB onto the right side-plane (inward normal (-0.8660,0,-0.5), offset
/// dot(n,eye)=3.7583) and the AABB's MAX projection is at corner
/// (x=-11.55,z=12.6) = -0.8660·(-11.55) - 0.5·12.6 = 3.7026 < 3.7583 ⇒ the whole
/// box lies outside the right plane (margin 0.056 m) ⇒ out of frustum. So the
/// pier glance stays SEVEN vessels; the crate never enters (its range here,
/// were it in, would be 3.2653 m).
/// The RITE V·V2 cat sits near the stall too (AABB center [-5, 0.2278, 23.21])
/// but is CULLED HARD: forward is -z, and the cat's NEAREST z-face is min
/// z=22.9420 — wholly BEHIND the eye at z=15 (z_off = 23.21−15 = +8.21 > 0). The
/// whole box lies on the far side of the eye's near plane ⇒ out of frustum by a
/// 7.94 m margin (22.9420 − 15). So the pier glance stays SEVEN; the cat never
/// enters (its range here, were it in, would be 11.66 m). The MIRROR PROOF
/// vessels are culled the same way: min z-faces 17 (mirror), 17 (mirror_minor)
/// and 17.75 (kami orb) all lie BEHIND the eye at z=15 (margins 2.0, 2.0,
/// 2.75 m) — the pier glance still counts SEVEN.
/// Pier range: center [-12,-0.8375,-2] − eye [-13,2.7,15] = [1,-3.5375,-17] →
///   √(1 + 12.51 + 289) = 17.3929 m. Lighthouse still ahead: rock 135.75 m,
///   tower 140.93 m, both bearing ≈ +5.5° (well inside the 30° half-FOV).
///
/// RITE VI · VI-1 grows this glance too: the stack (center [-13.65, y, 13],
/// y = 1.475/2.325/3.175) sits almost UNDER the pier eye — offset from
/// [-13,2.7,15] is [-0.65, y−2.7, −2] for every crate (only y differs), well
/// inside the z_view=2 cone (half-width tan30·2 = 1.1547 ≥ |x_off| = 0.65)
/// ⇒ all three in-frustum, nearer than the chrome orb:
///   stack_crate_0 [-0.65,-1.225,-2] → √(0.4225+1.500625+4) = 2.4337
///   stack_crate_1 [-0.65,-0.375,-2] → √(0.4225+0.140625+4) = 2.1361
///   stack_crate_2 [-0.65, 0.475,-2] → √(0.4225+0.225625+4) = 2.1560
/// entity_count = 13; the caption order leads with the two nearer crates
/// (crate_1 then crate_2 — their y-offsets from the eye are near-symmetric,
/// crate_1 sits slightly closer), then crate_0, then the chrome orb.
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
    // + lighthouse_beacon (range 145.9056) since the Living World merge,
    // + naruko_chrome_orb (range 3.2150, right beside the pier eye) since
    // PLEROMA L2, + the three SIGNAL RINGS (144.0571/143.5956/143.1343,
    // derived above) — ten meshed vessels — and VI-1 adds the three stack
    // crates (2.1361/2.1560/2.4337, derived above) — thirteen.
    assert_eq!(g.entity_count, 13, "moved-eye in-frustum count");

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
            "naruko_stack_crate_1",
            "naruko_stack_crate_2",
            "naruko_stack_crate_0",
            "naruko_chrome_orb",
            "naruko_pier",
            "lighthouse_rock",
            "lighthouse_tower",
            "signal_ring_c",
            "signal_ring_b",
            "signal_ring_a",
            "lighthouse_beacon",
            "naruko_sea",
        ],
        "moved-eye non-support caption set/order (VI-1: the stack 2.1361/2.1560/2.4337 leads, nearer than the chrome orb 3.2150; rings 143.13/143.60/144.06 between tower 140.93 and beacon 145.9056)"
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
    // VI-1 — the stack leads the WHOLE order (nearer than even the chrome orb).
    // AUTHORED/load-pose ranges only — NOT solver truth (this test never ticks
    // physics; the stack sits at its authored pose). The stack was authored at
    // its solver-rest, so the F6 rest-pose ordeal (rest_pose_canon.rs) confirms
    // these to < REST_TOL; that file covers each of these vessels' rest gaze.
    for (id, expect) in [
        ("naruko_stack_crate_0", 2.4337_f32),
        ("naruko_stack_crate_1", 2.1361),
        ("naruko_stack_crate_2", 2.1560),
    ] {
        let r = range_of(&plain, id);
        assert!(
            (r - expect).abs() < RANGE_TOL,
            "range({id}) live {r} != derived {expect} (tol {RANGE_TOL})"
        );
    }
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

/// CANON #8 — VII-0b THE FIRST GROUND (the render weld). `naruko_first_ground`
/// carries ONLY a `terrain` sigil — the oracle derives its world bounds
/// ANALYTICALLY from that sigil (`terrain_world_aabb`, `model.rs`), never by
/// building a mesh. Header derivation: tile_origin = (0,128)
/// (`tile_x·grid_resolution·cell_size_m` with grid_resolution=64,
/// cell_size_m=1.0 at the default tile_size_m=64), height_amplitude=9.6
/// (`64 × DEFAULT_SLOPE_FRACTION(0.15)`) ⇒ x[0,64] y[-9.6,9.6] z[128,192],
/// center [32,0,160], range from the canon eye [0,7,44] = √14529 = 120.5363.
/// Also proves the patch stays OUT of the default frustum set (confirmed
/// separately by CANON #1's unchanged entity_count=24), and that the oracle
/// never needed a mesh: the entity carries no `mesh` component at all.
#[test]
fn canon_terrain_patch_bounds_and_range_are_derived() {
    let world = canon();
    let geom = world
        .geometry("naruko_first_ground")
        .expect("naruko_first_ground is a registered entity");
    let bounds = geom
        .bounds
        .expect("a terrain sigil derives analytic bounds");

    const AABB_TOL: f32 = 1e-4;
    let want_min = [0.0_f32, -9.6, 128.0];
    let want_max = [64.0_f32, 9.6, 192.0];
    for axis in 0..3 {
        assert!(
            (bounds.min[axis] - want_min[axis]).abs() < AABB_TOL,
            "terrain min axis {axis}: live {} != derived {} (tol {AABB_TOL})",
            bounds.min[axis],
            want_min[axis]
        );
        assert!(
            (bounds.max[axis] - want_max[axis]).abs() < AABB_TOL,
            "terrain max axis {axis}: live {} != derived {} (tol {AABB_TOL})",
            bounds.max[axis],
            want_max[axis]
        );
    }

    let eye = world.spawn_pose().expect("canon spawn pose");
    let center = bounds.center();
    let d = [
        center[0] - eye.position[0],
        center[1] - eye.position[1],
        center[2] - eye.position[2],
    ];
    let range = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    // Same derived tolerance as CANON #2's RANGE_TOL (1e-3 m, ≈16× the
    // measured live-f32-vs-4-decimal-reference discrepancy).
    const RANGE_TOL: f32 = 1e-3;
    assert!(
        (range - 120.5363).abs() < RANGE_TOL,
        "terrain patch range: live {range} != derived 120.5363 (tol {RANGE_TOL})"
    );

    // The sigil is the SOLE authored artifact: no `mesh` component travels
    // with it (the NO-STORAGE law at the scene seam — the oracle's own
    // confirmation that it derived from data, not geometry it happened to
    // find).
    assert!(
        world
            .entities
            .iter()
            .find(|e| e.id == "naruko_first_ground")
            .expect("entity registered")
            .components
            .iter()
            .all(|c| c != "mesh"),
        "a terrain entity must carry no mesh component"
    );
}
