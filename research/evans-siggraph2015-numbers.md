# Alex Evans SIGGRAPH 2015 "Learning from Failure" — NUMBERS extract (sonnet, 07-16)

Source: https://advances.realtimerendering.com/s2015/AlexEvans_SIGGRAPH-2015-sml.pdf
(mirror of the original mediamolecule.com PDF, domain now dead) · retrieved
2026-07-16 · 34.7MB, 145 slides, pdftotext -layout clean extraction. Cross-ref
dreams-recon.md §Renderer (prose summary written pre-fetch, no numbers) — this
file supersedes it as the numeric source of record.

## Edit list / sculpt data model
- 1 – 100,000 edits per model (edits = CSG primitives: cubic strokes,
  cylinders, cones, cuboids, ellipsoids, tri-prisms, donuts, "markoids"
  [superellipsoids, variable power per axis], pyramids). add/subtract/color
  ops only, no domain deformation, no blur — CSG trees forced RIGHT-LEANING
  (flat list, not tree).
- "Dad's head" = 8,274 edits (recurring demo model through the whole talk).
- 100k edits × ~1000³ grid evaluated naively = 100 BILLION evaluations/model
  — stated as "too many," the reason hierarchical evaluation exists at all.

## Engine 1 (marching cubes, abandoned)
- Compound SDF stored in "83^3" fp16 volume texture blocks (verbatim OCR;
  almost certainly 8³ — UNVERIFIED which, glyph corruption in extraction),
  incrementally updated per edit, meshed independently per block via
  marching cubes IN a compute shader (histopyramid stream compaction for
  index buffers — advanced/buggy CS use on early PS4 devkits).
- Failure mode: mushy surfaces / degenerate slivers from marching cubes on
  fine SDF detail.

## The evaluator ("CS of doom") — hierarchical refinement, kept
- Full pipeline edit-list → renderable data: 40+ compute shaders chained.
- The "CS of doom" proper = a handful of 3000+-instruction shaders chained
  to produce the sparse SDF.
- Coarse start grid: 4×4×4 (first hierarchical prototype by Anton); actual
  production split is 4×4×4 PER STEP (matches GCN's 64-wide wavefront —
  4×4×4=64 — for coherent scalar branches on primitive type).
- Quality floor: renderer backends needed ≥1 to 1.5 voxels of valid data on
  each side of the mesh surface (undersplit → gaps in model; oversplit →
  orders-of-magnitude wasted evaluator work).
- Distance-field norm: evaluator uses MAX-NORM (d = max(|x|,|y|,|z|)), not
  L2 — because many non-uniform primitives (ellipsoids esp.) have far
  simpler closed-form distance under max-norm, and max-norm's iso-shape
  (cube) matches the hierarchy's own cube nodes. Rotation care needed (max
  norm isn't rotation-invariant; a valid field is a scale factor away).
- Soft blend (soft_min/soft_max, credit Dave Smith):
  `soft_min(a,b,r) = min(a,b) - e²·0.25/r`, `e = max(r - |a-b|, 0)`. Reverts
  to hard min/max once |a-b| > r. Breaks culling over a wide radius unless
  you track "future soft blend" reach — interval arithmetic + max-norm
  bounds (Simon) rescued the culling.

### Per-model dispatch/eval stats table (verbatim, PS4) — crystal's dad run:
eval dispatch 60 · sweep dispatch 91 · points dispatch 459 · bricker
dispatch 73.

### Full stats table (Filename / ElementCount / BlockCount / BlockEvalCount / VoxelCount / EvaluatorTime(s) / ClusterCount / BrickCount / TimePerPrimPerBlock / CullingEfficiency):
| model | elements | blocks | blockEval | voxels | evalTime(s) | clusters | bricks | time/prim/block | culling |
|---|---|---|---|---|---|---|---|---|---|
| crystals_dad | 8274 | 235286 | 1879414 | 5241466 | 0.09109472 | 11190 | 5497 | 0.00000004847 | 99.90% |
| big_mech | 850 | 156759 | 292582 | 4119346 | 0.01510454 | 8651 | 5741 | 0.00000005163 | 99.78% |
| sphere_woman | 24407 | 36625 | 6995798 | 848194 | 0.3423158 | 1408 | 694 | 0.00000004893 | 99.22% |
| hoverbot7b | 604 | 168234 | 349221 | 4317399 | 0.01616372 | 8411 | 5187 | 0.00000004629 | 99.66% |
| whirlwind | 791 | 410945 | 763795 | 10411468 | 0.0344279 | 19222 | 10431 | 0.00000004507 | 99.77% |
| head_lion | 53976 | 182451 | 4148168 | 4676180 | 0.2271382 | 10555 | 5866 | 0.00000005476 | 99.96% |
| hatching | 736 | 68793 | 143484 | 1729905 | 0.009373858 | 3939 | 4446 | 0.00000006533 | 99.72% |
| dots | 1480 | 149988 | 190265 | 3816626 | 0.01703693 | 6241 | 4436 | 0.00000008954 | 99.91% |
| bolt_lock | 1830 | 41936 | 1302908 | 1018246 | 0.02170936 | 2298 | 1216 | 0.00000001666 | 98.30% |
| motel_final | 610 | 373423 | 752668 | 9214195 | 0.03267893 | 17606 | 6882 | 0.00000004342 | 99.67% |
| derek_glamour_head8 | 3824 | 266152 | 5906261 | 6997031 | 0.1903296 | 12668 | 5052 | 0.00000003223 | 99.42% |
| Dan-OldDudeHat | 907 | 146283 | 1235569 | 3923554 | 0.04069558 | 7159 | 3606 | 0.00000003294 | 99.07% |
| head40 | 22484 | 96477 | 1593633 | 2464223 | 0.1425964 | 5087 | 2571 | 0.00000008948 | 99.93% |

- Range summary: 600–53,976 edits (worst outlier "around 120k, atypical")
  → 1M–10M surface voxels (±1.5 of surface). Culling vs brute force: well
  over 99%. THROUGHPUT: 10M–100M voxels evaluated/second on a PS4, on
  models with tens of thousands of edits.

## Failed renderer chronology (in order, ~4 years/2011→2014→2015 ship)
1. Marching cubes on 8³(?) fp16 blocks → mushy/sliver surfaces.
2. GigaVoxels (Crassin/Neyret/Lefebvre) prototype — single-object only,
   doesn't handle many overlapping objects (games need that).
3. "Brick engine" (Engine 2): brick tree from GigaVoxels, but instead of
   ray-marching from the eye, rasterize each 8×8×8 brick (view-dependent
   LOD cut chosen up front), PS then ray-marches ONLY from brick surface to
   SDF zero (POM/tiny-raymarch-in-cube, oDepth sets z-buffer). ~2 years in
   this direction; "too slow," "untextured Unreal but slower." Struggled
   with memory for gigavoxel bricks (explicit "we were struggling with the
   memory" — no MB figure given, UNVERIFIED exact budget).
4. OIT/fuzz refinement attempt on top of bricks engine: output partial-alpha
   voxel lists + z on full opacity, composite non-solid voxels. "Easily
   32× 1080p" non-opaque pixels to composite — the OIT depth-peel/sort
   became the bottleneck (CS load-balanced depth peel, 8 layers/wavefront).
   Overall verdict: 4×–10× too slow depending on optimism. Deferred lighting
   also too costly (lit every leaf voxel).
5. Froxel refinement volumetric renderer (Engine 3): post-projection voxels
   ("froxels" — Sony WWS term) instead of world-space voxels; subdivide 8
   children per froxel per CS pass; switches from 3D dense pointer volume
   to 2D representation at the 1/16-res → 1/8-res refinement step
   (128×64×128 → 256×128×256, dense 3D pointers too expensive past that).
   Verdict: also too slow, shelved.
6. SHIPPED: splat-based engine (Engine 4) — decided after "hitting rock
   bottom in January 2014" (per art director Kareem's push toward painterly
   look vs "untextured Unreal-engine" look of raw SDF/brick renders).

## Engine 4 — splats (shipped)
- Point cloud generation: ~900³ domain (up to) → ~2,000,000 surface voxels
  → ~2,000,000 points (1 pt/leaf-voxel, dual-grid 2×2×2 zero-crossing test).
- Points sorted into Hilbert order (4³ voxel bricks in Hilbert order,
  surface voxels inside each brick in raster order) then CUT into clusters
  of ~256 points (partial clusters allowed at Hilbert-brick seams).
- Per-point storage: ONE DWORD (bitpacked pos, normal, roughness, colour in
  a DXT1-style texture) per point.
- Separate point-cloud + cluster set generated PER LOD (mip pyramid: 900
  voxels across → 450 → 225 → …).
- Rendering: clusters arranged into a BVH per LOD; Russian-roulette
  stochastic LOD smoothly reduces cluster density 256 → 64 points/cluster
  before dropping a full LOD level (25% floor before drop).
- Splats packed into groups of ~64 for the CS splatter.
- Stochastic transparency: randomly discard pixels by splat alpha, TAA
  reconstructs — "works great in static scenes."
- Rendering pipeline: 64-bit atomic-min splat (== 1-pixel point) into a
  1080p buffer per point, "10s of millions of points per frame." Then
  z+id buffer → traditional 1080p gbuffer (normal/albedo/roughness/z) →
  classic deferred lighting → heavy TAA.
- MEASURED PERF: one scene = 28.2M point splats in 4.38ms → ≈640,000,000
  single-pixel splats/second (atomic_min splatter, this GPU/scene).
- Depth of field: splats jittered in a screen-space disc scaled by circle-
  of-confusion — "literally the objects exploding a little bit," free from
  representation, no TAA occlusion artefacts.
- Shadows: 4 (stated "3?") cascades for the hero sun, imperfect shadow maps
  (ISM), atomic-splatted then sampled w/ TAA smoothing noise. LOCAL LIGHTS:
  budget of 64 small 128×128 ISM shadow maps distributed across scene
  spotlights (brute-force splatted/sampled, paraboloid projection).
- AO: SSAO baseline (Morgan McGuire alchemy-style) replaced by 1
  cosine-weighted random ray/pixel z-buffer trace (TAA denoises) → then a
  longer-range 1-bit-per-voxel cascaded volume texture (~64³ per cascade,
  "mine-craft sized voxels" at finest, coarser further out), 4 world
  cascades, atomic-OR'd from the point cloud, cone-trace-like step-up
  through cascades for range.
- WIP/untried at talk time: RGB emissive low-res world cascade gathered
  along the AO rays for bounce light — "variance is INSANELY high," not
  shippable as-is even with TAA temporal averaging; 8×8 bilateral/stratified
  filtering helped.
- Final self-description: "a cloud of clouds of point clouds" — mini point
  clouds per splat expand to "a few thousand points" up close, degenerate
  to single pixels at distance/tight mode.

## Numbers NOT found / UNVERIFIED in this deck
- No explicit total memory budget in MB/GB for the shipped splat engine
  (only the qualitative "struggling with memory" complaint re: Engine 2/3
  gigavoxel bricks).
- Exact meaning of "83^3" fp16 blocks (Engine 1) — likely 8³, OCR-ambiguous,
  not re-verified against original slide image.
- Console target framerate/resolution for the shipped engine not stated
  numerically in this talk (only "1080p" buffer size, no fps target given).
