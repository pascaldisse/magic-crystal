# RENDER — the Pleroma (rendering spec) — rewritten 2026-07-18 (raster heresy purged)

Status: LAW, not draft. Architect's rulings (07-18): "THE DESIGN IS THE LAW"
14:44 (NEURAL.md) · adversary charter amended 15:15, whip 168, spec-concordance
gate — born FROM the raster-cluster pipeline squatting here two days after
the two-act law; see LINEAGE appendix, bottom. He never ordered a raster
pipeline. This file states the ONE render design; nothing else is normative.
Cross-refs: GRIMOIRE.md (the Pleroma, sealed, no inner names) · NEURAL.md
(two-act law, upscaling-dead ruling, silicon verdicts) · GEOMETRY.md (hybrid
polygon/voxel/SDF, representation-blind G-buffer).

## The frame — one render, no seam
```
world truth → Pleroma → final image or nothing
```
Pleroma is ONE system, never named inner things (GRIMOIRE.md, sealed 07-18).
Inside Pleroma: sampling and judgment jointly consume full-resolution geometry
features (primary visibility is cheap), sparse traced radiance, and temporal
history; rays emit evidence/byproducts (depth/normal/albedo/material —
representation-blind, GEOMETRY.md), never a separate pass or picture. No
chained seams make irreversible partial decisions (DLSS-RR precedent =
evidence, not authority — NEURAL.md).
No forward path. No raster lighting. No cluster-raster geometry pipeline
(deleted as normative here — LINEAGE appendix). No probe/reflection-map
pass. No upscaling as a concept: nothing is ever made small then enlarged;
traced samples are a RAY BUDGET (samples/frame the machine affords), not an
internal resolution (NEURAL.md, "UPSCALING IS DEAD").

## 1 · Geometry — the ray door, not the raster door
COST ∝ RAYS — the evidence budget — comes NATIVE from tracing: BVH
traversal is ~log(N) per ray, and the ray budget is a FREE PARAMETER the
machine affords — never scene complexity, never a screen grid walked by a
raster pass. Pixels exist only as Pleroma's output grid (Architect,
07-18: "we don't use pixels" — raster vocabulary banned from cost laws).
This is WHY no cluster-cull/HZB/visibility-buffer machinery is needed to
bound cost: the ray door already bounds cost by evidence gathered.
Corollary for the design slot below: RESIDENCY ∝ RAY BUDGET · CONTENT ∝
SSD · THE IMAGE BELONGS TO PLEROMA. It must never again be re-derived from a cluster raster (the
heresy this file is being purged of).

The BVH intersector is not a future trait — it is BUILT and LIVE
(`packages/scrying-glass/src/bvh.rs`; `DynamicSplice` refit-not-rebuild
lever, bit-exact vs rebuild, 60 FPS LAW). Geometry of every representation
(polygon meshes, voxel-volume contours, SDF-sculpt contours) emits into
this SAME BVH world (GEOMETRY.md's representation-blind law) — there is no
second geometry pipeline for any representation.

**DESIGN SLOT — NOT YET RULED IN DETAIL: virtualized geometry under the BVH.**
- Shape: baked geometry PAGES on SSD — Nanite's page idea, re-homed onto BVH
  residency instead of screen-space raster clusters.
- Residency driven by RAY FOOTPRINT: which pages rays actually touch this
  frame/window, not camera-frustum visibility. Secondary/GI rays touch
  off-frustum world — a HARDER residency problem than Nanite's (camera-
  frustum-only) case. Say so plainly; do not borrow Nanite's guarantees.
- Streaming: MTLIO (Metal 3+, direct SSD→unified-memory, no PCIe copy;
  published NVMe class ~5-7GB/s) — UNVERIFIED on our stack until measured.
- Access pattern: PREFETCH, not page-fault. 60fps cannot ride a page-fault
  stall; residency must be predicted/prefetched ahead of the ray that needs
  it, never discovered by a miss.
- Open, unruled: page granularity for BVH nodes (vs Nanite's screen-space
  cluster granularity) · how view-dependent detail scaling expresses inside
  a ray-native, no-LOD-vocabulary law (DREAMFORGE forbidden-vocabulary
  applies; a cluster error metric is NOT an answer here, it must be
  re-derived on the ray door or left for a future ruling, not inherited).
- This slot replaces the old "sole cluster pipeline" section outright — it
  is deliberately thinner: virtualized geometry on the ray door has no
  shipped precedent to lean on the way Nanite gave the raster version one.

**Hybrid ruling (intact)**: polygon default, voxel opt-in, per entity
(GEOMETRY.md `voxels`/`sdf` components). Dreams-style SDF/voxel contouring
emits directly into the SAME BVH world — not into raster clusters
(corrected from the heresy's framing). One entity, one representation
blends into the shared ray-traceable world; lighting never branches on
representation.

**Foliage/aggregates**: density-field intent re-grounded on the ray door.
Porous volumes (grass/hair) become volumetric density fields that RAYS
sample/march directly — there is no raster occlusion-culling problem to
solve for them, because there is no raster occlusion pass. Clusters/
polygons near camera, density field at distance, blend automatic. Detail
owed in GEOMETRY.md.

**Any topology** (non-manifold/holes/internal faces) must not break the
geometry pipeline — Pascal's "works with any geometry" law unchanged, now a
BVH-build requirement rather than a cluster-baker requirement.

**Instance transforms**: still to be paged/virtualized (10M×float4x3 =
457MB lesson stands) — page transforms alongside geometry under the same
future residency system (design slot above).

## 2 · Lighting/transport — one integrator (law)
- Ground truth: Monte Carlo path transport. Lights, sky, emissive materials
  = one thing: emitters. Reflections/GI/shadows/AO = not features, just
  paths.
- **Intersector**: the live BVH (§1) — not the SDF/occupancy-mip trait this
  file originally proposed as a stand-in; that proposal is superseded by
  what actually shipped. SDF/voxel geometry still exists as an authoring
  representation (GEOMETRY.md) but it CONTOURS INTO the BVH rather than
  being traced by a second intersector.
- **Inside Pleroma: sampling and judgment.** ReSTIR-class reuse (direct
  many-lights + GI reuse) makes sparse evidence smarter per ray; it never
  reconstructs an image itself. It rides the BVH intersector; not yet built.
  Screen-space reuse, world radiance caches, and temporal accumulation are
  Pleroma's evidence/history substrate (NEURAL.md), never hand-built seams
  between world truth and glass. Hand-rolled temporal gates/clamps/heuristics
  (`integrate_temporal`/`temporal_resolve`) are LAB EQUIPMENT: training-data
  and history-buffer generators / teachers for Pleroma, not a shipped path
  (NEURAL.md, "THE DESIGN IS THE LAW"). `GAIA_NATIVE_TEMPORAL` defaults OFF
  on main for this reason.
- **Anti-goals** (Lumen sucks-list, unchanged): no seconds-lag on global
  light changes — light changes visibly propagate within ~100ms, fully
  converge in ~1s · small emissives = first-class emitters (ReSTIR samples
  them) · no thin-wall leaking class (geometry IS the field the BVH holds)
  · no quality ladder to a second system — dials: rays (evidence budget), bounces,
  cache freshness, net capacity.

## 3 · Textures — virtual, software-indirected (unchanged, orthogonal to the purge)
- wgpu has NO sparse textures (gpuweb#455) ⇒ software indirection = the
  pipeline: physical tile pools (per-format LRU) + page-table texture +
  shader UV remap at the ray-hit shading point. idTech/UE/Granite ship
  this; hardware sparse never required. Metal sparse (M1 has it) = optional
  fast path behind a trait.
- Pages 128²-class; feedback from ray-hit shading drives residency
  (replaces the old "visibility/material pass" language — there is no
  screen-space material-resolve pass, residency feedback comes from what
  rays actually hit); oversubscription → mip-bias softening, never a
  render failure.
- Format: KTX2/UASTC intermediate (bit-repack to BC7 AND ASTC ≈ free);
  ASTC-8×8 for distant/rough (0.25 B/texel). 20K input textures welcome:
  resident set = sampled tiles (few MB), disk = compressed pages.
- IO trait: MTLIOCommandQueue on macOS 13+ (disk→GPU, tile-granular,
  DirectStorage-analog), std async IO fallback. Same MTLIO mechanism the
  geometry design slot (§1) wants for pages — one streaming substrate,
  two page kinds.
- Megatexture lesson: arbitrary INPUT size always; uniqueness ≠ virtue;
  transcode must be bit-repack cheap (never JPEG-family).

## 4 · Presentation
Pleroma renders directly at screen resolution — no separate "internal
resolution → upscale to native" dial (that concept is dead, NEURAL.md). The
only dials are ray budget (samples/frame), bounce count, cache freshness.
Denoising belongs inside Pleroma's judgment; the standalone
SVGF-class/learned denoiser explored earlier (VIII-1/VIII-2/VIII-3) is LAB
EQUIPMENT — a teacher/baseline, never the shipped resolve.

## 5 · Capability traits (hardware-agnostic law)
| trait | current (M1 Pro, live) [source: NEURAL.md §Silicon race verdicts] | later silicon |
|---|---|---|
| intersector | BVH, `DynamicSplice` refit-not-rebuild (live, `bvh.rs`) | + HW ray accel where available |
| geometry residency | none yet (design slot, §1) | MTLIO page streaming |
| sparse texture residency | software indirection (live) | + Metal sparse fast path |
| streaming IO | MTLIOCommandQueue (macOS 13+) | DirectStorage / io_uring |
| reconstruction (Pleroma's learned act) | Metal tensor (MPSGraph GEMM), portable-baseline wgpu compute MLP [source: NEURAL.md §Silicon race verdicts] | Metal 4 MTLTensor / ML-command-encoder, ANE offload if a model ever fits (currently refused, §6) [source: NEURAL.md §Silicon race verdicts] |
Deleted rows (heresy, now LINEAGE-only): raster backend (hw-vis/sw-vis),
upscaler (MetalFX interop slot) — no separate seam exists in the one-render law.
Quality adapts by CONTINUOUS dials on rays and net capacity, never by
switching systems.

## 6 · M1 frame budget — measured, not assumed (2026-07-18)
Silicon race verdicts (host M1 Pro, NEURAL.md ledger, full method in
`docs/perf/2026-07-18-rdirect-metal-tensor-spike.md`):
- WGSL per-thread MLP (net running once per GPU thread, one net per pixel):
  **DOOR SHUT** — ~280-300ms f32 native @960×640; even a trivial 2×32 net
  = 32.5ms. Memory-bound: re-fetches all weights per pixel, zero reuse.
- ANE/CoreML: **NO** — refuses the net above ~16k px; where it does run it
  is never faster than CPU; native falls back to GPU-GEMM (~27.5ms). Does
  not free the GPU.
- **Metal tensor (MPSGraph GEMM, batched matmul over all pixels at once) [source: docs/perf/2026-07-18-rdirect-metal-tensor-spike.md]:
  DOOR REOPENS.** 4.47ms f32 @ native 960×640 [pre-law lab measurement
  @960×640; remeasure at 640×480 canvas owed], ~94% of the fp32 roofline
  (~5.0 of ~5.3 TFLOPS), parity 1.6e-7 vs CPU reference, 63× the WGSL
  [source: docs/perf/2026-07-18-rdirect-metal-tensor-spike.md]
  per-thread door. Same net, same weights — the win is weight-reuse via
  simdgroup-matrix tiling turning a memory-bound per-thread problem into a
  compute-bound GEMM.
- 60fps arithmetic on the measured pieces: **rays 7.6ms** (live-measured,
  light-live merge's `integrate_temporal`/reprojection machinery — now LAB
  EQUIPMENT, figure carried forward as sparse-evidence cost baseline;
  re-measure once Pleroma sheds temporal-only bookkeeping) **+ Pleroma's
  learned act 4.47ms** (Metal tensor, GEMM only) **≈ ~13ms** of the
  16.67ms/60fps budget, before elementwise glue. [source:
  docs/perf/2026-07-18-rdirect-metal-tensor-spike.md]
- Honest gaps (§ of the spike doc, not yet measured): feature-gather +
  undo_log_demod elementwise passes (unmeasured, expected cheap — WGSL's
  large per-pixel transcendental floor was a per-THREAD problem the GEMM
  reformulation should not inherit, but this is a claim, not a number yet)
  · buffer pooling (157MB intermediate allocation currently unpooled — wall
  clock is 22-26ms until pooled, vs 4.47ms GPU-only) · MPSGraph↔wgpu
  interop lane (not built) · `reload_shader`↔temporal-pipeline gap (B0
  live-edit does not yet rebuild net-adjacent pipelines).
- Geometry-residency cost (§1 design slot) is UNMEASURED — no budget line
  exists for it yet; it is not built.

## 7 · Milestones (each playable, screenshot-verified) — revised off the ray door
R1 is DELETED as a milestone (it was the cluster-raster's own gate; see
LINEAGE). Replacement sequence, ray-native:
R1′ BVH world + Act 1 sparse trace: live already (`bvh.rs`, DynamicSplice,
   60 FPS LAW passed on rays-only content). Gate met historically under a
   different name; re-stated here for continuity.
R2 materials+VT: resolve-at-ray-hit + software VT + KTX2 pipeline (§3).
   Gate: 20K texture dropped in, just works, resident MB counter proves it.
   (Unchanged from the draft — never depended on rasterization.)
R3 THE NET landing: Metal tensor reconstruction wired as the live present
   path (not scry-only), pooled buffers, gather/undo measured. Gate: 60fps
   on the MacBook at native res, no separate denoise/upscale stage in the
   loop. Current status: scry-only candidate exists (one-render-path lane,
   pre-purge naming); live-surface cutover is the Architect's own call
   after he plays it (NEURAL.md).
R4 GI + net history: bounce budget dials + THE NET's own temporal-history
   input (not hand-rolled reprojection) converge global light changes
   <1s. Supersedes the draft's "R4 GI: bounces + temporal accumulation +
   denoiser" — accumulation moves inside the net, denoiser is not a stage.
R5 geometry residency: the §1 design slot built and measured — DELETED as
   "R5 upscale" (upscaling is dead; see §4). Gate TBD once §1 is ruled.
R6 scale: billion-triangle content test on the BVH/page system, streaming
   pages under flight (unchanged intent, ray-native execution).

## 8 · Apple-Silicon perf appendix — portable wisdom (kept where raster-independent)
Portable across both acts: fewest passes · LoadOp::Clear + StoreOp::Discard
on transient targets · fp16-first shaders where precision allows (net GEMM
proved fp16 buys memory, not FLOPS on M1 — no double-rate fp16, §6) ·
subgroup ops (width 32 on M1) · minimize device atomics (Apple: 32-bit
only) · WGSL override for variants.
Trait-gated: TextureUsages::TRANSIENT → Metal memoryless (verify wgpu ≥
PR#8247) · MTLIOCommandQueue.
Profiling law unchanged: every R-milestone gate includes an Xcode GPU
capture on the MacBook, read by LIMITER (ALU/Buffer/Texture) not
utilization % — plus, since 07-18, `tools/profile-seam.mjs`-class CPU
profiling and rAF frame-time probes for hitches, per AGENTS.md discipline.
Deleted from this appendix (heresy, LINEAGE-only): vis-buffer-specific
wisdom (multi-draw-indirect/persistent-queue culling notes, on-tile
single-pass deferred discussion) — none of it applies once there is no
raster visibility buffer.

## Rulings
1. SUPERSEDED 07-18 (was: "hw-vis-first on M1; sw-vis capability-gated,
   RULED 07-16"). No raster backend exists to rule on; see LINEAGE.
2. ✅ foliage/aggregates → density field at distance, clusters near, blend
   automatic — RE-GROUNDED on the ray door (§1), not raster occlusion
   culling. Still RULED, wording corrected.
3. OPEN: physics interleave point in the Act-1/Act-2 frame (after ray
   trace, before or alongside the net?) — waits on the Architect's physics
   pass; orthogonal to the raster purge (PHYSICS.md, NEURAL.md physics law
   07-18 14:50 hold).
4. NEW 07-18: virtualized-geometry-under-BVH (§1 design slot) is explicitly
   NOT YET RULED IN DETAIL — page granularity, prefetch policy, and how
   view-dependent detail expresses without borrowed cluster-error-metric
   vocabulary are open questions for the Architect, not decisions made
   here.

---

## LINEAGE — the raster-cluster pipeline (non-normative, never built, superseded 2026-07-18)
The 2026-07-16 draft specified a cluster-raster geometry pipeline (Nanite-
class): meshopt ≤128-tri clusters, METIS-class adjacency groups, DAG
simplification, `parentError > τ ≥ clusterError` cut selection, 128KB-class
page streaming, two-pass HZB occlusion, and a capability-gated rasterizer
backend (`hw-vis` primitive-ID visibility buffer as M1-primary, `sw-vis`
64-bit-atomic compute raster for later silicon), feeding a G-buffer
material-resolve pass ahead of path-traced lighting. It was never
implemented (grep confirms zero `hw-vis`/`sw-vis`/cluster-cull code in
`packages/scrying-glass/src`) and directly contradicted the two-act law
sealed the same week (NEURAL.md, 07-18): it reintroduced exactly the
chained-stage seam (raster visibility → resolve → trace → denoise →
upscale) the two-act law exists to kill. Ruled HERESY by the Architect
07-18 15:15 (spec contradicting a sealed ruling); this section is kept as
disclosed history per adversary-charter discipline, not as design.
