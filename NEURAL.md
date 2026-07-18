# NEURAL — DreamForge neural performance ledger (DRAFT, 2026-07-16)

> **RULED (Architect, 07-16):** NEURAL RENDERING = UPSCALING, DENOISING,
> PHYSICS ASSISTANCE ONLY. Frame interpolation/generation is BANNED
> forever, in every form. 60 FPS is the MINIMUM — the floor, never the
> target.

Target it serves (Pascal, restated 07-16): 60 FPS · M1 · 16 GB unified ·
the ONE traced lighting system always on — "no such thing as ray tracing
on or off"; neural = INTERNAL detail levels (denoiser/upscaler/cache
quality), never a toggle of the transport. Evidence: research/neural-recon
.md (+ gi/metal recons). Build order: LATER (Pascal) — this doc = the
write-down of what it buys, so budgets assume it correctly.

## Law
- The path integrator is truth. Neural components are VARIANCE REDUCTION
  and RECONSTRUCTION — they converge to (or reconstruct toward) the traced
  result, never replace the transport model.
- Physics: near-field / gameplay-critical = EXACT solver, always. Neural
  surrogates serve far-field, decorative, and fluid-detail tiers — every
  surrogate hot-swappable for the exact path (truth-checkable, testable).
  "Hallucination that looks like simulation" (Pascal) is admitted exactly
  where a wrong guess cannot change an outcome that matters.
- Neural execution, REVISED 07-16 (Pascal: "we find a way" — he was right;
  evidence: research/metal4-neural-recon.md): METAL 4 (macOS 26, M1+ floor
  confirmed via SDK headers) ships MTLTensor + MTL4MachineLearningCommandEncoder
  — runs networks IN the GPU command timeline, auto-dispatching GPU-or-ANE
  per model with machineLearning-stage barriers. A real per-frame ANE path.
  Placement: portable baseline = wgpu compute MLPs (fp16, subgroups — wgpu
  has NO tensor surface, confirmed); Metal-native fast path = ML-encoder
  package behind a capability trait (pillar 7 shape), models eligible for
  ANE offload = denoiser/upscaler/radiance-cache/surrogates — freeing GPU
  ALU for path tracing. MetalFX frame interpolation + RT-denoised upscaler
  = same macOS 26 floor (RT-denoised needs M3+ — triangle-RT gate, not
  ours). Requirement: Pascal's Mac on macOS 26 Tahoe for the fast path.
  UNCONFIRMED (flagged): M1 ANE per-frame latency under the encoder —
  first Metal-native spike measures it before anything depends on it.

## Render ledger (published numbers → our expected gains)
| Component | Evidence | Gain |
|---|---|---|
| Internal-res dial + temporal upscale | MetalFX: ~40-50% quality mode, >2× performance mode; wgpu interop exists (texture_from_raw) | THE biggest multiplier: path tracing at ~¼–½ native pixels ⇒ 2–4× ray budget back |
| Learned/temporal denoiser | SVGF ≈10ms baseline (too fat — ours must beat); DLSS-RR = proprietary proof learned>hand-tuned | 1–2 spp presents like ~10× spp; makes low-spp PT shippable |
| Neural radiance cache (NRC-class) | Müller 2021: ~2.6ms @1080p update+query, 6×64 fp16 fused MLP; ★ reimplemented in RUST compute shaders on mobile GPUs (Breda, SIGGRAPH Asia 2025), query 2–25× cheaper than tracing the paths | Long-bounce GI nearly free after 1st bounce; kills fireflies; the "remembers light" network |
| ReSTIR (not neural, same slot) | 9.3–166× MSE win at 1 spp | Infinite lights at fixed budget |
Combined M1@60 equation: ~1080p internal · 1-2 spp · ReSTIR · cache ·
denoise · upscale to native — every stage individually published-proven;
the composition is ours to prove per milestone (Xcode limiter captures).

## Physics ledger
| Component | Evidence | Gain | Tier |
|---|---|---|---|
| Subspace neural dynamics (Holden 2019) | 300–5,000× vs full sim; 2700 FPS vs 0.5 FPS cloth | Pascal's "1000%" is conservative | cloth/soft decorative + far-field |
| Graph-network simulators (GNS 2020→geoelements 2023) | >165× vs parallel-CPU MPM granular; generalizes ~10× particle counts | many-object far-field, debris fields | far-field |
| Coarse-sim + neural detail (fluids) | tempoGAN = offline super-res; NeuralVDB inference too slow realtime (decode-first) | fluid DETAIL layer = research milestone, coarse SPH/grid stays primary | detail tier |
| Neural collision | MLP-SDF wins only <~10k queries GPU | niche; not load-bearing | targeted |
Honesty line (unchanged): ZERO shipped-game precedent for learned sim in
the loop — we would be first. Staged experiments BEHIND the exact solver,
never a dependency; promotion requires side-by-side truth comparison.

## Memory note (16 GB unified)
MLPs are tiny (6×64 fp16 ≈ KBs); training buffers ≈ MBs. Neural layer is
compute-bound, not memory-bound — memory budget stays owned by VT pages +
geometry pages + voxel volumes (unified memory = zero-copy between sim and
render, the M1's actual gift).

## Staging (build later — order fixed now)
NR1 temporal upscaler (own, cross-platform) + MetalFX trait on Metal
NR2 denoiser: temporal+spatial compute → learned variant when NR1 stable
NR3 radiance-cache MLP (Breda-pattern fused fp16 compute; NRC numbers)
NR4 physics surrogates: cloth/debris far-field A/B vs exact solver;
    fluid neural-detail research spike
Each gated: same scene, neural on/off, frame-time + image/physics-error
metrics on the MacBook. No gate, no promotion.

## ★ THE TWO-ACT LAW (Architect's ruling, 07-18 — supersedes staged chain
as DESTINATION; NR1/NR2 demote to teachers/baselines, lab equipment only)
The render is TWO ACTS, NO SEAM: trace → ONE NET → screen.
- Act 1: Ananke's rays — the one integrator emits sparse radiance + G-buffer
  features (same rays, byproducts, never separate passes).
- Act 2: the ONE NET consumes everything jointly — full-res geometry
  features (primary visibility is cheap) + sparse traced radiance + temporal
  history — and RENDERS THE ONLY IMAGE, at screen resolution, directly. No
  chained stages: chains make irreversible decisions on partial information
  at every seam (the argument that killed separate-denoiser chains
  industry-wide — DLSS-RR precedent as evidence, not authority).
- ★ UPSCALING IS DEAD AS A CONCEPT (Architect, 07-18): no small picture is
  ever made, so nothing is ever enlarged. The traced samples are EVIDENCE,
  not an image — a RAY BUDGET (samples/frame the machine affords at 60fps),
  not an internal resolution; samples need not sit on a grid. Supersedes the
  07-17 "640×480 + upscale" framing: the budget survives, the costume dies.
  Last legal use of the word: the live window's pre-cutover bilinear
  scaffold, which dies when HE plays the neural frame.
- Temporal accumulation = substrate, not a stage: the integrator gathering
  its own samples — live-window convergence today + ground-truth teacher
  data + the net's history input.
- Performance rule unchanged: the net beats the rays/cost it replaces at
  equal quality or it dies. Cutover of old selectable paths = his call,
  after HE plays it.
Lane lineage: r-direct spike (07-18) = first embodiment; sharpening at fix
pass: output at present res so upscaling dissolves into the same act.
SPIKE VERDICT (07-18, docs/perf/2026-07-18-rdirect-spike-verdict.md, lane
r-direct @ f403c65 unmerged): net BEATS chain on ALL held-out poses + scene
edit at equal 1spp budget (RMSE .0389/.0405/.0372 vs chain .0481/.0540/.0545)
@ 1.23x MAC; 18k params (fits on-chip). 60fps UNVERIFIED: CPU-ref only;
fp16/fused GPU kernel = next atom.

## Silicon race verdicts (07-18, Architect's full-speed order; host = M1 Pro)
- WGSL per-thread MLP: DOOR SHUT at native (280-300ms; even 2x32 net 32.5ms).
- ANE/CoreML: NO — refuses the net >16k px; never faster than CPU where it
  runs; native falls to GPU-GEMM ~27.5ms. Does NOT free the GPU. (ane-race)
- METAL TENSOR (MPSGraph GEMM): DOOR REOPENS — 4.47ms f32 @ native 960x640,
  ~94% roofline, parity 1.6e-7; 63x over WGSL. 60fps arithmetic: rays 7.6ms
  + net 4.47ms + elementwise ≈ ~13ms < 16.67. Remaining: gather/demod
  measure · buffer pooling (157MB intermediates; wall 23ms until pooled) ·
  wgpu↔MPSGraph interop lane · reload_shader↔temporal-pipeline gap.
  (r-direct @ eed0bdc, docs/perf/2026-07-18-rdirect-metal-tensor-spike.md)

## UNIFIED WORLD-NET LEDGER (07-18, searched — the challenger category, kept honest)
The Architect's question "do we need 2 networks?" = the actual frontier.
Category EXISTS: GameNGen (Doom inside one diffusion net, 20fps, single TPU)
· Genie 3 (DeepMind 08-2025: text→interactive 3D world, ONE net, 24fps@720p
real-time on datacenter silicon, coherence ~minutes, drifts/hallucinates)
· DIAMOND/Oasis lineage. Verdicts vs OUR laws: (1) budget — 24fps@720p on
TPUs vs our 60fps@native on M1 Pro; (2) ULTRADETERMINISM — dreamed worlds
drift; state ≠ f(seed,entropy,journal); breaks replay BY CONSTRUCTION;
(3) shipped SOTA (DLSS-RR, 1000+ titles) = our exact shape: one net fuses
the lossy reconstruction seams, real rays kept as evidence, physics
separate. Neural physics field shape = per-contact/per-body GNNs (Allen 23
rigid contact · ContactNets · NeurIPS-24 SDF-scaled) = P-N0's shape.
Multi-task truth (corrected from overclaim): negative transfer is COMMON
and capacity-dependent, not universal (ICLR-20; independent small nets
often superior — conditional).
STANDING RULE: unified world-net = NAMED CHALLENGER, not heresy — re-search
this ledger when silicon/scale moves; death rule decides chairs, never
ideology. Two nets, one engine remains the buildable design under M1@60 +
replay law.
