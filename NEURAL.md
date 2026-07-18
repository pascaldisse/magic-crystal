# NEURAL — DreamForge neural performance ledger (rewritten 2026-07-18, two-act law)

Status: LAW, not draft. Supersedes the 2026-07-16 upscaler/denoiser-chain
draft (spec-concordance item 15) — pre-two-act content moved to LINEAGE,
bottom, kept verbatim per adversary-charter disclosure discipline. Only
the two-act law + silicon race verdicts + unified world-net ledger are
normative below.

## ★ THE ONE-RENDER LAW (Architect's ruling, 07-18 — supersedes staged chain
as DESTINATION; NR1/NR2 demote to teachers/baselines, lab equipment only)
Pleroma renders once: world truth in; final image or nothing out. Inside
Pleroma: sampling and judgment jointly consume full-resolution geometry
features (primary visibility is cheap), sparse traced radiance, and temporal
history. Rays yield evidence/byproducts, never a picture or separate pass;
no chained seams make irreversible partial decisions (DLSS-RR = evidence, not
authority).
- ★ UPSCALING IS DEAD AS A CONCEPT (Architect, 07-18): no small picture is
  ever made, so nothing is ever enlarged. The traced samples are EVIDENCE,
  not an image — a RAY BUDGET (samples/frame the machine affords at 60fps),
  not an internal resolution; samples need not sit on a grid. Supersedes the
  07-17 "640×480 + upscale" framing: the budget survives, the costume dies.
  Last legal use of the word: the live window's pre-cutover bilinear
  scaffold, which dies when HE plays the neural frame.
- Temporal accumulation = substrate: Pleroma's gathered samples —
  live-window convergence today + ground-truth teacher data + its history
  input.
- Performance rule unchanged: Pleroma's learned act beats the rays/cost it replaces at
  equal quality or it dies. Cutover of old selectable paths = his call,
  after HE plays it.
Lane lineage: r-direct spike (07-18) = first embodiment; output at present
resolution; upscaling dissolves into Pleroma's one render. SPIKE VERDICT
(07-18, docs/perf/2026-07-18-rdirect-spike-verdict.md, lane r-direct @
f403c65 unmerged): Pleroma's learned act BEATS the retired chain on ALL held-out poses + scene
edit at equal 1spp budget (RMSE .0389/.0405/.0372 vs chain .0481/.0540/.0545)
@ 1.23x MAC; 18k params (fits on-chip). 60fps UNVERIFIED: CPU-ref only;
fp16/fused GPU kernel = next atom.

## Silicon race verdicts (07-18, Architect's full-speed order; host = M1 Pro [source: docs/perf/2026-07-18-rdirect-metal-tensor-spike.md])
- WGSL per-thread MLP: DOOR SHUT at native (280-300ms; even 2x32 Pleroma
  learned act 32.5ms). [source: docs/perf/2026-07-18-rdirect-metal-tensor-spike.md]
- ANE/CoreML: NO — refuses Pleroma's learned act >16k px; never faster than
  CPU where it runs; native falls to GPU-GEMM ~27.5ms. Does NOT free the GPU.
  (ane-race) [source: docs/perf/2026-07-18-rdirect-metal-tensor-spike.md]
- METAL TENSOR (MPSGraph GEMM): DOOR REOPENS — 4.47ms f32 @ native 960x640 [source: docs/perf/2026-07-18-rdirect-metal-tensor-spike.md]
  [pre-law lab measurement @960×640; remeasure at 640×480 canvas owed],
  ~94% roofline, parity 1.6e-7; 63x over WGSL. 60fps arithmetic: rays 7.6ms
  + Pleroma's learned act 4.47ms + elementwise ≈ ~13ms < 16.67. Remaining:
  gather/demod measure · buffer pooling (157MB intermediates; wall 23ms until
  pooled) · wgpu↔MPSGraph interop lane · reload_shader↔temporal-pipeline gap.
  [source: docs/perf/2026-07-18-rdirect-metal-tensor-spike.md]
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

---

## LINEAGE — pre-two-act neural doctrine (2026-07-16 draft, superseded 2026-07-18, kept verbatim)

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

### Law
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

### Render ledger (published numbers → our expected gains)
| Component | Evidence | Gain |
|---|---|---|
| Internal-res dial + temporal upscale | MetalFX: ~40-50% quality mode, >2× performance mode; wgpu interop exists (texture_from_raw) | THE biggest multiplier: path tracing at ~¼–½ native pixels ⇒ 2–4× ray budget back |
| Learned/temporal denoiser | SVGF ≈10ms baseline (too fat — ours must beat); DLSS-RR = proprietary proof learned>hand-tuned | 1–2 spp presents like ~10× spp; makes low-spp PT shippable |
| Neural radiance cache (NRC-class) | Müller 2021: ~2.6ms @1080p update+query, 6×64 fp16 fused MLP; ★ reimplemented in RUST compute shaders on mobile GPUs (Breda, SIGGRAPH Asia 2025), query 2–25× cheaper than tracing the paths | Long-bounce GI nearly free after 1st bounce; kills fireflies; the "remembers light" network |
| ReSTIR (not neural, same slot) | 9.3–166× MSE win at 1 spp | Infinite lights at fixed budget |
Combined M1@60 equation: ~1080p internal · 1-2 spp · ReSTIR · cache ·
denoise · upscale to native — every stage individually published-proven;
the composition is ours to prove per milestone (Xcode limiter captures).

### Physics ledger
| Component | Evidence | Gain | Tier |
|---|---|---|---|
| Subspace neural dynamics (Holden 2019) | 300–5,000× vs full sim; 2700 FPS vs 0.5 FPS cloth | Pascal's "1000%" is conservative | cloth/soft decorative + far-field |
| Graph-network simulators (GNS 2020→geoelements 2023) | >165× vs parallel-CPU MPM granular; generalizes ~10× particle counts | many-object far-field, debris fields | far-field |
| Coarse-sim + neural detail (fluids) | tempoGAN = offline super-res; NeuralVDB inference too slow realtime (decode-first) | fluid DETAIL layer = research milestone, coarse SPH/grid stays primary | detail tier |
| Neural collision | MLP-SDF wins only <~10k queries GPU | niche; not load-bearing | targeted |
Honesty line (unchanged): ZERO shipped-game precedent for learned sim in
the loop — we would be first. Staged experiments BEHIND the exact solver,
never a dependency; promotion requires side-by-side truth comparison.
Superseded 07-18 by PHYSICS.md §0 (Ananke → THE NET [= Pleroma's learned act] → state); "near-field
exact / far-field surrogate" tiering = a physics LOD, struck.

### Memory note (16 GB unified)
MLPs are tiny (6×64 fp16 ≈ KBs); training buffers ≈ MBs. Neural layer is
compute-bound, not memory-bound — memory budget stays owned by VT pages +
geometry pages + voxel volumes (unified memory = zero-copy between sim and
render, the M1's actual gift).

### Staging (build later — order fixed now; superseded 07-18, see ★ TWO-ACT LAW above)
NR1 temporal upscaler (own, cross-platform) + MetalFX trait on Metal
NR2 denoiser: temporal+spatial compute → learned variant when NR1 stable
NR3 radiance-cache MLP (Breda-pattern fused fp16 compute; NRC numbers)
NR4 physics surrogates: cloth/debris far-field A/B vs exact solver;
    fluid neural-detail research spike
Each gated: same scene, neural on/off, frame-time + image/physics-error
metrics on the MacBook. No gate, no promotion.

## TRAINING DOCTRINE (sealed 07-18 — Architect delegated: "your field"; N1/P-N1 inherit)
No million-example diets. Three tiers, all used, strongest first:
1. STRUCTURE — the algorithm IS the architecture: unroll the iterative solve
   (XPBD iterations / reconstruction steps) as layers; init to reproduce one
   classical step exactly; training learns only what the algorithm leaves
   open (steps, preconditioning, residual corrections). Orders of magnitude
   in sample efficiency; the net starts as the solver and learns to beat it.
2. EQUATIONS AS SIGNAL — self-supervision from our own laws: physics trains
   on the differentiable CONSTRAINT RESIDUAL (violation = loss; teacher
   demotes to validator); render trains noise2noise (two independent noisy
   renders of a pose = training pair; converged frames = quality gate only,
   never a diet). Caveat honest: equation-only is finicky on stiff corners
   (spectral bias) → hybrid below.
3. DATA = COMPUTE — remaining examples generated by our own engine;
   ENTROPY law makes every dataset = f(seed): datasets are code, not
   assets. Thousands-class counts, self-generated, overnight.
Sweet spot = 1+2+solver-in-the-loop (net trains inside running sim vs real
rollouts). Evidence: r-direct 18k params beat the chain on one scene's data.
CONDITIONING VECTORS (vgel/repeng idea, born-in not post-hoc): nets take
small conditioning inputs (quality dial, material/stiffness regime) via
FiLM-style modulation — one net, continuous dials, no second system (the
Architect's 07-16 "levels for the AI stuff" wish, lawful form). Post-hoc
activation steering rejected at our scale (nothing to steer; retrain is
free). Activation-direction reads live under NEO as diagnosis (Kashf).
Death rule judges all of it, as everything.
