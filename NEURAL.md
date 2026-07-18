# NEURAL.md — ANE/CoreML race spike (R-Direct net, silicon lane 2/2)

Question: can the Neural Engine carry Act 2 (the R-Direct denoise+upscale MLP)
and free the GPU for rays, inside the 16.67ms/60fps budget?

## Net under test
- R-Direct MLP (real trained artifact `rdirect-weights-v1.bin`, GAIARDR1,
  73552B f32, sha `4f25cdb1…`): 23 feat → 5×(linear 64 + ReLU) → 3 LINEAR out.
  Absolute demod-log radiance. Feature layout per `rdirect.rs::pixel_features`
  (12 demod-log taps + 2 subpixel + 3 albedo + 3 normal + 1 log-depth + 2 motion).
- ONE forward per TARGET pixel; pixels = the batch dim → MLMultiArray [N,23]→[N,3].
- Shapes: 6144 px (96×64, the trained native res) · 614400 px (960×640, a real game res).

## Harness (authoring path: coremltools MIL → MLProgram fp16)
Location: `ane-spike/` in this worktree.
- `build_fixed.py` — reads the GAIARDR1 weights, hand-builds the MLP in coremltools
  MIL (`mb.linear`+`mb.relu`, weight row o = output o → [out,in]), converts to
  fp16 MLProgram (`ct.precision.FLOAT16`, macOS15 target). Fixed batch N per model.
  (`build_model.py` variant made the flexible-N `rdirect.mlpackage`; ANE wants
  static shapes so fixed models are the honest measure.)
- `ref_gen.rs` (rustc, no deps) — loads the REAL weights, runs the EXACT
  `rdirect.rs::Mlp::forward` (f32) → `golden.json` (64 feature/output pairs). Rust-side truth.
- `parity.py` / `bench.swift` / `bench2.swift` — parity + timing.
- `plan.swift` / `plan_ane.swift` — real silicon attribution via `MLComputePlan`.
- Host: macOS 26.5.1 (Tahoe), reported by CoreML as **Apple M1 Pro** (task said M1).

## Gate 1 — PARITY (fp16 CoreML vs Rust f32)  ✅ PASS
`N=64  maxAbs=0.006191  meanAbs=0.000320  maxRel=0.006357`  (tol rel<2e-2).
fp16 quantization holds the net to <0.7% worst rel error vs the trained f32 forward.

## Gate 2 — REAL per-frame ms (end-to-end predict, incl. marshal)
predict-only = pre-built input REUSED; marshal measured separately (naive Swift fill).

| shape | mode | predict-only median (ms) | marshal (ms) |
|---|---|---|---|
| 6144 (96×64) | cpuOnly | 0.589 | 0.166 |
| 6144 | cpuAndNeuralEngine | **0.486** | |
| 6144 | all | 0.504 | |
| 16384 (128×128) | cpuOnly | **1.437** | 0.457 |
| 16384 | cpuAndNeuralEngine | 1.685 | |
| 16384 | all | 1.695 | |
| 614400 (960×640) | cpuOnly | 28.010 | 17.28* |
| 614400 | cpuAndNeuralEngine | 27.513 | |
| 614400 | all | 28.191 | |

*marshal 17.3ms is a naive per-element Swift PRNG fill, NOT representative — a
real memcpy of 56.5MB (614400×23 f32) is ~few ms, or zero-copy (below) removes it.

## Gate 3 — SILICON ATTRIBUTION (real, via MLComputePlan — not just deltas)
Per-op device CoreML actually PLANS (`.cpuAndNeuralEngine` config unless noted):

- N=6144:  linear+relu → **NeuralEngine** (both `.all` and `.cpuAndNeuralEngine`).
- N=16384: linear+relu → **NeuralEngine** — but ANE (1.69ms) is SLOWER than cpuOnly (1.44ms).
- N=20480…262144: linear+relu → **CPU** (GPU-banned) — ANE REFUSED.
- N=614400: `.all` → **GPU (Apple M1 Pro)**;  `.cpuAndNeuralEngine` → **CPU**. ANE never chosen.

ANE row-batch ceiling for this net: **crossover between 16384 and 20480 rows**
(~128×128). Above it CoreML declines the ANE entirely. Honesty note: Xcode ANE
power counters are unavailable headless, but MLComputePlan device assignment is a
direct planner read, not an inference from timing — and it agrees with the deltas
(no ANE speedup anywhere).

## VERDICT
**No — the ANE cannot carry Act 2 at a real resolution, and does not free the GPU.**

1. At the trained 96×64 (6144 px): the net runs end-to-end on the ANE in ~0.65ms
   (0.49 predict + 0.17 marshal) — trivially inside the 16.67ms budget, GPU freed.
   But 96×64 is not a shippable frame; and even here ANE only ties/marginally beats
   CPU (0.49 vs 0.59ms).
2. At a real 960×640 (614400 px): CoreML **refuses the ANE** — it runs on the GPU
   (~27ms predict) or CPU (~27ms). That is **>16.67ms (misses 60fps)**, ~33ms fits 30fps.
   It DOES beat the 300ms GPU-compute/native disaster by ~11× — but that win is
   CoreML's optimized GPU GEMM (MPSGraph), NOT the ANE, and it lands ON the GPU, so
   it does **not** free the GPU for rays.
3. The ANE gives **no speedup at any batch** for this net: it is a memory-bandwidth
   bound wide GEMV (56MB input, ~157MB per-layer activations at native res), the
   regime where the ANE has no advantage and CoreML's planner won't select it past
   ~16k rows.

Path if ANE is still wanted: TILE the frame into ≤16k-px blocks to stay ANE-eligible
— but per-block dispatch overhead + no per-tile speedup make this a net loss vs the
one-shot GPU GEMM. The honest silicon answer for Act 2 at native res is the GPU
(27ms one net) — the ANE lane does not free it.

## Integration cost remaining
- Marshal: features must be packed row-major [N,23] contiguous (a GPU gather over
  the G-buffer + radiance taps). Zero-copy is feasible: back the MLMultiArray with an
  IOSurface/Metal buffer (`MLMultiArray(dataPointer:shape:…)`) so the CoreML input IS
  the render buffer — eliminates the 56MB copy. Output [N,3] = 7MB back to a texture.
- fp16 parity holds (<0.7%), so no accuracy blocker to fp16 CoreML deployment.

## GAPS
- Motion channels = 0 (static dataset) — untested under a moving-camera wave.
- Real marshal cost unmeasured (naive fill used); zero-copy claim is design-level,
  not yet benched.
- ANE power draw not directly read (headless); attribution is via MLComputePlan +
  timing deltas, which agree.
- Host is M1 **Pro** (7-8 GPU cores), not base M1 — base-M1 GPU numbers would be higher.
