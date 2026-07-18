# R-DIRECT Metal tensor-kernel spike — 2026-07-18 (the door WGSL shut, reopened)

Branch `r-direct` (worktree `magic-crystal-rdirect`), UNMERGED lane. Silicon
race lane 1 of 2. Follows `2026-07-18-rdirect-gpu-kernel.md` (the WGSL verdict:
native per-pixel MLP unaffordable — ~280 ms f32 @960×640, even a trivial 2×32
net = 32.5 ms; 36 KB net > 32 KB threadgroup; fp16 cache = occupancy collapse).

QUESTION: does Metal's TENSOR machinery — the SAME net reformulated as BATCHED
MATMULS (pixels×23 · 23×64 · … · 64×3) instead of one MLP per GPU thread —
reopen the native-res door?

## Harness
- `tools/metal4-probe/main.swift` (Swift 6.2, standalone) · API = **MPSGraph**
  (MetalPerformanceShadersGraph). Build:
  `swiftc -O main.swift -o metal4-probe -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation`.
- Parses the committed `rdirect-weights-v1.bin` (GAIARDR1) directly → same net
  (6 layers, 23→5×64 ReLU→3). Forward = chain of `matrixMultiplication`
  (X[N,in]·Wᵀ + b, ReLU on hidden). f32 and fp16-storage/f32-accumulate builds.
- Fixture exported by `cargo run -p scrying-glass --release --example
  rdirect_export_features`: traces the SAME naruko "front" pose the WGSL measure
  used at the spike shape (48×32 → 96×64, 6144 px), writes
  `tools/metal4-probe/data/{features.f32 [6144×23], expected.f32 [6144×3],
  rdirect-weights-v1.bin, meta.json}`. `expected` = the Rust CPU-reference
  `Mlp::forward` output (demod-log, the matmul-chain output — the harness
  reproduces the NET FORWARD, not the per-pixel feature gather / undo_log_demod).

### WHY MPSGraph, not native Metal 4 MTLTensor
The device reports `supportsFamily(.metal3) = true` and macOS Tahoe 26.5.1 has
the Metal 4 stack (MTLTensor, `MTL4MachineLearningCommandEncoder`). MPSGraph was
chosen because it COMPILES AND RUNS TODAY on this SDK with zero binding work and
its `matrixMultiplication` lowers to Apple's simdgroup-matrix / AMX GEMM units —
the exact tensor path wgpu 30 could not reach. The Metal 4 ML-encoder routes
through the same units; measuring MPSGraph measures that machinery. Native
MTLTensor bindings are a follow-up, not needed for the verdict.

## GATE 1 — PARITY (real exported features @ 96×64, vs Rust CPU `Mlp::forward`)
| build | parity_rel | max\|Δ\| | derived bound | verdict |
|---|---|---|---|---|
| f32  | **1.601e-7** | 1.311e-6 | 1.727e-2 | PASS — same net |
| fp16 | **1.302e-3** | 1.191e-2 | 2.502e-2 | PASS — same net |

Bound method = the WGSL verdict's (Higham): f32 ≈ (macs + transcendental
budget)·u32; fp16 ≈ 2·u16 + macs·u32; ×8 slack for MPS GEMM tiling reorder.
f32 is essentially bit-exact; fp16 storage drifts ~1e-3 as expected. The tiled
native buffers also reproduce the CPU `expected` on their first 6144 rows
(`cpu-parity: match` at both shapes).

## GATE 2 — REAL ms on M1 Pro (Apple M1 Pro, macOS 26.5.1)
Single-forward WHOLE-GRAPH GPU time: MPSGraphExecutable encoded on ONE
MTLCommandBuffer, `gpuEndTime − gpuStartTime`, warm-up 8 + median of 40. TFLOPS
= 2·macs·N / ms, sanity-checked against the M1 Pro ~5.3 TFLOPS fp32 roofline.

| shape | px | f32 GPU ms | fp16 GPU ms | TFLOPS | % of 16.67 ms |
|---|---|---|---|---|---|
| spike 96×64   | 6 144   | 0.33 | 0.24 | 0.7–0.9 (launch-bound, too small to saturate) | ~2% |
| native 960×640 | 614 400 | **4.47** | **4.47** | **5.0 (~94% roofline)** | **~27%** |

MEASUREMENT HONESTY (both directions checked, artifacts rejected):
- `MPSGraph.encode(to:)` (non-executable) reported 1.5 ms native = ~15 TFLOPS →
  **REJECTED**, >2× roofline: it splits the graph across internal command
  buffers, so our buffer's GPU timestamps capture only a fragment.
- "K identical forwards back-to-back on one buffer / K" reported 0.29–0.39 ms =
  56–76 TFLOPS → **REJECTED**, >10× roofline: identical input → the driver
  elides redundant work.
- The single-forward executable timer (4.47 ms, 5.0 TFLOPS ≈ 94% of peak) is the
  ONLY physically consistent figure → authoritative. f32 and fp16 tie because
  the M1 GPU has no double-rate fp16 (fp16 buys memory, not FLOPS).
- Per-call WALL clock (commit→wait, no readback) = 22–26 ms native: dominated by
  per-call graph encode + the 157 MB [614400×64] intermediate allocation. A
  warmed renderer with pooled buffers amortizes this to the 4.47 ms GPU cost;
  reported alongside so the alloc gap is not hidden.

## GATE 3 — VERDICT: the door REOPENS for the net forward.
- Native 960×640 net forward = **4.47 ms** (f32 or fp16) vs the WGSL per-pixel
  MLP's **~280 ms** → **~63× faster**, and **≤ 10 ms** with ~5.5 ms of the
  16.67 ms frame left for the 1-spp trace + feature gather.
- WHY matmul batching wins where per-thread MLPs died: the WGSL kernel ran the
  full net PER THREAD, re-fetching all 18k weights from device memory per pixel
  (memory-bound, zero weight reuse) with a register-spilling `array<f32,64>`
  scratch. The GEMM formulation reuses each weight across the whole 614 400-px
  batch via simdgroup-matrix tiling (weights stay resident, amortized) — turning
  a memory-bound per-thread problem into a compute-bound GEMM that runs at ~94%
  of the fp32 roofline. Same net, same weights, ~63× the throughput.

## Honest gaps (what 4.47 ms does NOT yet include)
1. **Feature gather + undo are out of scope of this number.** The harness times
   only the matmul chain. The WGSL verdict measured a large FIXED per-pixel
   transcendental/gather floor SEPARATE from MACs (2×2 log-demod taps = 16 `ln`,
   depth `ln`, 3 output `exp`; even a 0-MAC net paid ~30 ms native from it). In
   the GEMM design that floor becomes two cheap elementwise passes (featurize →
   X buffer; undo_log_demod on the [N,3] output) — NOT per-thread inside the MLP,
   so it should be far cheaper, but it is UNMEASURED here. A full fused native
   R-Direct = 4.47 ms (GEMM) + featurize pass + undo pass; only the GEMM is proven.
2. **Allocation must be pooled.** The 22–26 ms wall clock is real until the
   157 MB intermediate + result buffers are pooled/double-buffered across frames.
3. **fp16 gave no speedup on M1** (no double-rate) — only ~2× weight/activation
   memory. f32 is the honest default here.
4. Motion channels still zero (static-pose dataset) — unchanged; ms unaffected.
5. Native MTLTensor / MTL4MachineLearningCommandEncoder not exercised — MPSGraph
   already lands the verdict; the ANE path (potentially off-GPU) is the next spike.

## One-line for NEURAL.md
Per-pixel MLP inference recast as a batched GEMM (MPSGraph) runs the R-Direct
net at native 960×640 in **4.47 ms on M1 Pro (~94% fp32 roofline), ~63× the
WGSL per-thread kernel** — the tensor door the WGSL per-pixel path could not
open. Gaps: feature-gather/undo passes + buffer pooling unmeasured.
