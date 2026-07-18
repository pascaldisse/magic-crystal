# SILICON RACE II — THE ANE MATRIX · 2026-07-18

## Scope seal

- Architect steer 18:27 → Metal-native ML command encoder = sole neural-cores door.
- Prior high-level R-A path → VOID.
- R-B expanded → tiny + Pleroma shapes; CPU + MPSGraph controls; `MTL4MachineLearningCommandEncoder` door.
- R-C voice → SKIP by steer; no inference claim.
- Host → Apple M1 Pro · macOS 26.5.1 (25F80) · Xcode 26.3 (17C529) · Swift 6.2.4.
- Weights → deterministic arbitrary constants; performance-only; no quality claim.
- Shapes:
  - tiny → `14→32→32→3`; batches `64,128,256,512,1024,2048,4096`.
  - Pleroma → `23→5×64→3`; batch `640×480 = 307200`.

## Method

- CPU → f32 Accelerate SGEMM + bias/ReLU loops; reused input/activation storage; warm state; median single forward; tiny `n=200`, Pleroma `n=9`.
- GPU → MPSGraph batched matmuls; fp16 storage + f32 I/O; compiled executable; reused feed; warm state; one forward/command buffer; tiny median `n=200`, Pleroma median `n=30`.
- GPU columns:
  - `timeline` → `MTLCommandBuffer.gpuStartTime→gpuEndTime`.
  - `encode` → CPU time in executable encode.
  - `wall` → encode→commit→wait; immediate per-call dispatch/synchronization included.
  - `sync/submit` → `wall − encode − timeline`; derived control quantity.
- Correctness gate → full output GPU-vs-CPU max absolute delta `1–3×10⁻⁶` across every shape.
- Raw source + result → `tools/silicon-race-2/baselines.swift` · `proof/2026-07-18-silicon-race-2-baselines.txt` · commit `7de8e3b`.

## Full matrix — part × silicon × measured ms

`UNVERIFIED` = no network dispatch; never latency-inferred into a silicon claim.

| part | batch | silicon/path | per-call ms | throughput, items/s | HOW execution verified |
|---|---:|---|---:|---:|---|
| tiny | 64 | CPU | 0.016500 | 3,878,788 | Accelerate call on host CPU; wall timer |
| tiny | 64 | GPU | 0.033083 timeline · 0.305000 wall | 209,836 wall | Metal command-buffer GPU timestamps |
| tiny | 64 | ANE via MTL4 | **UNVERIFIED** | **UNVERIFIED** | package wall; no dispatch |
| tiny | 128 | CPU | 0.030167 | 4,243,047 | Accelerate call on host CPU; wall timer |
| tiny | 128 | GPU | 0.038750 timeline · 0.321208 wall | 398,496 wall | Metal command-buffer GPU timestamps |
| tiny | 128 | ANE via MTL4 | **UNVERIFIED** | **UNVERIFIED** | package wall; no dispatch |
| tiny | 256 | CPU | 0.058333 | 4,388,597 | Accelerate call on host CPU; wall timer |
| tiny | 256 | GPU | 0.039500 timeline · 0.306500 wall | 835,237 wall | Metal command-buffer GPU timestamps |
| tiny | 256 | ANE via MTL4 | **UNVERIFIED** | **UNVERIFIED** | package wall; no dispatch |
| tiny | 512 | CPU | 0.083833 | 6,107,380 | Accelerate call on host CPU; wall timer |
| tiny | 512 | GPU | 0.040542 timeline · 0.300208 wall | 1,705,484 wall | Metal command-buffer GPU timestamps |
| tiny | 512 | ANE via MTL4 | **UNVERIFIED** | **UNVERIFIED** | package wall; no dispatch |
| tiny | 1024 | CPU | 0.158333 | 6,467,382 | Accelerate call on host CPU; wall timer |
| tiny | 1024 | GPU | 0.061708 timeline · 0.332542 wall | 3,079,310 wall | Metal command-buffer GPU timestamps |
| tiny | 1024 | ANE via MTL4 | **UNVERIFIED** | **UNVERIFIED** | package wall; no dispatch |
| tiny | 2048 | CPU | 0.311708 | 6,570,252 | Accelerate call on host CPU; wall timer |
| tiny | 2048 | GPU | 0.050333 timeline · 0.326084 wall | 6,280,590 wall | Metal command-buffer GPU timestamps |
| tiny | 2048 | ANE via MTL4 | **UNVERIFIED** | **UNVERIFIED** | package wall; no dispatch |
| tiny | 4096 | CPU | 0.612083 | 6,691,903 | Accelerate call on host CPU; wall timer |
| tiny | 4096 | GPU | 0.061583 timeline · 0.347208 wall | 11,796,963 wall | Metal command-buffer GPU timestamps |
| tiny | 4096 | ANE via MTL4 | **UNVERIFIED** | **UNVERIFIED** | package wall; no dispatch |
| Pleroma | 307200 | CPU | 198.570417 | 1,547,058 | Accelerate call on host CPU; wall timer |
| Pleroma | 307200 | GPU | 4.302625 timeline · 12.410250 wall | 24,753,732 wall | Metal command-buffer GPU timestamps |
| Pleroma | 307200 | ANE via MTL4 | **UNVERIFIED** | **UNVERIFIED** | package wall; no dispatch |

### GPU call decomposition

| part | batch | encode CPU ms | GPU timeline ms | sync/submit ms | wall ms |
|---|---:|---:|---:|---:|---:|
| tiny | 64 | 0.055709 | 0.033083 | 0.216208 | 0.305000 |
| tiny | 128 | 0.057125 | 0.038750 | 0.225333 | 0.321208 |
| tiny | 256 | 0.056916 | 0.039500 | 0.210084 | 0.306500 |
| tiny | 512 | 0.058125 | 0.040542 | 0.201541 | 0.300208 |
| tiny | 1024 | 0.064041 | 0.061708 | 0.206793 | 0.332542 |
| tiny | 2048 | 0.070375 | 0.050333 | 0.205376 | 0.326084 |
| tiny | 4096 | 0.068542 | 0.061583 | 0.217083 | 0.347208 |
| Pleroma | 307200 | 2.471000 | 4.302625 | 5.636625 | 12.410250 |

## R-B · Metal 4 door report

### Reached

- SDK headers → present:
  - `MTL4MachineLearningCommandEncoder.h`
  - `MTL4MachineLearningPipeline.h`
  - `MTLTensor.h`
- Swift overlay → synchronous + async `makeMachineLearningPipelineState(descriptor:)` present.
- Probe compile → PASS under `swiftc -O -j 2`; Metal + Foundation only.
- Runtime M1 Pro/Tahoe → MTL4 queue · allocator · compiler · argument table all created.
- Runtime tensors → input/output allocation PASS for tiny-64, tiny-4096, Pleroma-307200.
- Runtime ML encoder → object created + ended beside dummy GPU compute in one MTL4 command buffer.
- Empty-encoder control only → `5.334 µs` CPU encode · `0.062833 ms` whole-control GPU timeline · `0.210958 ms` commit/wait. **Not network timing; no silicon-locus evidence.**

### Exact wall

1. `MTL4MachineLearningPipelineState` requires a specialized function from a Metal ML package.
2. Installed default toolchain search → `metal` + `metal-package-builder`; no Metal-native network-authoring frontend found.
3. Package-builder attempt against prior compiled artifact → no output package:

```text
metal-package-builder: error: Failed to read model package at ane-spike/f_16384.mlmodelc/ -- file:///Users/pascaldisse/projects/magic-crystal-ane/. Error: A valid manifest does not exist at path: /Users/pascaldisse/projects/magic-crystal-ane/ane-spike/f_16384.mlmodelc/Manifest.json
tool_exit_status=0 output_exists=no probe_exit_status=1
```

4. Ordinary Metal library function substitution → hard runtime wall, not a recoverable Swift error:

```text
negative_control=compile_ordinary_metal_function_as_ml_pipeline
NSInvalidArgumentException: -[_MTLLibrary executableWithDeviceSelection:]: unrecognized selector
... -[_MTL4Compiler newMachineLearningPipelineStateWithDescriptor:error:]
exit_status=134
```

5. Project law → non-Metal source-package authoring route forbidden. Result → no lawful source artifact for tiny/Pleroma package conversion in this lane.

Raw source + walls → `tools/silicon-race-2/metal4-door.swift` · `tools/silicon-race-2/probe-package-wall.sh` · `proof/2026-07-18-silicon-race-2-metal4-{door,negative}.txt` · `proof/2026-07-18-silicon-race-2-package-wall.txt` · commit `abae4ce`.

### Execution locus

| path | locus evidence | verdict |
|---|---|---|
| CPU baseline | direct Accelerate host calls | CPU CONFIRMED |
| MPSGraph baseline | Metal GPU start/end timestamps around executable | GPU CONFIRMED |
| MTL4 network | no dispatch | **ANE/GPU locus UNVERIFIED** |
| power counters | `sudo: a password is required` | unavailable; no proxy claim |

## Verdicts

### Tiny physics/mind class

- CPU wins end-to-end latency through batch 2048.
- Batch 64 → CPU `0.0165 ms`; GPU wall `0.3050 ms` → CPU `18.5×` lower latency.
- Batch 2048 → CPU `0.3117 ms`; GPU wall `0.3261 ms` → practical tie; CPU still `4.5%` lower.
- Batch 4096 → GPU wall `0.3472 ms`; CPU `0.6121 ms` → GPU `1.76×` faster.
- 60–120 Hz → every measured tiny path inside `16.67/8.33 ms`; CPU consumes `0.20–7.34%` of 120-Hz frame; synchronous GPU call consumes `3.60–4.17%`.
- Neural cores beating either → **UNVERIFIED**; no MTL4 network dispatch.
- Current honest choice → CPU ≤2048 bodies; GPU at 4096 only when synchronous wall latency, not raw device time, governs.

### Pleroma class · 307200 pixels

- CPU → `198.570 ms` → dead for frame path.
- GPU → `4.303 ms` device · `12.410 ms` immediate call wall → 60-Hz fit in isolation; only `4.260 ms` left for all other frame work; 120-Hz miss.
- GPU wall vs CPU → `16.0×` faster.
- GPU encode/allocation/synchronization gap → `8.108 ms` beyond device work; renderer-side pooling/async scheduling opportunity, **UNVERIFIED here**.
- ANE acceptance · latency · overlap with dummy GPU work · fallback behavior → **all UNVERIFIED** at package wall.

### Voice

- SKIP → Architect steer; same Metal-native door owed later; no tokens/s claim.

## Gaps ledger

- MTL4 network package for either shape → absent.
- Actual `dispatchNetwork(intermediatesHeap:)` → unexecuted.
- Network encode CPU cost → **UNVERIFIED**.
- GPU-timeline barrier/sync overhead with network + dummy GPU work → **UNVERIFIED**.
- ANE vs GPU scheduler choice → **UNVERIFIED**.
- Neural-core power/utilization → **UNVERIFIED**; privilege wall.
- Pleroma parity/quality → not asked; arbitrary weights; performance shape only.
- MPSGraph wall gap attribution → aggregate only; allocation vs driver vs wait split **UNVERIFIED**.

## Source trail

- API shape/recon → `client-rs/research/metal4-neural-recon.md`.
- Timing pattern → `tools/metal4-probe/main.swift` in R-Direct lane.
- Baseline raw → `proof/2026-07-18-silicon-race-2-baselines.txt`.
- MTL4 runtime raw → `proof/2026-07-18-silicon-race-2-metal4-door.txt`.
- Pipeline negative raw → `proof/2026-07-18-silicon-race-2-metal4-negative.txt`.
- Package wall raw → `proof/2026-07-18-silicon-race-2-package-wall.txt`.
