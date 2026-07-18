# R-DIRECT GPU kernel verdict — 2026-07-18 (the 60fps mountain)

Branch `r-direct` (worktree `magic-crystal-rdirect`), UNMERGED lane.
Port the ONE net (`src/rdirect.rs`) to a fused WGSL compute kernel and MEASURE
real ms on this M1. Prior CPU-ref proof: net beats the denoise+upscale chain at
equal budget (`2026-07-18-rdirect-spike-verdict.md`). The question here is only
SPEED — does the net render at 60fps?

Kernels `src/rdirect.wgsl` (f32 anchor) + `src/rdirect_fast.wgsl` (fp16 MODE A) ·
driver `src/rdirect_gpu.rs` · harness `examples/rdirect_kernel.rs` ·
ordeals `tests/rdirect_gpu_ordeals.rs` (4 tests, green).

## Kernel design
- FUSED single dispatch, one MLP forward per TARGET pixel, feature gather
  inline (2×2 low-res demod-log radiance taps + subpixel + G-buffer
  albedo/normal/log-depth/motion = 23) → 5×64 ReLU → 3, ABSOLUTE demod-log
  out. Same fixed accumulation order as `Mlp::forward` → GPU-vs-CPU parity.
- `@workgroup_size(8,8)`, dispatch `(⌈w/8⌉, ⌈h/8⌉)`. No chain, no
  intermediate buffers. Bindings 0..7: uniform, weights, low radiance,
  hi albedo/normal/depth/motion, out.
- Two memory strategies measured:
  - **f32 no-cache** (`rdirect.wgsl`): weights in device storage, read direct.
  - **fp16 MODE A, threadgroup prefix-cache** (`rdirect_fast.wgsl`): weights
    f16, a 16000-scalar (32000 B) prefix cooperatively loaded into
    `var<workgroup>`, tail streamed; f32 accumulate.

## PARITY GATE (GPU output vs CPU reference, spike 96×64, derived bounds)
| kernel | parity_rel | derived bound | verdict |
|---|---|---|---|
| f32 no-cache | 5.636e-7 | 2.159e-3 | PASS — same net |
| fp16 MODE A | 5.791e-4 | 3.128e-3 | PASS — same net |
| f32, native 960×640 | 8.727e-6 | 2.159e-3 | PASS |
| fp16, native 960×640 | 9.328e-4 | 3.128e-3 | PASS |

Bounds derived (never frozen literals): f32 = (macs + 16·4 transcendental ULP)·u32;
fp16 MODE A = 2·u16 + macs·u32 (Higham dot-product / fp16 verdict method).
GPU determinism: byte-identical across two runs (ordeal a). The GPU kernel is
the SAME net, not an approximation drift.

## REAL ms — GPU compute pass, TIMESTAMP_QUERY, warm-up + median/min of 40
(60fps budget = 16.67 ms/frame; "leaves room for trace" target ≤ ~10ms.)

| shape | target px | f32 no-cache | fp16 threadgroup-cache |
|---|---|---|---|
| spike 96×64  (low 48×32)   | 6 144   | **5.41 ms** (32% budget) | 20.4 ms (123%) |
| native 960×640 (low 640×480) | 614 400 | **280–315 ms** (~1750%) | ~2056 ms (~12300%) |

Method: GPU timestamp deltas around the compute pass ONLY (upload/readback
excluded); 8 warm-up dispatches dropped, median of 40. min≈median (tight,
reproducible). native = canonical present size (src/main.rs GAIA_NATIVE_WIDTH/
HEIGHT 960×640; low = GAIA_NATIVE_RENDER_W/H 640×480).

### Net-size sweep @ native 960×640 (f32 no-cache, timing-only, random weights)
| layers×w | macs/px | f16 size | median ms | % budget |
|---|---|---|---|---|
| 2×32 | 1 856  | 3.8 KB  | 32.5  | 195% |
| 3×32 | 2 880  | 5.8 KB  | 45.7  | 274% |
| 3×48 | 5 856  | 11.7 KB | 89.5  | 537% |
| 4×48 | 8 160  | 16.3 KB | 130.6 | 784% |
| 4×64 | 13 952 | 27.8 KB | 227.8 | 1367% |
| 5×64 | 18 048 | 35.9 KB | 280.9 | 1685% |

## VERDICT: 60fps is NOT within reach for R-Direct as specified on this M1.
- The prompt's premise ("18k params ≈ 36 KB FITS IN THREADGROUP / ON-CHIP") is
  **false on M1**: 36 KB > the 32 KB Metal threadgroup limit. The whole net
  cannot be cached (unlike the 13.8 KB VIII-3 upscaler, which fit and was
  saved by exactly this trick).
- Worse, filling threadgroup memory COLLAPSES occupancy: the 32 KB
  `var<workgroup>` cache forces ~1 resident workgroup per core → no latency
  hiding → the fp16 threadgroup-cache kernel is **4–7× SLOWER** than the f32
  no-cache kernel (20 ms vs 5 ms at the spike; 2056 ms vs ~300 ms at native).
  For THIS net size the threadgroup cache is an anti-optimization.
- The wall is not only weight memory. The net-size sweep shows even a trivial
  **2×32 (1856 MAC) net is 32.5 ms at native = 195% of budget** — there is a
  large fixed per-pixel floor (614 400 threads × {4 tap reads, 16 `ln`, 3
  `exp`, `array<f32,64>` scratch}). No net shape crosses 16.67 ms at native.
- Root cause vs the chain it beats: R-Direct runs the FULL net at EVERY native
  pixel; the chain runs its expensive denoiser at LOW res (¼ the pixels) and
  only a cheap upscale at native. "Fusing" quadrupled the heavy evaluations.
  The spike shape (96×64 = ¼-res-ish) IS in budget at f32 (5.4 ms) — the cost
  is native per-pixel evaluation, not the net per se.

## What remains for 60fps
1. Do NOT run the full net per native pixel. Evaluate at low res + a cheap
   (bilinear/tiny) upscale to native — i.e. structurally the chain's shape,
   which contradicts "the net is the renderer." A true native per-pixel MLP
   at 18k MAC is off by >15× on M1.
2. Cut the fixed per-pixel floor: precompute the 2×2 log-demod radiance taps
   into a low-res buffer (removes 12+ `ln` per native pixel), drop the
   `array<f32,64>` register scratch (spills), fuse `exp`/`undo` cheaply.
3. Threadgroup caching only helps a net small enough to leave occupancy
   headroom (≤ ~8–10 KB f16 ≈ ≤ 4–5k params) — and even then the sweep says
   such a net is still ~45–90 ms at native, so caching alone will not close
   the gap.
4. Metal cooperative/simdgroup matrix ops (not exposed in wgpu 30) would help
   the MAC term but not the transcendental/gather floor.
5. Trace-side cost is not even in the budget yet — the ≤10 ms kernel target
   assumed room for the 1-spp trace; with the kernel alone at ~300 ms native
   that headroom is moot.

## Honest gaps
- Motion channels are zero (static-pose dataset) — unchanged from the spike;
  ms is unaffected (the reads happen regardless).
- fp16 native timing measured at WARMUP8/TIMED40 (~2056 ms); the checked-in
  example skips it by default (≤100k px guard) to stay under the 300 s run cap
  — re-enable by removing the `time_this` guard.
- The threadgroup array is a fixed 16000 scalars at compile time, so even a
  small net pays full 32 KB occupancy cost in the fast kernel; a per-net-sized
  cache would need shader specialization (not attempted — the f32 no-cache
  sweep already shows the ceiling).
