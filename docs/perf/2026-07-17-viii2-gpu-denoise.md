# RITE VIII-2 — GPU denoiser port: parity derivation + ms measurement

Lane `rite8-viii2-gpu-denoise`. Port of the VIII-1 CPU reference denoiser
(`src/denoiser.rs`) to a wgpu/WGSL compute pass (`src/denoiser.wgsl` +
`src/denoiser_gpu.rs`). Same hash-pinned weights (`denoiser-weights-v1.bin`),
LOADED via `deserialize_weights` → `Mlp::flat_weights` upload, never re-derived.

## Parity verdict — NOT bit-exact; DERIVED fp32 tolerance, measured ~700× under

fp32 bit-exact GPU-vs-CPU is not feasible: Metal contracts `a*b + c` into a
fused multiply-add (FMA, single rounding) where the CPU reference does mul
then add (two roundings). Same math, different rounding — so parity is a
DERIVED tolerance, not equality.

Derivation (machine-computed in `tests/viii2_ordeals::derived_parity_rel_bound`,
never a frozen literal):
- forward pass = `macs` fused multiply-adds; `macs = Σ_layer in·out` read from
  `Mlp::layer_dims()` = 10·32 + 32·32·3 + 32·3 = **3488**.
- FMA vs mul-then-add differ by ≤ 1 unit-roundoff `u` per MAC (FMA is the more
  accurate; the DIFFERENCE is bounded by the same `u` — Higham, dot-product
  error analysis). `u = f32::EPSILON = 2^-23 ≈ 1.19e-7`.
- feature/undo transforms add 7 transcendentals (3 `ln` in, 1 `ln` depth, 3
  `exp` out); budget 4 ULP each = 28·u (the MAC term dominates).
- relative bound = `(macs + 28)·u` ≈ **4.19e-4**.

MEASURED relative spread `rmse(gpu,cpu)/rmse(cpu,0)` (from the standalone
example `viii2_gpu_dream`, which runs `front` + `orbit_-20`):
- front (train pose):    **5.965e-7**
- orbit_-20 (held-out):  **5.759e-7**

→ ~700× under the derived bound. The ordeal `b_and_c` re-derives and asserts
`parity_rel ≤ bound` itself, machine-checked over the OFFICIAL validation set
(`VALIDATION_POSE_NAMES` = `orbit_-20` + `orbit_+40` — see the quality-gate
section below for those numbers).

## Quality gate on GPU output — beats-noisy, machine-checked (the real gate)

The GPU-denoised RMSE vs the 128-frame reference, the two TRUE held-out
validation poses the ordeal gates (`VALIDATION_POSE_NAMES` = `orbit_-20` +
`orbit_+40` — NOT `front`, which is a TRAIN pose; `front` appears above only
in the parity measurement, taken from the standalone example), at the
pinned dataset resolution 96×64:
- orbit_-20:  noisy 0.052073 → GPU-denoised **0.042999** (matches CPU 0.042999)
- orbit_+40:  noisy 0.095662 → GPU-denoised **0.049997** (matches CPU 0.049997)

Both strictly beat noisy (ordeal `b_and_c`, `gpu_rmse < noisy_rmse`). The GPU
result equals the CPU reference to 6 decimals — the pinned VIII-1 bound
(0.049997, on `orbit_+40`) still holds on GPU output.

## GPU denoise-pass cost — measured (timestamp queries), M1

Bracketed compute pass only (upload/readback excluded), median of 32
dispatches, `wgpu::Features::TIMESTAMP_QUERY`:
- 96×64 (dataset res):   min 1.05 ms / median 1.32 ms
- **900×600 (perf-audit present res): min 26.54 ms / median 26.58 ms**

Against the 60-fps frame budget (16.67 ms; measured front headroom 11.26 ms):
the naive fp32 port at 900×600 is **~160% of the whole 16.67 ms budget /
~236% of the front headroom — it does NOT fit the budget as-is.** Reported
honestly per charter (measure + report; spp/bounce/pixel levers untouched).

### Why it's slow, and the follow-up (NOT in VIII-2 scope)
3488 MAC/pixel × 540k pixels ≈ 3.8 GFLOP — ideal ~1.5 ms on M1; the 26.5 ms
is ~18× off peak. Cause: per-invocation `array<f32,64>` activation scratch +
data-driven (non-unrolled) layer loops → register pressure, no vectorization,
each thread streaming all 14 KB of weights from device storage. Curiosity
noted honestly: shrinking the scratch to a tight `array<f32,32>` measured
WORSE (52.8 ms) on this naga/Metal combo, so 64 is shipped.

The fp16-first / subgroup-×32 / weights-in-threadgroup-memory optimization
(RENDER.md §8) is the path to budget — but the beats-noisy margin on
orbit_-20 is razor-thin (0.042999 vs 0.052073 at 96×64; ~0.002 headroom, and
fragile at higher res per the VIII-1 dream note), so fp16 CANNOT be adopted
without RE-DERIVING the bound and re-proving beats-noisy in fp16. Deferred to
a VIII-2 optimization follow-up; not attempted here to avoid loosening the
gate. This is the honest open gap of this wave.

## Proof
`proof/viii2-gpu-front.png`, `proof/viii2-gpu-heldout.png` — each = noisy |
GPU-denoised (96×64, exposure 1.6). Read by the conductor's own eyes: right
panels visibly smoother than left, structure (towers, sky gradient, emissive
glows) preserved.
