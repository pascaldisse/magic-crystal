# TEACHER/BENCHMARK LAB CHAIN — budget measurement + historical verdict

Status 07-18 → DE-CHARTERED by two-act law (§ NEURAL.md). Chain = lab teacher/
benchmark only → explicit `/scry?lab=teacher-benchmark`; never present/default.
Measurements below = historical evidence, not live-path charter.

Lane `one-render-path` historical goal →
`trace(640×480) → NEURAL DENOISE → NEURAL UPSCALE → present`, bilinear gone.

Measuring instruments (real GPU, M1, timestamp-bracketed compute passes,
median of 32): `examples/onepath_budget.rs`, `examples/onepath_fp16_verdict.rs`.
Host load noted per run (shared machine); numbers vary ±15% with load.

## STAGE 1 — upscaler → WGSL compute — DONE, gated (green)

`src/upscaler.wgsl` + `src/upscaler_gpu.rs`, house pattern (== the VIII-2
denoiser port). `tests/viii3b_ordeals.rs` on real M1 GPU:
- (a) GPU determinism byte-identical.
- (b) GPU-vs-CPU parity **7.47e-7 / 6.93e-7** vs DERIVED bound **1.657e-3**
  (`(macs 13824 + 19·4 transcendental ULP)·u`, u = f32::EPSILON). ~2200× under.
- (c) beats-bilinear on both held-out orbits: orbit_-20 0.054105→**0.029502**,
  orbit_+40 0.042729→**0.033101** (matches CPU provenance neural rmse).
- (d) BAN scan on `upscaler.wgsl` + `upscaler_gpu.rs`.

## STAGE 2 — fp16 DENOISER verdict — DONE (bound re-derivation + BAN re-proof)

`examples/onepath_fp16_verdict.rs` — fp16 simulated in the CPU reference (a
test oracle: the SAME rounding arithmetic a GPU f16 shader performs; answers
the numerical question, never a runtime path). Two modes, DERIVED bounds:
- **MODE A** (f16 storage/read, **f32 accumulate**): rel bound ≈ **1.392e-3**
  (`2·u16 + macs·u32`; the accumulator does not compound in f32). This is the
  single-round term; the more rigorous per-LAYER bound is `L·2u16` (each
  layer re-rounds its activation to f16 before the next dot product, L=5
  layers here) ≈ 4.9e-3 — still the same order, and the stated 1.392e-3 is
  conservative-in-direction (understates by ~3.5×) but not wrong-order; the
  measured 6.49e-4/6.72e-4 parity sits comfortably under both.
- **MODE B** (full f16 accumulate): ADVISORY CORRECTION — the Higham
  compounding term is `n·u16` where `n` is the length of one dot-product
  CHAIN (the layer's `in_dim`, ≤64 for this net: 10/32/32/32/32), not the
  total MACs (3488) summed across the whole network — layers don't share an
  accumulator. Honest worst case ≈ **0.03–0.12 rel** (per-layer `n·u16`
  compounded across L=5 re-roundings), not the 1.703e0 previously quoted —
  do not cite 1.703 as a tight bound. The REJECTION VERDICT stands: even the
  corrected 0.03–0.12 range is 20-100× the MODE A margin and can still eat
  the razor-thin 0.009 margin at untested poses; MODE B remains unsafe to
  adopt, on the corrected number, not the inflated one.

MEASURED on the two TRUE held-out orbits (96×64, the pinned-margin res):

| pose      | noisy    | fp32 den. | MODE A fp16 | margin (A) | beats? | MODE B | beats? |
|-----------|----------|-----------|-------------|------------|--------|--------|--------|
| orbit_-20 | 0.052093 | 0.043001  | 0.043029    | 0.009064   | YES    | 0.043016 | YES  |
| orbit_+40 | 0.095662 | 0.049997  | 0.050017    | 0.045644   | YES    | 0.050020 | YES  |

MODE A parity vs fp32: 6.49e-4 / 6.72e-4 — within the derived 1.392e-3 bound.

**fp16 VERDICT: MODE A is VIABLE.** The razor-thin 0.009 margin SURVIVES
(0.009064 vs fp32's 0.009092 on orbit_-20 — loses 0.3% of margin), with a
soundly-derived bound. **MODE B is REJECTED by derivation** — it happens to
pass on these two poses but its worst-case bound (≈0.03–0.12 relative, per
the corrected per-dot-chain derivation above — not the 1.703 figure
previously quoted) means the margin can vanish at untested poses/
resolutions; not safe to adopt. The sound fp16 lever is f16 storage + f32
accumulate.

## STAGE 2 — BUDGET phase table — the honest wall

`examples/onepath_budget.rs`, production shapes (trace/denoise 640×480 →
upscale ×2 → 1280×960), median of 32:

| phase                         | median ms | min ms  |
|-------------------------------|-----------|---------|
| denoise 640×480 (fp32 naive)  | **15.6**  | 15.4    |
| upscale 1280×960 (fp32 naive) | **371**   | 326     |
| upscale 1280×960 (f16-tg fast)| **2843**  | 2661    |

**The upscaler, not the denoiser, is the wall-breaker.** 1.23M target pixels ×
13824 MAC/pixel = 17 GFLOP/frame; the naive per-pixel port streams 55 KB of
fp32 weights per thread from device storage → ~64× redundant traffic → 371 ms,
57× off the M1 fp32 peak (~6.5 ms ideal for 17 GFLOP).

### Rejected lever: full-net f16 threadgroup cache (negative result, KEPT)
`src/upscaler_fast.wgsl` (`GpuUpscaler::new_fast`) loads the whole net once per
workgroup into `var<workgroup> array<f16, 16000>` (27.6 KB, fits the 32 KB
limit) — CORRECT (parity 5.47e-4 vs CPU) but **2843 ms, 7.6× SLOWER**: 27.6 KB
threadgroup memory per workgroup collapses occupancy to ~1 resident workgroup
per core, serializing the machine. Caching the whole net defeats itself. Kept
in-tree as a measured negative result, NOT a runtime path.

## Combined budget verdict — DOES NOT FIT (honest, per charter HONESTY clause)

Best true numbers: denoise 15.6 ms + upscale (naive fp32) 371 ms ≈ **387 ms**
= **~2.6 fps** at 1280×960. **26× over the 16.67 ms / 60-fps wall.** fp16 MODE A
on the denoiser (viable, above) trims only the 15.6 ms stage; it cannot touch
the 371 ms upscaler wall.

The Architect rules pixels (target resolution) and net size, never this lane.
The remaining exact levers all require a ruling I do not own:
1. Smaller / shallower / separable upscaler net (retrain — quality ruling).
2. Lower upscale target resolution or scale (pixel ruling).
3. simdgroup_matrix / subgroup-tiled weights that preserve occupancy (a real
   near-peak MLP kernel campaign — the theoretical 6.5 ms floor exists, but is
   a multi-day optimization, not landed here).

## STAGE 3 — wire as sole path — NOT DONE, deliberately (would break the law)

Wiring the 387 ms path as THE runtime path would run the live window at ~2.6
fps — a direct violation of the 60-FPS LAW that currently passes (b34d10c).
Per the charter's own HONESTY clause ("if the combined path cannot fit 16.67
with all exact levers, land the best true number + phase table and say so
plainly"), the runtime bilinear resolve is LEFT IN PLACE and the sole-path cut
is BLOCKED on a budget fix that needs the Architect's pixel/net ruling. The
neural path is proven CORRECT (Stage 1 ordeals green) but not yet budget-viable
at production resolution. Nothing merged.

## FINAL SWEEP — 2026-07-17 21:3x, post `git merge main` (c7189a5), release build

Merge clean (10 files, rimguard/floor-fallthrough from main). Build green in
21s. Host load HIGH at measure (`uptime` load ~15.9) — GPU numbers run ~±15%
hot vs the Stage-2 table; the VERDICT is unchanged (walls hold by ~160×).

ORDEALS (real M1 GPU, all green):
- denoiser (shipping path) `viii2_ordeals` 5/5 — byte-identical same-frame,
  parity+beats-noisy within derived bound, BAN (no temporal vocab + every
  `pub fn` current-frame-only), hash-pin weights == committed artifact.
- upscaler / neural resolve `viii3b_ordeals` 4/4 — GPU determinism
  byte-identical, GPU-vs-CPU parity + beats-bilinear on both held-out orbits,
  BAN, full neural path deterministic end-to-end.
- hash-identity BOTH path selections — `live_loop_hash_identity` run under
  `GAIA_NATIVE_UPSCALE=bilinear` and `=neural`: 24/24 frames bit-identical
  serial-vs-overlap AND the per-frame hashes are IDENTICAL across the two
  selections (frame0 336ef6cab3e95ac1, frame3 28a2ed7cf62257e4). ADVISORY
  REWORD: this run hashes the surface frame, which is produced UPSTREAM of
  the resolve selector in the frame loop — the hash match demonstrates the
  two env-var runs produce identical surface bytes, it does not itself
  exercise or observe the neural resolve path executing. The stronger claim
  — that the live surface is INVARIANT to the resolve selection because the
  selector is structurally unreachable from the frame loop (neural only
  ever writes /scry A/B, wired nowhere near `run_render_loop`) — is a
  STRUCTURAL property of the code path (see wiring, not this run's output),
  not something this hash-identity run demonstrates by itself. Full
  workspace suite 400/0.

### PHASE TABLE — per path (live-loop reality)

**BILINEAR path = the live surface loop** (`live_loop_audit`, 640×480 int,
spp 2, 960×640 surface), identical under both selections:

| phase                          | SERIAL median ms |
|--------------------------------|------------------|
| skin                           | 0.522            |
| tick (physics+kami)            | 3.542            |
| splice (refit/merge)           | 0.277            |
| upload                         | 0.487            |
| trace (2 spp, no medium)       | 8.880            |
| blit (2× present)              | 0.495            |
| **SERIAL TOTAL**               | **14.202 (70.4 fps) PASS** |
| **OVERLAP wall (pipelined)**   | **8.016 (124.7 fps) PASS** |

**NEURAL path = the /scry A/B capture** (`onepath_budget`, trace/denoise
640×480 → upscale ×2 → 1280×960, median of 32, hot host):

| phase                          | median ms | min ms |
|--------------------------------|-----------|--------|
| denoise 640×480 (fp32)         | 16.198    | 15.769 |
| upscale 1280×960 (naive fp32)  | 334.901   | 324.649|
| upscale 1280×960 (FAST f16-tg) | 2687.696  | 2654.999 (parity 5.47e-4) |
| **combined (best true)**       | **~351 ms (~2.85 fps)** | — |

VERDICT UNCHANGED: neural exceeds the 16.67 ms wall by ~21× (best true
naive-fp32 combined). The upscaler remains the wall-breaker; fp16 MODE A is
the only sound denoiser lever and cannot touch it. Sole-path cut stays BLOCKED
on the Architect's pixel/net ruling; runtime bilinear resolve left in place;
chain remains a proven-correct explicit teacher/benchmark lab surface; default
`/scry` never selects it. Nothing merged.
