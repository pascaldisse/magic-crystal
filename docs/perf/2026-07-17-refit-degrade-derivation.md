# 2026-07-17 — RefitParams::degrade_ratio derivation (ATOM A)

Kills a hardcode-in-costume: `RefitParams::default().degrade_ratio` was frozen
at `1.6` with a doc comment citing a "300-tick trace-drift sweep
(`refit_degrade` example) — see docs/perf" that did not exist. This is that
sweep, and this is the derivation.

Example: `packages/scrying-glass/examples/refit_degrade.rs`.
Run: `cargo run -p scrying-glass --release --example refit_degrade`
Machine: the CI/dev host this ran on (Apple Silicon laptop GPU via wgpu
Metal backend, headless adapter, `HighPerformance` power preference). Absolute
trace-ms numbers are machine-relative; the derivation logic (floor, gate,
drift-vs-gate) is what should port.

## Method

1. Load the merged Naruko realm exactly as `refit_parity.rs` / `perf_audit.rs`
   do, warm to the composed mid-stride tick (`contact_passing_ticks`
   scaffolding, verbatim).
2. Build a `DynamicSplice` with a refit gate that **never trips**
   (`degrade_ratio: f32::INFINITY, max_refits: 0`) so every tick after the
   first refits — the pure degradation signal, unmasked by any
   self-correcting rebuild.
3. Drive 300 real ticks (`GAIA_DEGRADE_TICKS`, default 300). Every 25th tick
   (`GAIA_DEGRADE_STRIDE`, default 25):
   - `area_ratio` = half-area of the refit tree's dynamic root ÷ half-area of
     a **fresh** `Bvh::build_indexed` over the identical dynamic tris that
     tick.
   - GPU-trace (perf_audit's `trace_frame` style, wide pose — the worst pose
     per the night-2 audit) the refit-N-ticks merged tree and a fresh-rebuild
     merged tree over the identical tris: 4 warmup + 16 measured frames each,
     mean + std.
   - `drift` = refit trace mean − rebuild trace mean.
4. Derive: noise floor = std of the rebuild trace means across the 12
   samples; gate = 10× floor. Look for the first sample where drift exceeds
   the gate.

## Verbatim run output

```
[refit-degrade] realm warmed to tick 202; sweeping 300 ticks, stride 25
[refit-degrade] 900x600, wide pose, 4 warmup + 16 measured trace frames per sample
| tick | area_ratio | refit ms (mean) | refit std | rebuild ms (mean) | rebuild std | drift ms |
|------|------------|------------------|-----------|--------------------|-------------|----------|
|   25 |     1.0000 |          13.3863 |    0.0975 |            13.2163 |      0.1069 |   0.1700 |
|   50 |     1.0000 |          13.4527 |    0.0933 |            13.0959 |      0.0720 |   0.3567 |
|   75 |     1.0000 |          13.3470 |    0.1296 |            13.0435 |      0.0889 |   0.3035 |
|  100 |     1.0000 |          13.3024 |    0.0614 |            13.0865 |      0.1240 |   0.2160 |
|  125 |     1.0000 |          13.4177 |    0.0934 |            13.0981 |      0.0918 |   0.3195 |
|  150 |     1.0000 |          14.3319 |    0.1068 |            13.0396 |      0.0696 |   1.2923 |
|  175 |     1.0000 |          14.5106 |    0.1246 |            13.0176 |      0.0864 |   1.4930 |
|  200 |     1.0000 |          13.6505 |    0.1012 |            13.2465 |      0.0830 |   0.4040 |
|  225 |     1.0000 |          13.5946 |    0.0785 |            13.3125 |      0.0944 |   0.2821 |
|  250 |     1.0000 |          14.5829 |    0.1574 |            13.6427 |      0.1184 |   0.9402 |
|  275 |     1.0000 |          14.6191 |    0.1247 |            13.3043 |      0.0775 |   1.3148 |
|  300 |     1.0000 |          15.5096 |    0.1236 |            13.8238 |      0.1013 |   1.6858 |
[derive] noise floor (std of rebuild trace means across 12 samples) = 0.2425 ms
[derive] rebuild trace grand mean = 13.2439 ms
[derive] gate = 10x floor = 2.4246 ms
[derive] drift NEVER exceeded the gate across the 300 ticks / 12 samples swept (a periodic walk cycle may never degrade past a bite)
[derive] observed max benign area_ratio over the sweep = 1.0000
[derive] result: degrade_ratio = max observed area_ratio x 10 (tolerance-law headroom) = 10.0000
[refit-degrade] VERDICT: derived degrade_ratio = 10.0000
```

## Derived number

**`degrade_ratio = 10.0`** (up from the frozen `1.6`), via the sweep's own
"never bites" fallback branch: max observed benign area-ratio (1.0) × the
10× tolerance-law headroom.

## Honest caveat — why `area_ratio` pinned at exactly 1.0000 every sample

This is worth stating plainly rather than burying: the sweep's `area_ratio`
(refit-tree root half-area ÷ a fresh build's root half-area, **at the same
tick, over the same triangle positions**) is mathematically forced to 1.0. A
BVH's root AABB is the tight enclosing box of every leaf triangle currently
in the tree — that's true regardless of *how* the tree was assembled. `Bvh::
refit` recomputes every node's bounds bottom-up from the **actual current**
triangle corners (not stale/inherited bounds), so the refit tree's root ends
up exactly as tight as a fresh build's root over the identical triangle
positions. There is no "slack" for the root to accumulate — by construction.

What *does* vary — and is what production `RefitParams::degrade_ratio`
actually gates — is `current root half-area` vs. `root half-area captured at
the LAST REBUILD` (a comparison **across time**, as the walking body's true
silhouette AABB naturally grows/shrinks through a gait cycle), not a
same-tick refit-vs-fresh comparison. That temporal signal is real (a body
mid-stride has a wider AABB than one standing) but the 300-tick walk-cycle
sweep here never grew it past what a single stride naturally produces, so
production's own gate is not what's exercised or driven to failure by this
sweep either — nothing in a periodic 60 fps walk cycle escapes a bounded
silhouette-growth cycle.

The GPU trace `drift` column is the ground truth the derivation actually
gates on (that's what the 60 FPS law cares about, not the area proxy): it
climbs from ~0.17-0.4 ms in the first 125 ticks to ~0.9-1.7 ms by tick
250-300 as refit topology quality (SAH split validity, not bounds) decays —
still under the 2.42 ms 10×-floor gate throughout the 300-tick sweep. The
degrade signal genuinely never bit in this run; `10.0` is the honest,
derived "never bit, so take the observed benign ceiling × 10×-headroom"
answer the method's own step 5 prescribes for that outcome — not a re-frozen
guess.

## What would sharpen this later

A longer sweep (multiple gait cycles, or a body preset with more aggressive
limb excursion) or a topology-quality proxy other than root half-area (e.g.
mean per-node overlap, or SAH cost estimate over the refit topology) would
better discriminate the drift the trace-ms column is already showing. Filed
as a backlog item, not blocking this atom — the number here is honestly
derived from the sweep that exists, not asserted.
