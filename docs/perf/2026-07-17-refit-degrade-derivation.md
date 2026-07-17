# 2026-07-17 — RefitParams::degrade_ratio derivation (ATOM A, revision 2)

## REVISION 2 (adversary MUST-FIX, this pass)

An adversary review of revision 1 (the `total_node_half_area` pass, below)
found the derivation measured the WRONG ratio: `DynamicSplice::update`'s gate
actually tests

```
current total-node-half-area SUM  >  rebuild_reference_area × degrade_ratio
```

where `rebuild_reference_area` is the sum captured AT THE LAST REBUILD (a
comparison ACROSS TIME — `bvh.rs`'s `rebuild_area` field). Revision 1's sweep
instead computed `area_ratio = refit_current / fresh_build_THIS_TICK` — a
SAME-TICK comparison that measures topology staleness only, not growth since
the last rebuild. These are different physical signals: the same-tick ratio
stayed pinned near 1.0 (0.99–1.01) for the full 1200-tick sweep, while the
actually-gated ratio (computed below) ranges 0.91–1.07 over the same run,
because it also carries the body's silhouette oscillating relative to
whatever pose the last rebuild happened to freeze as the reference. Revision
1's derived `10.0964` was rigorous arithmetic on the wrong number, and
because the sweep runs with `degrade_ratio: f32::INFINITY` the gated ratio
was never even computed anywhere in that pass.

This revision:
- exposes `DynamicSplice::rebuild_reference_area()` (read-only) so measurement
  code can compute the EXACT ratio the gate divides by,
- reports THREE numbers per sample: `stale` (topology decay: refit/fresh),
  `pose` (silhouette oscillation: fresh/rebuildRef), `gated` = `stale × pose`
  = the actual gate quantity,
- re-derives `degrade_ratio` from an EXCURSION form: `1 + K × max(0, max
  benign gated_ratio − 1.0)`, `K = 10` — a headroom on the observed benign
  band above 1.0, not a flat multiplier (so a ratio hovering near 1.0 doesn't
  get inflated by 10× for no reason). This `K = 10` is a SEPARATE constant
  from the drift-gate's own `10× noise floor` below — the two aren't the same
  10× and are named distinctly in the sweep's output,
- adds two discriminating tests for the DEFAULT params (not an inline
  fixture): one proving the default gate HOLDS across a bounded periodic
  oscillation, one proving it FIRES on a genuine blowup,
- makes `DEFAULT_DEGRADE_RATIO` `pub` so `main.rs` references it instead of
  re-typing the literal,
- argues the `max_refits: 0` (unlimited) call explicitly (see below) instead
  of leaving it unexamined,
- notes the `perf_audit` OVERLAP margin for the `wide` pose honestly (it is
  the thinner of the two poses).

Sections below marked "(revision 1)" are the superseded first pass, kept for
the historical record of the mistake rather than deleted.

Kills a hardcode-in-costume: `RefitParams::default().degrade_ratio` was frozen
at `1.6` with a doc comment citing a "300-tick trace-drift sweep
(`refit_degrade` example) — see docs/perf" that did not exist. This is that
sweep, and this is the derivation.

**Revision (this pass):** the first version of this derivation used the
dynamic-tree's ROOT half-area as the degradation signal and got
`degrade_ratio = 10.0` — but that signal is structurally dead (see "Why
root-half-area was replaced" below). This revision replaces it with the
TOTAL half-area summed over every node of the dynamic tree
(`Bvh::total_node_half_area`), re-runs the sweep, and re-derives.

Example: `packages/scrying-glass/examples/refit_degrade.rs`.
Run: `cargo run -p scrying-glass --release --example refit_degrade`
Machine: the CI/dev host this ran on (Apple Silicon laptop GPU via wgpu
Metal backend, headless adapter, `HighPerformance` power preference). Absolute
trace-ms numbers are machine-relative; the derivation logic (floor, gate,
drift-vs-gate) is what should port.

## Why root-half-area was replaced (revision 1 context — still correct, kept)

`Bvh::refit` recomputes EVERY node's AABB bottom-up from the triangles'
ACTUAL current corners (never inherited/stale bounds). A BVH's root AABB is,
by definition, the tight union of every leaf triangle currently in the tree
— true regardless of how the tree was assembled. So a refit tree's root ends
up exactly as tight as a fresh build's root over the identical positions,
with no possible slack: `root_half_area(refit tree) / root_half_area(fresh
build over same tris)` is mathematically forced to `1.0000`, every sample,
by construction. The original sweep run proved this empirically (all 12
`area_ratio` samples read exactly `1.0000`), which meant the watchdog in
`DynamicSplice::update` — which compared this pinned ratio against
`degrade_ratio` — could never trip regardless of how badly the tree's
INTERIOR topology (sibling bounds, overlap) had decayed after many refits.
Combined with `max_refits: 0` (unlimited refits), this let real GPU trace
drift leak unbounded in long play.

The replacement, `Bvh::total_node_half_area`, sums half-area over EVERY node
(root + every internal + every leaf), not just the root. Sibling bounds can
loosen and start overlapping after many refits even while the root (their
union) stays tight — that overlap is exactly what inflates GPU traversal
cost (more boxes a ray must test to find the same triangle), so the sum
tracks it. `Bvh::refit` already walks every node bottom-up to recompute
bounds; it now returns the sum from that same pass, so `DynamicSplice`
doesn't pay a second full scan per tick.

## Method (revision 1 — superseded ratio, kept for history)

1. Load the merged Naruko realm exactly as `refit_parity.rs` / `perf_audit.rs`
   do, warm to the composed mid-stride tick (`contact_passing_ticks`
   scaffolding, verbatim).
2. Build a `DynamicSplice` with a refit gate that **never trips**
   (`degrade_ratio: f32::INFINITY, max_refits: 0`) so every tick after the
   first refits — the pure degradation signal, unmasked by any
   self-correcting rebuild.
3. Drive N real ticks (`GAIA_DEGRADE_TICKS`, default 300; also run extended to
   1200 — see below). Every 25th tick (`GAIA_DEGRADE_STRIDE`, default 25):
   - `area_ratio` = `DynamicSplice::dyn_total_half_area()` (the refit tree's
     current total-node-half-area sum) ÷ `Bvh::total_node_half_area()` of a
     **fresh** `Bvh::build_indexed` over the identical dynamic tris that tick.
   - GPU-trace (perf_audit's `trace_frame` style, wide pose — the worst pose
     per the night-2 audit) the refit-N-ticks merged tree and a fresh-rebuild
     merged tree over the identical tris: 4 warmup + 16 measured frames each,
     mean + std.
   - `drift` = refit trace mean − rebuild trace mean.
4. Derive: noise floor = std of the rebuild trace means across the samples;
   gate = 10× floor. Look for the first sample where drift exceeds the gate.

## Verbatim run output — 300 ticks (revision 1, SUPERSEDED — wrong ratio, see revision 2 below)

```
[refit-degrade] realm warmed to tick 202; sweeping 300 ticks, stride 25
[refit-degrade] 900x600, wide pose, 4 warmup + 16 measured trace frames per sample
| tick | area_ratio | refit ms (mean) | refit std | rebuild ms (mean) | rebuild std | drift ms |
|------|------------|------------------|-----------|--------------------|-------------|----------|
|   25 |     1.0020 |          13.3638 |    0.1187 |            13.3145 |      0.1808 |   0.0494 |
|   50 |     1.0043 |          13.4359 |    0.1089 |            13.1141 |      0.0963 |   0.3218 |
|   75 |     1.0056 |          13.3470 |    0.1303 |            13.0656 |      0.1090 |   0.2814 |
|  100 |     1.0023 |          13.2736 |    0.0592 |            13.0710 |      0.1164 |   0.2026 |
|  125 |     1.0002 |          13.4083 |    0.0870 |            13.0955 |      0.1180 |   0.3127 |
|  150 |     1.0060 |          14.2335 |    0.0715 |            13.0767 |      0.0983 |   1.1568 |
|  175 |     1.0050 |          14.4641 |    0.0476 |            13.0519 |      0.1378 |   1.4122 |
|  200 |     1.0049 |          13.6420 |    0.1299 |            13.2216 |      0.1076 |   0.4205 |
|  225 |     1.0009 |          13.6173 |    0.0925 |            13.3856 |      0.1745 |   0.2317 |
|  250 |     1.0009 |          14.5425 |    0.1417 |            13.5878 |      0.0839 |   0.9547 |
|  275 |     0.9994 |          14.5838 |    0.0991 |            13.3473 |      0.0965 |   1.2365 |
|  300 |     0.9972 |          15.5711 |    0.1254 |            14.7882 |      0.6950 |   0.7829 |
[derive] noise floor (std of rebuild trace means across 12 samples) = 0.4642 ms
[derive] rebuild trace grand mean = 13.3433 ms
[derive] gate = 10x floor = 4.6424 ms
[derive] drift NEVER exceeded the gate across the 300 ticks / 12 samples swept (a periodic walk cycle may never degrade past a bite)
[derive] observed max benign area_ratio over the sweep = 1.0060
[derive] result: degrade_ratio = max observed area_ratio x 10 (tolerance-law headroom) = 10.0601
[refit-degrade] VERDICT: derived degrade_ratio = 10.0601
```

Note `area_ratio` now genuinely MOVES tick to tick (0.9972-1.0060), unlike
the old root-only signal, which read exactly `1.0000` every sample.

## Extended run — `GAIA_DEGRADE_TICKS=1200` (revision 1, SUPERSEDED — wrong ratio, see revision 2 below)

Run once to check whether the signal saturates or keeps growing across more
than one gait cycle:

```
[refit-degrade] realm warmed to tick 202; sweeping 1200 ticks, stride 25
[refit-degrade] 900x600, wide pose, 4 warmup + 16 measured trace frames per sample
| tick | area_ratio | refit ms (mean) | refit std | rebuild ms (mean) | rebuild std | drift ms |
|------|------------|------------------|-----------|--------------------|-------------|----------|
|   25 |     1.0020 |          13.3463 |    0.0825 |            13.1807 |      0.0998 |   0.1656 |
|   50 |     1.0043 |          13.4123 |    0.1184 |            13.1261 |      0.0883 |   0.2862 |
|   75 |     1.0056 |          13.3195 |    0.0999 |            12.9915 |      0.0786 |   0.3279 |
|  100 |     1.0023 |          13.2958 |    0.0991 |            13.0972 |      0.1009 |   0.1986 |
|  125 |     1.0002 |          13.4903 |    0.1183 |            13.1127 |      0.1381 |   0.3776 |
|  150 |     1.0060 |          14.3264 |    0.1250 |            13.0276 |      0.1056 |   1.2987 |
|  175 |     1.0050 |          14.4742 |    0.0993 |            13.0296 |      0.1248 |   1.4446 |
|  200 |     1.0049 |          13.6051 |    0.1200 |            13.3015 |      0.1163 |   0.3037 |
|  225 |     1.0009 |          13.5023 |    0.0747 |            13.2991 |      0.1030 |   0.2033 |
|  250 |     1.0009 |          14.5319 |    0.1591 |            13.6349 |      0.1465 |   0.8970 |
|  275 |     0.9994 |          14.5792 |    0.1007 |            13.3705 |      0.1157 |   1.2088 |
|  300 |     0.9972 |          15.5837 |    0.1185 |            13.8054 |      0.1017 |   1.7782 |
|  325 |     1.0000 |          15.6538 |    0.1522 |            14.0427 |      0.1063 |   1.6111 |
|  350 |     1.0020 |          14.0870 |    0.0872 |            13.4839 |      0.1950 |   0.6031 |
|  375 |     1.0015 |          13.7145 |    0.0761 |            13.3274 |      0.0869 |   0.3871 |
|  400 |     0.9916 |          14.8612 |    0.1019 |            13.3172 |      0.0741 |   1.5440 |
|  425 |     0.9924 |          14.5278 |    0.0994 |            13.3623 |      0.1148 |   1.1655 |
|  450 |     0.9980 |          14.7097 |    0.1598 |            13.1900 |      0.1342 |   1.5197 |
|  475 |     1.0040 |          14.2437 |    0.1009 |            13.0924 |      0.0655 |   1.1513 |
|  500 |     1.0096 |          13.4291 |    0.1084 |            13.0405 |      0.1382 |   0.3886 |
|  525 |     1.0089 |          13.2396 |    0.0964 |            12.8451 |      0.0741 |   0.3945 |
|  550 |     1.0045 |          13.4891 |    0.1146 |            12.8993 |      0.0999 |   0.5898 |
|  575 |     1.0054 |          13.3502 |    0.1483 |            12.8665 |      0.0763 |   0.4837 |
|  600 |     1.0020 |          13.8529 |    0.0940 |            13.0089 |      0.1136 |   0.8440 |
|  625 |     1.0021 |          13.2996 |    0.1253 |            12.9887 |      0.0817 |   0.3109 |
|  650 |     1.0034 |          13.3755 |    0.1329 |            13.0862 |      0.1331 |   0.2893 |
|  675 |     0.9966 |          13.3138 |    0.0574 |            13.1101 |      0.1010 |   0.2036 |
|  700 |     0.9988 |          13.3081 |    0.1292 |            13.0488 |      0.1104 |   0.2593 |
|  725 |     0.9997 |          13.4160 |    0.1303 |            13.2578 |      0.1559 |   0.1581 |
|  750 |     0.9967 |          13.4015 |    0.0946 |            13.2924 |      0.1174 |   0.1090 |
|  775 |     0.9996 |          13.5275 |    0.0951 |            13.4419 |      0.0838 |   0.0856 |
|  800 |     1.0021 |          13.6210 |    0.1044 |            13.3411 |      0.0721 |   0.2798 |
|  825 |     1.0005 |          13.4929 |    0.0854 |            13.3668 |      0.1286 |   0.1261 |
|  850 |     0.9977 |          13.3935 |    0.1567 |            13.4387 |      0.1010 |  -0.0453 |
|  875 |     1.0005 |          13.3605 |    0.1078 |            13.3750 |      0.1323 |  -0.0145 |
|  900 |     1.0004 |          13.4008 |    0.1055 |            13.3710 |      0.2583 |   0.0298 |
|  925 |     1.0010 |          13.4180 |    0.1244 |            13.2084 |      0.1002 |   0.2096 |
|  950 |     1.0042 |          13.4364 |    0.0721 |            13.1020 |      0.0941 |   0.3343 |
|  975 |     1.0076 |          13.3435 |    0.0827 |            13.0049 |      0.0581 |   0.3387 |
| 1000 |     1.0016 |          13.1996 |    0.1053 |            12.9922 |      0.1106 |   0.2074 |
| 1025 |     1.0000 |          13.5761 |    0.1491 |            13.1763 |      0.1127 |   0.3998 |
| 1050 |     0.9979 |          14.1986 |    0.0863 |            13.2807 |      0.1473 |   0.9179 |
| 1075 |     1.0011 |          13.7075 |    0.1024 |            13.1553 |      0.1197 |   0.5522 |
| 1100 |     1.0053 |          13.6658 |    0.1188 |            25.1930 |      5.6720 | -11.5273 |
| 1125 |     1.0027 |          28.1379 |    3.6129 |            25.8426 |      4.7680 |   2.2953 |
| 1150 |     0.9974 |          26.8229 |    6.7085 |            26.9271 |      3.3364 |  -0.1042 |
| 1175 |     0.9992 |          24.9725 |    6.8369 |            26.8569 |      4.0592 |  -1.8844 |
| 1200 |     0.9927 |          28.0116 |    6.7389 |            27.7278 |      3.5355 |   0.2838 |
[derive] noise floor (std of rebuild trace means across 48 samples) = 4.0756 ms
[derive] rebuild trace grand mean = 14.6050 ms
[derive] gate = 10x floor = 40.7564 ms
[derive] drift NEVER exceeded the gate across the 1200 ticks / 48 samples swept (a periodic walk cycle may never degrade past a bite)
[derive] observed max benign area_ratio over the sweep = 1.0096
[derive] result: degrade_ratio = max observed area_ratio x 10 (tolerance-law headroom) = 10.0964
[refit-degrade] VERDICT: derived degrade_ratio = 10.0964
```

**Honest read of the extended run:** `area_ratio` stays bounded in a narrow
band (0.9916–1.0096) across the full 1200 ticks (4 gait cycles) — it does
NOT grow unboundedly, it oscillates with the walk cycle and saturates. That
supports the "never bites, take the ceiling" derivation rather than
suggesting a longer sweep would eventually cross the gate.

Past tick ~1075 BOTH the refit and rebuild trace means jump together (from
~13ms to ~25-28ms, with std also jumping 10-70x) and `drift` briefly goes
negative — this is host noise (thermal throttling / background GPU
contention on the dev laptop late in a long run), not signal: a real
refit-vs-rebuild divergence would move refit away from rebuild, not move
both arms together. This inflates the reported noise floor (4.08ms vs
0.46ms in the clean 300-tick run) and therefore the gate (40.8ms vs 4.6ms)
for the extended run — reported honestly rather than hidden, but the 300-tick
run's floor/gate pair is the cleaner measurement for anyone using this
methodology on quieter hardware.

## Derived number (revision 1 — SUPERSEDED, see "Derived number — revision 2" below)

**`degrade_ratio = 10.0964`**, from the 1200-tick extended sweep's own "never
bites" fallback branch: max observed benign area-ratio (1.0096) × the 10×
tolerance-law headroom. This supersedes the previous root-half-area-based
derivation of `10.0` (`degrade_ratio` barely moves numerically since both
signals happen to plateau near a benign ratio just above 1.0, but the new
number is honestly re-derived from the total-node-half-area signal, not
carried over).

The 300-tick default-length run alone would have derived `10.0601`; the
1200-tick extension found a slightly higher benign ceiling (1.0096 vs
1.0060) so `10.0964` is used as the more complete observation.

---

## Method — revision 2 (the corrected ratio)

Same scaffolding (warm to the composed mid-stride tick, `DynamicSplice` with
`degrade_ratio: f32::INFINITY, max_refits: 0` so `rebuild_reference_area`
stays frozen at the tick-202 warmup value for the whole sweep — the pure
degradation signal, both components, unmasked by any self-correcting
rebuild). Every `GAIA_DEGRADE_STRIDE`-th tick now computes THREE ratios via
`DynamicSplice::rebuild_reference_area()` (new, read-only getter):

- `stale` = `splice.dyn_total_half_area()` / `Bvh::total_node_half_area()` of
  a fresh build over the identical dyn tris that tick — topology decay only.
- `pose` = that same fresh-build sum / `splice.rebuild_reference_area()` —
  silhouette oscillation relative to the frozen reference, only.
- `gated` = `splice.dyn_total_half_area()` / `splice.rebuild_reference_area()`
  = `stale × pose` — the EXACT quantity `DynamicSplice::update`'s gate tests.

The gait is deterministic and periodic: `GaitParams::walk()` has
`cadence = 1.0` Hz at `dt = 1/60` s, so one gait cycle is exactly 60 ticks.
The 300-tick sweep covers 5 cycles; the 1200-tick extension covers 20 —
enough that a repeating, deterministic motion has nowhere new to go (any
future cycle revisits phases already swept).

Derivation (unchanged drift-gate method, now applied to the right ratio):
noise floor = std of rebuild trace means; `drift_gate` = 10× floor (a name
distinct from the derivation headroom below, so the two 10×s are never
conflated). If drift never crosses `drift_gate`: `degrade_ratio = 1 + K ×
max(0, max observed benign `gated` − 1.0)`, `K = 10`.

## Verbatim run output — 300 ticks, revision 2 (corrected ratio)

```
[refit-degrade] realm warmed to tick 202; sweeping 300 ticks, stride 25
[refit-degrade] 900x600, wide pose, 4 warmup + 16 measured trace frames per sample
| tick | stale (refit/fresh) | pose (fresh/rebuildRef) | gated (refit/rebuildRef) | refit ms (mean) | refit std | rebuild ms (mean) | rebuild std | drift ms |
|------|----------------------|--------------------------|----------------------------|------------------|-----------|--------------------|-------------|----------|
|   25 |               1.0020 |                   0.9741 |                     0.9761 |          13.2797 |    0.0642 |            13.1592 |      0.0667 |   0.1205 |
|   50 |               1.0043 |                   0.9506 |                     0.9547 |          13.3713 |    0.0534 |            13.0918 |      0.0522 |   0.2795 |
|   75 |               1.0056 |                   0.9304 |                     0.9355 |          13.2710 |    0.0432 |            13.0421 |      0.0566 |   0.2289 |
|  100 |               1.0023 |                   0.9200 |                     0.9220 |          13.2875 |    0.0478 |            13.0675 |      0.0410 |   0.2200 |
|  125 |               1.0002 |                   0.9172 |                     0.9174 |          13.4186 |    0.0541 |            13.0324 |      0.0380 |   0.3862 |
|  150 |               1.0060 |                   0.9154 |                     0.9209 |          14.2759 |    0.0549 |            13.0236 |      0.0597 |   1.2523 |
|  175 |               1.0050 |                   0.9298 |                     0.9345 |          14.4632 |    0.0635 |            13.0201 |      0.0761 |   1.4430 |
|  200 |               1.0049 |                   0.9488 |                     0.9535 |          13.5499 |    0.0510 |            13.2484 |      0.0779 |   0.3015 |
|  225 |               1.0009 |                   0.9730 |                     0.9739 |          13.5336 |    0.0454 |            13.2502 |      0.0311 |   0.2834 |
|  250 |               1.0009 |                   0.9950 |                     0.9959 |          14.5693 |    0.1712 |            13.6505 |      0.0669 |   0.9188 |
|  275 |               0.9994 |                   1.0210 |                     1.0205 |          14.5593 |    0.0407 |            13.3371 |      0.0536 |   1.2221 |
|  300 |               0.9972 |                   1.0457 |                     1.0427 |          15.4936 |    0.0731 |            13.7842 |      0.0749 |   1.7095 |
[derive] noise floor (std of rebuild trace means across 12 samples) = 0.2428 ms
[derive] rebuild trace grand mean = 13.2256 ms
[derive] drift_gate = 10x floor = 2.4279 ms
[derive] drift NEVER exceeded drift_gate across the 300 ticks / 12 samples swept (a periodic walk cycle may never degrade past a bite)
[derive] observed max benign gated_ratio over the sweep = 1.0427
[derive] excursion above 1.0 = 0.0427; headroom K = 10.0 (tolerance-law, distinct from the drift_gate's own 10x)
[derive] result: degrade_ratio = 1 + K x excursion = 1 + 10.0 x 0.0427 = 1.4271
[refit-degrade] VERDICT: derived degrade_ratio = 1.4271
```

`stale` stays essentially flat (0.997–1.006) the whole run — interior
topology decay alone contributes almost nothing here. `pose` is what moves
(0.9154–1.0457): the silhouette oscillates relative to the tick-202 reference
because tick 202 happened to be a wider-than-average mid-stride moment, so
`pose` dips below 1.0 for most of the sweep before climbing back past it by
tick 300. `gated = stale × pose` tracks `pose` almost exactly.

## Extended run — `GAIA_DEGRADE_TICKS=1200` (20 gait cycles), revision 2

```
[refit-degrade] realm warmed to tick 202; sweeping 1200 ticks, stride 25
[refit-degrade] 900x600, wide pose, 4 warmup + 16 measured trace frames per sample
| tick | stale (refit/fresh) | pose (fresh/rebuildRef) | gated (refit/rebuildRef) | refit ms (mean) | refit std | rebuild ms (mean) | rebuild std | drift ms |
|------|----------------------|--------------------------|----------------------------|------------------|-----------|--------------------|-------------|----------|
|   25 |               1.0020 |                   0.9741 |                     0.9761 |          13.3118 |    0.0396 |            13.2095 |      0.0591 |   0.1023 |
|   50 |               1.0043 |                   0.9506 |                     0.9547 |          13.4480 |    0.1485 |            13.1060 |      0.0801 |   0.3420 |
|   75 |               1.0056 |                   0.9304 |                     0.9355 |          13.3242 |    0.0434 |            13.0624 |      0.0457 |   0.2618 |
|  100 |               1.0023 |                   0.9200 |                     0.9220 |          13.2781 |    0.0454 |            13.0572 |      0.0797 |   0.2209 |
|  125 |               1.0002 |                   0.9172 |                     0.9174 |          13.4950 |    0.1760 |            13.0547 |      0.0564 |   0.4403 |
|  150 |               1.0060 |                   0.9154 |                     0.9209 |          14.2513 |    0.0599 |            13.0619 |      0.2901 |   1.1894 |
|  175 |               1.0050 |                   0.9298 |                     0.9345 |          14.5479 |    0.1664 |            13.0252 |      0.0407 |   1.5228 |
|  200 |               1.0049 |                   0.9488 |                     0.9535 |          15.4453 |    2.8111 |            13.3283 |      0.1124 |   2.1170 |
|  225 |               1.0009 |                   0.9730 |                     0.9739 |          13.6033 |    0.0554 |            13.2965 |      0.0317 |   0.3068 |
|  250 |               1.0009 |                   0.9950 |                     0.9959 |          14.7084 |    0.5513 |            13.6848 |      0.0810 |   1.0236 |
|  275 |               0.9994 |                   1.0210 |                     1.0205 |          14.6029 |    0.0476 |            13.3930 |      0.0711 |   1.2099 |
|  300 |               0.9972 |                   1.0457 |                     1.0427 |          15.5212 |    0.0569 |            13.8599 |      0.0571 |   1.6614 |
|  325 |               1.0000 |                   1.0608 |                     1.0608 |          15.6181 |    0.0632 |            14.0897 |      0.0584 |   1.5284 |
|  350 |               1.0020 |                   1.0660 |                     1.0681 |          14.1528 |    0.0418 |            13.4667 |      0.0482 |   0.6861 |
|  375 |               1.0015 |                   1.0601 |                     1.0617 |          13.7796 |    0.0477 |            13.3701 |      0.0429 |   0.4094 |
|  400 |               0.9916 |                   1.0539 |                     1.0451 |          14.8667 |    0.0722 |            13.3403 |      0.0724 |   1.5263 |
|  425 |               0.9924 |                   1.0315 |                     1.0236 |          14.5778 |    0.1934 |            13.1912 |      0.0577 |   1.3866 |
|  450 |               0.9980 |                   1.0013 |                     0.9992 |          14.5451 |    0.1561 |            12.9825 |      0.0335 |   1.5627 |
|  475 |               1.0040 |                   0.9725 |                     0.9765 |          14.1850 |    0.0721 |            13.0712 |      0.1130 |   1.1138 |
|  500 |               1.0096 |                   0.9463 |                     0.9554 |          13.2956 |    0.0779 |            12.9416 |      0.0790 |   0.3540 |
|  525 |               1.0089 |                   0.9286 |                     0.9368 |          13.1512 |    0.0895 |            12.8087 |      0.0416 |   0.3424 |
|  550 |               1.0045 |                   0.9231 |                     0.9273 |          13.4934 |    0.0901 |            12.7486 |      0.0543 |   0.7448 |
|  575 |               1.0054 |                   0.9125 |                     0.9174 |          13.2624 |    0.0805 |            12.7880 |      0.0522 |   0.4744 |
|  600 |               1.0020 |                   0.9170 |                     0.9188 |          13.7905 |    0.0642 |            12.8478 |      0.0628 |   0.9428 |
|  625 |               1.0021 |                   0.9287 |                     0.9307 |          13.1677 |    0.0469 |            12.9054 |      0.0654 |   0.2623 |
|  650 |               1.0034 |                   0.9465 |                     0.9497 |          13.2951 |    0.0916 |            12.9859 |      0.0872 |   0.3092 |
|  675 |               0.9966 |                   0.9751 |                     0.9718 |          13.2000 |    0.0689 |            12.9744 |      0.0537 |   0.2256 |
|  700 |               0.9988 |                   0.9970 |                     0.9958 |          13.2210 |    0.1365 |            12.9943 |      0.0764 |   0.2267 |
|  725 |               0.9997 |                   1.0219 |                     1.0216 |          13.4257 |    0.0478 |            13.2929 |      0.0452 |   0.1328 |
|  750 |               0.9967 |                   1.0476 |                     1.0441 |          13.4838 |    0.2242 |            13.2826 |      0.0895 |   0.2011 |
|  775 |               0.9996 |                   1.0628 |                     1.0624 |          13.5626 |    0.0782 |            13.2628 |      0.0622 |   0.2999 |
|  800 |               1.0021 |                   1.0681 |                     1.0703 |          13.5713 |    0.0571 |            13.2763 |      0.0539 |   0.2950 |
|  825 |               1.0005 |                   1.0638 |                     1.0643 |          13.4092 |    0.0698 |            13.3557 |      0.0603 |   0.0535 |
|  850 |               0.9977 |                   1.0498 |                     1.0474 |          13.2803 |    0.0630 |            13.3613 |      0.0721 |  -0.0809 |
|  875 |               1.0005 |                   1.0248 |                     1.0253 |          13.4391 |    0.4955 |            13.2702 |      0.0908 |   0.1689 |
|  900 |               1.0004 |                   0.9988 |                     0.9992 |          13.2858 |    0.0806 |            13.1262 |      0.0520 |   0.1595 |
|  925 |               1.0010 |                   0.9726 |                     0.9735 |          13.3637 |    0.0805 |            13.1213 |      0.0853 |   0.2424 |
|  950 |               1.0042 |                   0.9449 |                     0.9489 |          13.3824 |    0.0519 |            13.0438 |      0.0706 |   0.3387 |
|  975 |               1.0076 |                   0.9201 |                     0.9271 |          13.2361 |    0.0772 |            12.9814 |      0.0652 |   0.2547 |
| 1000 |               1.0016 |                   0.9114 |                     0.9129 |          13.1320 |    0.0588 |            12.9220 |      0.0757 |   0.2100 |
| 1025 |               1.0000 |                   0.9102 |                     0.9102 |          13.4439 |    0.0539 |            13.1094 |      0.0642 |   0.3345 |
| 1050 |               0.9979 |                   0.9182 |                     0.9162 |          14.0790 |    0.0645 |            13.0531 |      0.0827 |   1.0259 |
| 1075 |               1.0011 |                   0.9315 |                     0.9325 |          13.5998 |    0.0455 |            13.0220 |      0.0621 |   0.5778 |
| 1100 |               1.0053 |                   0.9510 |                     0.9561 |          13.4857 |    0.0644 |            13.0787 |      0.0592 |   0.4070 |
| 1125 |               1.0027 |                   0.9803 |                     0.9829 |          13.4333 |    0.0500 |            13.1189 |      0.0533 |   0.3144 |
| 1150 |               0.9974 |                   1.0139 |                     1.0113 |          14.6069 |    0.1113 |            13.2006 |      0.0433 |   1.4063 |
| 1175 |               0.9992 |                   1.0401 |                     1.0393 |          14.3824 |    0.0676 |            13.4658 |      0.0703 |   0.9166 |
| 1200 |               0.9927 |                   1.0685 |                     1.0607 |          15.4987 |    0.0576 |            13.7369 |      0.0765 |   1.7618 |
[derive] noise floor (std of rebuild trace means across 48 samples) = 0.2676 ms
[derive] rebuild trace grand mean = 13.1818 ms
[derive] drift_gate = 10x floor = 2.6765 ms
[derive] drift NEVER exceeded drift_gate across the 1200 ticks / 48 samples swept (a periodic walk cycle may never degrade past a bite)
[derive] observed max benign gated_ratio over the sweep = 1.0703
[derive] excursion above 1.0 = 0.0703; headroom K = 10.0 (tolerance-law, distinct from the drift_gate's own 10x)
[derive] result: degrade_ratio = 1 + K x excursion = 1 + 10.0 x 0.0703 = 1.7030
[refit-degrade] VERDICT: derived degrade_ratio = 1.7030
```

`pose` (and therefore `gated`) is periodic and bounded across the full 20
cycles — it oscillates in the same ~0.91–1.07 band the 300-tick run already
found, with no monotonic drift across additional cycles. This run was clean
throughout (no thermal-throttling jump like revision 1's extended run
showed past tick 1075 — the noise floor here, 0.2676ms, is close to the
300-tick run's 0.2428ms, both clean).

## Derived number — revision 2

**`degrade_ratio = 1.7030`**, from the 1200-tick extended sweep (20 gait
cycles): `1 + 10 × (1.0703 − 1.0) = 1.7030`. The 300-tick run alone would
give `1.4271`; the 1200-tick extension is used as the more complete
observation (20 cycles vs 5, same bounded/periodic conclusion, slightly
higher observed peak).

This SUPERSEDES both `10.0` (root-half-area, structurally dead) and `10.0964`
(revision 1, total-node-half-area but the wrong same-tick ratio). The new
number is over 5× TIGHTER than the previous wrong derivation, because the
corrected ratio actually carries the real signal (pose oscillation up to
±7–8%) that the same-tick ratio was blind to.

## Why `max_refits: 0` (unlimited) stays

The gait is deterministic and exactly periodic (60 ticks/cycle at the walk
preset's cadence). The 1200-tick sweep (20 cycles) found:
- `stale` (interior topology decay from consecutive un-rebuilt refits) stays
  in a tight 0.99–1.01 band for the ENTIRE 1200 ticks — no sign of unbounded
  growth from refits alone. This is the component `max_refits` was meant to
  backstop (an "area creep that never trips the ratio"); it is empirically
  flat, not creeping.
- `pose` (silhouette oscillation) is bounded by the animation's own
  periodicity — a repeating gait cannot exceed whatever its widest stride
  produces relative to the frozen reference, and 20 cycles already sampled
  every phase.
- The derived `degrade_ratio = 1.7030` carries ~1.6× headroom over the
  observed 20-cycle peak (`1.0703`), so the gate will fire well before a
  genuinely pathological topology (proven directly by
  `dynamic_splice_default_gate_fires_on_blowup`, below) while staying clear
  of ordinary periodic motion (proven by
  `dynamic_splice_default_holds_across_bounded_oscillation`).

Given the CORRECTED gate now measures and bounds the right quantity with
real headroom (unlike the earlier derivations, which were either
structurally blind or measuring the wrong ratio), `max_refits: 0` is
redundant rather than decorative: the ratio gate itself is now doing the
work the cap was meant to backstop. Kept at `0` (unlimited) rather than
deriving a cap — a cap would only matter for a NON-periodic motion class
(e.g. continuous, non-looping deformation) this realm doesn't exercise;
that's the same backlog item noted below, not invented here to avoid an
honest "no cap needed given what we tested" conclusion.

## Discriminating tests (both directions) for the DEFAULT params

Added to `packages/scrying-glass/src/bvh.rs`'s test module — both use
`RefitParams::default()` directly (the real `DEFAULT_DEGRADE_RATIO`), no
inline fixture:

- **`dynamic_splice_default_holds_across_bounded_oscillation`** — six
  independent "limb" quads, each swinging around its own fixed base offset
  with its own phase (a rigid whole-body translation would leave every
  node's half-area unchanged and prove nothing; independent per-limb motion
  is what actually reshapes internal node boxes, same as a real gait).
  Amplitude/period are the same order as the real sweep's observed envelope.
  Driven for 4 synthetic 60-tick "cycles" (≥2, matching the real sweep's
  multi-cycle coverage) — asserts **0 rebuilds**.
- **`dynamic_splice_default_gate_fires_on_blowup`** — a modest move first
  proves the default tolerates ordinary motion (refit holds), then
  scattering the triangles far apart (the pre-existing blowup fixture, now
  run against `RefitParams::default()` instead of an inline `1.6`) asserts
  the default gate fires a **rebuild**, proving the default is not
  decorative and does not rely on `max_refits` to ever catch anything.

Both pass. (The prior `dynamic_splice_rebuilds_on_degradation` test, which
used an inline `degrade_ratio: 1.6` fixture, is folded into
`dynamic_splice_default_gate_fires_on_blowup` above — same scatter scenario,
now proving the DEFAULT fires rather than an arbitrary fixture.)

## `perf_audit` OVERLAP margin — honest note (ADVISORY 6)

The LEVER 2 `perf_audit` OVERLAP wall-clock mean PASSES the 16.67ms budget
for both poses, but the `wide` pose's margin is noticeably thinner than
`front`'s across repeated runs on this machine: `front` OVERLAP has read as
low as ~11.2ms (>5ms of headroom), while `wide` OVERLAP has read between
~13.0ms and ~14.0ms (roughly 2.7–3.6ms of headroom, and an adversary-observed
run read `14.15ms`, ~2.5ms of headroom) — background system load visibly
moves it. `wide` is the more geometry-heavy composed pose (per the night-2
audit); its thinner margin is real and worth tracking if the realm grows
further, not a one-off. Not a regression from this atom (LEVER 1's
`degrade_ratio` fix does not change `perf_audit`'s trace cost), but flagged
here since the adversary asked for it to be stated plainly.

## What would sharpen this later

The GPU trace `drift` column (the ground truth `drift_gate` actually tests
— that's what the 60 FPS law cares about, not the ratio proxy) never crossed
`drift_gate` in either revision-2 run (300-tick peak drift ~1.71ms vs a
2.43ms gate; 1200-tick / 20-cycle peak drift ~2.12ms vs a 2.68ms gate — both
clean runs this time, no thermal-throttling artifact). A body preset with
more aggressive limb excursion, or genuinely non-periodic motion (not a
looping walk cycle — the backlog item `max_refits` would actually matter
for, see above), might degrade the interior topology enough to bite;
filed as backlog, not blocking this atom. The `stale` (topology-decay)
component staying flat at 0.99–1.01 across 20 full cycles is itself decent
evidence that un-rebuilt refit alone isn't where the risk lives for THIS
realm — the risk (if any) is in silhouette excursion from non-periodic
motion, which this sweep by construction can't exercise (the realm's only
locomotion is the periodic walk cycle).
