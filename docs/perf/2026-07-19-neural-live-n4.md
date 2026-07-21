# neural-live N4 — THE TEACHER-GATED FIREFLY LOSS (v5): VERDICT BLACK STANDS

Scope: N4 attempt to escape the N3 sparkle↔resid Pareto front with a
per-pixel TEACHER-GATED firefly clamp. Branch `neural-live`.
Weights `data/rdirect-weights-v5.bin` sha `01b67a4550f8` (5×64, 27-in, warm
from v3, resumed from the salvaged ep9 checkpoint). Bars UNTOUCHED.

## The idea (§ rdirect_train_v5.rs, accumulate_backward_firefly_gated)
LOSS = MSE(out,teacher) + gate·ff_w·Σ_c relu(out_c − cap_c)²
- cap_c = teacher k×k neighbourhood max (demod-log) + margin.
- gate = 1 ONLY where teacher k×k is genuinely DARK (max-lum < scene
  percentile ceiling), 0 where teacher is bright (real neon/windows/cyan).
- Where teacher bright → plain MSE (render the real light exactly, no smear).
- Where teacher dark → invented dot over cap crushed.
- IRON params: ff_w 15, cap 5×5, margin 0.05, dark_pct 0.80.

## Training (resume ep9 → 90 ep, wall 499s, monitor@10) — § scratch/v5-train-resume.log
val (orbit_-20, 480×360, K=4 settle):
```
start(=ep9 salvage)  sparkle 75.2  resid 0.0394
ep9   sparkle  40.5  resid 0.0411  *BEST(fallback lowest-sparkle)
ep19  sparkle 329.9  resid 0.0358
ep29  sparkle  63.7  resid 0.0379
ep39  sparkle 167.8  resid 0.0374
ep49  sparkle 185.2  resid 0.0356
ep59  sparkle  52.1  resid 0.0366
ep69  sparkle 318.3  resid 0.0373
ep79  sparkle 196.8  resid 0.0351
ep89  sparkle 156.2  resid 0.0352
done: pass=FALSE (no ep cleared sp<16 AND resid<0.036); best = ep9 fallback.
```
THE SEESAW: sparkle oscillates 40–330 while resid sits 0.035–0.041 — as MSE
refits resid falls and sparkle explodes; as the clamp bites sparkle falls and
resid rises. NO joint minimum. The teacher gate MOVED along the Pareto front
(recovered real cyan → resid better than v4) but did NOT escape it.

## REAL ORDEAL @640×480 (bars untouched, 738.9s) — § scratch/v5-ordeal.log
```
resid_still    0.04470  bar 0.03500  FAIL  (+0.00970)
sparkle_still 48.82812  bar 40.00000 FAIL  (+8.82812)
tvar_still     0.00001  bar 0.00050  PASS  (−0.00049)
resid_move     0.04501  bar 0.06000  PASS  (−0.01499)
ghost_excess   0.00072  bar 0.01200  PASS  (−0.01128)
VERDICT: FAIL — NO stamp — window BLACK by law.
```
No `rdirect-weights-v5.bin.stamp` on disk → `verify_stamp` false → present
black. Gate re-pinned: real_image_gate 2/2 (unstamped denied).

## v4 (N3) vs v5 (N4) — moved along the front, not off it
```
          resid_still        sparkle_still
v4 (N3)   0.05099 FAIL       39.06 PASS
v5 (N4)   0.04470 FAIL       48.83 FAIL
```
The gate traded v4's resid-fail down (0.051→0.0447, ~0.006 of the over-clamped
cyan waterline recovered) for a sparkle-fail up (39→49, invented dots survive
in mid-bright neighbourhoods the gate scored as not-dark). Two-of-five fail now
where v4 failed one. Same front, different point.

## s24 both eyes @640×480 net-vs-teacher (§ proof/neural-live/s24-*.png) — defects first
- FIREFLIES NOT GONE: stray cyan/blue specks scatter across the water and
  along the waterline (net), absent in teacher — the visible sparkle 48.8.
- CYAN WATERLINE STILL BROKEN: dashes sparse/fragmented, right building-base
  glow largely missing; teacher has clean continuous dashes + right-base glow —
  the visible resid climb. (Better than v4's dim smear — real cyan partly back.)
- LIT WINDOWS CRISP: two yellow windows sharp at correct positions (gate off,
  MSE rules) — PASS, unchanged from v3/v4.

## Gates (no regression)
rite5 17/17 PASS · medium_parity 2/2 PASS · real_image_gate 2/2 PASS (BLACK held).

## NEXT (the front is real — a scalar clamp cannot win)
A single per-channel excess-over-cap penalty, gated or not, rides one Pareto
front: killing an invented dot in a mid-bright neighbourhood also dims the real
emissive there. Escape needs a loss that separates INVENTED energy from REAL
energy structurally — temporal (a dot that flickers frame-to-frame vs a stable
emissive) or a matched-teacher high-freq residual — not a spatial luminance cap.
