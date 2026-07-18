# R-DIRECT spike verdict — 2026-07-18

Branch `r-direct` (worktree `magic-crystal-rdirect`), UNMERGED lane.
The Architect's ruling (07-18): the net RENDERS DIRECTLY — G-buffer +
sparse 1-spp radiance in, THE IMAGE out; weighed vs the shipped
denoise+upscale CHAIN at the SAME ray budget. Performance rule: a net that
loses at equal quality+cost DIES.

Module `src/rdirect.rs` · harness `examples/rdirect_spike.rs` ·
full log `proof-rdirect-run.log` (160-epoch run) · proof quads
`proof/rdirect-*.png`.

## Architecture (direct-render per-target-pixel MLP)
- Fuses VIII-1 denoise + VIII-3 upscale into ONE net. Input = LOW-res 1-spp
  traced radiance (the sparse guide) + FULL-res G-buffer. Output = final
  native image directly (absolute, NOT a residual over bilinear — the net
  IS the renderer, honoring the ruling).
- Features per native pixel (23, ALL current-frame): 2×2 low-res 1-spp
  demod-log radiance taps (12) + subpixel offset (2) + hi-res albedo (3) +
  normal (3) + log-depth (1) + motion (2). Output = 3 (demod-log radiance,
  re-modulated by hi-res albedo → RGB).
- Shape: 5 hidden × 64 ReLU. **18048 MAC / native pixel**, ~18k params.
- Output space = VIII-1's albedo-demod log-radiance (HDR-safe). Absolute.
- MOTION channels = zero in this static-pose dataset (honest gap — a
  moving-camera wave, out of the current-frame ban's static scope).
- BAN-SCOPED (current-frame only) — passes the VIII-0 grep-gate.

## Dataset
naruko realm, low 48×32 → native 96×64 (scale 2 — where VIII-1/VIII-3 pin),
128-frame converged reference. TRAIN = {front, wide, orbit_+20}; held-out
VAL = {orbit_-20, orbit_+40}; SCENE-EDIT gate = front pose, 160 leaf-tris
displaced by (6,3,0) — reference perturbed by rmse 0.0251 (asserted >0, not
a no-op).

## Gate (a) QUALITY — RMSE vs converged reference (equal 1-spp ray budget)
| pose | bilinear(1spp) | chain | net | net<chain |
|---|---|---|---|---|
| orbit_-20 (held-out) | 0.054106 | 0.048072 | **0.038871** | yes |
| orbit_+40 (held-out) | 0.042729 | 0.054037 | **0.040453** | yes |
| front_edit (scene edit) | 0.067038 | 0.054533 | **0.037185** | yes |
| front (train) | 0.068492 | 0.052964 | 0.032610 | yes |
| wide (train) | 0.068553 | 0.056059 | 0.049792 | yes |
| orbit_+20 (train) | 0.062546 | 0.048428 | 0.036464 | yes |

→ net beats the chain on EVERY pose, held-out AND train. Worst held-out net
RMSE (pinned bound) = **0.040453**. Chain sometimes WORSE than bilinear
(orbit_+40, front_edit) — a warm horizontal horizon smear it introduces;
the net does not.

## Gate (b) COST
- MAC/native-pixel: chain (denoise@low + upscale@native) = 14696 · net =
  18048 · ratio **1.23×** (net costs ~23% more compute than the chain).
- CPU f32 reference @96×64 (median of 5): net 174.15 ms · chain 178.26 ms
  (CPU-ref only — NOT the GPU number; the chain is two nets' overhead).
- GPU PROJECTION (naive-port scaling from VIII-2's measured 26.5ms /
  3488-MAC / 540k-px, UNVERIFIED extrapolation) @900×600: net ~137 ms ·
  chain ~112 ms. 60-fps budget = 16.67 ms → BOTH are ~7–8× over in naive
  fp32; reaching budget needs the fp16 + subgroup + threadgroup-weights opt
  VIII-2 already flagged (unproven at this MAC count).

## Gate (c) GENERALIZATION — no memorization
Held-out camera poses (orbit_±) + one held-out scene edit (block displaced,
never seen in that config) — net wins all three. Proof quad `front_edit`
shows the displaced towers reconstructed correctly. Not memorization.

## Gate (d) SUITE
Green. 51 lib tests (incl. 4 new rdirect) + every integration ordeal
(viii0 ban-gate through viii3b) pass. No regression.

## Proof (own-eyes read, panels = 1spp-input | net | ground-truth | chain)
`proof/rdirect-{orbit_-20,orbit_+40,front_edit,...}.png`. The 1-spp panel is
heavy colored grain; the NET panel is visibly smooth — towers crisp, sky
gradient clean, close to truth. The CHAIN panel is denoised too but carries
a blotchy warm horizon smear truth lacks. Net's honest weakness: it slightly
UNDER-warms the foreground ground strip (orbit_-20) and shows a mild cyan
waterline streak (orbit_+40) vs truth's warmer tone. Net is cleaner overall
and closer to truth than the chain — matching the RMSE.

## VERDICT
**The direct net BEATS the chain on quality at equal ray budget** — every
held-out pose + scene edit, at 1.23× the compute. Per the performance rule
it LIVES on quality. **60 fps = UNVERIFIED**: CPU-reference only, no GPU/WGSL
port; naive-fp32 projection ~137 ms @900×600 needs ~8× from the fp16/subgroup
opt (same path VIII-2 owes) to reach 16.67 ms. What 60 fps would cost = that
GPU optimization, measured — not this spike.

## Gaps
- No GPU/WGSL port — all ms are CPU-ref or projected (UNVERIFIED on GPU).
- Motion channel zero (static dataset) — untested; needs a moving-camera wave.
- One realm (naruko), 96×64 dataset res — small; broader validation owed.
- Absolute-output direct form beat the chain; a residual-over-bilinear
  R-direct variant was not A/B'd.
- Chain at scale-2 both nets pinned; not swept across scales.

Branch parks UNMERGED as reference (rite8-viii2-ari / neural-radiance precedent).
Artifacts: `data/rdirect-weights-v1.bin` (+ `.provenance.json`).
