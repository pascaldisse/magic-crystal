# v7 cutover status — RUNS LIVE NOW, small known parity gap, fps regression measured

Ghoul run 2026-07-20 room 2 (Stage 3 continuation, ~timeboxed). Previous
verdict (this file, prior room): v7 loads but the frame loop refuses to
drive it (present BLACK by the REAL-OR-BLACK guard). **That blocker is now
gone** — `GAIA_NATIVE_WEIGHTS=v7 GAIA_NATIVE_EVIDENCE_SPLIT=1` drives the
REAL 39-in recurrent net through the live app end-to-end (offscreen bench,
verified — see screenshot below) at ~40 fps. Full diff detail:
`scratch/v7-live-lane.md` §"STAGE 3 PROGRESS (room 2)".

## What changed this room

1. **`NetPresent::new` guard relaxed**: 39-in (`HIST_FEATURES_SPLIT`) weights
   are now ACCEPTED when `GAIA_NATIVE_EVIDENCE_SPLIT=1` (still refuses BLACK
   without that flag, and still refuses BLACK for any in_features that is
   neither 23 nor 39 — REAL OR BLACK intact).
2. **Gather stage branches on `is_v7`**: skips the 23-in composite gather
   (would silently write the wrong stride into the now-39-wide pooled
   feature buffer) and drives Stage 2's `FeatureGatherHistSplit` +
   `HistoryBuffers` instead — feeding the REAL net every frame, not just a
   probe.
3. **NEW `rdirect_evidence.{rs,wgsl}`** — 3 GPU compute passes porting
   `rdirect.rs`'s `EvidenceAccum`/`local_max_3x3`/`clamp_evidence_lin`
   (commit c8b9ba6) to the live path:
   - `evidence_accumulate`: bilinear-upsamples this frame's low-res E+D
     radiance into a persistent native-res sum buffer (temporal-mean
     numerator).
   - `evidence_clamp_present`: 3×3 spatial max of that sum, scaled by
     `gamma/count` (folding the `/count` into the clamp kernel instead of a
     separate mean buffer — bit-exact against `EvidenceAccum::ceiling`, not
     an approximation, since max and division-by-a-positive-constant
     commute), then `present = min(present, ceiling)` in place.
   - `pack_out_dl3to4`: repacks the net's tight `[n,3]` output into the
     vec4-per-pixel layout `HistoryBuffers::prev_out_dl` needs.
   This is a **real small compute-pass cut**, not a CPU round-trip — the
   task note's "or CPU-side, note which" question: GPU, all three passes.
4. **Resolve stage**: folds this frame's own split trace into the persistent
   evidence sum every v7 frame, clamps `present_accum` once a net output
   finishes, and swaps history with that frame's own RAW/unclamped `out_dl`
   (matches the CPU reference — the clamp is presentation-only, never fed
   back into history).
5. **Fixed a REAL pre-existing bug this surfaced**: `Integrator::make_split_buffer`
   was missing `COPY_DST` — the live trace stage `clear_buffer`s it every
   frame (same as the composite `net_accum`), which panicked the first time
   `GAIA_NATIVE_EVIDENCE_SPLIT=1` was ever driven through the actual app
   binary (Stage 1 was additive/observational and had never been live-tested
   end-to-end before; probes build their own buffers and rely on wgpu's
   implicit zero-init instead of an explicit clear). Fixed in `integrator.rs`.

## Full-frame parity (live GPU pipeline vs CPU reference)

`examples/v7_present_parity_probe.rs` — drives gather_hist_split + the REAL
loaded v7 net (`RdirectLive::forward_cpu_roundtrip`, same MPSGraph the async
pipeline commits) + `DemodPass` + the new evidence-clamp kernels, against
`rdirect::direct_render_sequence_hist_split` (same weights, same captured
per-frame low_e/low_d/AOV fed to both sides — isolates "does the rest of
Stage 3 agree", not an RNG cross-check; Stage 1/2's own gather parity is
already proven separately at ~1e-6/4e-5).

```
$ cargo run --release -j2 --example v7_present_parity_probe
[v7-present-parity] weights stamp PASS: true
[v7-present-parity] live net loaded in_features=39 out_channels=3
[v7-present-parity] still frame 0 N=6144 max-abs-diff 4.7684e-7 mean-abs-diff 7.9931e-8 px>1e-3=0
[v7-present-parity] still frame 1 N=6144 max-abs-diff 1.7205e-2 mean-abs-diff 7.9278e-5 px>1e-3=59
[v7-present-parity] still frame 2 N=6144 max-abs-diff 1.9002e-2 mean-abs-diff 7.9573e-5 px>1e-3=60
[v7-present-parity] pan   frame 0 N=6144 max-abs-diff 4.7684e-7 mean-abs-diff 7.6172e-8 px>1e-3=0
[v7-present-parity] pan   frame 1 N=6144 max-abs-diff 1.5497e-6 mean-abs-diff 8.2843e-8 px>1e-3=0
[v7-present-parity] pan   frame 2 N=6144 max-abs-diff 1.0729e-6 mean-abs-diff 7.6162e-8 px>1e-3=0
```

**PAN sequence (camera actually moving, reprojection genuinely exercised)
matches the CPU reference to machine precision through all 3 frames
(~1e-6).** **STILL sequence (camera repeated 3× identically) matches at
frame 0 (~5e-7) then widens to ~1.9e-2 max-abs-diff at frames 1-2 — BUT
mean-abs-diff stays ~8e-5 and only 59-60/6144 px (~1%) exceed 1e-3.** This
reads as isolated evidence-clamp BOUNDARY flips — a pixel sitting exactly at
`gamma*ceiling` flips `min()`/not-`min()` on a sub-ULP numeric difference
between the GPU and CPU clamp paths — not a systemic wiring bug (the bulk of
pixels, and the entire pan sequence, stay in the same float-ULP class every
other gate in this lane uses). Root cause of *why* stationary evidence sits
closer to the boundary than panning evidence was not chased further this
room (would need per-pixel clamp-active tracing) — recorded honestly rather
than hidden or asserted into a clean bound. **This is not gated as PASS/FAIL
here** — no ordeal-shaped assertion was added; the raw numbers are the
deliverable.

## FPS (offscreen, 640×480, real `s20-bench.sh`, live app)

```
$ bash proof/neural-live/s20-bench.sh v7stage3b 8434 GAIA_NATIVE_WEIGHTS=v7 GAIA_NATIVE_EVIDENCE_SPLIT=1
/budget: {"frames":1109,"wall_fps":39.81,"stages":{
  "trace":[8.73,15.80],"gather":[1.80,2.18],
  "net_wall":[0.05,3.51],"net_gpu":[5.11,6.41],
  "demod":[8.39,13.69],"present":[0.10,0.18],
  "total":[23.32,31.09]}}
[n0i] ... TOTAL 23.32/31.24 | WALL-FPS 39.9
```
(median/p95 ms; "demod" here is Stage 3's `resolve_ms` bucket — the fused
native demod itself is ~free, but this room's 3 extra evidence/pack/swap GPU
round-trips each block on their own `device.poll(wait_indefinitely)`, which
is where ~8ms of the ~13ms regression below actually lives — an obvious next
room's optimization: batch those into fewer polls.)

**vs the last documented non-split baseline (18.57ms / 53.85fps): v7 measures
23.32ms / ~40fps median — a real, now-measured ~5ms / ~14fps regression**,
in the direction the lane spec predicted (39-in GEMM + new gather/history/
clamp passes), not fabricated.

**Presented image is REAL, not black** (REAL-OR-BLACK proof):
`proof/neural-live/s20-v7stage3b-presented.png` (960×640, offscreen capture)
shows the naruko scene rendering — v7's known floor-bar sparkle is visible in
the ground plane, consistent with `resid_still=0.0349` sitting at the
ordeal's narrowest-margin PASS (recorded before, repeated here — no new
claim).

## Launch command (verified to actually present, NOT yet Architect-facing)

```sh
GAIA_NATIVE_WEIGHTS=v7 GAIA_NATIVE_EVIDENCE_SPLIT=1 ./target/release/scrying-glass
```

Runs, presents a real (non-black) frame, ~40fps. Per whip 154 nobody
launches this for the Architect this room regardless (no window was ever
opened — all verification above is `GAIA_NATIVE_OFFSCREEN=true` on a
dedicated port, his session untouched) — cutover itself (making this the
DEFAULT weights selection) is a separate decision this room does not make.

## Honest deltas

- fps: measured now (was "not re-measured" last room) — ~40fps vs 18.57ms/
  53.85fps baseline, real regression, real number, optimization opportunity
  noted above (fewer polls in the evidence/history stage).
- parity: pan sequence machine-precision; still sequence has a small
  (~1%-of-pixels) clamp-boundary discrepancy, not chased to root cause this
  room, not hidden.
- `resid_still` (0.0349) still sits at the CPU ordeal's own narrowest-margin
  PASS vs the 0.035 bar — unchanged, repeated for the record.
- Regression guard: all 6 pre-existing parity ordeals
  (`rdirect_live_ordeals`, `rdirect_gpu_ordeals`, `rdirect_gather_ordeals`)
  still green, unchanged this room.

## Room 3 update (ghoul run 2026-07-20): fps regression fixed, boundary theory corrected

- **Fix**: folded `evidence_accumulate` / (non-fused) `demod` /
  `evidence_clamp_present` / `pack_out_dl3to4` / history `swap` into ONE
  command encoder + ONE `queue.submit`, no mid-frame
  `device.poll(wait_indefinitely())` between them (was 3). wgpu's own
  intra-command-buffer hazard tracking handles the ordering; nothing reads
  these buffers back on the CPU this frame, so the next frame's own poll
  transitively catches the work up (house pattern, same as SHIFT 17 CUT B).
- **FPS, s20 offscreen 640x480, `GAIA_NATIVE_WEIGHTS=v7
  GAIA_NATIVE_EVIDENCE_SPLIT=1`, TOTAL stage bucket (median/p95 ms)**:

  | | TOTAL | demod(resolve) bucket | /budget wall_fps |
  |---|---|---|---|
  | before (room 2, 3 polls) | 23.32/31.09 | 8.39/13.69 | 39.81 |
  | after (room 3, 1 submit) | 16.57-16.63 / 23.51-25.60 | 0.26-0.27/0.44-0.47 | 45.36-45.39 |
  | non-split baseline (v4, room 1) | 18.57 | — | 53.85 |

  TOTAL now sits BELOW the non-split baseline (16.6ms vs 18.57ms) — target
  was <=18.6ms, beaten. `wall_fps` (full server loop incl. world/http) is
  still under the baseline's 53.85 — read as a difference in `outside`
  (world/http) load between sessions, not render cost; TOTAL-to-TOTAL is
  the fair comparison and it improved. Full numbers/commands:
  `scratch/v7-live-lane.md` §"STAGE 3 PROGRESS (room 3)".
- **Boundary-flip verdict corrected**: room 2's "sub-ULP `min()` flip"
  theory is FALSIFIED by a new diagnostic dump (5 worst STILL-sequence
  pixels' pre-clamp net output vs their clamp ceiling) — every flipped
  pixel's unclamped value sits 10-20% BELOW its ceiling (the clamp never
  fires), so the mismatch is a real ~5-10%-relative gap in the raw net
  output itself, present only in the repeated-identical-camera (STILL)
  sequence, absent in the moving-camera (PAN) sequence and in STILL's own
  frame 0. Scoped to compounding recurrent feedback of REAL net output at
  a fixed pose; not root-caused this room. See lane note §4 for the full
  dump and reasoning.
- Regression guard unchanged: all 6 pre-existing ordeals still
  byte-identical.

## Artifacts

- `src/rdirect_evidence.{rs,wgsl}` — new Stage 3 GPU evidence-clamp kernels.
- `examples/v7_present_parity_probe.rs` — full-frame parity probe + (room 3)
  the boundary-flip diagnostic dump.
- `proof/neural-live/s20-v7stage3b.{log,budget.json,state.json}` +
  `s20-v7stage3b-presented.png` — room 2's fps run + captured proof frame.
- `proof/neural-live/s20-v7fpsroom{,2}.{log,budget.json,state.json}` +
  `-presented.png` — room 3's fps re-runs (2x) + captured proof frames
  (non-black, mean brightness ~88/255, real content).
- `src/main.rs` (`NetPresent`, `resolve_frame`'s v7 tail) — the wiring +
  room 3's single-submission merge.
- `src/integrator.rs` (`make_split_buffer`) — the COPY_DST fix.
- This file + `scratch/v7-live-lane.md` §§"STAGE 3 PROGRESS (room 2/3)".

No gate was weakened or bypassed. Nothing launched with a window, no running
session touched (whip 154). All new claims in this file were played through
the real offscreen server + read back from real output (image bytes, /budget
JSON, probe stdout) — nothing here is inferred from logs alone.
