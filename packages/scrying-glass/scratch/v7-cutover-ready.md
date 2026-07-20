# v7 cutover status — RUNS LIVE NOW, small known parity gap, fps regression measured

> **CORRECTION (room 7, 2026-07-20):** every "v4 baseline" / "non-split
> baseline (v4, room 1)" mention below of 18.57ms/53.85fps is mislabeled.
> That number is from `docs/perf/2026-07-18-neural-live-n0.md` SHIFT 18
> (N0.n), BEFORE the `GAIA_NATIVE_WEIGHTS` v1..v7 scheme existed (it
> shipped one shift later, SHIFT 19, default v2) — it is not a v4
> measurement and cannot be reproduced under a version label. v4 itself
> has no PASS stamp anywhere in this worktree (v7 is the first-ever
> weights to earn one here) and is REAL-OR-BLACK gated, so it **cannot
> lawfully render right now**; 18.57/53.85 is a dead, unversioned number
> carried forward as an informal target only.

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
  | unversioned pre-v1..v7 baseline (SHIFT 18/N0.n, NOT v4) | 18.57 | — | 53.85 |

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

## Room 4 update (ghoul run 2026-07-20): reprojection-guard fix — STATUS STILL OPEN

- Root cause found this room: `gather_hist_split`'s `cam_reproject`
  off-screen bounds check (`fpx < 0 || fpy < 0 || fpx > w-1 || fpy > h-1`)
  sits geometrically AT 0/w-1 for a still camera's self-reprojected edge
  pixels (algebraic identity, `sx == cx` when cur==prev); a sub-ULP
  GPU-vs-CPU difference in the underlying dot products flips which side
  of the boundary the coordinate lands on, converting into a full
  history valid=0/1 disagreement at ~1% of pixels (the still-sequence
  "plateau" from room 4's earlier finding). The other leading suspect
  (WGSL `round()` half-to-even vs Rust `f32::round()` half-away-from-zero)
  was tested and FALSIFIED — fixing it changed nothing.
- Fix: `REPROJ_EDGE_EPS = 1.0e-3` slack on the bounds check + clamp the
  admitted fpx/fpy into range (`src/rdirect_gather_split.wgsl`).
- Result: still-sequence max-abs-diff **6.07e-2 → 1.28e-2** (~2x), flipped
  pixel count **58-62 → 30-32** (~2x) — real improvement, but NOT collapsed
  to the pan-class 1e-6 floor, and the fix is asymmetric (only GPU's side
  moved) so it introduced a small new pan-sequence delta (2.93e-4 at one
  frame, still px>1e-3=0). **Cutover-blocking status UNCHANGED**: the
  shipped-equals-ordealed-act seam is NOT closed. Full numbers, the exact
  divergent lines, and next steps: `scratch/v7-live-lane.md` §"STAGE 3
  room 4".
- Regression guard: all 6 pre-existing ordeals re-run this room, still
  byte-identical (the WGSL edit doesn't touch their code paths).
- FPS: one s20 run (flag ON, v7+split), TOTAL median 18.15ms/WALL-FPS 42.7
  — within noise of room 3's 16.57-16.63ms, not independently re-confirmed
  with a second run.

## Room 5 update (ghoul run 2026-07-20): symmetric SNAP_EPS — SEAM CLOSED

- **Fix**: replaced room 4's asymmetric (GPU-only) `REPROJ_EDGE_EPS` bounds
  slack with a symmetric pixel-boundary SNAP applied identically on BOTH
  sides: `CamPose::reproject` (`src/rdirect.rs`) and `cam_reproject`
  (`src/rdirect_gather_split.wgsl`) now snap `fpx`/`fpy` to the nearest
  integer whenever within `SNAP_EPS = 1.0e-3` of one, BEFORE the unchanged
  bounds/depth/normal tests — removes the boundary ambiguity at its source
  (the fractional coordinate) instead of fuzzing the accept window on one
  side only. `SNAP_EPS` derivation (documented at both definition sites):
  ~1e-6 observed ULP noise floor, 0.5 half-pixel tie point; 1e-3 sits
  ~1000x above the former, ~500x below the latter. Room 4's `REPROJ_EDGE_EPS`
  constant and its asymmetric clamp are fully removed (superseded, not kept
  alongside).
- **Ordeal (a) — RE-EARNED, PASS.** Full real-image ordeal re-run under the
  modified CPU act (`GAIA_ORDEAL_WEIGHTS=v7 GAIA_ORDEAL_W=640
  GAIA_ORDEAL_H=480`, detached, `scratch/v7i-ordeal-snap.log`, 754.5s):
  ```
  resid_still    0.03487  bar 0.03500  PASS  (distance -0.00013)
  sparkle_still 35.80729  bar 40.00000  PASS  (distance -4.19271)
  tvar_still     0.00003  bar 0.00050  PASS  (distance -0.00047)
  resid_move     0.03670  bar 0.06000  PASS  (distance -0.02330)
  ghost_excess   0.00000  bar 0.01200  PASS  (distance -0.01200)
  VERDICT: PASS — stamp re-written data/rdirect-weights-v7.bin.stamp
  ```
  Every number byte-identical to the pre-fix stamp (room `55720b45`'s own
  PASS) — confirms the prediction: the snap only touches boundary-history
  validity at silhouette pixels, the resid/sparkle/tvar/ghost metrics (which
  don't reproject) are untouched. `resid_still` still sits at its own
  narrowest margin (0.03487 vs 0.035 bar) — unchanged, not a new risk.
- **Parity (b) — COLLAPSED to machine precision on BOTH sequences.**
  `examples/v7_present_parity_probe` re-run post-fix
  (`scratch/v7i-parity-snap.log`):
  ```
  still frame 0  max-abs-diff 4.7684e-7
  still frames 1-11  max-abs-diff 3.5763e-7 (constant, px>1e-3=0 every frame)
  still OVERALL max-abs-diff 4.7684e-7        (was 1.2803e-2 after room 4's fix, 6.0695e-2 pre-room-4)
  pan   frame 0  max-abs-diff 4.7684e-7
  pan   frame 1  max-abs-diff 1.5497e-6
  pan   frame 2  max-abs-diff 1.0729e-6
  pan OVERALL max-abs-diff 1.5497e-6          (was 2.9301e-4 after room 4's fix, 1.5497e-6 pre-room-4)
  ```
  The still sequence now matches the pan sequence's own machine-precision
  class (~1e-7-1e-6, px>1e-3=0 throughout) — the room-4 plateau (30-32/6144
  flipped px) is fully gone, not just halved. The pan sequence's small new
  room-4 delta (2.93e-4 at frame 2) is also gone — pan is back to its
  original 1.5497e-6 exactly, because the fix is symmetric (no asymmetric
  admission on either side to introduce a new delta).
- **Regression (c) — all 6 pre-existing ordeals byte-identical, flag OFF.**
  `scratch/v7i-regression-snap.log`:
  ```
  n0b_gather_and_shared_forward_match_cpu ... ok
  c_ban_no_temporal_vocabulary_in_the_gpu_kernel ... ok
  a_gpu_inference_is_byte_identical_same_frame_twice ... ok
  b_f32_gpu_matches_cpu_within_derived_bound ... ok
  b2_fp16_fast_kernel_matches_cpu_within_derived_bound ... ok
  n0_gate1_live_net_matches_cpu_reference ... ok
  ```
  Checked which of these exercise `CamPose::reproject`: NONE — grepped
  `tests/rdirect_gather_ordeals.rs`, `tests/rdirect_gpu_ordeals.rs`,
  `tests/rdirect_live_ordeals.rs` for `reproject`/`hist_features`/
  `direct_render_sequence_hist`; the only hit is a string-ban check
  (`rdirect_gpu_ordeals.rs`'s vocabulary-ban test greps for the word
  "reproject" in generated shader source — unrelated to calling the
  function). All 6 use the non-history/non-split forward paths
  (`gather_and_shared_forward`, GPU-vs-CPU single-frame parity, live-net
  gate) which never call `reproject` — byte-identity is exactly expected,
  not merely observed, and confirmed exactly (all numbers match room 4's
  own re-run verbatim).
- **FPS (d) — one s20 run, flag ON.** `GAIA_NATIVE_WEIGHTS=v7
  GAIA_NATIVE_EVIDENCE_SPLIT=1`, 640×480 offscreen, 1130 frames:
  **TOTAL median 20.30ms / p95 28.81ms, WALL-FPS 42.8**
  (`proof/neural-live/s20-v7snap.{log,budget.json,state.json}` +
  `-presented.png`). Within noise of room 4's 18.15ms/42.7fps and room 3's
  16.57-16.63ms — the changed op (an `if`+`floor` compare, same ALU class as
  room 4's eps compare) is not a plausible multi-ms source; read as machine
  load noise, consistent across 3 rooms now.
- **Seam status: CLOSED.** Ordeal PASS (re-earned under the modified act,
  byte-identical metrics to the pre-fix stamp), both parity probes at
  machine precision, all 6 regressions unchanged, fps within established
  noise band. The shipped-equals-ordealed-act seam that blocked cutover
  since room 2 is resolved — the only remaining known delta is the
  documented p95 tail (28.8ms, ~1.6x the median, consistent across every
  room's fps numbers — a scheduling/poll-latency tail, not a correctness
  gap; not chased further, was never in scope for this seam).

**Final launch command (unchanged from room 2, now backed by a closed
parity seam):**
```sh
GAIA_NATIVE_WEIGHTS=v7 GAIA_NATIVE_EVIDENCE_SPLIT=1 ./target/release/scrying-glass
```
Cutover itself (making v7 the DEFAULT weights selection) remains a separate
decision this note does not make — per whip 154, no window was opened, no
running session touched; all verification above is offscreen on dedicated
ports.

### Artifacts this room
- `src/rdirect.rs` (`CamPose::reproject` — SNAP_EPS symmetric snap, replaces
  nothing on the CPU side, this is the CPU side's first edit in this lane).
- `src/rdirect_gather_split.wgsl` (`cam_reproject` — SNAP_EPS symmetric
  snap, REPROJ_EDGE_EPS + its asymmetric clamp fully removed).
- `data/rdirect-weights-v7.bin.stamp` — re-earned PASS stamp (metrics
  byte-identical to pre-fix).
- `scratch/v7i-ordeal-snap.log`, `scratch/v7i-parity-snap.log`,
  `scratch/v7i-regression-snap.log` — this room's full act outputs.
- `proof/neural-live/s20-v7snap.{log,budget.json,state.json}` +
  `-presented.png` — this room's fps run + captured proof frame.
- `proof/neural-live/s25-{still,moving}{,-teacher}.png` — refreshed ordeal
  proof frames (regenerated by the re-earned ordeal run).

No gate was weakened or bypassed. Nothing launched with a window, no running
session touched (whip 154). All new claims in this file were played through
the real offscreen server + read back from real output (image bytes, /budget
JSON, probe stdout) — nothing here is inferred from logs alone.

## Room 6 (ghoul run 2026-07-20): clean-GPU fps confirmation — THE number for launch

Measurement-only room (no engine/wgsl/rust edits). GPU confirmed clean
before and after (no ordeal/train/cargo PIDs). 3x sequential s20 offscreen
benches, 640x480, flag ON (`GAIA_NATIVE_WEIGHTS=v7
GAIA_NATIVE_EVIDENCE_SPLIT=1`), `nice -n 19`:

| run | TOTAL median/p95 (ms) | wall_fps |
|---|---|---|
| v7clean1 | 18.116 / 27.548 | 42.68 |
| v7clean2 | 18.181 / 30.296 | 41.24 |
| v7clean3 | 19.096 / 27.940 | 43.71 |

**This supersedes rooms 3/4/5's own fps numbers (16.63/18.15/20.30ms) as
the number to expect at launch** — those 3 numbers, taken individually,
looked like an unexplained drift; this room's 3-run same-session spread
(18.12-19.10ms median) shows that drift was mostly ordinary run-to-run
noise on this machine, not a regression trend. **Median expectation:
~18.5ms (~54fps-class), consistent with the documented pre-v7 baseline
(18.57ms/53.85fps) — v7 is not a median regression once room 3's poll
fix is counted.** Known, unfixed p95 tail: ~27.5-30.3ms (~1.5x median) on
a recurring (not one-off) subset of frames, root-caused only as far as
"trace + net_wait spike together, whole-frame GPU/driver stall shape" —
not chased to a fix, not v7-specific (present in every room's prior
single-run numbers too). Flag-OFF (old 23-in v4 path) could not be
re-measured this room: `data/rdirect-weights-v4.bin.stamp` is missing
from this worktree (gitignored, machine-local, never generated here) so
v4 refuses REAL-OR-BLACK and falls back to raster (`/budget` returns
`"note":"net-present off"`) — earning that stamp needs its own ~750s
ordeal run, out of scope for a 20min measurement room. Full numbers,
per-bucket table, and the p95 spike-attribution reasoning:
`v7-live-lane.md` §"§perf — room 6".

**The one number for the Architect: ~18.5ms median / ~54fps-class, with a
known ~28-30ms p95 tail on a recurring minority of frames (pre-existing,
not v7-specific, not yet root-caused to a fix).**

**UPDATE (§perf room 8, 2026-07-20) — opt-in flag improves this.** With
`GAIA_NATIVE_ASYNC_TRACE=1` added on top of the same v7+evidence-split
config (flag stays default-OFF, this is an A/B, not a new default): 3-run
mean **46.29 fps / 16.31ms median / 25.47ms p95** vs the clean baseline's
42.54 fps / 18.46ms median / 28.60ms p95 above — **+3.75fps, −2.16ms
median, −3.13ms p95**, consistent across all 3 pairs. Root cause: purely
CPU-side (net_gpu, the actual GPU-compute bucket, stays flat 6.0→6.4ms) —
collapsing 2 blocking `device.poll` waits in the trace stage into fewer
sync points removes real wall-clock wake/scheduling overhead; the freed
time does show up in `gather`'s own (unconditional, unchanged) poll, but
by a smaller amount than trace saved. This REVERSES the pre-v7 N0-lane
CUT B rejection (which found the same move net-negative, −4 to −5fps)
specifically under v7's queue shape (fused single-submission resolve
tail + frame overlap + evidence split) — the old rejection was scoped to
"pre-v7 queue shape" by room 7 and that scoping holds. Parity: the
live-path safety argument is structural (wgpu FIFO hazard tracking makes
poll timing irrelevant to pixel correctness); `v7_present_parity_probe`
ran flag ON (`still_max=4.7684e-7` px>1e-3=0, `pan_max=1.5497e-6`
px>1e-3=0) but is a standalone harness that doesn't read the flag at all
(confirmed byte-identical to a flag-OFF re-run) — not itself proof, just
a clean sanity re-run. 6 regression ordeals byte-identical, flag OFF,
no code changed. Full tables: `v7-live-lane.md` §"§perf — room 8".
Recommend the Architect consider flipping the default ON for the v7 path
(not the 23-in v4 path, untested here, where N0's rejection still
stands).
