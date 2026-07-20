# v8 training lane — resume/status note (ghoul run 2026-07-20, room 1)

## What shipped this room (committed, NOT the training itself)
Commit `ad6979a` on `neural-live` (worktree
`/Users/pascaldisse/projects/magic-crystal-neural-live`):
- `packages/scrying-glass/examples/rdirect_train_v8.rs` — new trainer,
  smallest-diff evolution of v7e. Implements all three HANDOFF.md
  §07-20 EVE roots:
  (a) MOVING-CAMERA HISTORY: `render_pose_seq` drifts camera yaw across the
      K unroll steps (`GAIA_V8_PANSTEP`, default 0.004 rad/step, matches the
      ordeal's own `GAIA_ORDEAL_PANSTEP`); `reproject_prev` +
      `history_forward` run the REAL `CamPose::reproject` + depth/normal
      guard + `sky_history_reject()` path during TRAINING (not just eval).
      `pan_step=0` (the validation pose) degenerates exactly to a still
      camera via SNAP_EPS — one code path serves both.
  (b) SPECULAR EVIDENCE + MIRROR POSES: new "mirror" training pose
      (`denoiser_dataset::mirror_camera`, the `naruko_show_chrome`-framing
      `spawn_eye`), evidence taps AND teacher/target rendered at
      `GAIA_V8_MIRROR_SPP` (default 4) instead of 1.
  (c) HIGHLIGHT-TARGETED SAMPLING: `GAIA_V8_HIGHLIGHT_FRAC` (default 0.3) of
      each epoch's per-pose subsample drawn from that pose's own top-
      `GAIA_V8_HIGHLIGHT_PCTL` (default 5%) brightest teacher pixels instead
      of uniform-random. No loss-shape change (still plain MSE, no cap/gate/
      firefly-weight — banned per v7c/N3/N4).
  FRESH init only (no resume path — v7's recorded lesson). Corner-crawl
  lr 1e-4. Monitor every epoch. Cross-run bar-normalized score floor.
- `src/rdirect.rs`: `bilinear_vec3` made `pub` (visibility only, body
  byte-identical).
- `src/denoiser_dataset.rs`: `mirror_camera()` added (standalone additive
  helper; `law_poses`/TRAIN/VALIDATION untouched).
- `examples/real_image_ordeal.rs`: `GAIA_ORDEAL_WEIGHTS=v8` convenience
  mapping to `data/rdirect-weights-v8.bin`.

Verified before commit: clean release build; smoke run (96x72, 3 epochs)
and one full-resolution 2-epoch run (384x288) both completed without
panics, loss decreasing, `highlight_ratio` climbing 0.011→0.375 in 2
epochs; ordeal plumbing smoke-tested end to end at tiny resolution
(expected FAIL, no stamp written, v7's own stamp file untouched — verified
`data/rdirect-weights-v7.bin.stamp` mtime unchanged by this room); all 6
pre-existing regression ordeals (`rdirect_gather_ordeals`,
`rdirect_gpu_ordeals` ×4, `rdirect_live_ordeals`) re-run, byte-identical to
every prior room's recorded numbers.

## Training launch (DETACHED, still running as of this note)
```
cd /Users/pascaldisse/projects/magic-crystal-neural-live/packages/scrying-glass
GAIA_V7_SKY_HISTORY=reject nohup nice -n 19 cargo run --release -j2 \
  --example rdirect_train_v8 > scratch/v8-train.log 2>&1 &
```
PID (process launched, actual binary PID once cargo handed off): `23764`.
Log: `packages/scrying-glass/scratch/v8-train.log`.
Config: all defaults — 200 epochs, K=3 steps, 384×288 native / 192×144 low,
ref_frames=64, D-blur r=2, pan_step=0.004, mirror_spp=4, highlight_frac=0.3
(top 5%), lr0=1e-4, ema=0.999, wall budget 10800s (3h — not expected to be
hit; ETA below is well under it).

**Progress at note time**: epoch 1/200 finished (mse 0.130→0.082,
`highlight_ratio` 0.011→0.128→(pending, epoch1 monitor not yet printed at
snapshot time)), history precompute ≈16.4s/epoch (fixed, weight-independent
cost — full 384×288×4-pose forward pass), total per-epoch wall ≈33-37s.
**ETA for all 200 epochs ≈ 110 minutes** from launch (well inside the
10800s wall budget — expected to finish on epoch count, not the wall
timer). This room's own interactive session does not span that long, so
training was launched detached per the task's own instruction ("walls eat
rooms, not work") and this note stands in for the "not complete inside
wall" report — training is expected to complete on its own; check
`scratch/v8-train.log` for final status.

## Resume / check-in commands (next room or later in this one)
Check progress:
```
tail -40 packages/scrying-glass/scratch/v8-train.log
ps aux | grep rdirect_train_v8
```
If it finished (log ends with `[v8] wrote ... provenance.` and
`data/rdirect-weights-v8.bin` exists), run the ordeal exactly as the
mandate specifies (640×480, sky-reject semantics matching training):
```
cd packages/scrying-glass
GAIA_ORDEAL_WEIGHTS=v8 GAIA_V7_SKY_HISTORY=reject \
  GAIA_ORDEAL_W=640 GAIA_ORDEAL_H=480 \
  nice -n 19 cargo run --release -j2 --example real_image_ordeal \
  2>&1 | tee scratch/v8-ordeal-640.log
```
PASS ⇒ `data/rdirect-weights-v8.bin.stamp` is written (sidecar, sha256-bound
to the weights bytes) — that stamp + the log is the only lawful basis for
a "v8 passes" claim. FAIL ⇒ no stamp; read the per-bar distances printed
(resid_still/sparkle_still/tvar_still/resid_move/ghost_excess) to see which
bar(s) are still short, and whether another training round (this trainer,
`GAIA_V8_EPOCHS` raised, or resumed via a NEW run — this trainer has no
built-in resume, would need `GAIA_V8_RESUME`-style support added if a
second round from THIS checkpoint is wanted; per the mandate's own "FRESH
init ALWAYS" law, a second round should probably also be fresh, not a
fine-tune, unless that law is explicitly revisited) is warranted.

If the process was killed (machine restart, wall hit, etc.) and needs
relaunching from scratch (no partial-epoch resume exists — FRESH init is
required by the mandate, so "resume" means "relaunch the same command",
not "continue from a checkpoint"):
```
cd packages/scrying-glass
GAIA_V7_SKY_HISTORY=reject nohup nice -n 19 cargo run --release -j2 \
  --example rdirect_train_v8 > scratch/v8-train.log 2>&1 &
```

## UPDATE (same room, +12min watch) — seesaw divergence observed, diagnosed, NOT fixed this room

Watched epochs 1→19 land (log tail, `scratch/v8-train.log`). Clear,
reproducible signature — the SAME sparkle<->resid seesaw v7's own history
already named and cured once (v7-resume's pre-v7d run: "sparkle climbs
monotonically epoch over epoch... resid creeps down... a noise-ball
outlier growing under an unconstrained late-training LR"):

```
epoch  val_sparkle/Mpx  val_resid  score   mirror_sparkle  mirror_resid
2      45.2             0.0812     2.320   72.3            0.0759   <- last *BEST->saved
3      117.5            0.0671     2.939   126.6           0.0608
5      180.8            0.0507     4.521   171.8           0.0430
9      262.2            0.0422     6.556   63.3            0.0316
13     479.2            0.0408     11.981  0.0             0.0264
17     723.4            0.0406     18.084  0.0             0.0249
```
val_resid PLATEAUS around 0.040-0.041 from ~epoch 10 onward (never reaches
the 0.035 bar) while val_sparkle climbs roughly monotonically (45→723+ and
still rising at epoch 19) — score is dominated by the sparkle term and
keeps getting worse. The cross-run bar-normalized score-floor mechanism is
working AS DESIGNED: `best_bytes`/`data/rdirect-weights-v8.bin` have not
updated since epoch 2 (confirmed: no further `*BEST->saved` tag in the log
past epoch 2) — the saved checkpoint cannot regress even though the live
EMA-monitored net keeps getting worse. That checkpoint itself is NOT a pass
(sparkle 45.2 > bar 40, resid 0.0812 > bar 0.035) — it is only the least-bad
state seen so far.

**Diagnosis (not fixed this room, high-confidence lead for the next
round)**: `history_forward` builds each epoch's frozen reprojection-source
chain from the RAW, actively-optimized `mlp` (epoch-start snapshot) — NOT
from `ema_mlp`. v7d's entire cure principle is "the checkpoint that gets
measured/saved is a slow-moving average... which cannot itself develop a
sharp outlier the way the raw SGD/Adam iterate can" — applied there only to
the MONITOR/CHECKPOINT selection. v8's moving-camera history reads the RAW
iterate as its recurrent-feedback SOURCE, once per epoch — if that raw net
has any local sparkle/overshoot that epoch, `hist_features_split`'s
`prev_dl` feature bakes that overshoot into the TRAINING INPUT for every
later step of every pixel that reprojects near it, for the entire epoch —
a plausible feedback path for exactly the runaway-sparkle signature
observed, and NEW relative to v7e (whose identity self-feedback read the
SAME per-pixel forward call being trained that instant, not a separately-
sourced frame-old raw snapshot). **Recommended next-round fix**: build
`history_forward`'s chain from `ema_mlp` instead of raw `mlp` (the smoothed
net, already computed every batch this room's code already maintains) —
consistent with v7d's own cure philosophy, applied one level deeper. This
was NOT implemented/tested this room (would require another edit+rebuild+
relaunch cycle, and the mandate's other two roots — b, c — are unaffected
either way, so this is scoped as a targeted follow-up, not a rewrite).
Secondary suspects, lower confidence, worth checking if the primary fix
doesn't resolve it: K=3 evidence-accumulator steps (vs v7e's 4) may leave
the clamp ceiling noisier (temporal-mean variance ~1/K) than v7e ever
tested; lr0=1e-4 combined with FRESH init (vs v7d/e's fine-tune-from-good
start) may simply need a longer warmup or steeper decay before the corner-
crawl regime is genuinely stable.

**This is NOT a fixed/passing result — do not report v8 numbers from this
log as a pass.** The training process is still running (detached, harmless
— the score floor protects the saved checkpoint from further regression)
and may be left to finish its 200 epochs unattended, but based on the
epoch 2→19 trend a PASS inside this run is unlikely without the fix above.

## Honest state at hand-off
- Code: committed (`ad6979a`), reviewed, smoke-tested, regression-clean.
- Training: RUNNING (detached, PID 23764 as of launch), not complete.
  Observed epochs 2-19 show a sparkle<->resid seesaw (diagnosed above,
  high-confidence lead: history chain should read `ema_mlp` not raw `mlp`)
  — the saved checkpoint is frozen at epoch 2's score (2.320, sparkle 45.2,
  resid 0.0812), which is NOT a pass. No sparkle/resid/ghost bars from a
  genuinely converged v8 round exist. Left running unattended (harmless,
  protected by the score floor) but a PASS inside this specific run is
  unlikely on the observed trend — the honest expectation is this round
  ends with the epoch-2-era checkpoint still saved, still failing bars,
  and the real fix (EMA-sourced history) landing in a NEXT round.
- No ordeal has been run against a genuinely-trained v8 checkpoint, no
  stamp exists or is claimed.
- Port 8430 / the live scrying-glass window: NOT touched this room (this
  trainer is a standalone offscreen `wgpu` process, never binds a server;
  verified no listener on 8430 during the run).

## 2026-07-20 (ghoul room 2) — EMA fix FALSIFIED, ablation matrix launched

**EMA-sourced-history fix (`cf8bd7b`) does NOT cure the runaway.** v8b
(fresh detached relaunch of the exact same trainer post-fix, PID 24874,
log `scratch/v8b-train.log`) reproduces the IDENTICAL seesaw shape as the
pre-fix v8 run (`scratch/v8-train.log`), epoch-for-epoch, through epoch 28:

```
epoch  v8  (pre-fix)          v8b (post-fix, ema_mlp-sourced history)
9      val_sparkle 262.2      val_sparkle 334.6
13     val_sparkle 479.2      val_sparkle 542.5
17     val_sparkle 723.4      val_sparkle 633.0
21     val_sparkle 795.7      val_sparkle 759.5   resid stuck ~0.040 both runs
```
Both: val_resid plateaus ~0.039-0.041 (never reaches 0.035 bar), val_sparkle
climbs roughly monotonically well past the sp<16 target while mirror-pose
val (STILL, diagnostic) stays low/zero and resid keeps improving normally.
The EMA-vs-raw history-source hypothesis (previous room's leading
diagnosis) is **falsified as sole root cause** — whatever detonates val
sparkle survives sourcing the recurrent feedback chain from the smoothed
net. v8b killed at **epoch 28** (PID 24874, confirmed dead via `pgrep -fl
rdirect_train_v8` returning empty) — two matching diseased curves through
epoch 21+ is evidence enough; the GPU was freed for ablation instead of
riding out 200 epochs of a run already known to fail the bars.

### Ablation design
v7e (prior trainer) was stable; v8 added exactly three new components:
(a) moving-camera pan history, (b) mirror training pose, (c) highlight-
targeted sampling. One of these (or their interaction) is the detonator.
Added env switches to `examples/rdirect_train_v8.rs` (commit `bcc1c04`),
IRON-law defaults (omitting all of them reproduces the exact mandate
trainer): `GAIA_V8_TAG` (renames log prefix + `data/rdirect-weights-<TAG>.bin`
+ its provenance.json + the cross-run score floor — ablation runs can never
read or clobber the real v8 checkpoint), `GAIA_V8_MIRROR_POSE` (default 1,
0 drops mirror from the TRAINING cam set only; the diagnostic mirror-monitor
pose still renders/reports either way). `GAIA_V8_PANSTEP=0` and
`GAIA_V8_HIGHLIGHT_FRAC=0` already existed and already fully disable (a)/(c)
respectively (SNAP_EPS still-camera degeneration; zero highlight-oversample
draws) — no new switches needed for those two.

Four 25-epoch runs, SERIAL (one GPU), fresh init each (INIT_SEED is a
compile-time const shared by construction across all runs — same seed),
`GAIA_V7_SKY_HISTORY=reject` always:

| run | tag      | pan_step | mirror_pose | highlight_frac | isolates |
|-----|----------|----------|-------------|-----------------|----------|
| A0  | v8ablA0  | 0        | 0           | 0               | v7e-parity baseline (all 3 off) |
| A1  | v8ablA1  | 0.004    | 0           | 0               | (a) moving-camera history alone |
| A2  | v8ablA2  | 0        | 0           | 0.30            | (c) highlight sampling alone |
| A3  | v8ablA3  | 0        | 1           | 0               | (b) mirror pose alone |

Detonation signature to compare once all four finish: val sparkle
>300/Mpx by epoch 10 (both diseased v8/v8b runs hit that); healthy
expectation ≪100 (A0, the v7e-parity baseline, should stay low throughout
if the theory holds — its own early epochs already look healthy, see
below). Whichever of A1/A2/A3 reproduces the >300/Mpx-by-epoch-10 signature
while A0 stays clean identifies the detonator; if A0 ALSO detonates, the
bug is upstream of all three (shared plumbing/config), not one of the
three mandate components.

Driver: `scratch/v8-ablate.sh` (new, committed `bcc1c04`) runs all four
runs serially, logs to `scratch/v8-ablate-{A0,A1,A2,A3}.log`, launched
detached: `nohup ./scratch/v8-ablate.sh > scratch/v8-ablate-driver.log 2>&1 &`
— driver PID **26543** (the trainer subprocess it spawns per run gets its
own PID, currently 26546 for A0). Total matrix ≈ 4×(25/200 × prior-run
wall) — prior 200-epoch runs ran ~37s/epoch cumulative-elapsed so 25
epochs ≈ 13-16min per run, ≈ 1-1.1h for all four; leave unattended, it
finishes on its own (fresh 384x288 native res, same as the full v8 runs —
NOT shrunk, so the signature stays comparable).

**A0 early read (epochs -1..2, watched live)**: val_sparkle **0.0/Mpx**
through epoch 2, val_resid falling normally 0.1253→0.1198→0.1105→0.0972
(monotonic improvement, no plateau yet at this early stage — 25-epoch runs
starting from scratch won't reach the 0.035 bar in this few epochs
regardless, that's expected and not itself a verdict). This is NOT yet
comparable to the diseased runs' own epoch-2 numbers (val_sparkle was
already 45-72/Mpx by epoch 2 in both v8/v8b) — A0's epoch 2 sparkle being
exactly 0.0 vs their 45-72 is the first positive signal for the "one of
(a)/(b)/(c) is the detonator, not shared plumbing" hypothesis, but this is
NOT a verdict — only 3 epochs observed, no cure/pass claim, wait for all
four logs at epoch 10+ before comparing against the >300/Mpx-by-epoch-10
signature.

Checkpoint/floor isolation verified before launch: `GAIA_V8_TAG=smoketest`
tiny run (64x48, 1 epoch) wrote `rdirect-weights-smoketest.bin` +
`.provenance.json`, left `data/rdirect-weights-v8.bin`'s md5
(`69dbd3f0f952479c8d679c1a9238896d`) byte-identical; smoketest artifacts
deleted after the check.

**Status: matrix running, unwatched past A0 epoch 2. No cure/pass claim.**
This room only isolates the variable — read all four
`scratch/v8-ablate-{A0,A1,A2,A3}.log` in a later room before drawing any
conclusion about which component (or combination) is the detonator.

## 2026-07-20 (ghoul room 3) — ablation VERDICT read + v8c DOCTRINE ROUND shipped, training launched

### Ablation verdict (all four 25-epoch arms finished, logs read in full)
Val-sparkle/Mpx at matched epochs (bar: <16; ">300 by epoch 10" = the
v8/v8b detonation signature):

| epoch | A0 (baseline, all off) | A1 (pan history only) | A2 (highlight sampling only) | A3 (mirror pose only) |
|-------|------------------------|------------------------|-------------------------------|-------------------------|
| 0     | 0.0                    | 0.0                     | 0.0                            | 0.0                      |
| 4     | 0.0                     | 0.0                     | 126.6                          | 0.0                      |
| 8     | 18.1                    | 18.1                    | 253.2                          | 54.3                     |
| 10    | 72.3                    | 72.3                    | 352.6                          | 63.3                     |
| 13    | 63.3                    | 63.3                    | 533.5                          | 72.3                     |
| 24    | 117.5 (best score 1.442)| 108.5 (best score 1.450)| 651.0 (best score 2.487)       | 117.5 (best score 1.356) |

**VERDICT: A2 (highlight-targeted sampling) is the CONVICTED DETONATOR.**
Only A2 crosses >300/Mpx before epoch 10 (352.6 at epoch 10, already past
the signature threshold) and keeps climbing to 651.0 by epoch 24 — an
order of magnitude past A0/A1/A3, all three of which track each other
closely (63.3-117.5 at epoch 24, none showing runaway). A1 (moving-camera
history) and A3 (mirror pose) are **INNOCENT** — A3 is in fact the
best-behaved arm of the three (lowest best-score, 1.356). Matches the
task's own stated verdict exactly.

### v8c shipped (commit `4568a3d`) — TIER1 + TIER2, first lane under the 07-20 enforcement clause

**DOCTRINE-CONCORDANCE:** v8c implements TIER 1 (estimator-init,
`Mlp::evidence_mean_init_split` in `src/rdirect.rs` — analytic weight
construction, output==box-mean(4 E-taps)+box-mean(4 D-taps) BY
CONSTRUCTION, no He-random, no pretrain loop) + TIER 2 (noise2noise,
`examples/rdirect_train_v8c.rs` — two independent sparse evidence draws
per pose/step, seed families 0x7abc/0xb222; loss = MSE(net(draw_A
features), draw_B radiance); teacher [ref_frames=64] demoted to VALIDATOR
ONLY — measured every MONITOR line and by the cross-run score floor,
never touches `accumulate_backward_clamped_slice`'s target argument) +
TIER 3 kept as structure/validator only, NOT the training signal (moving-
camera pan history — A1 innocent; mirror pose spp-4 evidence — A3
innocent, kept unconditionally; EMA-sourced history chain, cf8bd7b;
gamma=1.5 evidence clamp, v7e/IRON). Highlight-biased sampling DROPPED
ENTIRELY (A2 convicted above) — no `GAIA_V8_HIGHLIGHT_*` env vars in this
file. `box_blur`/`GAIA_V8_BLUR` also dropped: its referent (teacher-blur
target construction) is exactly what tier 2 replaces, and an unfiltered
draw-B label is required for the noise2noise unbiasedness argument to
actually hold — a disclosed, narrowly-scoped deviation from "keep all env
params", not a silent one.

Verified before launch: `cargo build --release` clean; new unit test
`rdirect::tests::evidence_mean_init_split_copies_the_box_mean_exactly`
passes (synthetic feature vector, exact match to 1e-6); all 6 pre-existing
regression ordeals (`rdirect_gather_ordeals`, `rdirect_gpu_ordeals` x4,
`rdirect_live_ordeals`) re-run, still passing; tiny smoke run (96x72, 3
epochs, tag `v8c_smoke`) completed end-to-end, TIER1 ORDEAL-ASSERT
max|diff|=0.0; full-res 5-epoch check (384x288, tag `v8c_check5`,
`scratch/v8c-check5.log`) run and read before the real launch. Checkpoint
isolation confirmed throughout: `data/rdirect-weights-v8.bin` md5
`69dbd3f0f952479c8d679c1a9238896d` unchanged by any of the above.

### TIER1 init verification (printed by the trainer, real render, not synthetic)
```
[v8c] TIER1 ORDEAL-ASSERT: output==evidence-mean(box-mean of 4 E-taps + 4 D-taps), BY CONSTRUCTION, on 36 real-render channel samples: max|diff|=0.00000000 mean|diff|=0.00000000
[v8c] TIER1 construction VERIFIED — output IS the classical evidence-averaging estimator before epoch 0.
[v8c] MONITOR epoch -1 (TIER1 fresh, BORN AS ESTIMATOR): val sparkle 18.1/Mpx resid 0.0428 highlight_ratio 0.784 score=1.222 | mirror sparkle 9.0/Mpx resid 0.0192 highlight_ratio 0.948 (tgt sp<16 resid<0.035) *BEST->saved
```
Exact-zero diff confirms the analytic construction is correct, not
approximately-close. The MEASURED (teacher-vs-net) numbers at epoch -1 —
sparkle 18.1/Mpx, resid 0.0428 — are already inside shouting distance of
the bars (sp<16, resid<0.035) BEFORE a single gradient step, vs v8/A0's
own random-init baseline of resid 0.1253 (sparkle read 0.0 there too, but
for the opposite, uninteresting reason — a near-zero-output net has no
local error PEAKS to be sparkle-outliers, not because its output is any
good; resid 0.1253 is the honest number for that baseline).

### Early curve (epoch -1..14, DETACHED run PID 35150, launched
`GAIA_V7_SKY_HISTORY=reject nohup nice -n 19 cargo run --release -j2
--example rdirect_train_v8c > scratch/v8c-train.log 2>&1 &`), read live,
honest report per the mandate:

| epoch | v8c val sparkle | v8c val resid | A0 val sparkle (same epoch) | A0 val resid (same epoch) |
|-------|------------------|-----------------|-------------------------------|------------------------------|
| -1/0  | 18.1             | 0.0428/0.0427   | (n/a)/0.0                      | (n/a)/0.1198                 |
| 2     | 9.0              | 0.0427          | 0.0                             | 0.0972                        |
| 4     | 0.0              | 0.0430          | 0.0                             | 0.0751                        |
| 6     | 9.0              | 0.0433          | 18.1                            | 0.0612                        |
| 8     | 27.1             | 0.0435          | 18.1                            | 0.0529                        |
| 10    | 36.2             | 0.0437          | 72.3                            | 0.0489                        |
| 13    | 45.2             | 0.0438          | 63.3                            | 0.0467                        |

**Honest read**: v8c's val sparkle does NOT reproduce A0's late creep
pattern in this window (13 epochs observed) — it stays in the same 0-45
order of magnitude as A0 itself, nowhere near A2's >300-by-epoch-10
signature. v8c's val resid, unlike A0's (which drops steeply from a bad
random-init 0.12 down toward 0.047 — still short of the 0.035 bar at
epoch 13), sits almost FLAT around 0.043 with a slow uptick (0.0427 ->
0.0438 over 13 epochs) — TIER1 init starts it far closer to the bar than
A0 ever gets to in this window (v8c epoch -1's 0.0428 already beats A0's
OWN epoch-13 resid of 0.0467), but the noise2noise signal is not yet
pulling it down further in the first 13 epochs; whether it breaks through
0.035 depends on the rest of the 200-epoch run (the corner-crawl lr
schedule is deliberately slow/flat — "mild harmonic decay" from 1e-4 — so
13 epochs is early). This is reported honestly, not as a pass: the slight
resid uptick is a real, if small, trend to keep watching in later epochs,
not yet diagnosed as benign or a problem.

### State at hand-off (this room ends here — training left running unattended)
- Code: committed (`4568a3d`), reviewed, unit-tested, regression-clean.
- Training: RUNNING DETACHED, PID 35150, log `scratch/v8c-train.log`,
  default tag `v8c` (writes `data/rdirect-weights-v8c.bin` +
  `.provenance.json` — NEVER the real `v8` files; tag-scoped score floor).
  200 epochs, ETA ~200*37s ≈ 7400s (~2h3m), well inside the 10800s (3h)
  wall budget. Confirmed alive and progressing normally through epoch 14
  as of this note (elapsed 9m21s at that point).
- **No converged checkpoint exists inside this room's wall — no ordeal
  run, no pass/stamp claim of any kind.** This note is the resume point,
  not a result.
- Port 8430 / any live scrying-glass window: NOT touched this room (this
  trainer is a standalone offscreen `wgpu` process, never binds a server).

### Resume / check-in commands (next room)
```sh
tail -60 packages/scrying-glass/scratch/v8c-train.log
ps aux | grep rdirect_train_v8c
```
If finished (log ends with `[v8c] wrote ... provenance.` and
`data/rdirect-weights-v8c.bin` exists — check its mtime moved past this
note's), run the ordeal exactly as v8's own resume note specifies, just
with `GAIA_ORDEAL_WEIGHTS` pointed at the v8c artifact (check
`real_image_ordeal.rs` for whether a `v8c` convenience mapping needs
adding, or pass the path directly):
```sh
cd packages/scrying-glass
GAIA_V7_SKY_HISTORY=reject GAIA_ORDEAL_W=640 GAIA_ORDEAL_H=480 \
  nice -n 19 cargo run --release -j2 --example real_image_ordeal \
  2>&1 | tee scratch/v8c-ordeal-640.log
```
PASS ⇒ `data/rdirect-weights-v8c.bin.stamp` written — that + the log is
the only lawful basis for a "v8c passes" claim. FAIL ⇒ read the per-bar
distances, decide whether another round (raised epochs, or the resid-
uptick trend above needs its own follow-up diagnosis) is warranted. If
the process died (machine restart, wall hit) before finishing, FRESH init
is the tier-1 construction itself — relaunch verbatim, same command as
above (no partial-epoch resume path exists, matches v8's own law).
