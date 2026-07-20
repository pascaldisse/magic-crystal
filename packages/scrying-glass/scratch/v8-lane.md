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
