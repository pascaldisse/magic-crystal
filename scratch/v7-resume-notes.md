# v7 iteration resume notes (2026-07-19T10:45Z, room wall cutoff)

## State on disk right now
- `packages/scrying-glass/data/rdirect-weights-v7.bin` sha256=`636c8743faf25fa3311d8752a9e5d6ff0abd4b4ad52fb2466e2cfcaf4bbdf8ce`
  — this is the MONITOR epoch-9 checkpoint of the K=8/subsample=20000 warm-start
  run below (val sparkle 46.3/Mpx resid 0.0361), NOT the epoch-9 run's natural
  completion. Process was SIGTERM'd at the 25-min room wall, mid epoch 10.
- `data/rdirect-weights-v7.provenance.json` is STALE — still describes the
  PRIOR run (ba701a929d1d..., "FRESH init", K=4/subsample=6000, epoch-29 best
  100.9 sparkle). Do not trust it against the current .bin sha. Regenerate by
  letting a run reach natural completion/wall-stop (it rewrites both files
  together at the end of `main`).
- Old baseline (ba701a92, K=4/subsample=6000) ordeal is COMPLETE and committed:
  resid_still 0.03738/0.035 FAIL, sparkle_still 97.65/40 FAIL, tvar/resid_move/
  ghost all PASS. See `scratch/v7-ordeal-640.log` (commit feaa523).

## What changed in code (committed, safe to reuse)
- `examples/rdirect_train_v7.rs`: added `GAIA_V7_RESUME=1` warm-start (loads
  `data/rdirect-weights-v7.bin` if present instead of fresh init), mirrors the
  v5 trainer's resume pattern. Provenance `init` field now says WARM vs FRESH.
- `examples/real_image_ordeal.rs`: added the `"v7" => rdirect-weights-v7.bin`
  arm to `GAIA_ORDEAL_WEIGHTS` (was missing before this room).

## Progress signal (encouraging, NOT yet passing)
Raising data (K 4->8, subsample 6000->20000/pose) while warm-starting from the
old v7 best cut sparkle from 100.9 -> 46.3/Mpx in just 10 epochs (resid held
flat ~0.036). Bars: sparkle<16 (still 2.9x over), resid<0.035 (0.0361, close).
This is the first run where sparkle moved this much this fast — worth
continuing the SAME warm-start rather than starting over.

## Exact resume command (next room)
```sh
cd packages/scrying-glass
nice -n 19 env GAIA_V7_RESUME=1 GAIA_V7_STILL=8 GAIA_V7_SUBSAMPLE=20000 GAIA_V7_WALL=1200 \
  ../../target/release/examples/rdirect_train_v7 2>&1 | tee /tmp/v7c-train.log
```
(rebuild first if the binary is stale: `nice -n 19 cargo build --release
--example rdirect_train_v7 -j2`). This will RESUME from the epoch-9 checkpoint
above (636c8743...) since GAIA_V7_RESUME=1 loads whatever is currently at
`data/rdirect-weights-v7.bin`. Let it run to full wall budget (1200s) or
completion this time — do not cut at epoch 10 again unless the room wall
forces it. Raise `GAIA_V7_WALL` if more of the 20-min iteration budget is
available in one sitting.

## If sparkle plateaus (per original instructions)
Only if sparkle re-plateaus above 40 with the tripled data: try
`GAIA_V7_BLUR=3` (D-blur radius 2->3) as the ONE extra lever, one run, still
warm-started from whatever is in `data/rdirect-weights-v7.bin` at that point.
Do NOT add scalar luminance/firefly penalties (v3-v6 died on that Pareto
front — recorded law).

## After the next training run completes
1. Full ordeal: `nice -n 19 env GAIA_ORDEAL_WEIGHTS=v7 GAIA_ORDEAL_W=640
   GAIA_ORDEAL_H=480 cargo run --release --example real_image_ordeal -j2`
   (takes ~13 min at 640x480, 10 still + 6 pan frames x2 poses — budget for it).
2. If PASS: stamp is auto-written by the ordeal binary next to the weights;
   commit weights + provenance + stamp + logs immediately (commit-first law),
   then produce the both-eyes proof paths:
   `packages/scrying-glass/proof/neural-live/s25-{still,still-teacher,moving,moving-teacher}.png`.
3. If FAIL: paste the raw bar table, commit whatever trained further, and
   write a fresh resume-notes.md for the next room the same way this one was
   written — never leave the .bin/.provenance.json pair inconsistent without
   a note explaining why.
