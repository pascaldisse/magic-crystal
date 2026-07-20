# v7 cutover status — STILL NOT READY (blocker narrowed, verified live)

Ghoul run 2026-07-20 (Stage 3, ~25min timebox). Previous verdict (commit
7333316): live path can't even LOAD v7 weights (architecture mismatch). That
specific blocker is now FIXED (commits d0a9240, 602d8bf) — but the frame loop
still doesn't drive the net correctly, so this is still not launchable for
the Architect. Full detail: `scratch/v7-live-lane.md` §"STAGE 3 PROGRESS".

## What changed this room

- `RdirectLive::build` (rdirect_live.rs) now derives `in_features` from the
  loaded weights' own first-layer dim instead of hard-asserting 23. Verified
  live:
  ```
  $ cargo run --release -j2 --example v7_live_load_probe
  LOADED OK in_features=39 out_channels=3
  ```
  (was `REJECTED: ... in_dim Some((39, 64)) != INPUT_FEATURES 23` before this
  room). The 23-in weights (v1-v4) load exactly as before — no regression,
  confirmed by all 6 pre-existing parity ordeals staying byte-identical.
- `NetPresent::new` (main.rs) gained `GAIA_NATIVE_WEIGHTS=v7` (loads +
  stamp-checks `data/rdirect-weights-v7.bin`, same REAL-IMAGE BAR path as v4)
  — AND a hard guard right after: if the loaded net's `in_features` doesn't
  match what the (still 23-in-only) frame loop below actually feeds it, the
  constructor returns `Err` → `present_black`, same failure shape as an
  unstamped/missing weights file. So today `GAIA_NATIVE_WEIGHTS=v7` loads the
  net, proves the stamp, and then **cleanly presents black** — not a
  corrupted/wrong image. This was a deliberate choice: without the guard,
  selecting v7 would silently feed a 39-wide net 23-wide gathered rows (a
  per-pixel stride mismatch, not a smaller act) — a genuinely WRONG image,
  which is worse than black under this codebase's own REAL OR BLACK law.

## THE REMAINING BLOCKER (verified live, same shape as before, smaller scope)

The frame loop `NetPresent::resolve_frame` drives (FeatureGather → 23-wide
gather buffer → `live.forward` → DemodPass) is **unchanged** this room. Three
real pieces of engineering remain, none started:

1. Wire Stage 2's `FeatureGatherHistSplit` + `HistoryBuffers` (currently only
   exercised by `examples/v7_live_hist_probe.rs`) into `NetPresent`'s actual
   frame loop — a new 39-wide buffer family + a history ping-pong instance,
   pooled once, feeding `live.forward` every frame and swapping after present.
2. Port `clamp_evidence_lin` (rdirect.rs) to GPU at the present stage, which
   itself needs a NEW per-pixel temporal-mean evidence accumulator (not just
   a spatial max — the lane spec is explicit that time-mean, not time-max, is
   the CPU reference's actual recipe). No GPU code for this exists yet.
3. A full-frame parity gate (GPU v7 present vs `direct_render_sequence_hist_split`
   CPU reference, post-clamp) and an fps re-measurement — both depend on 1-2
   existing first. **No parity numbers, no fps table this room** — fabricating
   either would be worse than reporting the honest gap.

Per the v7-live-lane.md's own Stage 3 spec, this is "real engineering days,
not a port" — confirmed again this room; the narrowed scope (loader generalized,
guard in place) is real forward progress, not the finish line.

## Launch command (still NOT ready — do not run for the Architect)

```sh
GAIA_NATIVE_WEIGHTS=v7 \
GAIA_NATIVE_ASYNC_TRACE=1 \
./target/release/scrying-glass
```

Today this produces a **clean black window** (verified failure mode: the new
in_features guard trips, `NetPresent::new` returns `Err`, same
Pleroma-or-BLACK fallback as any unstamped weights file) — not a corrupted
image, and not the ordealed act either. Per whip 154 nobody launches this for
the Architect regardless — recorded for when it's actually true.

## Honest deltas (repeated from the previous room's note, still true)

- Frame timing last measured (pre-v7, non-split net): 18.57ms / 53.85fps. Not
  re-measured for v7 this room either (still can't run it live — the frame
  loop wiring above is the reason now, not the load-time rejection). Expect a
  regression once it runs: 39-in GEMM is larger than 23-in, plus the new
  gather/history/clamp passes.
- v7's `resid_still` (0.0349) sits at the CPU ordeal's own narrowest-margin
  PASS vs the 0.035 bar — no live-path claim, repeated for the record.

## Artifacts

- `examples/v7_live_load_probe.rs` — now reproduces the LOAD SUCCESS (was the
  architecture rejection before this room).
- `examples/v7_stamp_probe.rs` — reproduces the stamp-bar PASS (unchanged).
- `examples/v7_live_hist_probe.rs`, `examples/v7_live_ed_probe.rs` — Stage
  1/2 probes, unchanged, still the only callers of the 39-in gather types.
- This file + `scratch/v7-live-lane.md` §"STAGE 3 PROGRESS" (fuller diff/log).

No weights, no shipped code path, no gate was weakened or bypassed to get
here. Nothing launched, no running session touched (whip 154).
