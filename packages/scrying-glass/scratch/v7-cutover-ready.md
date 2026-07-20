# v7 cutover status — NOT READY (architecture blocker, verified)

Ghoul run 2026-07-20. TASK: port c8b9ba6's evidence clamp into rdirect_live.rs
so the live present path matches the ordealed act for stamped v7 weights.
BLOCKED before the clamp itself — the live path cannot run the v7 net at all.

## What's confirmed GOOD

- Stamp bar: `data/rdirect-weights-v7.bin.stamp` verifies TRUE against
  `data/rdirect-weights-v7.bin` (sha 55720b45) via `rdirect::verify_stamp` —
  proven live (`examples/v7_stamp_probe.rs`):
  ```
  verify_stamp = true
  ```
  Not weakened, not bypassed. `NetPresent::new`'s REAL-IMAGE BAR check in
  main.rs (unconditional, no env override) will accept it once loadable.
- `packages/scrying-glass` release binary builds clean (0 errors, pre-existing
  warnings only): `target/release/scrying-glass` present, `cargo build
  --release -j2 --bin scrying-glass` exits 0.

## THE BLOCKER (verified live, not inferred)

v7's stamped/passing weights are the **N6 split-evidence recurrent** net:
`input_features=39` (`HIST_FEATURES_SPLIT` = E-taps + D-taps split (24) + 11
geo + 3 prev-radiance history + 1 validity), trained and ordealed through
`direct_render_sequence_hist_split` — recurrence (`prev_dl` fed forward frame
to frame) is not optional, it's load-bearing for the architecture (N5/N6
"recurrent" is the whole point).

`rdirect_live.rs` — the ENTIRE live path (`RdirectLive::build`, the
`rdirect_gather.wgsl` feature gather, the `SharedPool` buffer sizing) — is
built for the OLDER **N1-era plain** net: `INPUT_FEATURES = 23` (4 radiance
taps × 3, no E/D split, no history, no validity), no recurrent state at all.
`RdirectLive::build` hard-asserts this:

```rust
if dims.first().map(|d| d.0 as usize) != Some(INPUT_FEATURES) {
    return Err(format!(
        "rdirect_live: first layer in_dim {:?} != INPUT_FEATURES {}",
        dims.first(), INPUT_FEATURES
    ));
}
```

Proven live (`examples/v7_live_load_probe.rs`, `cargo run --release -j2
--example v7_live_load_probe`):

```
REJECTED: rdirect_live: first layer in_dim Some((39, 64)) != INPUT_FEATURES 23
```

So today, if the Architect set `GAIA_NATIVE_WEIGHTS=data/rdirect-weights-v7.bin`
and launched the live window, `RdirectLive::from_wgpu_queue` errors →
`NetPresent::new` returns `Err` → **present BLACK** (the documented
Pleroma-or-BLACK fallback) — not the ordealed act, not even the unclamped
overshoot. The clamp is not the blocking atom; the net SHAPE is.

## Why I did not write dead clamp code

Porting `clamp_evidence_lin`/`EvidenceAccum` into `rdirect_live.rs`'s demod
stage (WGSL or the fused MSL kernel) is mechanically easy — but there is no
live evidence composite to clamp against: `rdirect_gather.wgsl` never builds
`low_e`/`low_d` (the gather shader only assembles the 23-feature plain-net
input), and there is no per-pixel history buffer carrying `prev_dl` across
frames for the recurrent input either. Writing the clamp math against a
placeholder/self-referential "evidence" (e.g. the net's own trace-res accum,
not the actual E/D split taps the CPU path clamps against) would NOT be the
same act as the ordeal — it would violate the exact design law this task
exists to uphold ("shipped image must equal the ordealed act"). Landing that
would be worse than not landing it: a clamp that silently diverges from the
CPU reference is a *new* correctness bug hiding behind a PASS-shaped name.

## Real prerequisite (separate work item, not a 25-min port)

Before the clamp port is even meaningful, the live path needs:
1. `rdirect_gather.wgsl` (+ `FeatureGather`) rewritten for the 39-in split
   layout: separate low-res E/D radiance taps (not just one `accum`), same
   bilinear reconstruction `pixel_features_split`/`hist_features_split` do.
2. A per-pixel recurrent history buffer (`prev_dl`, `valid`) that survives
   frame-to-frame and reprojects (the CPU sequence path's `prev` state) —
   new GPU-resident state, new buffer in `SharedPool`, new reprojection logic
   (currently the live path is fully stateless, CURRENT-FRAME ONLY per its
   own doc comment).
3. THEN the evidence composite (`evidence_composite_frame` + temporal-mean
   `EvidenceAccum` + `local_max_3x3`) can be built on GPU from the same E/D
   taps, and THE CLAMP ported into `rdirect_demod.wgsl` / `DEMOD_FUSED_MSL`
   at the exact point `net_lin` becomes `present[i]` — mirroring
   `clamp_evidence_lin(net_lin, evidence_max[i], gamma)`, same
   `GAIA_V7_CLAMP_GAMMA` env override, same default 1.5.
4. THEN the parity gate (n0-gate1 pattern, `tests/rdirect_live_ordeals.rs`)
   extends to a v7-shaped fixture (39-in features + history state) and to the
   POST-clamp presented pixel, not just the raw net forward.

None of steps 1-3 exist yet. Estimate is real engineering days, not a port.

## Launch command (once actually ready — NOT today)

```sh
GAIA_NATIVE_WEIGHTS=data/rdirect-weights-v7.bin \
GAIA_NATIVE_ASYNC_TRACE=1 \
./target/release/scrying-glass
```
Today this produces a **black window** (verified failure mode above), not the
ordealed act. Per whip 154 nobody launches this for the Architect regardless
— this command is recorded for when it's actually true.

## Honest deltas already known (from the v7e commit message, unrelated to
## this blocker, still true once cutover is real)

- Temporal accumulation / history state: v7's `resid_still` (0.0349) still
  misses the CPU-side `resid` bar's original 0.035 threshold by the
  narrowest possible margin at `still` framing per the v7h ordeal log
  (PASS was on the full table, see stamp above) — no live-path regression
  claim, this is the CPU ordeal's own number, repeated here for the record.
- Frame timing last measured (pre-v7, N0.i S13 era, non-split net):
  18.57ms / 53.85fps. Not re-measured for v7 (can't run it live yet — see
  blocker). Expect it to regress: 39-in vs 23-in GEMM is larger, plus
  whatever the history buffer costs once built.

## Artifacts

- `examples/v7_live_load_probe.rs` — reproduces the architecture rejection.
- `examples/v7_stamp_probe.rs` — reproduces the stamp-bar PASS.
- This file.

No weights, no shipped code path, no gate was weakened or bypassed to get
here. Nothing launched, no running session touched (whip 154).
