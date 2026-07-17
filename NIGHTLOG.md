# NIGHTLOG — the night of 2026-07-17 (Fable conducts; NIGHTRUN.md is the order)

## Landed

### 1. rite-specs → main (queue item 1, part a)
- **What**: RITE VI (STRIFE) + RITE VII (THE PLANET-WALKER) spec proposals — the
  night queue's law inputs; branch `rite-specs` @ 76a9d75 was never merged
  (NIGHTRUN cited docs/proposals/RITE-VI-STRIFE.md, which only existed there).
  `rite-specs-2` (VIII + IX) was already on main — verified, nothing to land.
- **Merge**: ceb102d (docs-only, additive), PUSHED.
- **Suite**: 279 passed / 0 failed, counted from the running lines
  (first count attempt truncated itself via `tail -60` — the vacuous-tail law
  bit its own auditor; re-ran with full capture: scratchpad suite-main-ceb102d.log).
- **Proof**: docs-only; no pixels owed.

## In flight
- **perf-exact** (queue item 1, part b): dirty worktree SALVAGED @ cb6770e —
  LEVER 1 refit-not-rebuild (DynamicSplice, build_indexed/refit, parity-gate
  example, audit + player wiring) compiled clean and committed. First audit run
  with the lever: **DYN-ON 14.05 ms front / 15.80 ms wide — ALL SIX CELLS PASS
  the 60 FPS law** (56 refits / 0 rebuilds per pose; realm now 3492 static +
  9096 dynamic tris, 2 bodies). Builder atom in flight: degradation metric was
  root-half-area — structurally blind (root AABB of a refit tree ≡ fresh build's
  by construction; the sweep proved ratio pinned at 1.0000 while real trace
  drift climbed 0.17→1.69 ms) — being reworked to total-node-half-area (SAH
  cost proxy) with the degrade_ratio derived from the re-run sweep. Then:
  CPU/GPU overlap atom (audit-truthfulness: player loop already overlaps —
  non-blocking poll, Fifo, no readback; the audit's serial sum overstates),
  adversary review, merge.
