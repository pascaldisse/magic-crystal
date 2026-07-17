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

### 2. perf-exact → main (queue item 1, part b) — THE 60 FPS LAW PASSES
- **What**: two exact levers, nothing stolen from the pixels. LEVER 1
  refit-not-rebuild (persistent DynamicSplice; build_indexed + refit; watchdog
  on total-node-half-area vs the rebuild reference; degrade_ratio=1.7030
  DERIVED from the gate's own ratio, 1200-tick/20-cycle sweep, revision 2 after
  an adversary MUST-FIX caught the first derivation measuring a structurally-
  pinned proxy; discriminating tests both ways at defaults). LEVER 2 CPU/GPU
  overlap (audit measures the player-shaped pipelined frame; hash-identity
  serial-vs-overlap MATCH; Metal validation clean).
- **Merge**: 7fe8275, PUSHED. Suite on main post-merge: 283 passed / 0 failed.
- **Adversary**: 2 MUST-FIX (derivation category error; inert watchdog) fixed
  by derivation, never loosening; 4 ADVISORY addressed. Adversary independently
  re-ran every gate.
- **Key numbers**: refit_parity BIT-EXACT all three law poses (0 diverging
  pixels; hashes e8ca…/226a…/5dca… equal on both arms). Audit idle-host:
  OVERLAP 11.20/13.02 ms PASS, serial 14.45/15.82 ms.
- **Proofs read**: parity + audit verdict tables (this lane's proofs are
  numbers, not scryings); auditor re-ran perf_audit on merged main himself.

### 3. Queue item 2 — 60 FPS verification on merged main
- `perf_audit` on main @ 7fe8275: front OVERLAP **11.26 ms PASS** · wide
  OVERLAP **13.23 ms PASS** (budget 16.67), hash-identity MATCH both poses,
  56 refits / 0 rebuilds per pose. Serial DYN-ON read 20.39/20.12 under
  three concurrent cargo builds (idle-host serial: 14.45/15.82 — recorded
  above); the law is judged on the player-shaped pipelined frame, which is
  what the window actually runs. **NO WALL REMAINS — RITE IX not required
  tonight** (stays on the shelf as proposal).

## In flight
- **rite6-vi1** (queue item 3, wave VI-1 THE STACK TOPPLES): built @ d642551 —
  impulse plumbing (Solver::apply_impulse → Physics → Op::Impulse →
  tick_with_ops), NEW rigid-vs-rigid collision pass (solve_body_collisions —
  beyond original plumbing scope, flagged), naruko_stack_crate_0..2 authored at
  derived chained rest heights, 6 new ordeals, canon re-derived, 285 green
  in-lane, three proof scryings READ by the conductor's own eyes (stack stands
  / topples / rests). ADVERSARY REVIEWING now (focus: new collision pass
  determinism/conservation/hardcodes; P-gate 5.1 ms/tick was DEBUG-measured —
  release re-measure demanded).
- **backlog-walkable** (queue item 6, ruling 6): built @ 0aafbd5 — contact-patch
  floor gate; DEFAULT_CONTACT_RADIUS=0.09 measured from nari's foot-bone vertex
  half-extents (0.0807 max, rounded up); slope-derived tolerance; first builder
  died on a real infinite loop (exclusion step 1e-4 < acceptance epsilon 1e-3 —
  the just-rejected candidate re-qualified forever; 46 CPU-min before the
  conductor killed it), salvaged then fixed structurally (named COLUMN_EPSILON,
  step = 2×, loop bounded); 6 patch ordeals + pose-trace canon byte-unchanged,
  285 green in-lane. ADVERSARY REVIEWING now (focus: tolerance looseness
  ~0.29 m — does the mirror die for the right reason; disconnected-sliver
  conspiracies; per-tick probe cost).
- **rite8-viii0** (queue item 5, wave VIII-0 THE NOISE AND THE TRUTH): builder
  in flight — AOV export (albedo/normal/depth, current-frame-only with the
  grep-gate ban ordeal planted from day one), error metric with 0e0 self-test,
  converged reference oracle, viii0-truth.png proof.
- **Rite VII**: recon complete (anchors mapped; coordinate-law payment is
  greenfield across transmute/ring/scene/player). Held until current lanes
  merge — the 64-bit/camera-relative refactor touches every file in flight.
