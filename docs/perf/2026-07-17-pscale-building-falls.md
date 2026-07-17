# P-SCALE — THE BUILDING FALLS · phase-broken cost of a bonded-building collapse
Branch `physics-scale` · 2026-07-17 · lane: honest numbers → where neural physics pays.

## What was built
- `packages/elements/src/building.rs` — parameterized bonded multi-storey
  structure ON the solver (`BuildingSpec` → `erect`/`settle`/`topple`, every
  dial a documented default, no hardcode-as-law). Base layer anchored
  (`inv_mass=0` — the physics-recon "anchors" problem). Mass DERIVED
  (`density×volume`, `spawn_bonded_box`'s own law).
- `packages/elements/src/solver.rs` — `PhaseProfile` + `Solver::step_profiled`:
  opt-in per-phase wall-clock timers wrapping the SAME private phase methods
  `step` calls, same order. `step` (production) untouched, zero timing
  overhead. Bit-identity to `step` locked by ordeal.
- `packages/elements/examples/pscale_measure.rs` — the cost table over the full
  collapse at 3 scales.
- `packages/scrying-glass/examples/pscale_building.rs` — 3 offline PNGs (cubes
  per particle, coloured by fragment, traced directly — no ECS, renders the
  exact anchored solver scenario).
- `packages/elements/tests/pscale_ordeals.rs` — 5 ordeals.

## The building
Solid bonded lattice, footprint 8×8 m, height 24 m (tall tower), density 2000
(masonry). Base anchored. Knocked down by an authored lateral impulse (`topple`,
+30 m/s on the upper half → support-shear). N set by lattice resolution; scales
profiled: **1024** (8×16×8), **2000** (10×20×10), **3456** (12×24×12).
Largest tractable ≈3456 chosen because at the default 8 substeps N=3456's whole
tick already runs ~62–71 ms (~4× the 16.667 ms budget) driven by the O(k²) body
pass — larger N is a minutes-per-collapse curve, not a tractable diorama. The
10k ceiling is reachable but only as a cost DATUM, not a playable scene.

## Cost table (per-tick solver CPU, wall-clock, single core, vs 16.667 ms budget)
`dt=1/60, substeps=8, topple +30 m/s upper 50%, 400 ticks over the collapse.`

| phase | N=1024 med / worst | N=2000 med / worst | N=3456 med / worst |
|---|---|---|---|
| constraint solve | 0.30 / 1.95 | 0.78 / 1.17 | 1.63 / 8.33 |
| fragment floodfill | 0.61 / 16.83 | 1.43 / 13.56 | 2.85 / 10.14 |
| **body-vs-body O(k²)** | **8.91 / 34.12** | **27.75 / 73.27** | **53.95 / 115.80** |
| static collision | 0.81 / 10.16 | 1.53 / 5.73 | 8.30 / 33.57 |
| integrate | 0.02 / 2.20 | 0.04 / 0.10 | 0.06 / 2.32 |
| velocity passes | 0.06 / 5.64 | 0.14 / 0.45 | 0.27 / 0.83 |
| fracture pass | 0.002 / 0.02 | 0.004 / 0.02 | 0.007 / 0.03 |
| **WHOLE TICK** | **10.48 / 53.12** | **31.59 / 80.19** | **70.94 / 134.77** |

Whole-tick median vs budget: 0.63× · 1.90× · 4.26×.
Body-pair-checks/tick (the O(k²) driver, `(k choose 2)×it×sub`): 4.19M · 15.99M · 47.76M.

## The curve (own reading)
- **`body-vs-body O(k²)` DOMINATES at every scale and explodes with N**:
  8.9 → 27.7 → 53.9 ms median (≈82–85% of the whole tick). It is the ONLY
  super-linear phase. Doubling+ N roughly quadruples it — the `(k choose 2)`
  brute-force pair loop, exactly as flagged "not yet exercised at scale."
- **Crucially, the O(k²) cost is nearly all ITERATION OVERHEAD, not collision
  work.** While the building is whole (and largely after — see fragmentation
  below), almost every clustered pair is SAME-cluster and hit the early
  `continue`, yet the loop still VISITS all `(k choose 2)` pairs. At N=3456
  the collapse produced only 2 fragments, so ~all 47.76M pair-checks/tick were
  same-cluster skips — and it still cost 53.9 ms. The bottleneck is the
  brute-force broad-phase itself, independent of how much actually touches.
- Everything else stays cheap and ~linear: constraint solve, floodfill,
  static collision, integrate, velocity, fracture all ≤8.3 ms even at N=3456,
  most ≤3 ms. Fragment floodfill is O(N+bonds) and stays sub-3 ms median.
- Worst-tick spikes (floodfill 16.8 ms, static 33.6 ms) are impact-tick
  allocation/contact bursts, not sustained — median is the honest steady cost.

## Honest secondary finding — fragmentation is resolution-dependent
At a FIXED material fracture bar (`love×threshold`), coarser lattices fracture
far more readily (higher per-bond strife): the same collapse yields **83 / 6 / 2**
fragments at N=1024 / 2000 / 3456. Peak strife exceeds the bar at every scale,
but at fine resolution only a handful of bonds cross it. N=1024 gives a full
rubble field; the finer towers fold with few fractures. A learned fracture
model would have to be resolution-aware, not a fixed per-bond threshold.

## Metastability (honest gap, measured, NOT hidden)
A tall anchored bonded tower holds its height rock-solid for ~1.3 s (~80 ticks)
then slowly buckles/creeps downward — Euler instability of a slender heavy
column the single-iteration Gauss-Seidel XPBD cannot resist indefinitely (more
iterations do NOT fix it — verified 1/4/10 its). The scenario settles to the
STANDING PLATEAU (~40 ticks, top ≈23.5 m of 24 m) and topples from there; the
"rest stays at rest" ordeal checks a BOUNDED 60-tick window over that plateau
plus a ≤20% height-retention bound. Sustained static equilibrium of tall stacks
is itself a candidate for a learned corrector.

## Ordeals (5, all green; suite 368 → 373)
- `ordeal_building_at_rest_stays_at_rest` — standing tower held 60 ticks at 2
  scales: whole, height ≥80%, max speed < 10× a known-stable reference floor.
- `ordeal_building_collapse_replay_byte_identical` — full settle+topple+collapse
  folds to identical per-tick state hashes AND fragment partition across 2 runs.
- `ordeal_building_fragments_do_not_interpenetrate` — post-collapse, no two
  cross-fragment particles overlap beyond 10× a measured STACK-overlap floor
  (derived-gate pattern from `vi2_break_ordeals`, but a `depth`-deep column, not
  a pair — a rubble pile stacks more contact residual than two bodies).
- `ordeal_step_profiled_matches_step` — the profiled path is bit-identical to
  `step` (the timers do not perturb the physics; the numbers are trustworthy).
- `ordeal_per_tick_cost_recorded` — the breakdown is populated & self-consistent
  (total ≥ Σphases; body-pair-checks = analytic `(k choose 2)×it×sub`). RECORDED,
  never gated (a perf gate at unknown scale would be a guess).

## Proof (read with own eyes)
`proof/pscale-standing.png` — a tall intact warm-stone multi-storey tower,
storey grid visible, on the ground under a blue sky (1 fragment).
`proof/pscale-collapse.png` — mid-collapse: the tower folded to a stepped,
offset progressive-collapse mound (fragments breaking free, ~55% height).
`proof/pscale-rubble.png` — a spread debris field of ~110 distinct hue-coded
chunks (mottled yellow-green with pink/blue/tan pieces) — many separate blocks,
NOT a smeared soup. It is a flat pancake-collapse debris field (the real failure
mode of a shove-toppled anchored tower), not a dramatic 3-D heap.

## NEURAL VERDICT (data → recommendation; Pascal rules)
- Dominant phase at every N, and the only super-linear one: the **O(k²)
  body-vs-body contact broad-phase** (82–85% of the tick, growing quadratically).
  Its cost is iteration overhead of the brute-force pair loop, NOT the contact
  math — it pays full price even when almost nothing collides.
- Therefore the FIRST thing to replace is **NOT** a subspace/fracture net —
  those phases (constraint solve, floodfill, fracture) are already cheap and
  linear. The first, highest-leverage replacement is the **CONTACT BROAD-PHASE**:
  either a classic spatial hash/BVH broad-phase (exact, O(k) expected — the
  cheapest honest win, ordeal-gatable) OR a learned neighbour/contact predictor
  that skips the same-cluster and far-apart pairs the current loop wastes 90%+
  of its time visiting. Subspace dynamics and learned fracture-stress are
  second-order here: they target phases that are not the bottleneck at building
  scale.
- Recommendation: prove a spatial broad-phase first (exact, bounded, ordeal-
  proven), THEN weigh a learned contact model only for the residual narrow-phase.
  Learned fracture is worth revisiting only alongside the resolution-dependence
  finding above.

## Honest gaps carried
- Metastable static equilibrium (tall-stack creep) — documented, bounded, not
  solved.
- Fragmentation resolution-dependence at a fixed threshold — documented.
- Rubble is a flat pancake field, not a 3-D heap (the collapse mechanism's real
  behaviour), and fragment hues cluster toward yellow-green (golden-ratio start).
- The ECS `body` sigil has no anchor field (physics-recon open) — the anchored
  scenario lives at the solver level; the render bypasses ECS to draw it.
- Worst-tick numbers carry allocation/GC noise; median is the load-bearing stat.
- 10k particles reachable as a cost datum only (minutes/collapse), not playable.
