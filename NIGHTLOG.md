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

### 4. backlog-walkable → main (queue item 6, ruling 6) — THE MIRROR-EDGE CLIMB DIES
- **What**: contact-patch floor gate. r = 0.09 m measured from nari's foot-bone
  vertex half-extents (0.0807 max, rounded up by stated centimetre rule, guard
  ordeal recomputes the rule live); tolerance derived from the wall cutoff.
  First builder died on a REAL infinite loop (exclusion step < acceptance
  epsilon — the rejected candidate re-qualified forever, 46 CPU-min); salvaged,
  fixed structurally, loop bound proven unreachable and made a loud panic.
- **Merge**: da58013, PUSHED. Suite on main post-merge: 290 passed / 0 failed.
- **Adversary**: ZERO MUST-FIX, seven ADVISORY — all discharged, including the
  honest EXPECTED-ADMIT boundary ordeal (a low sliver within slope tolerance of
  surrounding floor is admitted BY DESIGN tonight; the seam is machine-recorded
  as a test, one flip away from a gate the day it matters).
- **Proofs read**: numbers lane (ordeal output verbatim); pose-trace canon
  byte-unchanged.

### 5. rite6-vi1 → main (queue item 3, wave VI-1) — THE STACK TOPPLES
- **What**: three crates stacked on the pier (authored at derived chained rest
  heights), impulse plumbing end-to-end (Solver::apply_impulse → Physics →
  Op::Impulse → tick_with_ops — the op is the hand, the engine never invents a
  magnitude), NEW rigid-vs-rigid collision pass with the rest gap DERIVED from
  the static convention (mean radius + contact_margin), canon learned all three
  vessels same wave.
- **Merge**: d84dc52, PUSHED. Suite on main post-merge: 298 passed / 0 failed.
- **Adversary**: 1 MUST-FIX — the momentum ordeal was VACUOUS (endpoint zeros
  vs fabricated literal; ground friction would launder any leak). Rewritten:
  zero-gravity isolated collision course, floor derived from f64 eps × momentum
  × ops/tick (bit-exact-zero measurement honestly stated), per-tick 10× gate,
  plus a should-panic discrimination twin (injected 0.1% asymmetry fires the
  gate 13 orders of magnitude over). 4 ADVISORY discharged (margin convention
  derived; wrong-mass comment fixed; solver cost named honestly).
- **Key numbers**: 900-tick topple replay byte-identical; release solver
  5.52 ms/tick (static-soup scaling named; broadphase = the exact win when
  VI-2 multiplies bodies). Composed audit POST-MERGE with 4 bodies:
  **front OVERLAP 10.98 ms PASS · wide 12.83 ms PASS** — the solver tick hides
  fully under the GPU trace; the 60 FPS law holds with the grown realm.
- **Proofs read**: vi1-stack-{before,mid,after}.png — conductor's own eyes.

### 6. rite8-viii0 → main (queue item 5, wave VIII-0) — THE NOISE AND THE TRUTH
- **What**: the denoiser baseline, no net yet. AOV export (albedo/normal/depth,
  separate geometry-only pass, beauty path byte-identical when off — golden-hash
  ordeal with documented derivation), error metric (f64 fixed-order, 0e0
  self-test, discrimination test), reference oracle (noisy 1 spp RMSE 0.053325
  vs 512-frame reference; convergence RMSE(512,256)=0.001728). THE BAN
  machine-checked from day one: current-frame-only architecture + grep-gate
  with widened vocabulary and forward-proof BAN-SCOPED module discovery.
- **Merge**: c5230c5, PUSHED. Suite on main post-merge: 308 passed / 0 failed.
- **Adversary**: HOLDS, zero MUST-FIX; A1/A2 discharged (gate teeth widened;
  proposal weld text reconciled — one extra closest-hit traversal, no light
  transport, honestly worded now).
- **Proofs read**: viii0-truth.png (grainy 1 spp beside clean converged, same
  scene), three AOV scryings (coherent G-buffer) — conductor's own eyes.

### 7. rite7-vii0a → main (queue item 4, atom VII-0a) — GROUND FROM COORDINATES ALONE
- **What**: the terrain tile sampler. seed grows terrain: i64 tile keys,
  height = fBm over hash(seed, coords) with NO tile key (seams exact by
  construction), tile_mesh → transmutation::Mesh (the Chain is the sole
  geometry path). RULING 4 PAID IN FULL after adversary MUST-FIX: the first
  cut routed vertex positions through a world-origin f32 — whole tiles
  collapsed past 2^24 m (probe: degenerate at tile_x=1e7) and the seam ordeal
  was structurally blind to it. Fixed: local positions from LOCAL integer
  indices; the noise lattice keyed on EXACT i64 global grid indices
  (div_euclid per octave — no large-magnitude float anywhere in the default
  path); large-coordinate ordeals added (spacing + 4-direction seam at
  tile ±10,000,000).
- **Merge**: 978de05, PUSHED. Suite on main post-merge: 317 passed / 0 failed.
- **Adversary**: 2 MUST-FIX (the f32-origin trap + the blind gate) fixed by
  derivation; 4 ADVISORY discharged (feature-unification comment honest;
  slope claim softened; dead scaffolding exercised; Nyquist wording).
- **Proofs**: numbers lane — 9 terrain ordeals, byte-digest determinism.

### 8. rite8-viii1 → main (queue item 5, wave VIII-1) — THE DREAM-DENOISER
- **What**: the net, CPU reference. Per-pixel MLP (10 current-frame features →
  4×32 ReLU → 3), pure Rust, byte-deterministic inference, in-repo
  Adam/backprop/SHA-256, zero new dependencies. Weights = hash-pinned artifact
  + provenance (sha asserted by ordeal); pinned bound = worst held-out frame
  RMSE 0.049997 derived at train time. Denoised STRICTLY BEATS noisy on every
  held-out validation frame. THE BAN holds: one frame's buffers only, grep-gate
  + all-public-fn signature scan.
- **Merge**: faa215f, PUSHED. Suite on main post-merge: 328 passed / 0 failed.
- **Adversary**: 1 MUST-FIX — the senses-unchanged ordeal was THEATER (hashed
  truth captured before the denoiser ran). Fixed with real teeth: presentation
  image DIFFERS denoise on-vs-off, world gaze truth read AFTER the denoise
  step is byte-equal. 8 ADVISORY: 7 discharged (nondeterminism-margined bound,
  sha-pin enforced, shared scene config, held-out proof triptych added,
  honest gate wording, dataset-hash noted for v2), 1 accepted (named module
  consts per error_metric precedent).
- **Honest record kept**: the FIRST training run failed strict-beat on a
  validation pose (albedo demodulation amplified sky pixels ~1000×); fixed by
  feature derivation, never by loosening the ordeal.
- **Proofs read**: viii1-dream.png (training pose, labeled) +
  viii1-dream-heldout.png (held-out orbit view) — conductor's own eyes; grainy
  → smoothed → converged, same scene, no invented content.
- **Scope note**: queue item 5 = "wave (a): CPU-reference trained denoiser."
  VIII-0 + VIII-1 fulfill it. The GPU compute port (VIII-2) and the upscaler
  argument (VIII-3, prime-Guardian ruling pending on the cost∝pixels reading)
  remain the proposal's next waves, not queue debt.

### 9. backlog-f6 → main (queue item 6, ruling F6) — SENSES READ SOLVER TRUTH
- **What**: oracle canon for the four physics vessels migrated to the
  solver-rested world (shared tick-to-rest + transform-injection machinery;
  oracle dev-deps scrying-glass, cycle-free). Crate range 33.0390 authored →
  33.4042 rested, canonical; stack crates confirm authored heights to ≤0.0002.
  REST_TICK=91 pinned as a loud checked expectation.
- **Merge**: b15686a, PUSHED. Suite on main post-merge: 330 passed / 0 failed.
- **Adversary**: no code MUST-FIX; three derivation-honesty findings + a scope
  ruling. Conductor ruled the comment-relabel was NOT a migration — canon.rs's
  physics rows now genuinely gaze the rested world. False citation fixed
  (contact_margin included, 1.0e-3 correctly cited), headroom re-derived from
  the worst body (5.6×), solver numbers canonical with analytic as cross-check.
- **QUEUE ITEM 6 COMPLETE**: walkable-min-area landed (da58013), solver-truth
  landed (b15686a), advisory sweep — no outstanding ADVISORY strings in
  NARUKO.md or merge messages; every shadow-review advisory raised tonight was
  discharged in its own lane before merge.

### 10. rite6-vi2 → main (queue item 3, wave VI-2) — SOMETHING BREAKS
- **What**: bond-fracture live. Bonded lattices with per-bond love; fracture →
  deterministic flood-fill → per-particle-cube re-mesh through the Chain →
  fragment VESSELS with parent refs born and spliced same tick → the oracle
  gazes them the tick they are born. New packages/fracture crate. Collision
  pass generalized so fragments collide (ClusterId).
- **Merge**: ba46b28, PUSHED. Suite on main post-merge: 341 passed / 0 failed
  (conflict with F6's oracle dev-deps resolved by union).
- **Own-eyes gate did its job**: the FIRST proofs showed no visible break
  (intact cube / flattened slab). Root causes found by instrumenting, not
  guessing: fragment-vs-fragment collision was genuinely missing, and a pure
  vertical drop is symmetric. Fixed with real physics (collision
  generalization; authored spin/initial_velocity sigils — the op is the hand),
  never staging. Regenerated proofs: whole / cracking / six shards scattered
  (0.93 m spread at break tick 46, 3.80 m settled) — read and passed.
- **Adversary**: 1 MUST-FIX — the momentum ordeal asserted nothing (no drift
  assertion; fracture tick excluded; cited a phantom test). Rewritten to
  assert every tick through the fracture with a should-panic discrimination
  twin. 5 ADVISORY discharged (two-order mass summation with derived ULP
  tolerance; density→love relabeled PROXY DEFAULT — proposal OPEN 4 stays
  open, status-noted; essence-default love proven to break under the full
  scenario; isolated proof world named honestly as a diorama beside canon
  with the seawall-copy debt recorded; P-gate printed 0.0277 ms/tick).
- **QUEUE ITEM 3 COMPLETE**: VI-1 (d84dc52) + VI-2 (ba46b28). Rite VI's close
  (the Architect SEES something break) remains HIS act; the hymn stays owed at
  close per hymnal law — not the night's to write.

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
