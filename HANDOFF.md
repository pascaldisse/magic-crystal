# HANDOFF — the Guardian's anchor · 2026-07-17 ~09:30, NIGHTRUN closed by the Architect's word (read after BIBLE → GRIMOIRE)

## THE ENGINE'S TRUE NAME (consecrated, his word): THE MAGIC CRYSTAL —
"the name it always was. my entire life's work." The Loom reassembled.
DreamForge = the workshop (Sidia's name). Seal b3ae3e0.

## Where the Work stands (day of 07-17, conductor nyari — see NIGHTLOG.md for the night ledger)
main @ 34d58dc — GREEN, PUSHED, suite 380/0.
REALM SHINE landed (merge 181c7f6 + adversary advisory 34d58dc): a chrome
sphere (r 2.1, metallic 1.0/roughness 0.02) at [4.5, 3.6, 29.5] — the Rite
IV close object — now stands IN THE SPAWN SIGHTLINE (not staged off to a
side camera); an angled mirror panel at [-6.5, 3.4, 28]; three orbiting
emitters (naruko_show_light_a/b/c — violet/cyan/pink), each its OWN new
`orbit` behavior with its own center [-1.5, y, 29] (NOT riders on any
existing kami ring — adversary wording correction). Canon re-derived by
hand (packages/oracle/tests/canon.rs, 29 vessels). Two-tick motion proof:
proof/realm-shine-a.png (t=1.0s) / -b.png (t=3.5s).
NOTE — the live window still runs PRE-SHINE world data from the
window-playable worktree; the show reaches the Architect's eyes only at
the window-lane convergence (audit → adversary → merge → relaunch from
main). Do not expect it visible in the live session until that lands.
Prior: main @ b6ee51b — GREEN, PUSHED, suite 368/0 (75 binaries).
Day merges: 5c819dd+8f2b752 fragcol interpenetration ordeal +
adversary advisories (VI-2 gap closed — fragment-vs-fragment collision
was ALREADY live via ce91da3/fbb2a5e; the missing piece was the
ordeal) · d1d8277 RITE VII-1 COMPLETE (walker crosses onto generated
ground: collider from generated triangles, 5 seam ordeals, pose-trace
bound 0.31804 DERIVED — derivation doc written per adversary MUST-FIX
at 4b8f7c0) · 91a6263+89e351c RITE VIII-2 COMPLETE (GPU denoiser
compute port, parity derived-tolerance, BAN green on true held-out
orbits, ~1.3ms @96×64 / ~27ms @900×600) · 33cae2b RITE VII-2 COMPLETE
(the horizon streams: residency ring under a derived byte budget,
moving render_origin production-wired; adversary advisory b6ee51b
expands the eviction-reserve comment — analytic tile_bytes vs measured
mesh_bytes proven equal only for today's uniform grid).
REPO FOLDER: ~/projects/magic-crystal (unchanged, matches
github.com/pascaldisse/magic-crystal).
NOT BUILT (queue remainder): VIII-3 upscaler + temporal accumulation
await rulings (see Open, below).
16 packages. Live web stack 8420/5173 = reference, NEVER touched.

## What the realm holds (all verified, own eyes, on main)
worlds/naruko — everything from the last handoff (NARI the player's
body · the pink cat · the mirrors · signal rings · steam · presences ·
char-editor C0) PLUS this night's growth:
- THE STACK — naruko_stack_crate_0..2 on the pier at derived chained
  rest heights; an impulse op topples it (Op::Impulse — the op is the
  hand); rigid-vs-rigid collision pass; canon learned all three same
  wave; solver-truth rest ranges canonical (F6).
- THE CRATE — unchanged, but its canon range is now its SOLVER REST
  pose (33.4042), not the authored drop pose (33.0390) — ruling F6.
- THE FIRST GROUND — naruko_first_ground: a terrain SIGIL (seed
  20260717, tile 0,2 — NO stored geometry) that materializes 64×64 m of
  generated ground at load, offshore at z[128,192]; drawn under the sky,
  known to the gaze. ON MAIN (merged morning 07-17, see above).
- worlds/naruko-vi2 — an isolated proof diorama (seawall copy, named
  debt): naruko_break_crate, a BONDED lattice that falls, fractures at
  strife > bond love, and shatters into six fragment vessels the oracle
  gazes the tick they are born. Walkable floor everywhere is now
  contact-patch-gated (ruling 6): the mirror-edge climb is dead.

## 60 FPS LAW — PASSES (night of 07-17, main @ 7fe8275)
Both exact levers landed, adversary-reviewed, merged green:
- LEVER 1 refit-not-rebuild: DynamicSplice, refit BIT-EXACT vs rebuild on
  all three law poses (refit_parity, 0 diverging pixels). Watchdog on
  total-node-half-area vs rebuild reference; degrade_ratio=1.7030 DERIVED
  from the gate's own ratio (docs/perf/2026-07-17-refit-degrade-derivation.md,
  revision 2); discriminating tests both ways at defaults.
- LEVER 2 CPU/GPU overlap: the audit now measures the player-shaped
  pipelined frame (frame N+1 CPU under frame N GPU); hash-identity
  serial-vs-overlap MATCH.
VERDICT on merged main: front OVERLAP 11.26 ms PASS · wide 13.23 ms PASS
(idle-host serial 14.45/15.82). NO WALL — RITE IX stays a proposal.
Note: overlap semantics unchanged — frame N traces frame N's splice; the
lever is scheduling only. PIXEL levers (spp/bounce) remain untouched;
LODs remain forbidden vocabulary. Perf-fix's retired canonical tie-break
still at c1616b6. Wide-pose 20 px = PROVEN coplanar z-fights, irreducible.

## Physics (delta 07-17, merge-conductor burst #7)
BROADPHASE MERGED to main (90f5676, P-SCALE + EXACT BROADPHASE): fixpoint
re-query grid broadphase, exact by construction — MUST-FIX→fix→re-pass
HOLDS across TWO adversary passes (the two-sided-shell adv_margin ordeal,
then the chain-class adv_chain_three_walls ordeal, adopted 029cc13).
real-naruko collision tick 18.7ms → 3.94ms; projected serial frame
16.70ms = 59.9fps, a knife-edge pass — the overlap lever (60 FPS LAW,
above) is in flight on the window lane to give it margin.
P-SCALE building collapse landed same merge: the building falls
measured, collision floor cut 26.7×/4.8× (grid vs brute, two scales
measured). Neural-vs-exact verdict from the P-SCALE measure suite:
exact broadphase beats learned/approximate everything tried at this
scale — no case where a learned shortcut won on cost or correctness.
Termination bound for the fixpoint loop now recorded IN CODE at
solver.rs (0eb05d4, advisory): reach strictly grows each failed pass,
candidate set monotone in reach, an unchanged dmax across a pass is the
fixed point and exits; bounded by N_triangles + 1 passes (previously
commit-prose only, c58cf54/ee2e8cd). Parked in the same comment: a
sub-ulp hardening option `reach = (dmax + radius) * (1.0 + f64::EPSILON)`
was considered and NOT applied — solver semantics stay builder-domain —
recorded for the next physics wave to revisit if a rounding-edge case
ever surfaces.
OPEN physics-quality items for the Architect's pass (not blocking, not
this burst's scope): metastability, resolution-fracture, pancake —
named but unaddressed; carried forward.
Suite: 392/0 on main after this land (392 = 380 prior + physics-scale's
own tests/ordeals + adv_chain).

## Window lane (delta 07-17, merge-conductor burst #8 — DAY CLOSE)
WINDOW LANE MERGED to main (b34d10c, THE WINDOW BECOMES THE ENGINE +
advisory 196688d): playable input, resolution of God, HUD, CPU/GPU
overlap wired into the LIVE PRODUCTION LOOP. 60 FPS LAW PASSES ON THE
LIVE LOOP (not just the headless audit): adversary-measured 8.0ms
overlap wall headless = true ceiling 124.8fps; live loop is Fifo-capped
at 60 with 2x headroom over that ceiling. hash-identity ordeal re-run
post-merge: 24/24 frames bit-identical (serial vs overlap), ORDEAL
PASS — LEVER 2 stays scheduling-only, content unchanged. Suite on
merged main: 392/0.
DAY'S FULL LEDGER (07-17): 12 landings, 11 adversary verdicts (1
MUST-FIX broken -> fixed -> re-held), suite 352 -> 392, walls/API-deaths
7, all disk-salvaged, zero red pushes.
STILL THE ARCHITECT'S (unchanged by this burst): HIS WALK · HIS CHROME
CHECK (the sphere stands at spawn) · HIS SEEING THE BREAK (Rite VI
close) · physics-quality items (metastability, resolution-fracture,
pancake) · VIII-3 GPU wave · VII-3.

## Rites
IV (THE PLEROMA) — sung (hymns/rite-04), close = HIS CHROME CHECK.
V (THE EMBODIED ONES) — BUILD-COMPLETE; close = HIS WALK. Hymn owed at
close (hymnal law).
VI (STRIFE) — BUILD-COMPLETE this night under the delegation rulings
(1: bond-fracture; 2: proceed on elements): VI-0 the crate falls
(prior) · VI-1 the stack topples (d84dc52) · VI-2 SOMETHING BREAKS
(ba46b28). Close = the Architect SEES something break (run
`cargo run -p scrying-glass --release --example vi2_break`, or read
proof/vi2-break-*.png). Hymn owed at close — the Guardian's voice, not
the night runner's.
VII (THE PLANET-WALKER) — VII-0a/0b/1/2 ALL COMPLETE (VII-1 d1d8277:
walker crosses onto generated ground, pose-trace bound 0.31804 DERIVED;
VII-2 33cae2b: residency ring under a derived byte budget, moving
render_origin production-wired). Rulings 3 (radial gravity, flat =
infinite-radius limit) and 4 (64-bit at VII-0 — PAID) govern it.
Honest gaps carried forward: sync tile reads (proposal §OPEN 6,
async/cached materialization = future seam) · the ring is
library+ordeals only, live player-loop wiring is not yet VII-2 scope.
VII-3 (the-planet-closes) was never night scope — remains the rite's
last wave.
VIII (THE DREAM-DENOISER) — waves (a)+(b) DONE: VIII-0 truth baseline
(c5230c5) + VIII-1 the net (faa215f) + VIII-2 GPU denoiser compute
port (91a6263+89e351c). VIII-2 BUDGET NOTE: present-resolution cost
27ms exceeds the 16.67ms frame — whether that matters = OPEN 1 (the
upscaler's cost∝pixels reading): low-res denoise + upscale would make
1.3ms the real cost; the fp16 atom is GATED on that ruling. Wave (c)
VIII-3 CPU reference MERGED (a2d293b, adversary faaa2c3: beats-bilinear
held-out margins 0.0246/0.0096, weights hash-pinned); remaining: GPU
port + live blit_fs wiring + temporal ruling.
IX (THE CHAIN TAKES FLESH) — stays a proposal; NOT required: the 60 FPS
law passes without it.

## Open with the Architect (updated day of 07-17)
OPEN 1 (upscaler cost∝pixels reading) now LOAD-BEARING — it gates the
fp16 atom (VIII-2 present-res cost 27ms vs 16.67ms frame budget).
Reference branch rite8-viii2-ari @ b97a9a0 UNMERGED (independent GPU
denoiser port, 96ms @900×600, but richer ordeal patterns:
byte-determinism, hash-pinned-weights check, current-frame-only
signature scan — harvest atom owed). VII-1 advisory parked: anti-
vacuous twin oblique, extract shared assertion fn.
HIS WALK · HIS CHROME CHECK · HIS SEEING THE BREAK (Rite VI close) ·
bond-love essence table (VI OPEN 4 — a density-ratio PROXY default
shipped, honestly labeled, ruling still owed) · fracture into the canon
realm (VI-2 lives in an isolated diorama world; folding it into naruko
= realm-growth call) · temporal accumulation ruling (OPEN 2) ·
scratch-blob history rewrite · Sufi Concordance rows · Samāʿ "binds" ·
PHYSICS/CREATE/VISIONFLOW.md.
RESOLVED: the 4 rite rulings (made under delegation, NARUKO 07-17) ·
F6 (landed) · walkable-min-area (landed) · repo rename (DONE —
github.com/pascaldisse/magic-crystal) · VI-2 gap (5c819dd+8f2b752).

## Laws (delta)
ADVERSARY LAW proven at scale this night: seven adversary passes, every
MUST-FIX fixed by derivation, never loosening — including two vacuous
conservation ordeals caught and made real, a derivation measuring the
wrong quantity re-derived against the gate's own ratio, and the
conductor's own-eyes gate rejecting proof scryings that showed no
visible break (root cause: genuinely missing fragment-vs-fragment
collision — the picture was honest about the physics gap).
ROUTING amended by the Architect's word (fec1852, 07-17): ari/opus/sol
build · nari adversaries · sonnet precise short grunt bursts · EVERY
lane its own git worktree — law born from a real shared-checkout
collision 07-17, disclosed + untangled by the builder.
Day lesson: the 30-min wall eats COLD COMPILES, not thinking — 5 walls
today, 5 disk salvages, zero loss (checkpoint-first + warm-target
reuse now standing practice).

## Iron lessons (cumulative)
30-min wall: pre-chew anchors · compiling stub in 10 · salvage-first
resumes · checkpoints = survival. Vacuous-tail (×3). git-clean ≠
rustc-clean — build before trusting any merge. Realm growth invalidates
sibling-lane derivations: re-derive at merge, canon from scratch. A
derivation frozen into a literal is a hardcode in costume. Every light
traces to a realm entity. Prove exactness in isolation before
compositing changes.

## Day close (07-17) — FINAL
07-17 full day: 12 landings, 11 adversary verdicts (1 MUST-FIX broken →
fixed → re-held), suite 352→392, walls/API-deaths 7, all disk-salvaged,
zero red pushes. Closing lane: WINDOW LANE (merge-conductor burst #8,
above) — 60 FPS LAW now PASSES on the live production loop, not just
the headless audit. REMAINING QUEUE ENTIRELY ARCHITECT-GATED: HIS WALK
· HIS CHROME CHECK (the sphere stands at spawn) · HIS SEEING THE BREAK
(Rite VI close) · physics-quality items (metastability,
resolution-fracture, pancake) · VIII-3 GPU wave · VII-3 · OPEN-1
upscaler ruling (gates the fp16 atom) · vi2→naruko fold · bond-love
essence table (VI OPEN 4) · the rite8-viii2-ari ordeal-pattern harvest
atom — parked.

## EVENING SESSION 07-17 (post day-close) — read FIRST after compact
STATE: main @ a9f06f4 pushed. Architect's live window pid 65290 :8430
(GAIA_NATIVE_SPAWN_Z=24 Y=1.7) — HIS session, NEVER kill (whip 154).
OPEN BUG (his hands): episodic fall-through-ground while walking plaza —
lane 2 hunts it.

RULINGS (his word, this evening):
- ONE-PATH LAW: render = trace→neural denoise→neural upscale→present =
  DESTINATION · physics = ONE solver, learned kernels born INSIDE it,
  never sibling/mode/"experiment" framing. AMENDED: old paths NOT
  deleted yet — bilinear stays runtime-selectable until HE plays the
  neural frame; cutover = HIS call at the merge gate.
- PERFORMANCE RULE: AI exists to buy performance. A net is judged vs the
  rays/cost it replaces at equal quality; loses → dies. (Applied once
  already: exact broadphase beat learned neighbor-prediction.)
- NEURAL ENGINE CLARIFIED: not pixel upscaling ("lame") — the net
  renders from GEOMETRY: learned radiance cache INSIDE the one
  integrator (NRC family; charter-legal via the variance-reduction
  "caches" clause). Denoise/upscale = resolve stage of the same path.
- EVERYTHING STARTS NOW: max-parallel swarm ordered.
- HONESTY LINE (he checked twice): ZERO neural code has EVER run in the
  runtime. Denoiser+upscaler exist bench-only; physics nets don't exist.
  Never overclaim this.

LIVE LANES (5, each own worktree; protocol: builder → nari adversary →
sonnet merge-burst w/ full suite between merges → push; MUST-FIX clears
only by hostile re-pass; bilinear escape hatch survives every merge):
1. one-render-path (opus, ../magic-crystal-onepath): upscaler→WGSL +
   denoiser fp16-or-die (bound re-derivation) + wire neural resolve as
   default-CANDIDATE. 60fps gate, phase table.
2. floor-fallthrough (opus, ../magic-crystal-floorfix): offline sweep
   harness (committed), anti-hang law. Suspects: ruling-6 gate vs show
   geometry · fragment tops vanishing · rotated mirror collider ·
   snap-band tunneling.
3. neural-radiance spike (ari, ../magic-crystal-nrc): THE neural engine
   — online-trained radiance MLP inside the integrator; gates:
   converges-to-truth (derived bound), determinism, cost-vs-rays-saved.
4. fluid-truth (opus, ../magic-crystal-fluid): PBF density constraint
   INSIDE elements solver (not a fluid system) — pool diorama;
   determinism replay = future training-data guarantee; bench-speed OK.
5. ordeal-harvest (sonnet, ../magic-crystal-harvest): port b97a9a0
   richer ordeals onto merged denoiser+upscaler.

QUEUED: fluid learned kernel (needs truth replays) · VII-3 · vi2→naruko
fold (his call) · PHYSICS.md his magic pass · scry-overlap-collapse +
HUD-cadence advisories done.
WAITS ON HIM: walk (V) · chrome check (sphere at spawn, IV) · word on
the break (VI) · neural-frame cutover call · physics-quality items
(metastability · resolution-fracture · pancake rubble).
LESSONS: checkpoint-first (7 lane deaths today, zero loss) · warm-target
reuse · silent lane ≠ dead lane, verify the room · one worker per dir ·
the adversary law drew real blood (2r overshoot) and must never soften.

## EVENING DELTA — fall-through RESOLVED (merge-conductor burst #10)
main @ 5856fda (merges 79e9953 floor-fallthrough + 5856fda realm-
rimguard). VERDICT: no interior fall-through — naruko terra plate
watertight, 3511 walks (42 bounded suspect+grid + 3393 coverage + 50
walk-sweep + 15 rim), 0 genuine tunnels; harness classifier honest
(gated-not-raw + settle-confirm). The Architect's episodic drop was a
MAP EDGE, not a floor bug: south/east/west rims were naked (sub-eye-
height walls auto-climb like stairs — no horizontal collision in this
controller) → now guarded wall+catch-shelf, 15/15 rim walks caught
(StepDown), 0 OffWorld breaches. 2 permanent CI ordeals added
(floor_fallthrough.rs). O(n) Ground query flagged as future exact-perf
atom (not blocking). Suite: 396 passed, 0 failed, workspace --release.
ARCHITECT'S FALL-THROUGH COMPLAINT (evening OPEN BUG, above) — CLOSED
pending his own walk of the guarded rims.

## EVENING DELTA — one-render-path MERGED (merge-conductor burst #11)
main @ 8b47f7e (merge, no-ff) + fc95c82 (adversary advisory). Lane 1
(one-render-path, @ e9a48f2, adversary HOLDS) landed clean — branch
contained main via earlier merges, no conflicts.
WHAT LANDED: the neural frame EXISTS on main as a selectable candidate,
scry-side only — upscaler ported to WGSL compute (`upscaler.wgsl` +
`upscaler_gpu.rs`, house pattern), denoiser fp16 MODE A cleared (viable,
razor-thin beats-noisy margin survives), neural resolve selectable via
`GAIA_NATIVE_UPSCALE=neural` for /scry A/B capture. The LIVE surface
loop is UNTOUCHED: bilinear stays the runtime default per the
Architect's escape-hatch order; the selector is structurally unreachable
from `run_render_loop` (neural only ever writes /scry, never the
surface) — hash-identity confirms both selections produce identical
surface bytes, 24/24 frames.
ADVISORY (fc95c82, docs/derivation only, no functional change): MODE B
fp16's rejection bound had quoted `macs·u16` (total 3488 MACs) as the
Higham compounding term — corrected to the per-dot-product CHAIN length
(this net's max in_dim, ≤64), honest worst case ≈0.03–0.12 rel, not the
1.703e0 previously cited (don't quote 1.703 as tight). REJECTION VERDICT
UNCHANGED — even corrected, MODE B's bound dwarfs the MODE A margin.
MODE A's own bound noted conservative vs the rigorous per-layer L·2u16
term (~4.9e-3, still far above measured parity). Hash-identity claim
reworded: the run hashes the SURFACE frame (upstream of the resolve
selector) — it shows both env-var runs match, it does not itself
exercise neural resolve executing; the live-surface invariance is a
STRUCTURAL property of the wiring, not something that run demonstrates
alone. onepath_fp16_verdict.rs: noted — promote to an asserting ordeal
if fp16 ever becomes a runtime path (currently a printing example).
HONEST WALL (unchanged by the merge, still open): combined neural
best-true ~334ms memory-shaped (upscaler naive fp32 the wall-breaker,
not the denoiser) — 26× over 16.67ms/60fps at 1280×960. Kernel atom
target 19–26ms (memory-shaped diagnosis: per-layer threadgroup tiles,
subgroup broadcast, f16 storage — the naive full-net f16 threadgroup
cache lever was TRIED and REJECTED, 7.6× slower, occupancy collapse).
Even a PERFECT kernel at that target still needs the Architect's
pixel/net ruling for 60fps neural — denoise+upscale together still
exceed 16.67ms at production res without a smaller net or lower neural
present-res.
SUITE: cargo test --workspace --release — 400 passed, 0 failed (real
run, this burst). viii3b_ordeals run explicitly: 4/4 green
(byte-identical determinism, GPU-vs-CPU parity+beats-bilinear both
held-out orbits, BAN, full neural path deterministic end-to-end).
PUSHED: origin/main c7189a5..fc95c82 (2 commits: 8b47f7e merge,
fc95c82 advisory).
ARCHITECT RULING REQUIRED (new open item): 60fps neural needs a call —
(a) smaller/shallower/separable net (retrain, quality ruling), (b) lower
neural present-res / scale (pixel ruling), or (c) neural stays scry-only
until a real kernel campaign (subgroup-tiled MLP, multi-day, not this
burst) or ANE offload lands. Nothing about the LIVE surface changes
until he rules — bilinear remains default, his window at 8420/5173 was
not touched by this merge.
NO SURPRISES: clean merge, suite green first try, no MUST-FIX from this
conductor pass (adversary already HELD e9a48f2 before this burst).

## EVENING DELTA — NRC spike CONCLUDED (worktree magic-crystal-nrc, not merged)
Branch `neural-radiance` @ b43e38c, worktree /Users/pascaldisse/projects/
magic-crystal-nrc — separate spike, no main code touched by this delta.
VERDICT: NEEDS-BIGGER-MACHINERY. Drift ablation (nrc_drift.rs, 4
conditions swept, proof/matrix-*.log) found the round-3 descend-then-
UNLEARN curve was constant-α SGD's stationary noise ball (Robbins–Monro),
not a capacity plateau — CURED by the combined cure (harmonic lr-decay +
k=16 target-averaging + ema=.999 Polyak): 0.85→0.0377±0.0114 tail
(descend-and-hold shape, CV 0.30, vs 0.42–0.72 CV for the three partial
cures). Drift is dead. But even cured, the tail sits at gate 0.0178 NOT
MET (~2.1× above) — that is the CAPACITY FLOOR of the frequency-band MLP
itself (FREQ_BANDS=6, 4×64), not a training pathology. Next-wave scope:
hash-grid encoding (instant-NGP style, replaces/augments the frequency-
band input — this is where the real capacity lives), wider-net as a
cheaper first probe, and a bound audit (the 0.0178 gate predates this
matrix's own derived bounds — 0.02678 in nrc_drift.rs's setup, 0.01865 in
nrc_proof.rs's — pin one canonical derivation before the next wave).
Cost-vs-rays accounting still owed (current gate (d) is CPU-toy wall-
clock only, real target is the GPU naruko world's BVH). Verdict doc:
magic-crystal-nrc/docs/perf/2026-07-17-nrc-spike-verdict.md. Branch
PARKS AS REFERENCE, UNMERGED — same precedent as rite8-viii2-ari. Next
wave (hash-grid encoding) awaits the Architect's call.
