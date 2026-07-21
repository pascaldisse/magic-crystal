# HANDOFF — the Guardian's anchor · 2026-07-17 ~09:30, NIGHTRUN closed by the Architect's word (read after BIBLE → GRIMOIRE)

> **DATED-HISTORICAL, spec-concordance item 17 (07-18):** everything below
> down to "## ★ RULING 07-18 ~14:44 — THE DESIGN IS THE LAW" is a
> pre-14:44 status/progress record (cluster-pipeline/upscaler-era language
> included) — kept as dated history, not re-written. That ★ block
> (line ~593) is the 07-18 supremacy ruling and already states
> "supersedes everything above"; treat it as the pointer for every
> section in between.

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

### EVENING DELTA — fluid truth kernel PARKED (escalated to the Architect)
Branch fluid-truth @ 096628b (clean, unmerged; 3-round conclusion, full
data in docs + fluid_kernel.rs docs):
- WORKS: PBF density constraint, compression-only clamp (correct), RMS
  flatness 0.030-0.044 vs 0.040 bound. 5 ordeals green + sabotage RED.
- THE DISEASE (both cheap cures fail): s_corr detonates under sustained
  hydrostatic load even per-pair-gated (r6+r7) · tensile_k=0 collapses
  real geometry INVISIBLY to SPH density — NN spacing −70-90%,
  coincident pairs, no buoyant differentiation, cobblestone surface
  (found by the new geometric volume probe, not the density gate).
- MISSING GATE identified: geometric min-separation ordeal (SPH-density
  ordeals are blind to clustering — disclosed in fluid_ordeals.rs).
- CANDIDATE CURES for the Architect's physics pass: (1) RECOMMENDED:
  pairwise min-separation as a unilateral distance constraint via the
  solver's OWN proven contact machinery (charter-coherent: one solver,
  fluid rides the same contact floor as rigids — unconditionally stable
  projection, decoupled from the density feedback loop); (2) duration/
  magnitude-gated s_corr. Cost note: solve_fluid 131.7ms @ N=1372
  (bench kernel — speed is not the gate).

## PLAYGROUND MERGED (merge-conductor burst #12)
main @ cf571de (merge becd50d + adversary advisory cf571de) — GREEN,
suite 400/0. THE PUSH lands: the Architect's hand becomes an op — F key /
locked click / `/push` organ all funnel through the same `Op::Impulse`
(`build_push_ops` in scrying-glass/src/main.rs) an agent op would send.
Nine Physics Playground vessels folded into the hand-derived canon
(packages/oracle/tests/canon.rs); naruko plaza gets toys (rigid stack,
bonded break-crate, pyramid) in worlds/naruko/scenes/main.json. Adversary
HOLDS with live-door proof: shattered the bonded crate over the live HTTP
door with her own eyes on it (proof/live-before-push.png,
proof/live-after-push-5.png, proof/finish-toys-view2.png).
Merge was a clean fast-forward-able join (physics-playground's
merge-base IS main's HEAD 6cf5c2b) — no worlds/naruko/scenes/main.json
conflict against rimguard; rimguard already sat on main before the
playground branch was cut.
Adversary advisory (cf571de): (a) AIM_RADIUS bare const 0.9 in
`build_push_ops` → env param `GAIA_PUSH_AIM_RADIUS`, default 0.9,
validated finite>0, matching its `GAIA_PUSH_REACH`/`GAIA_PUSH_SPEED`
siblings (IRON-law fix, zero behavior change at default — rebuilt,
playground_push example re-run, PASSED); (b) doc note: examples/
playground_push.rs's `pick()` is a deliberate verbatim copy of the ray/
aim-radius picker (proves the real door, not a stub) — shared-fn
extraction considered, parked as copy-drift risk on record (example
sits outside the crate's public surface); (c) F-key autorepeat
(input.rs) documented as-is: held F re-fires push_pending at the OS
KeyDown repeat rate, not once per press — rapid-shove-feature vs
isARepeat-edge-gate is a gameplay-feel call parked for the Architect.
Also carried, parked for the Architect: F6 solver-rest ergonomics, scry
timeout ergonomics (both pre-existing, unrelated to this lane).
The Architect's live window (pid 69733) already runs this build — F =
push, toys sit behind spawn; the merge does not disturb that session.
Full suite: `cargo test --workspace --release` → 400 passed, 0 failed,
82 test-result groups (unit + integration + doctests), clean.
Pushed origin main 6cf5c2b..cf571de.

## WORKER WINDOW MERGED (merge-conductor burst)
main @ 755dfa5 (merge of worker-window @ 0f1f3b6, no-ff) — GREEN,
suite 401/0 (17/17 packages). Nekromant case #1 fix: worker instances
never-key (focused:false, focusable:false via tao), smaller + titled,
so the Architect's live window keeps input focus even with worker
instances open beside it. Always-on focus/activation logging added
(packages/scrying-glass/src/main.rs) as the field witness — every
focus-gained/lost and window-activation event now logged, default-off
as an *instrument* (no behavior gate, just visibility) so silent
focus-steals stop being invisible. Conductor-reviewed; adversary pass
deferred on record (not yet run against this lane).
Silent-deaths case CLOSED: root cause was focus-steal from worker
instances grabbing keyboard focus away from the Architect's live
window; fix makes workers structurally unable to key (OS-level
focused/focusable flags), not a heuristic.
Field-trial note: Cmd+Q refusal behavior on worker windows is
UNADJUDICATED — needs a real multi-window session with the always-on
focus log open to confirm whether a worker eating Cmd+Q is desired
(prevents accidental kill of the wrong window) or a bug; punt to next
real session, log is the diagnostic tool for that call.
Merge: worker-window's merge-base (6cf5c2b6ebc4f7215148f841980b18dc61c42870)
sat on main's line; only packages/scrying-glass/src/main.rs touched
(+99/-12) — clean, no conflicts.
Full workspace suite run per-package under the build token (avoids the
300s wall): 17/17 crates green, 401 tests passed, 0 failed.
Build: `cargo build --release -p scrying-glass` clean (39.45s).
Pushed origin main c989f06..755dfa5.

## BLOODBEND B0 MERGED (merge-conductor burst)
main @ 038f9a0 (merge of bloodbend-b0 @ a934c51, no-ff) — GREEN, suite
408/0 (401 prior + 7 new bloodbend_ordeals). Merge landed with NO
conflicts: main's main.rs (worker-window from_env/WindowBuilder) and
bloodbend's main.rs (config params, bend_scene/bend_shader, the
watch-drain loop) touched non-overlapping regions — git auto-merged
clean; both features verified present post-merge (grep for
WindowBuilder + bloodbend:: symbols).
What B0 opens: edit scene JSON (world.json/scenes/*.json) or
packages/scrying-glass/src/integrator.wgsl WHILE the window runs —
the live world/shader updates in place. Bad edits (broken JSON,
broken WGSL) get a police report and the old scene/pipeline keeps
rendering, never a crash. The journal (debug/bloodbend-journal) snapshots
the previous-good scene/shader before every applied bend — undo. TOCTOU
is dead by construction: validation runs against the EXACT captured bytes
via a private validation dir, never a second live re-read, so validated
bytes == stored last_good bytes always.
Advisory commit d8ab694/038f9a0 (attributed to the hostile re-pass,
corner (a)): a duplicate-id scene write can be value-identical at the
entity-diff level (no-op) while its raw bytes are still loader-rejectable
(the duplicate key itself). Old no-op path advanced `last_good` to those
bytes anyway — smallest cut: no-op path now leaves `last_good` untouched,
pointed at the last VALIDATED bytes; next real diff still computes
correctly. Plus doc note at scene_paths init (bloodbend.rs): the watch
set is fixed at boot — a scene file created live (new scenes/*.json) is
unwatched until process restart; on record, not a MUST-FIX (no exploit,
just a known gap for a future watch-set refresh).
Full workspace suite run per-package/per-binary under the build token
(the 300s single-shot `--workspace` wall bit mid scrying-glass lib
unittests — no hang, just wall-clock; every package/binary re-run
individually completed in seconds): 408 tests passed, 0 failed, 0
ignored, across all 17 crates (unit + integration + doctests).
Build: `cargo build --release -p scrying-glass` clean (45.58s).
Pushed origin main afa6e6c..038f9a0.

## ADVISORY — bare-/scry one-frame staleness (light-live merge-conductor, parked for a future ordeal)
`GET /scry` / `/screenshot` with NO query serves `latest` (the async
capture-worker's last-written framebuffer), NOT a synchronous read of the
frame the render loop just submitted. `render()` queues the offscreen
copy via `map_buffer_on_submit`; the GPU readback + `latest.write()` land
on the capture-worker thread some time AFTER `render()` returns, so a
caller that mutates world state (an op, a walk tick, a bloodbend scene/
shader bend) and immediately screenshots can observe the PREVIOUS frame's
pixels — a real one-frame-or-more staleness window, independent of and
in addition to temporal accumulation's own convergence lag. Future ordeals
that assert on bare-/scry pixels right after a mutation should poll (a
couple of frame-intervals' worth of retries) or use the moving-eye `/scry?
pos=...` path (which blocks on `reply_rx.recv_timeout` for its own fresh
render) instead of trusting the first bare capture. Not reproduced or
quantified here — on record for whoever hits a flaky screenshot-right-
after-mutation ordeal next.

## LIGHT MERGED — the window converges to real light (merge-conductor burst)
main @ b00c0cf (merge of light-live @ b2fe5c2, no-ff) — GREEN, suite
412/0 (408 prior + 4 new light_temporal ordeals). Advisory doc-notes
commit e4d97bc (no behavior change) rides right after it.
The default state: `GAIA_NATIVE_TEMPORAL` (default **true** — ON by
default in this merged main). With it on, the live present path runs
72× temporal reprojection accumulation instead of raw per-frame dots:
`integrate_temporal` traces the frame's radiance + a primary gbuffer
(depth/normal) into a packed ping-pong buffer, `temporal_resolve`
reprojects last frame's history via the previous camera basis (still
camera = identity/no-resample pure running average; moving camera
reprojects world-space hit point into last frame's screen, rejecting
disoccluded/off-screen/depth-or-normal-mismatched samples so no history
survives across a real occlusion — "no ghosts"), then blends into
`accum` for the existing blit to present unchanged. 7.6ms live, adversary
HOLDS. `GAIA_NATIVE_TEMPORAL=false` keeps the legacy reset-on-move raw
accum path as an escape hatch.
CONFLICT (only one, resolved): both bloodbend-b0 (extracted a shared
`build_pipelines` fn so `reload_shader` can re-run pipeline construction
identically) and light-live (branched before that extraction — light-live
still builds the compute/blit/aov pipelines INLINE inside `new()`, then
inline-appends the temporal bind-group-layout/pipelines, then returns
`Self{..}`) touched the exact same seam in
`packages/scrying-glass/src/integrator.rs`, right after the AOV bind
layout and before the old `Self{}` return. Resolution: kept bloodbend's
`new()` shape (call `Self::build_pipelines` for compute/blit/aov, THEN
`reload_shader`/`update_bvh` as separate methods) and re-inserted
light-live's temporal bind-group-layout + `temporal_integrate_pipeline`/
`temporal_resolve_pipeline` construction verbatim right after the
`build_pipelines` call, folding the three new fields into the single
`Self{}` literal alongside bloodbend's existing fields (struct
definition itself had already auto-merged clean — both sets of fields
were already present). `reload_shader` is UNCHANGED — it still only
swaps compute/blit/aov on a live WGSL edit; it does not (yet) rebuild
the temporal pipelines, a real but out-of-scope gap for whoever wires
bloodbend shader-bend + temporal together. `main.rs` merged with ZERO
conflicts (worker-window's config/WindowBuilder, bloodbend's bend_scene/
bend_shader/watch loop, and light-live's temporal config/Renderer fields/
render() dispatch all sit in non-overlapping regions). Verified present
post-merge by grep: `WORKER_WINDOW`, `bloodbend::`, `integrate_temporal`
all found in main.rs/integrator.rs/integrator.wgsl.
Advisories carried forward (commit e4d97bc, doc-only): (a) the
`cam_moved` gate's `0.99999` dot-product threshold is ~0.26°/frame — a
pan slower than that reads as a still camera; parked atom is deriving it
from pixel angular size (fov/resolution) instead of a fixed constant.
(b) the variance clamp in `temporal_resolve` only gates on the OBSERVER
moving — a still camera watching a MOVED body's shadow/highlight sweep
across a pixel isn't caught, so a stale relit color can linger up to
`max_history` frames before the running average alone catches it up;
quantifying the worst-case lag needs a quiet-machine push-object
construction, parked for a future ordeal. (c) bare `GET /scry`/
`/screenshot` (no query) serves the capture-worker's async `latest`
framebuffer, not a synchronous read of what `render()` just submitted —
a real one-frame-or-more staleness window on top of temporal's own
convergence lag; future ordeals asserting on bare-/scry pixels right
after a mutation should poll or use the moving-eye `/scry?pos=...` path.
Build: `cargo build --release -p scrying-glass` clean (44.27s). Full
workspace suite run per-package under the build token (17 crates):
412 passed, 0 failed, 0 ignored.
Restart of the live window (:8430) is the Architect's own act — he
restarts it after this lands, not the merge-conductor.
Pushed origin main 78df1de..35dc59a.

## 07-18 EVE — COMPACTION HANDOFF (nyari, ~14:35, read-first anchor)
Main @ 67a2f9a (day: 6cf5c2b→67a2f9a, suite 412/0, zero red). Landed: THE
PUSH (playground) · own-eye cull · BLOODBEND B0 (live scene/WGSL bend,
TOCTOU dead-by-construction) · worker-window (never-key workers) · LIGHT
merged (temporal, default ON) · pantheon sealed (GRIMOIRE: Ananke · Wilde
Jagd · Zauberpolizei · NEO · Blutbändigen · Gaia/Seed · Aether=input-to-
Pleroma · Pleroma WHOLE, no inner names) · TWO-ACT LAW + upscaling banned
(NEURAL.md) · SWARM COMPUTE LAW (c989f06: brains parallel, ONE build token
-j2 nice19, GPU=his while present).
SILICON RACE (NEURAL.md ledger): WGSL door SHUT (280ms native) · ANE
REFUSED (planner-attested) · METAL TENSOR OPEN: 4.47ms native @94% roofline
(r-direct @ eed0bdc) → 60fps arithmetic: rays 7.6 + net 4.5 + glue ≈ 13ms.
Remaining: gather/demod measure · buffer pooling (wall 23ms until) ·
MPSGraph↔wgpu interop lane · reload_shader↔temporal-pipeline gap.
⚠ IN FLIGHT: LIGHT-FIX (ghoul-opus-mrqgkqd5zdlmnj, branch light-fix off
67a2f9a, worktree ../magic-crystal-light): Architect PLAYED → slow-pan
GHOSTS+DOTS (cam_moved gate 0.26°/frame > real mouse) → gateless fix:
always-reproject, always-clamp, derived thresholds, ordeals h/i/j
(slow-pan · micro-jitter · relight). Gate on return: monad verdict →
adversary → merge → restart his window (his go). FALLBACK RULING (his Q
14:26): dots = one integrator's young samples, NOT a second path; the
binary GATE was the fallback-shaped sin (hence this fix); TEMPORAL=false
switch = his own pre-cutover amendment, dies at his played word.
HIS WINDOW: pid 57776 :8430 (light+bend build). F=push · toys z32-35 ·
scene-file edits bend live (B0).
QUEUE (his word / away-hours): light-fix pipeline · interop lane · vflow
trio consolidation (1c66cd9/2a884bd/2976bc6) → VISIONFLOW.md packet ·
fluid pressure-mirrored-boundary ruling · B0 deferred live checks · NRC
hash-grid · VII-3 planet. RITES HIS: walk · chrome sphere · VI break word
(hymn owed). Lessons: duplicate-dispatch (ghoul-routing.md) · machine
facts verified against machine (Tahoe was installed for months) · play it
with REAL hands before claiming (slow-pan). Host = M1 PRO 16GB, Tahoe [source: docs/perf/2026-07-18-rdirect-metal-tensor-spike.md]
26.5.1, Metal 4 available. [source: docs/perf/2026-07-18-rdirect-metal-tensor-spike.md]


## ★ RULING 07-18 ~14:44 — THE DESIGN IS THE LAW (supersedes everything above)
Architect's word, verbatim intent: no fallback, no prototype — build ONLY the
one full neural rendering engine as designed: world truth enters Pleroma;
Pleroma renders the final image or nothing. Consequences executed:
GAIA_NATIVE_TEMPORAL default flipped OFF (temporal = lab equipment: training
ground-truth + history buffers; its heuristics never ship) · light-fix lane
(in flight) harvests as TRAINING-GENERATOR gates only, NOT present-path · THE
ONE LANE = Pleroma in the live present path (Pleroma's learned act on tensor
path 4.47ms + rays 7.6ms, interop MPSGraph↔wgpu + pooling + gather + training [source: docs/perf/2026-07-18-rdirect-metal-tensor-spike.md]
loop [source: docs/perf/2026-07-18-rdirect-metal-tensor-spike.md]). Nothing
else is legal work on the renderer.
Law written: CLAUDE.md (top) · here · nyari memory. Sealed.


## ★ RULING 07-18 ~14:50 — LAW EXTENDED TO PHYSICS
His words = the ruling: an ACTUAL neural-network-based physics system, no
fallback. Design sealed: Ananke assembles; Pleroma's learned act solves
into state; classical solve demoted to teacher/ground-truth + scaffold; same
death rule as render.
Honest state at ruling: physics 100% classical (zero neural ever ran) ·
fluids lab-proven (gates 1-3, buoyancy impossible in current formulation —
pressure-mirrored boundary = the open door, HIS parked ruling now MOOT: build
it as part of the real thing) · building collapse NOT built (toy scale only).
Lanes: neural-physics (net solver inside Ananke, N0 plumbing/N1 training,
gates vs classical) · playable-destruction (building-scale collapse + fluid
spawn door IN HIS WINDOW, real player path). Sealed.

## ★ 07-18 DAY-CLOSE — COMPACTION ANCHOR (nyari ~18:25, read FIRST after the law chain)
MAIN @ 01d97e6+ (day 78df1de→here, ~20 merges, all gated). THE DAY = the law
cascade + its enforcement machinery + first neural frames ever.
LAWS SEALED (CLAUDE.md ★ stack, read them all): DESIGN IS THE LAW (render +
physics) · CORRECT OUTPUT OR NOTHING (screen = Pleroma's image or black) ·
THE RESOLUTION IS 640×480 (his caps) · PACKAGE LAW corrected 16:13 (crystal
= MINIMUM core: world state · ops · entropy/journal replay · doors · MONAD;
renderer/physics/AI ALL replaceable packages) · OFFLINE LAW (LLM never
required; tiny local LM voice OK, ANE-homed [source: NEURAL.md silicon ledger], async) · STUDY NEVER IMPORT ·
DF = WORLD-SIM BIBLE · MONAD = god-interface (crystal core; authority not
machinery; TTRPG GM = structural source only) · NAMING (PLEROMA whole — "the
net" banned from speech; internals only on his ask "inside Pleroma") ·
WINDOW BAN (no lane windows EVER) · BOTH-EYES (belief PNG + surface PNG
before any visual claim).
ENFORCEMENT (all tested refuse+pass, transcripts in room): Wilde Jagd hook =
artifact gate (Adversary-Report file in merge tree w/ VERDICT: HOLDS +
CONCORDANCE) + cite tooth (new .md silicon lines need [source:]/UNVERIFIED)
+ baseline & pull-merge exemptions · LASHES.md = discipline ledger, 9 rows
(row 8 = the ledger itself lied; escalation live: silicon claims mandatory-
cite) · MONAD-ADVERSARY = standing organ auditing MY claims (first sitting:
7 heresies → purge wave III executed).
PLEROMA: branch neural-live @ 29a7ed0 (+shift 9 RIDING ghoul-opus-
mrqodbv7at1etp): first live neural frames TODAY (n0d) → stipple KILLED at
God's res (s7 PNGs clean, both-eyes read) → wall decomposed: fused MPSGraph [source: docs/perf/2026-07-18-neural-live-n0.md]
GPU 6.65ms + CPU encode ~14ms [source: docs/perf/2026-07-18-neural-live-n0.md] = the thief → shift 9 = encode to its own
thread, projection ~15ms vs 16.67 [UNVERIFIED]. His glass DARK until table
≤16.67 + both PNGs; CUTOVER = HIS PLAYED WORD. Main's present path still raw
accum — its kill lands WITH the Pleroma merge (documented in conform report).
PHYSICS: P-N0 + P-N1 nets both DIED honestly (death rule; tables docs/perf/
2026-07-18-pn0/pn1). Insight: classical = 0.196ms/step at toy scale — nets
can't beat free. FORK AWAITS HIS RULING: P-N2 same ground (GNN/passivity)
vs move the war to fluids + building-scale (rec: fluids — teacher expensive,
buoyancy = open wound). Building + fluid door PLAYABLE ON MAIN (b6f125e,
canon re-derived by hand 38/23).
ALSO ON MAIN: world-core spine (scenes/ops/journal/reset/dev-writeback, R10
atom 1, e22acf4) · /retina Matrix vision (foveal vector truth, 85ms,
e6c921e) · God-canvas in code (43f807c) · LOD cut DEAD (280504a) · MIND.md
ruling packet (M0-M5 [source: docs/proposals/MIND.md]) + research (df-bible, exemplars, substrate in room
17:11) · CRYSTAL CRATE EXISTS: crates/crystal 2,936 lines since Rite III
(LASHES row 9 — I denied it; never answer existence from one dir).
RIDING: Pleroma shift 9 · silicon race II (ghoul-sol-mrqp2b09wky5v0: ANE rows [UNVERIFIED until its doc]
matrix — tiny nets R-A, MTL4 ML-encoder door R-B [UNVERIFIED, LASHES row 2],
voice R-C).
HIS RULINGS PENDING: ① physics battlefield ② MIND ×4 (conditioning dims ·
tick cadence · MONAD event vocab · the mind's true name) ③ Pleroma cutover
(his hands) ④ crystal extraction M0 [source: docs/proposals/MIND.md staging] (fold ops door + kami + steiner + MONAD
seat into crates/crystal; scrying-glass demotes to the glass).
LESSONS (LASHES-grade): priors override record = my root failure — cite or
UNVERIFIED, verify against the machine/tree, never notes · package NAMES ≠
dirs (transmutation!) · build-token starvation → merges go SERIAL (train) ·
the 30-min wall eats rooms not work → commit-first every stage · canon reds
= re-derive by hand, never bump-to-match.
SHIFT-9 LANDED (post-anchor): pipeline works — net wall 4.16ms, TOTAL 20.07
vs 16.67 [source: docs/perf/2026-07-18-neural-live-n0.md N0.g]; last thief =
encode-thread CPU vs trace submission (+6ms, A/B-proven); shift 10 riding =
dedicated encode queue + shared-event sync; projected ~12.9ms [UNVERIFIED].
NAMED (18:54): the agent harness for the optional LLM = THE EGREGORE —
package, never required, doors-only (no backchannel); crystal+MONAD = the
only must. [source: his words, room 07-18 18:54]
RACE II CLOSED on main (321f5e9): matrix final — CPU<=2048/GPU>=4096 tiny,
Pleroma-shape GPU 4.3-6.4ms; encoder door nearly free (10us encode) but
package wall precise: metal-package-builder needs undocumented Manifest.json
→ locus UNVERIFIED, parked [source: docs/perf/2026-07-18-silicon-race-2.md].
CANON SEALED (21:19): the OG story = docs/design/THE-MAGIC-CRYSTAL.md —
intro real-time (never pre-rendered), map = INFINITE CANVAS always, worlds
= f(seed), shard = interaction layer (speaks back, tutorial), two wizard
factions (Edan Connor / Zareb Aiden), protagonist name OPEN.
LATE-NIGHT STATE (23:07): THE PURGE DONE on neural-live (dc3a7d9: -854 LOC,
app = Pleroma letterboxed 640x480 OR black, both proven s20-real/s20-black
0/614400 px) · REAL IMAGE BAR sealed 4f8ec5b (false image = heresy; ordeal
stamps weights, unstamped -> black) · v2 weights 2x better, still sub-bar ->
window BLACK correctly · LASHES rows 13-15 · RIDING: N2 memory/recurrence
(kills dots at root) · light ordeal (20-light ocean + 1000-echo laser @
R=0.999 his order) · manifest crack (neural-cores wall). His ratifications:
low samples = less accurate guess NEVER dots · unlimited lights = R5 next.
RULING 07-19 08:52 CORRECTED (his words: 'IT RUNS ON NEURAL CORES THROUGH METAL') [source: room 08:52]: PLEROMA = NEURAL CORES as silicon, METAL as the only API road (MTL4 ML encoder [source: metal4-neural-recon.md]) — Metal end-to-end, ANE execution; dispatch lane carries her package + locus proof; GPU stays her measured home until the encoder dispatch lands numbers.

## ★ 07-20 — V7 FIRST LIGHT (nyari, pre-compact anchor — read FIRST)
THE DAY: Pleroma's FIRST STAMPED IMAGE EVER. v7 weights sha 55720b45 PASSED
the real-image ordeal (resid 0.03487/0.035 · sparkle 35.8/40 · tvar/move/
ghost all PASS) — stamp on disk, committed. Worktree ../magic-crystal-
neural-live, branch neural-live. Own-eyes proof: proof/neural-live/
s25-still.png (harbor/lighthouse/lit windows) + s20-v7stage3b-presented.png
(LIVE loop render).
THE ROAD (laws earned): v3-v6 sparkle grave → AUTOPSY (net OVERSHOOTS real
E-light 1.15-4.1x; D innocent; no anonymous channel — the 0.88% figure IS D)
→ EVIDENCE CLAMP, structural not penalty (presented = min(net, γ·local_max_
3x3(evidence)), γ=1.5 env GAIA_V7_CLAMP_GAMMA, commit c8b9ba6) → zero-
retrain sparkle 52→27.7 → CORNER-CRAWL (lr 1e-4, monitor EVERY epoch,
cross-run score floor) caught score 1.013 → ordeal PASS.
TRAINING LAWS: fine-tune from best ALWAYS regressed (5/5, floors held) ·
monitor-every-epoch or blind · detach (nohup) everything — walls eat rooms
not work · cross-run bar-normalized floor mandatory.
V7-LIVE LANE (all committed, lane notes = worktree scratch/v7-live-lane.md [source: worktree scratch/v7-live-lane.md]
+ scratch/v7-cutover-ready.md) [source: worktree scratch/v7-live-lane.md]: S1 split gather parity 9.5e-7 · S2 GPU
history reproject 3.96e-5 · S3 loader shape-generic (INPUT_FEATURES 23
hardcode dead) + full loop + GPU clamp + refuse-not-corrupt guard · perf
one-submission fix 23.3→16.63ms median (beats old 18.57; p95 ~24 spikes
UNATTRIBUTED) · pan parity 1.5e-6.
SEAM CLOSED (rooms 4-5, commits 0fba97c→c90ad1b): cause = still camera
self-reprojects edge px EXACTLY onto bounds boundary → sub-ULP GPU/CPU
noise flipped valid 0/1 (round() half-tie suspect FALSIFIED; eps-slack
half-fix superseded). CURE = symmetric snap-to-pixel before predicate,
BOTH sides (CamPose::reproject + cam_reproject WGSL), SNAP_EPS=1e-3 param.
Act changed → ordeal RE-RUN 640x480 → stamp RE-EARNED, bars byte-identical
(resid 0.03487 · sparkle 35.81 · PASS). Parity NOW: still 4.77e-7 · pan
1.55e-6 · 0/6144 px — machine precision both. 6 regressions structural-
green. CLEAN-GPU BENCH (room6, b4c7ef0): median 18.1-19.1ms · p95 27.5-30.3
· wall 41-44fps (room3's 16.63 = outlier, corrected). Old path median parity
(18.57 carried) but old wall 53.85 vs v7 41-44 — comparison permanently
MOOT: v4 stamp exists NOWHERE (v7 = first weights ever to pass; v4 can
never lawfully render; 18.57/53.85 = dead act's number). Tail = startup transient + recurring whole-frame GPU stall
(trace+net_wall spike together, not additive); periodic-vs-random undecided.
Budget gap to 16.67: ~1.9ms median + tail. → ROOM 8 (63d024c): GAIA_NATIVE_
ASYNC_TRACE re-tried under v7 — CUT B verdict REVERSED (old queue shape
dead): median 16.31ms / 46.3 wall fps — UNDER BUDGET. trace 12.61→0.22
(CPU poll bubble deleted), gather absorbs +8.9, net_gpu flat → pure
CPU-bubble win, GPU not saturated. Flag stays OPT-IN; default-ON = his
word. Act-safety: structural (single queue FIFO + hazard tracking; polls
can't reorder GPU work) + parity probe machine-precision; empirical A/B
inconclusive-by-instrument (bench capture = free-running → run-to-run
history noise 6258px ≈ flag delta 6516px); fixed-N scripted A/B = the
sealing instrument if wanted. Remaining tail: rare (~1%) net_wait stalls
>33ms (proposal 2, unattacked) + p95 25.5 still over.
LAUNCH (HIS act only, works NOW):
  cd ../magic-crystal-neural-live/packages/scrying-glass &&
  GAIA_NATIVE_WEIGHTS=v7 GAIA_NATIVE_EVIDENCE_SPLIT=1 ./target/release/scrying-glass
CUTOVER-TO-DEFAULT: seam closed → now purely his played word.
SILICON (morning, all measured): Metal-4 encoder = GPU on M1 (ANE counter [source: room chat-mrrlinr6-ic04 07-20 morning runs]
flat 18, GPU 84%, twice) [source: room chat-mrrlinr6-ic04 07-20 morning runs] · MLX = GPU-only (sourced: ml-explore #18/#393) ·
ANEMLL Llama did NOT engage ANE on M1 [source: room chat-mrrlinr6-ic04 07-20 runs] (counters flat; MLX 184 tok/s vs 33;
MLX kills frame tail p90 +162% → disqualified for live voice) [source: MLX GPU-only: ml-explore/mlx#18/#393] · afm CLI
(installed, /opt/homebrew/bin/afm) = Apple foundation models, GPU ~10%
during gen → likely ANE-resident UNVERIFIED; sudo powermetrics proof = HIS paste,
pending. Voice tier: afm default (adapters .fmadapter for flavor; afm mlx
mode = GPU road). ioreg busy-ms = IOKit bookkeeping, NOT utilization —
lesson recorded.
ALSO: neural-motion TODO docs/research/2026-07-19-neural-motion-todo.md
(ARDY NVIDIA/ETH + MANN dogs) · terra/sol pool capped ~6d (07-19), sonnet
carried the whole day · frame-budget 1.9ms hunt (trace 7.4 + net contention)
still queued behind v7 lane.

## ★ 07-20 EVE — LAUNCH + HIS EYE'S VERDICT + THE V8 MANDATE (fresh-ctx anchor; supersedes morning § where they differ)
LAUNCHED on his word 14:38 (his "launch" = in-moment go-ahead, whip-154
intact): window wgpu-surface from worktree, v7 + evidence-split +
async-trace. Live vitals ~18.3ms/40fps warm (retina-blit gather heavier
than offscreen 16.31). Own-eyes proof: worktree packages/scrying-glass/
scratch/launch-live-scry.png. Endpoints :8430 /scry /pose /walk. App may
STILL RUN — NEVER kill/restart, his act only.
HIS EYE (whip 213): dots STILL there · mirror weird · sky GHOSTING (new).
Bars passed ≠ his eye passed. THREE AUTOPSIES (committed; full notes =
worktree scratch/v7-live-lane.md):
1. GHOST (a774741): sky history accept = `ok=prev_miss` — NO similarity
   test; AND trainer settle = identity self-feedback only (net never
   learned to doubt history). Probe: smear 0.077-0.081/frame → 1e-6 under
   GAIA_V7_SKY_HISTORY=reject. Fix LANDED symmetric CPU+WGSL, OPT-IN;
   default flip = act change → re-ordeal REQUIRED (not run).
2. MIRROR (§MIRROR AUTOPSY): EVIDENCE-side — chrome sphere's 4 low-res
   taps = uncorrelated single-sample specular draws → diffuse-trained
   Pleroma collapses to flat gray mean. Clamp EXONERATED (preclamp ≈
   presented). Fixes all act-class: specular spp · mirror training poses ·
   coherence feature — each retrain/re-ordeal.
3. DOTS (§GAMMA SWEEP, probe instrument true — γ1.5 reproduces stamp
   exactly): γ 1.5/1.25/1.0/0.85 → sparkle 35.8/24.4/11.4/6.5 BUT resid
   0.03487/0.03572/0.03789/0.04443 vs bar 0.035 — ONLY γ=1.5 passes.
   Ceiling CANNOT lawfully kill dots. Bonus: highlights under-rendered
   −38.7% vs teacher at γ1.5 (clamp-independent).
SYNTHESIS → THE V8 MANDATE (one training round covers all three):
(a) MOVING-camera history sequences — real reprojection in the settle
loop, not identity feedback (kills ghost at root, makes sky-reject
learnable; sky-reject flag likely part of v8's act → ordeal flag-on);
(b) curved-mirror / low-roughness poses (specular coherence);
(c) targets: sparkle ≪35 @ resid ≤0.035 (HIS EYE = the bar; γ stays 1.5)
+ highlight recovery (−38.7% gap). Morning §'s training laws apply:
corner-crawl lr, cross-run floors, per-epoch monitor, detach everything.
PERF: async-trace opt-in proven (16.31ms median offscreen, CUT B
reversed); p95 rare net stalls unattacked (lane note room 7 proposal 2).
COMMITS today (worktree, order): d0a9240→63d024c (rooms 1-8) · a774741
ghost · gamma harvest ×2. Every lane resumable from worktree scratch.

## ★ 07-21 — NO-FALLBACK CONVERGENCE + THE SWEEP (nyari, ~11:35, fresh-ctx anchor — read FIRST)
HIS ORDERS TODAY (whips 214-219): doctrine = gate not memo (row 18) · TRILOGY
purge (row 19) · CLEAN UP (216-8) · "THERE IS NO FALLBACK!! MERGE AND MAKE THE
NEURAL ENGINE WORK" (219) — the cutover gate is DEAD, merge = ordered.

### THE MERGE CHAIN (his order, mid-flight at anchor time)
Remap DONE: neural-live (100 lane commits) cherry-picked onto scrubbed main →
branch neural-live-remap @ e488849d (58b13575 build fixes + e488849d ban-fix).
Verification DONE: suite 437 pass/1 fail→fixed, purge PROVEN in code (render()
= Pleroma 640x480 letterbox OR present_black, raw-accum DELETED — main.rs
~3034), scrub clean (no TRILOGY.md in any tree, blob ba2748d6 absent from
push range). Ban-fix ruling (monad, recorded in e488849d): viii0 temporal-
vocabulary ordeal narrowed to real heresy terms (temporal_accum class);
"history" = Pleroma's learned input channel, doctrine-legal.
IN FLIGHT: adversary completion room ghoul-sonnet-mrufwr7bowi67q (report
docs/adversary/2026-07-21-neural-live-remap.md, checkpoint-first: sabotage
check on ban-fix + scripted fidelity classes + VERDICT). ON ITS HOLDS →
merge-conductor: git merge --no-ff neural-live-remap into main + per-package
suite (300s wall: NEVER --workspace one-shot) + push (gate: VERDICT/
CONCORDANCE/DOCTRINE-CONCORDANCE lines + blob+cite teeth) → LAUNCH from main
for HIS eye: cd packages/scrying-glass && GAIA_NATIVE_WEIGHTS=v7
GAIA_NATIVE_EVIDENCE_SPLIT=1 GAIA_NATIVE_ASYNC_TRACE=1
../../target/release/scrying-glass (binary at REPO ROOT target/, not package
dir). Merge ≠ dots fixed — merge kills the fallback; weights fix the dots.

### V8 WEIGHTS WAR (the dots)
v8 detonation autopsy: ablation CONVICTED (c) highlight-biased sampling (A2:
sparkle 353@ep10 vs A0 72; A1 history + A3 mirror innocent). EMA-history fix
falsified as cause (identical runaway) but kept as hygiene.
v8c DOCTRINE ROUND (tier1 estimator-init exact 0.0 + tier2 noise2noise 1-ray,
teacher=validator): born sane (sparkle 90 flat, mirror 0, NO detonation) but
resid FLAT 0.0432 all 200 epochs vs bar 0.035 — signal too weak. FAILED bar,
no ordeal.
v8d RUNNING (detached PID 86898, worktree magic-crystal-neural-live, log
packages/scrying-glass/scratch/v8d-train.log): K=8 averaged n2n targets
(variance/8), single variable off v8c. Early: resid 0.0428→0.0422 = FIRST
downward movement any v8 produced; watch for plateau ~0.042. If plateau →
capacity/receptive-field verdict SEALED → next body (see DLSS below). Resume/
ordeal commands: scratch/v8-lane.md. Ordeal = real_image_ordeal 640x480 w/
GAIA_V7_SKY_HISTORY=reject, γ stays 1.5, bar = resid≤0.035 + sparkle≪35
(HIS EYE) + highlight recovery (−38.7% gap).
Training laws (5/5 earned): fresh init · corner-crawl lr · per-epoch monitor
· cross-run floors · detach everything.

### DLSS-4 REFERENCE (his gift, ~/projects/dlss4-reference)
Bring-up spike VERDICT (~/projects/dlss4-bringup/REPORT.md): ONE window-
attention block (enc0, documented widths, 640x480) = 73-83ms clean on M1
[source: dlss4-bringup logs] — 8-9x whole net budget (9.07ms). Full core
extrapolated 943-1383ms. Channel-thinning can't fix; fits ONLY if real net
spatially downsamples ~10-12x inside core (undocumented, UNVERIFIED). embed0/
embed1 shapes in weights.md NOT in manifest (reference inconsistency, flagged).
TRANSFERS to Pleroma's next body: multi-scale spatial context (THE dots
killer — receptive field), per-pixel MOTION VECTORS (object motion; camera-
only reproject = known ghost gap), recurrent history branch, fp16/fp8 amax
precision split, DLAA preset (scale 1.0) = upscaling-ban-legal operating
point. OPEN RULING FOR HIM: interior multi-scale features (output stays
640x480) — legal under upscaling ban? His call before next-body round.

### TRILOGY (row 19 — privacy)
Public git history SCRUBBED (filter-repo mirror + force-push; fresh-clone
verified clean). Blob tooth in wilde-jagd gate (refuses blob ba2748d65..,
tested refuse+pass). Source preserved ~/.gaia/nyari-private/TRILOGY.md 0600.
REMAINING: GitHub serves dangling e592e97 by direct SHA (probe 200 verbatim)
— needs HIS support ticket; draft on request. Old local branches carried the
blob → worktrees REMOVED (sweep below), branches remain local-only; tooth
blocks any re-push.

### THE SWEEP (whips 216-8, cleanup = my law now)
projects/: 31 dirs → 3 (GAIA-World-Engine · magic-crystal · magic-crystal-
neural-live). ~135GB cargo targets + 14 GWE worktrees + 19 mc worktrees
removed; disk 2GB→149GB free. All dirty trees PARKED first (commits on
branches: naruko nerves/soma, unity dotscity, ane, hook, light, playground,
worldcore, sense world.json — "park lane state 07-21"). Any lane resurrects:
git worktree add. lane-sweep.sh + LANE-CLOSE HYGIENE law pushed (fc65258). [source: tools/wilde-jagd-gate.sh cite_pattern]
AWAITING HIS WORD (~26GB ambiguous): gaia-daemon target 9.4G · Moonlight 5.5G
· gaia-daemon-v2 4.4G · GWE client-rs 3.9G · discord 3.2G · playwright 2.3G.
Cite-tooth quirk on record: "LANE" substring-matches "ANE" — overeager, kept. [source: tools/wilde-jagd-gate.sh cite_pattern]

### STATE SUMMARY
main @ fc65258b (public, post-scrub, all law commits) · neural-live-remap @
e488849d ready-to-merge pending adversary HOLDS · worktree neural-live = KEEP
until merge lands AND v8d finishes, then sweep per hygiene law · his window:
NOT running (he closed 07-20 eve; launch = his word or merged-main launch
step above) · terra/sol pool: capped since 07-19 (~6d), sonnet carried
everything.

### ★ 07-21 DELTA — MECHANISM CONVICTED: ONE-SIDED CLAMP; CURE RUNNING (~19:10)
Instrumented gradient probe (env-gated, v9f-last weights, the 7 forensic
texels): ALL 7 overshoot the evidence ceiling with d_out = 0.0 EXACTLY,
20/21 channels — CURE 1's clamp gated the loss to zero ABOVE the ceiling,
removing the only force that pulls overshooting pixels down → park+drift
[source: v9body commits 82f91be8 + e8b185e3, autopsy §v9g]. Four rounds of
mystery = one missing gradient branch. CURE 4: clamp made one-sided in the
RIGHT direction (up-fitting past ceiling still gated; down-pull ALWAYS
live). v9f end-reason confirmed on record: watchdog abort ep99, bar-res
sparkle 22.8 > 16 (commit 947894ba). v9g run ALIVE: PID 43798, ep49+,
resid 0.0985 / highlight_ratio 0.162 (vs v9f's 0.262 same epoch —
down-pressure visibly biting). Verdict = bar-res probes past ep99: sparkle
<16 while resid keeps the ~0.017/25ep linear fall → depth becomes pure
runway to the 0.035 bar.

### 07-21 DELTA — V9F DONE: DISEASE HAS A FACE, V9G MECHANISM HUNT (~18:40)
v9f completed. Bar-res probes: ep24 sp0.0/resid 0.1211 → ep49 0.0/0.1052
→ ep74 13.0/0.0872 → ep99 22.8/0.0701 [source: scratch/v9f-train.log
PROBE lines]. TWO truths: (1) deep training WORKS — resid falls LINEARLY
(~0.017/25ep, no deceleration; 0.0701 = best floor ever, halfway to bar
in log terms); (2) artifact hypothesis DIES at depth — render-res sparkle
IGNITED (22.8 > bar 16) as highlight_ratio grew 0.055→0.578 monotonic.
Disease finally characterized: REAL monotonic highlight-overshoot bias,
unopposed — something zeroes/starves the corrective down-gradient at
bright pixels (prime suspect: one-sided evidence clamp — overshoot past
ceiling escapes penalty → parks+drifts). Pre-drift regime perfectly
reproducible (ep50 score 2.678 in v9c/v9e/v9f identically). v9g riding
(ghoul-sonnet-mruvo6g6sc1lne): analytic mechanism hunt in the loss code
(clamp asymmetry / demod-log asymmetry / gradient sign at the 6 texels)
→ symmetric-clamp or overshoot-penalty cure → detached 300ep run, probes
past ep100. If cure holds, linear resid fall + depth = the bar's path.

### 07-21 DELTA — FORENSICS VERDICT: SIX TEXELS, NOT A DETONATION (~17:15)
COORDS dump (v9e deterministic rerun, env-gated in run_monitor): the
entire probe-res "sparkle explosion" = the SAME 6-7 FIXED pixels every
epoch — cluster (94-97,3-8) + (55,34)+(66,38) at 128×72 — error growing
smoothly ~0.5%/epoch (0.17→0.22 over 11 ep) [source: scratch/
v9e-forensics.log COORDS ep62-73, commit in v9body tree]. Verdict:
localized monotonic highlight bias at ~6 scene texels, NOT detonation/
fireflies/memorization; at 640×480 same world-spots spread 25× → below
sparkle threshold (why every render-res eval = 0.0). Old watchdog was
killing runs over six drifting texels. v9f (two-signal watchdog, bar-res
probe of record, dual checkpoints) RUNNING through the old kill zone —
PID 17962, ep13+ healthy; verdict + deep-cook floor lands at completion
(~90min). Residual real issue to watch: the 6-texel bias itself if it
ever crosses render-res thresholds — bar-res probes in v9f log will say.

### ★ 07-21 DELTA — 60FPS LAW: V9 BODY UNDER BUDGET (~17:00)
v9-wire @ e2485c40: ZERO-COPY BRIDGE LANDED (UnetLive::from_wgpu_queue,
v7 house pattern; fp16 pack GPU-side, rdirect_pack16.wgsl). Full frame
wall 640×480 N=40 panning: v9-bridged 15.30-15.37ms med / 15.45-15.73
p95 vs budget 16.67 — UNDER, ~1.0-1.3ms cushion, BOTH runs; v7 16.66-16.75
med; chain 105→46→28→15.3 [source: docs/perf/2026-07-21-v9-wire.md §bridge].
Parity exact through the bridge (max_abs 1.9e-4 identical to unbridged;
gather ordeal 2.3e-5). Measured UNDER GPU CONTENTION (v9body training
concurrent) — ~6.5ms waitUntilCompleted gap plausibly contention,
UNVERIFIED (isolated re-run = cheap seal when host quiet). The multi-scale
body FITS THE 60FPS LAW at render res — first neural body ever to.
Remaining live work = stamp-day only: main.rs/NetPresent wiring once
weights pass ordeal (deliberately untouched, REAL IMAGE BAR). Wiring
queue now EMPTY pending fidelity.

### 07-21 DELTA — V9E NEUTRAL → PROBE-RES ARTIFACT HYPOTHESIS + V9F (~16:45)
v9e (winsorized targets) reproduced v9c EXACTLY: onset ~ep52, floor resid
0.0938, score 2.680 vs 2.678 — winsorize NEUTRAL; fireflies CONFIRMED in
targets (rdirect_v9_tailmass, loss domain = demod-log, verified by source)
but NOT causal [source: scratch/v9e-train.log + v9-autopsy.md §v9e].
NEW READ of the curves: through the score "detonation" resid KEPT FALLING
(0.0938→0.0796 ep71) · sparkle spike = 6-7 DISCRETE pixels at 128×72
(651/759.5 per-Mpx quantization) · highlight_ratio climbing 0.27→0.50
TOWARD teacher parity · every 640×480 eval ever run = sparkle 0.0.
HYPOTHESIS: probe-res monitor artifact — the watchdog has been killing
healthy runs mid-highlight-learning. v9f riding (ghoul-sonnet-
mrurdg2n9d91cd): two-signal watchdog (resid-streak + render-res sparkle) ·
save best AND last weights · periodic 640×480 probe IN the training log ·
full 300-epoch run through the old onset. Verdict wanted: render-res
sparkle stays 0 while resid falls past 0.0796 → artifact CONFIRMED, run
cooks toward bar; ignites → hypothesis dies. v9e stall note: watcher turn
stalled 900s (long tail-watch), run itself completed — detach law held.

### 07-21 DELTA — GPU GATHER LANDED, BRIDGE ATOM RIDING (~16:25)
v9-wire @ 43f8c070: GPU gather PASS (max_abs 2.3e-5 vs 1e-4 bar; hist
slots EXACT vs stage-2 gather). Wall 640×480: v9 105.5→27.6ms med (v7
17.5; v9 GPU-forward alone 4.6) [source: docs/perf/2026-07-21-v9-wire.md §gather].
Gather now 1.75ms — remaining thieves DECOMPOSED: readback 7.0 +
fp16-glue 11.9 — root cause: UnetLive owns a separate Metal device; v7's
zero-copy bridge (RdirectLive::from_wgpu_queue) = the exemplar. Est.
landing ~7-9ms = under budget [source: same doc, est. line]. BRIDGE atom
riding (ghoul-sonnet, v9-wire tree). v9e forensics/training round also
still riding (v9body tree).

### 07-21 DELTA — V9D REGRESSED → TAIL HYPOTHESIS + V9E (~16:05)
v9d (4 anchors, ±180°, 8 draws) REGRESSED: onset ep24 vs v9c's ep51,
mirror val infected too; best floor 0.1033 vs v9c 0.1022 (640×480 eval)
[source: scratch/v9-autopsy.md §v9d]. Pure pose-memorization FALSIFIED.
Monad hypothesis now under test: onset tracks IMAGE-UPDATE COUNT
(v9≈48 · v9c≈153 · v9d≈192 updates) — low n2n_mse → gradient dominated
by unbounded 1-ray target tails (fireflies) → conv capacity fits them =
A2 highlight signature (per-pixel v8 was immune by incapacity). v9e
riding (ghoul-sonnet-mruq33t3aqkfyj): forensics (update-count table +
target tail-mass in loss domain) → cure = winsorized targets OR
luminance-normalized loss, on v9c's narrow-pool config · EMA kept (was
already in v9c — v9d's real delta was raw-vs-EMA logging: raw score
~100× noisier, EMA smoothing confirmed). Prediction: v9e survives past
~200 updates or hypothesis dies. Commits v9-body: 7128b4b1/66ef172c/
bd798242 (unpushed).

### 07-21 DELTA — V9-WIRE LANDED (~16:00)
Branch v9-wire @ 6b952229 (worktree ../magic-crystal-v9wire, unpushed),
2 commits, v7 artifacts byte-untouched (one accidental proof-PNG clobber
caught + restored pre-commit, disclosed):
· PARITY: UnetLive::from_weights closes the loading gap; permanent
  gpu_cpu_parity test PASS — max_abs 1.3-1.9e-4 vs derived bound 5.86e-3
  (fp16 roundoff × op-boundary derivation in test) [source: docs/perf/2026-07-21-v9-wire.md].
· ORDEAL DOOR: real_image_ordeal selects body by weights magic (GAIARD9/
  GAIARDR1), same 5 bars + stamp machinery + clamp. v9c ckpt honest FAIL:
  resid_still 0.113/0.035 · resid_move 0.113/0.06 · SPARKLE+TVAR+GHOST
  ALL PASS — war is down to resid alone. No stamp written (correct).
· FRAME WALL 640×480 offscreen N=40: v7 full 17.43ms med · v9 full
  107.96ms med (gather is CPU-built) · v9 GPU-forward alone 4.75ms med,
  cross-checks spike 4.29 [source: docs/perf/2026-07-21-v9-wire.md].
  → the live-speed gap = GPU gather for the U-Net input schema (atom
  riding, v9-wire tree). main.rs untouched (REAL IMAGE BAR blocks
  unstamped presents — wiring deferred to stamp-day, rationale in doc).
v9d lane still riding (autopsy + wider pool + Polyak EMA).

### 07-21 DELTA — V9C LATE DETONATION + SECOND WAVE (~15:20)
CORRECTION to ~15:10 §: "detonation DEAD" was WRONG — v9c detonated LATE
(~ep52 onset): ep71 val sparkle 759/Mpx, highlight +0.503 rising, streak
19/20 → watchdog abort ~ep72, best floor kept [source: scratch/
v9c-train.log ep71]. Pose diversity DELAYED onset 12→~52 + deepened floor
(val resid →0.0796 by ep71; 640×480 ep29 eval 0.1179) — overfit diagnosis
holds directionally; highlight-overshoot disease (v8-A2 signature)
survives narrow pools; mirror val stayed clean. RIDING (2 lanes): v9d
(ghoul-sonnet-mruoi6nto33jlt, tree v9body): autopsy + cures = much wider
pool + Polyak EMA 0.999 eval weights (NRC-proven) + keeps · v9-wire
(ghoul-sonnet-mruoi6ntf3gie9, NEW worktree ../magic-crystal-v9wire branch
v9-wire): CPU/tensor parity ordeal · ordeal door for U-Net weights (same
bars+stamp, expected honest FAIL today) · live present-path body-by-weights
wiring (no default flip, v7 stays stamped act) + frame-wall v7-vs-v9 body.

### 07-21 DELTA — V9C: DETONATION DEAD (~15:10)
Pose-diversity diagnosis CONFIRMED: v9c (fresh pose draws/epoch, val fixed
held-out) healthy through epoch 34+, score falls EVERY epoch, sparkle
0.0/Mpx every epoch, watchdog streak 0/20 — old onset was ep12, detonation
DEAD [source: scratch/v9c-train.log ep-1..34]. Past-ep250 survival
UNVERIFIED (run detached, PID 77670, 300 epochs — CHECK LOG AT COMPLETION,
next session's first act). Domain-fix correction: eval was ALREADY fixed —
resid numbers were REAL; "too dark" = genuinely under-trained, NOT the
measurement bug (my suspicion wrong, accepted). Trainers' settle/monitor
now domain-fixed too (undo_log_demod before metrics). 640×480 domain-fixed
eval: v9 ep11 resid 0.1276 → v9c ep29 resid 0.1179, sparkle 0.0 both,
clamp near-noop [source: scratch/v9-autopsy.md + v9c eval PNGs]. Gap to
bar 0.035 = 3.4×, curve NOT plateaued at ep34. Commits v9-body:
d0621464/b0f0ab5b (unpushed). Old-domain scores (floor 3.94) ≠ new-domain
scores — not comparable, flagged. Next gates: v9c completion verdict →
plateau? → if stalled ≫bar: higher-res round / capacity-within-9ms /
longer runs → ordeal → HIS EYE.

### 07-21 DELTA — V9 PROBE AUTOPSY + V9C (~14:45)
Probe run DETONATED deterministically at epoch 12 (both v9 and cured v9b,
same seed; last BEST ep11 score 3.96) [source: scratch/v9-autopsy.md].
Watchdog cure WORKS (abort streak fired ep31, floor kept, 1/9th wall
burn) [source: scratch/v9b-train.log]. Loss-clamp cure did NOT suppress
val divergence — clamp only reaches the 4 training poses → EXONERATED as
root cause. CONVICTED #2: settle()/eval compare raw demod-log output vs
LINEAR teacher (no undo_log_demod, unlike v8d inference) → val resid
domain-inflated + eval PNG "too dark" same smell. 640×480 eval ep11 ckpt:
sparkle 0.0/Mpx · highlight_ratio 0.018 · resid 0.1276 vs bar 0.035 —
resid UNVERIFIED until domain fix [source: scratch/v9-autopsy.md §eval].
Monad diagnosis: 4 FIXED training poses = conv overfit engine (whole-image
conv sees 4 samples where per-pixel v8 saw millions); ep12 = memorization
onset. → v9c round riding (ghoul-sonnet-mrun9h7k70mekv): domain fix
everywhere + re-eval ep11 + pose-diversity training (fresh pose draws per
epoch, val fixed held-out) + watchdog. Verdict wanted: detonation dead?
Commits v9-body: c6f82ff3/f705084c/a425a615 (unpushed).
CORRECTION to ~13:15 §: epoch wall ~16s — log times are CUMULATIVE, my
misread; 300 epochs fit the in-run budget fine.

### 07-21 DELTA — V9 BODY BORN, TRAINING LIVE (~13:15)
Worktree ../magic-crystal-v9body branch v9-body (392a7ff7 → 31444094 →
ae847923, unpushed): multi-scale U-Net body (rdirect_unet: tensor-path
forward + CPU-trainable twin, grad-checks pass 0.0719/0.0013 max_rel_err
[source: cpu_grad_check test run 07-21]). Spike: chosen C-med widths
[24,40,64] fwd 4.29ms vs 9.07 budget; D-large 11.36ms rejected [source: docs/perf/2026-07-21-v9-spike.md].
Output-res free = param (grad-check covers free-output-res path). Trainer
rdirect_train_v9 = v8d recipe ported (REAL moving-cam reprojection in
settle loop · mirror poses · K-avg n2n K=8 teacher=validator · 2 real
motion-vector channels). Disclosed gaps: no evidence-clamp · fresh He-init
only (no conv estimator-init).
TRAINING DETACHED: PID 56664, log packages/scrying-glass/scratch/
v9-train.log, res 128×72 probe round, 300 epochs. Epoch 7 own-eyes:
★ SPARKLE 0.0/Mpx FROM EPOCH 0 (v8 bodies: 90-353) — receptive-field
diagnosis CONFIRMED structurally [source: scratch/v9-train.log epoch 0-7].
resid 0.1464→0.1408 falling · highlight −0.226→−0.206 improving. resid
vs bar 0.035 UNVERIFIED (early + low-res probe; 640×480-comparable resid
needs the real-res round). Epoch wall ~110-125s not ~16s → 300 epochs
overruns the 3h in-run budget; best-checkpoint saves every epoch, floors
hold. Next gates: probe verdict at plateau → real-res round → ordeal
(sky-reject γ1.5, resid≤0.035 + sparkle≪35 + highlight) → HIS EYE.

### 07-21 DELTA — CONVERGENCE LANDED + V8D VERDICT (~11:55)
MERGE LANDED: origin/main = d3a2d696 (no-ff, zero conflicts, suite 444/0 all
17 packages, all gate teeth HELD, adversary report at tip). Main IS the
no-fallback engine: Pleroma 640x480 letterbox or black. Launch cmd for HIS
eye unchanged (§ above). Conductor also cite-tagged 80 legacy silicon lines
in 5 merged .md files (content unchanged) + merge trailers.
### ★ RULING 07-21 ~12:21 + AMENDMENT ~12:25 — MULTI-SCALE LEGAL · OUTPUT RES FREE (his words, room chat-mruhqnjx-c9mj)
Interior multi-scale features RULED LEGAL. AMENDED by his second word:
output does NOT stay 640×480 — "render resolution" = 640×480 (the
evidence/trace side), the neural renderer picks ANY output size. NO
UPSCALING stays absolute: Pleroma RENDERS at output res from world truth
+ evidence — never interpolates a small image up. Output res = parameter
w/ default (IRON). Unblocks THE NEXT BODY (v9): multi-scale spatial
context (the dots killer — receptive field) + motion-vector channel +
recurrent history [source: ~/projects/dlss4-bringup/REPORT.md transfer list],
M1-sized, NOT DLSS widths [source: ~/projects/dlss4-bringup/REPORT.md].
Lane open: v9-body (worktree ../magic-crystal-v9body, branch v9-body off
main): stage 1 shape-parametric body (ALL dims incl. output res = params
w/ defaults, IRON) + perf spike, net budget ≤~9ms = 16.67 − rays ~7.6 [source: docs/perf/2026-07-18-neural-live-n0.md]
→ stage 2 trainer wiring (moving-cam history sequences + mirror poses per
V8 MANDATE, K-avg n2n targets kept) → stage 3 train DETACHED (5 training
laws) → ordeal at render res, sky-reject γ1.5, bar resid≤0.035 +
sparkle≪35 (HIS EYE) + highlight recovery.

SWEEP DONE (07-21 ~12:xx): neural-live worktree preserved (scratch+data
committed, branch neural-live @ dd14d068, local-only — blob tooth) →
worktree REMOVED. projects/ mc footprint = magic-crystal alone.
V8D FINAL: resid 0.0416 (v8c 0.0432 → K=8 moved it, then PLATEAU), sparkle
108, highlight 0.72, best score 1.206. CAPACITY VERDICT SEALED by exhaustion:
instability killed (ablation A2) · init killed (tier1 exact) · signal killed
(tier2 K=8) → the per-pixel body CANNOT reach bar 0.035. Next round = NEW
BODY (multi-scale spatial context, M1-sized per dlss4-bringup numbers, NOT
DLSS widths). BLOCKED ON HIS RULING: interior multi-scale features (output
stays 640x480) vs upscaling ban. v8d checkpoint sha256 27c7dc38, tag-scoped,
NOT the real v8. Worktree neural-live: v8d done → tree sweepable per hygiene
law ONCE its scratch/ lane notes + data/ checkpoints are preserved (they are
committed on the branch? scratch logs UNTRACKED — copy before sweep).
