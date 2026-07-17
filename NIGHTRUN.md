# NIGHTRUN — overnight charter for the Claude Code runner (Fable)
Read AFTER: BIBLE.md → GRIMOIRE.md → HANDOFF.md → DREAMFORGE.md →
NARUKO.md (incl. §GUARDIAN RULINGS UNDER DELEGATION 07-17) → docs/proposals/.
Those documents are LAW. This file is your marching order. main @ 7b08d68+.

## Mission
Implement EVERYTHING remaining in the queue below, all night, without
stopping. Every item: built → adversary-reviewed → fixed → final-reviewed
by YOU → merged with the full suite between merges → pushed GREEN → logged.
Never idle. If blocked on one item, advance another and record the blocker.

## Pipeline (the Architect's own words)
1. DECOMPOSE each queue item into bounded atoms with pre-chewed anchors
   (file paths, line refs, exemplar commits — recon is YOUR job, not the
   builder's).
2. BUILD: spawn a `builder` subagent (sonnet) per atom. Tight spec,
   checkpoint commit ≤15 min, compiling stub in 10, salvage-first resumes.
3. ADVERSARY: spawn an `adversary` subagent (opus, fresh context) per
   landed atom: spec conformance vs law docs · independent gate re-run ·
   derivation audit (numbers re-derived by hand) · law hunt. Findings
   MUST-FIX/ADVISORY only — the adversary NEVER edits.
4. FIX: must-fix findings back to a builder with the critique verbatim.
5. FINAL REVIEW = YOU: run the full gates yourself, READ every proof
   image with your own eyes (Read tool on the PNG), then merge + push.

## The Queue (priority order)
1. LAND IN-FLIGHT WORK: check `git worktree list` + origin branches for
   `perf-exact` (BVH refit + CPU/GPU overlap) and `rite-specs-2` (Rite
   VIII + IX proposals). If unfinished: salvage-first (commit dirty trees
   that compile, then finish). Merge per protocol.
2. 60 FPS — THE LAW: after perf-exact lands, run
   `packages/scrying-glass/examples/perf_audit.rs` — DYN-ON must read
   ≤16.6 ms both poses. If a wall remains: implement RITE IX (the Chain
   Takes Flesh — skinned bodies through the cluster pipeline, per its
   proposal) as the structural fix. NO LODs (forbidden vocabulary), NO
   spp/bounce cuts, NO resolution tricks.
3. RITE VI — STRIFE (per docs/proposals/RITE-VI-STRIFE.md + rulings:
   bond-fracture, proceed-on-elements): VI-1 the stack topples → VI-2
   something BREAKS (fracture → flood-fill fragments → transmute re-mesh
   → BVH splice → fragments are vessels → canon learns them SAME WAVE).
4. RITE VII — THE PLANET-WALKER (per its proposal + rulings: radial
   gravity as up(r) with flat = infinite-radius limit; 64-bit/camera-
   relative coords PAID at VII-0): VII-0 first ground → VII-1 walk onto
   it → VII-2 the horizon streams.
5. RITE VIII — wave (a) only (the denoiser, per proposal): CPU-reference
   trained denoiser as a compute pass. THE BAN IS ABSOLUTE: input =
   current frame only; no temporal synthesis; interpolation/generation
   are forbidden words made ordeals.
6. BACKLOG: walkable-min-area parameter (floor = surface that holds the
   body's contact patch) · senses read SOLVER TRUTH (migrate oracle canon
   per ruling F6 — crate range from its REST pose) · any advisory left in
   the shadow reviews (grep NARUKO.md + merge messages for ADVISORY).
7. DO NOT write hymns (Guardian's voice, not yours). DO NOT close Rites
   IV/V (the Architect's acts). DO NOT rewrite git history.

## Absolute laws (violating any = stop and log, never work around)
- 60 FPS = MINIMUM · cost ∝ pixels, never content · NO LODs · NO baking.
- NEURAL INTERPOLATION/GENERATION BANNED FOREVER.
- ONE light pass. Every light traces to a realm entity.
- NEVER HARDCODE: every varying value = parameter w/ default; LOVE=1 is
  the sole allowed literal. A derivation frozen into a literal is a
  hardcode in costume.
- Tolerances DERIVED (measure floor → gate ~10× → prove a break bites).
  Gates must discriminate. Canon numbers derived from realm data with
  the derivation shown (exemplars: 388eb85, d95ef72, 71fd3bb, 76b528d).
- ORACLE CANON LEARNS EVERY NEW VESSEL THE SAME WAVE.
- FULL SUITE BETWEEN MERGES (never batch). Push ONLY green. Build before
  trusting any merge — git-clean ≠ rustc-clean.
- Vacuous-tail check: an empty test-result pipe is a lie; read the
  Running lines. `cargo test --workspace` counted honestly.
- Plain-English identifiers (mythology lives in docs, never code).
- Scratch under the repo, NEVER committed to a branch tip. Never /tmp.
- NEVER touch ports 8420/5173 or any process you did not start. NEVER
  restart the user's apps. Own test servers on 8460+, killed clean.
- Realm growth invalidates sibling-lane derivations: re-derive at merge,
  canon from scratch, never concatenation.

## Logging (so the morning can verify everything)
Append to NIGHTLOG.md after EVERY landed item: what · merge hash · suite
count · key numbers · proof filenames read. Update HANDOFF.md at each
major milestone (rite wave landed, 60fps state change). Commit both.

## Stop conditions
Queue exhausted, or a genuine wall on every remaining item. Record the
final state in NIGHTLOG.md + HANDOFF.md, push, and stop. An honest wall
beats a dishonest finish — this house's oldest law.
