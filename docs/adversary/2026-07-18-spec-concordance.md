# SPEC CONCORDANCE — 2026-07-18 heresy report (the Wilde Jagd ledger)

Adversary charter (CLAUDE.md ★ AMENDED 07-18 15:15, whip 168): spec
CONCORDANCE is a standing gate — does the implementation match the law
chain, does the spec contradict a sealed ruling. A spec contradicting a
ruling = HERESY, reported like a broken test. This doc = the ledger the
Wilde Jagd re-checks; each row = one heresy found + its final disposition.
19 doc items (this pass, DOC PURGE WAVE II) + 3 code items (dispatched to
separate lanes, tracked here for concordance).

## Doc items (this pass)

| # | Doc | Heresy | Disposition | Status |
|---|---|---|---|---|
| 4 | PHYSICS.md | XPBD framed as destination solver, not scaffold | Rewritten §0: Ananke assembles → THE NET solves → state; classical XPBD = teacher/live scaffold until P-N1 cutover (death rule owns eviction) | ✅ DONE |
| 5 | GEOMETRY.md | Entire doc = pre-pivot deferred-raster ruling; §5 explicitly rejects ray tracing as "a second renderer" | SUPERSEDED banner + whole body demoted to LINEAGE verbatim; RENDER.md §1 sole normative owner | ✅ DONE |
| 9 | PHYSICS.md | SPH-primary fluid, render-pose interpolation, far-field physics LOD all present as normative | Deleted as normative in §0 (interpolation ban + no-LODs violations); old body moved to LINEAGE | ✅ DONE |
| 10 | docs/proposals/RITE-VIII-THE-DREAM-DENOISER.md | Chained trace→denoise-pass design; no banner despite already being lineage-class | SUPERSEDED banner added: denoise/upscale chain = teacher-artifact material only | ✅ DONE |
| 11 | docs/proposals/RITE-IX-THE-CHAIN-TAKES-FLESH.md | View-dependent coarse/fine cluster cut + "interim until chain lands" design | SUPERSEDED banner: geometry emits ray-native into the one BVH world, no cut; weld INTENT stands, cut MECHANISM does not | ✅ DONE |
| 12 | DREAMFORGE.md | Normative §§14-35 (cluster-sole-pipeline, clustered-deferred cost, physics exact/far-coarse) + :67,170,181-192 (denoise/upscale chain, CPU-exact+GPU-surrogate tiers) | All amended in place to two-act render / Ananke→NET physics / no-LODs language; originals preserved verbatim under a dated LINEAGE section at doc bottom | ✅ DONE |
| 14 | features/CLIENT.md | Header read as a native-client target spec, not a frozen parity ledger | Header renamed: "FROZEN OLD-CLIENT PARITY LEDGER (never native target spec)" | ✅ DONE |
| 15 | NEURAL.md | Pre-two-act §§ (upscaler/denoiser chain, NR1-NR4 staging, exact/surrogate physics tiers) sat as normative alongside the two-act law | Doc rewritten: two-act law + silicon race verdicts + unified world-net ledger are the only normative sections; pre-pivot content moved under a dated LINEAGE heading, kept verbatim | ✅ DONE |
| 17 | NARUKO.md (:46,109-130,173-187), HANDOFF.md (pre-14:44) | Historical wave/ruling records read as live design without dating | Dated-historical prefixes inserted at each cited span + HANDOFF.md top, each pointing to the 07-18 ~14:44 supremacy block (CLAUDE.md ★ / HANDOFF.md line ~593, "supersedes everything above"); bodies NOT rewritten | ✅ DONE |
| 18 | CREATE.md:11, GRIMOIRE.md:34, RAIN.md:95, docs/perf/2026-07-17-onepath-budget.md | Tendril references to raster-cluster/vis-buffer vocabulary (sole cluster pipeline, meshlets, vis-buffer taps, chained denoise→upscale budget doc) | Inline dated-lineage annotations at each site pointing to RENDER.md §1 two-act law; perf doc gets a top banner (kept as a dated measurement record, not normative path) | ✅ DONE |
| 19 | VISIONFLOW.md:73-75 | "semantic LOD" cites the cluster-DAG rationale as its justification | Renamed to "semantic zoom"; cluster-DAG rationale reference severed — law now stands alone as a graph-navigation/perception law | ✅ DONE |

## Doc items (earlier passes this wave, referenced for continuity)

| # | Doc | Heresy | Disposition | Status |
|---|---|---|---|---|
| — | RENDER.md | Raster-cluster pipeline squatting two days after the two-act law | Purged; two-act law (trace→NET→screen) sole normative render design; raster-cluster content moved to LINEAGE appendix | ✅ DONE (4b765dd) |
| — | RENDER.md | Cost law stated in pixel/raster vocabulary ("we don't use pixels" ruling) | Cost law re-derived from rays: residency ∝ ray budget, content ∝ SSD, image belongs to the net | ✅ DONE (96fe612) |

## Code items (dispatched to separate lanes — tracked here for concordance, not executed by this pass)

| # | Item | Heresy | Disposition (ruled) | Status as of this ledger |
|---|---|---|---|---|
| A | Present-path upscaler (`one-render-path` lane: `trace→NEURAL DENOISE→NEURAL UPSCALE→present`) | Chained-stage present path contradicts two-act law | RULED 07-18 ~14:44: demoted to TRAINING-GENERATOR gates only, NOT present-path; present path = THE ONE LANE (full neural engine, net-on-tensor-path) | Ruling sealed (CLAUDE.md, HANDOFF.md ~14:44); code demotion (GAIA_NATIVE_TEMPORAL default OFF) landed (575e336). Perf record of the struck path kept dated (docs/perf/2026-07-17-onepath-budget.md, this pass). Present-path NET replacement in progress: r-direct spike beats the chain (d39ec88), Metal tensor door reopened (1966fcf), NEURAL-LIVE N0.a MPSGraph forward landed (1bd54f5) — cutover not yet closed |
| B | Temporal isolation → neural-live lane | Temporal reconstruction (gates/clamps/heuristics) risked shipping as present-path machinery | RULED 07-18 ~14:44: temporal = lab equipment (training ground-truth + history buffers) only, default OFF; light-fix lane harvests as training-generator, never present-path | Ruling sealed + default flip landed (575e336). neural-live lane active (1bd54f5 N0.a); full cutover (interop + pooling + gather + training loop) still queued per HANDOFF.md ★14:44 block |
| C | Great Chain LOD → chain lane | View-dependent cluster cut (Rite III "the Great Chain") is a physics/geometry LOD scheme, contradicts no-LODs + two-act ray-native law | RULED via this pass: GEOMETRY.md superseded (item 5), RITE-IX cut struck (item 11) — geometry emits ray-native into the ONE BVH world, no cut | Doc-level ruling DONE this pass. No dedicated "chain lane" code migration found in git log as of this ledger (searched for chain-lane/geometry-lane commits — none) — status UNCONFIRMED at the code level; report, not improvised. Re-check when a chain-lane branch/commit appears |

## Notes for the re-check
- Every LINEAGE/SUPERSEDED marker in this wave preserves original text verbatim — no silent erasure, adversary-charter discipline honored.
- Code items A and B have sealed rulings + partial landings; full cutover to the present-path NET is still open work, tracked in HANDOFF.md's ★14:44/★14:50 blocks, not this doc's job to close.
- Code item C has no independent code lane found — the Wilde Jagd should confirm with whoever dispatched it whether "chain lane" work exists under a different name, or was folded into the doc-only ruling above.
