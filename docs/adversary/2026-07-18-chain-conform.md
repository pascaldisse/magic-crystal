# ADVERSARY REPORT — chain-conform · 2026-07-18

VERDICT: HOLDS

## CONCORDANCE

- No-LODs law → `CLAUDE.md:92` (“NO LODs (cluster law)”); former projection/view-selected runtime cut deleted (`e2530da`); renderer consumes full loss-free level-0 leaves always → `packages/transmute/FORMAT.md:5-8,110-112`.
- One-partitioner law → METIS sole mandatory backend; no feature gate/fallback; backend failure = typed transmutation failure → `packages/transmute/src/partition.rs:1-5,57-96`; serialized directory names sole backend `"metis"` → `packages/transmute/FORMAT.md:23-25,70-78`.
- ENTROPY law → deterministic state coordinate; no randomness, every value = `hash(seed, entropy, entity)` → `ENTROPY.md:17-29`; METIS seed derives only from `(world_seed, entropy, canonical_geometry_identity)` → `packages/transmute/src/partition.rs:25-50`; independent entropy-state partition ordeal → `packages/transmute/tests/inquisition.rs` (`finding8_entropy_state_is_byte_identical`).

## PERFORMANCE AUDIT

- Source → `proof/2026-07-18-lod-after-perf-audit-final.txt`; 900×600, spp 2, 16.67ms / 60 FPS budget.
- Front overlap wall → 15.154ms; PASS; static → 15.651ms; PASS.
- Wide overlap wall → 16.581ms; PASS; static → 15.724ms; PASS.
- Wide overlap margin → 0.089ms; razor-thin; max 27.080ms observed.

## SUITE

- Build token every cargo → `/Users/pascaldisse/projects/magic-crystal/.build-lock`; `nice -19 -j2` attempted; host denied priority change (`nice: setpriority: Permission denied`), cargo continued at host scheduler priority.
- Scrying Glass → lib + 18 integration-test binaries; each timeout 900s.
- Packages → crystal, aether, char-editor, homunculus, vessel, sama, elements, fracture, transmutation, kami, seed, pleroma, wired, jormungandr, oracle, steiner; each timeout 300s.
- Grand total → 35 commands; 35 pass; 0 failed; 573s.
- Log → `proof/2026-07-18-chain-conform-suite.log`.

## GAP

- None.
