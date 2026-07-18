# ADVERSARY REPORT — matrix-retina · 2026-07-18

VERDICT: HOLDS

## CONCORDANCE

- Act-evidence → typed SenseEvent/ActResult boundary; retina returns primary-ray geometry evidence only; no framebuffer, radiance, secondary rays, or network state. `docs/proposals/MIND.md:58-61`; `packages/scrying-glass/src/retina.rs:1-2`; `packages/scrying-glass/src/main.rs:1770-1796`.
- Sense exposure → structured eye-pose channels, never color/pixel round-trip. `RAIN.md:59-70`; `/retina` request/reply organ `packages/scrying-glass/src/main.rs:509-524`.
- No Pleroma-internals naming → response fields = depth/normal/world-pos/entity/material IDs + local tables; no acceleration-structure or renderer-private field crosses the organ. `packages/scrying-glass/src/retina.rs:62-86, 90-136`.
- Foveal law → requested layers + resolution pyramid; foveal windows trace independently at higher resolution, never upscale base cells. `RAIN.md:71-83`; `packages/scrying-glass/src/retina.rs:94-96`; `packages/scrying-glass/src/main.rs:1791-1795, 2188-2222`.
- IRON params → fixed God-canvas remains env-parametrized at 640×480; retina dimensions remain endpoint-local, not present-path selectors. `packages/scrying-glass/src/main.rs:141-149`; `docs/adversary/2026-07-18-present-conform.md:3-6`.

## LATENCY

- Naruko 64×64 core, depth-exact cache comparison → 269.8ms old per pull → 85.2ms cached warm; 3.17× reduction.
- Exactness guard → old depth array == cached depth array; four same-epoch pulls → one build. `packages/scrying-glass/src/retina.rs:172-211`.

## SUITE

- Build token → `/Users/pascaldisse/projects/magic-crystal/.build-lock`; every cargo invocation `nice -19 -j2`.
- Scrying Glass → lib + 18 integration-test binaries; `viii3b_ordeals` timeout 900s; all other commands timeout 300s.
- Other packages by metadata name → crystal, aether, char-editor, homunculus, vessel, sama, elements, fracture, transmutation, kami, seed, pleroma, wired, jormungandr, oracle, steiner.
- Grand total → 35 commands; 35 pass; 416 passed; 0 failed; 535s.
- Log → ignored `target/matrix-retina/full-suite-final-2026-07-18.log`; summary → ignored `target/matrix-retina/full-suite-final-2026-07-18.summary`.

## GAP

- Native-window `/retina` post-merge live pull → UNVERIFIED: NO WINDOWS constraint; suite green only.
