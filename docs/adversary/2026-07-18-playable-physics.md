# PLAY — playable physics close-out

2026-07-18 · branch `playable-physics` · scope: building push + fluid door.

## Verdict

**VERDICT: HOLDS — 419 / 419 green classes.**

- suite discovery: 423 classes.
- passed: 419.
- failed: 0.
- ignored: 4; green denominator: 419.
- rerun logs/count: `proof/PLAY-d-suite/final.summary` → `proof/PLAY-d-suite/full/*.log`.

## Canon re-derivation

| ordeal | old canon | scene/code derivation | canon |
|---|---:|---|---:|
| oracle frustum vessels | 36 | old 36 one-vessel ids: `naruko_terra`, `naruko_seawall`, `naruko_sea`, `lighthouse_rock`, `lighthouse_tower`, `naruko_pier`, `naruko_chain_posts`, `naruko_city_massing`, `naruko_lantern`, `naruko_stall_massing`, `lighthouse_beacon`, `naruko_chrome_orb`, `nari`, `naruko_cat`, `signal_ring_a/b/c`, `naruko_mirror`, `naruko_mirror_minor`, `naruko_kami_orb`, `naruko_crate`, `naruko_stack_crate_0/1/2`, `naruko_show_chrome`, `naruko_show_mirror`, `naruko_show_light_a/b/c`, `playground_stack_1/2/3/4`, `playground_pyramid_0/1/2`; scene adds mesh `bldg_tower` + mesh `bldg_basin` → 36 + 2 | 38 |
| scrying dynamic split | 22 | code selects `behavior`/physical-`body`: 9 behavior meshes (`lighthouse_beacon`, `signal_ring_a/b/c`, `naruko_lantern`, `naruko_kami_orb`, `naruko_show_light_a/b/c`) + 13 old physical bodies (`naruko_crate`, `naruko_stack_crate_0/1/2`, `playground_stack_0..4`, `playground_break_crate`, `playground_pyramid_0..2`) + `bldg_tower` body; `bldg_basin` mesh-only → static | 23 |

- result → stale world-truth counts; no physics delta.

## Concordance

- building → `bldg_tower`, authored scene data; existing `elements::building::erect` bonded structure + vessel/fragment machinery; player push door unchanged: view-ray pick → `Physics::push_targets()` → `Op::Impulse` → `RenderScene::tick_with_ops` (§ `packages/scrying-glass/examples/building_push.rs`).
- fluid → `bldg_basin`, authored scene/container data; existing `elements` fluid fill/settle machinery; entering volume through the same data/op surface (§ `packages/scrying-glass/examples/fluid_door.rs`).
- one door / no new system → `CLAUDE.md`, DESIGN LAW: one full physics engine; no fallback/prototype path. `GRIMOIRE.md` ANANKE row: constraints → one ops door. Scene data + existing vessel/weld/fluid machinery; no parallel gameplay system.
- params → **IRON**. Push dials: reach/speed/aim radius; door volume: dimensions/centre/velocity/radius factor; fluid spacing, wall height, settle ticks. No hidden tuning claim.
- buoyancy → absent; not claimed. Source: `docs/perf/2026-07-18-fluid-container-boundary-verdict.md` → gate 4 OPEN; compression-only stable gates 1–3; container boundary detonation; `ordeal_buoyancy_rises` expected-red ignored.

## Collapse witness

`proof/PLAY-report.md`:

- control → 396-particle, 3-storey tower: authored top `6.600 m` → settled `6.567 m`.
- push → first whole-body failure tick `49`; fracture journal starts tick `90`; 959 events tick `69`, 963 tick `900`.
- collapse → top `6.567 m` → `1.855 m` tick `69` → `0.261 m` settled; drop `6.306 m`; all 396 particles traceable.
- debris → 23 → 367 dynamics; final floor minimum `0.451 m`; max speed `0`.

## Fluid witness

`proof/PLAY-report.md`:

- residual film: 882 particles; settled surface `0.2093 m`.
- burst: 648 particles; `2.4 × 0.8 × 0.8 m`; centre `(0, 1.5, 1.0)`; velocity `(0, -2.5, -3.0)`; total 1530.
- splash: tick 31 surface `1.8066 m`; basin fraction `1.000`; max speed `4.012 m/s`.
- settle: tick 580 surface `0.3913 m`; basin fraction `1.000`; max speed `0.0659 m/s`; KE `0.1679 J`; flatness `0.2329 m`.

## Boundary

- HEADLESS state proof only; no pixels/window playthrough.
- `bldg_basin` mesh lacks live world `body` wiring; witness uses matching physics container.
- no float/sink claim.
- canonical-count gate → discharged by scene/code hand derivation + targeted ordeal/neighbor rerun; 419 / 419 green.
