# PLAY.c — headless playable-physics proof

2026-07-18 · branch `playable-physics` · physics-state snapshots; no window.

## Building push — `bldg_tower`

Witness → `packages/scrying-glass/examples/building_push.rs` · stdout →
`proof/PLAY-c-building-push.log` · snapshots →
`proof/building-push-{pre-push,mid-collapse,settled}.json`.

- control → 396-particle, 3-storey tower holds: authored top `6.600 m` → settled
  top `6.567 m`; whole target remains.
- push → window-equivalent pick selects `bldg_tower`; first whole-body failure:
  harness tick `49` (solver fracture journal starts at tick `90`).
- cascade → first 12 journal entries: tick `90`, y=`0.000..0.061 m`;
  ground-floor/base bonds first. `959` fracture events by snapshot tick `69`;
  `963` by tick `900`.
- progressive drop → top y: `6.567 m` rest → `1.855 m` at tick `69` →
  `0.261 m` settled; measured drop `6.306 m`. All 396 original tower particles
  remain traceable through the fragment cascade.
- debris → dynamic entity count `23` → `367` at break; final debris floor
  minimum `0.451 m`; settled snapshot max speed `0`, moving particles over
  `0.01 m/s`: `0`.

## Fluid door — `bldg_basin`

Witness → `packages/scrying-glass/examples/fluid_door.rs` · stdout →
`proof/PLAY-c-fluid-door.log` · snapshots →
`proof/fluid-door-{pre-burst,mid-splash,settled}.json`.

- residual film → `882` particles; settled surface `0.2093 m` at tick `30`.
- burst → `648` particles; `2.4 × 0.8 × 0.8 m`; centre `(0, 1.5, 1.0) m`;
  velocity `(0, -2.5, -3.0) m/s`; combined pool `1530` particles.
- splash → tick `31`: surface `1.8066 m`; peak also `1.8066 m` at tick `31`;
  `1.000` fraction inside basin; max speed `4.012 m/s`.
- spread/settle → surface `0.4965 m` at tick `156` → `0.3913 m` at tick `580`;
  inside fraction `1.000`; max speed `0.0659 m/s`; KE `0.1679 J`; sampled
  surface flatness `0.2329 m`.

## Claim boundary

True → existing bonded-fracture/fragment machinery shears an anchored tower;
existing fluid fill/settle machinery receives a parameterized entering volume,
splashes, spreads, and settles in an equivalent basin container.

Not claimed → pixels/window playthrough; authored `bldg_basin` mesh has no live
world `body` wiring, so the fluid witness uses its matching physics container;
buoyancy/Archimedes absent — no float/sink assertion.
