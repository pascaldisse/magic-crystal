# VII-1 seam-step ε derivation

Adversary MUST-FIX (rite6-fragcol-sonnet review of VII-1): doc claimed,
0 bytes on disk = costume violation. This is the real derivation, verified.

## Bound

```
floor_delta_bound = g_max × per_tick_advance + slack
```

- `g_max = tan(acos(WALL_NORMAL_Y_COS_CUTOFF))`, `WALL_NORMAL_Y_COS_CUTOFF = 0.3`
  (`packages/scrying-glass/src/player.rs:31`) → `g_max = 3.1798`.
  ONE source of truth — the same constant `Ground` uses to drop wall
  triangles (`player.rs:285`, `player.rs:312`).
- `per_tick_advance = walk_speed / 60 Hz`, `walk_speed` = env param
  `GAIA_PLAYER_WALK`, default `6.0` m/s (`player.rs:203`) → `0.1` m.
- product: `3.1798 × 0.1 = 0.31798`.
- `slack = magnitude × ε × 16` — pure fp headroom, `6.1e-5`
  (`0.02%` of the product).
- `total = 0.31798 + 0.000061 = 0.31804`.

## Measured

- crossing fwd (authored→generated): `0.01817` m
- crossing rev (generated→authored): `0.00910` m
- both ~17× under the `0.31804` bound.

## Anti-vacuous twin

Guard discriminates, not a tail that always passes: mis-author the seam
1 m off the field → the SAME guard fires, caught via the airborne clause
(a 1 m down-step exceeds `ground_snap = 0.35`, so the walker is airborne
that tick and the per-tick-delta guard trips).

## Source

Adversary's verified derivation (rite6-fragcol-sonnet review), transcribed
house voice for the seam_step ordeals in
`packages/scrying-glass/tests/seam_step.rs`.
