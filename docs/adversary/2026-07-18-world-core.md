# WORLD-CORE ADVERSARY REPORT · 2026-07-18

## VERDICT: HOLDS

## CONCORDANCE

### Ops door shape ↔ package law

- Law chain + green-suite merge gate → `CLAUDE.md:1-3,27-31`.
- Crystal = core only; replaceable functions stay package spirits → `README.md:53-69`.
- One ops door → `GRIMOIRE.md:294`; HTTP contract accepts one op, array, or `{ops,from,dev}` and returns `applied` → `features/SERVER.md:17`; WS contract = `{type:'ops',ops,from,dev?}` → `features/SERVER.md:20`.
- World-core: `OpBatch` enters `WorldCore::apply`; normalized applied ops only → `packages/steiner/src/world_core.rs:103-160`. HTTP batch ordeal → `packages/steiner/tests/world_core.rs` `http_door_batch_shape_preserves_dev_set_and_reset`.
- Concordant → Crystal owns typed ops; Steiner authority owns application/persistence; Wired transports the same batch shape. No parallel mutation door.

### Dev write-back ↔ runtime scene truth

- Scene-model contract → dev batches write authored scene entities; player-layer changes never write scenes → `features/SERVER.md:40,46-52`.
- World-core: live recorder mutates first; only `batch.dev` invokes scene write-back → `packages/steiner/src/world_core.rs:103-160`; write-back excludes presence/persist and writes owning scene only → `packages/steiner/src/world_core.rs:285-333`.
- Live proof → runtime red→green leaves authored court red; reset restores disk-red; dev blue writes authored court blue → `proof/world-core-live/README.md:7-13`.
- Concordant → runtime state remains live/journaled; explicit dev batches alone alter scene documents.

### Journal ↔ ENTROPY

- Entropy = deterministic x-axis; `(seed, entropy, recorded input journal)` reconstructs state; save = `(seed,journal)` → `ENTROPY.md:17-25`; arrow = growing journal → `ENTROPY.md:31-39`.
- World-core increments entropy once per non-empty batch, records normalized batch, then emits bounded observation events → `packages/steiner/src/world_core.rs:135-160,182-203,270-282`.
- Steiner replay contract → same `(seed,journal)` rebuilds bit-identical state → `packages/steiner/src/lib.rs:3-17`.
- Concordant → events are bounded observation; Steiner journal is replay ledger; reset materializes disk truth as concrete recorded ops.

## SUITE

- `419 passed · 0 failed` → `proof/world-core-live/finisher-suite-counts.tsv`.
- Scrying Glass isolated per lib/test binary; `rite5` cat-circuit binary under 900s; all other cargo invocations under 300s; build token + `nice -n 19` + `-j2` per cargo.
- Scrying target `viii3b_ordeals` rerun after cleanup-trap false nonzero; both executions: `4 passed`; suite count records target once.

## LIVE PROOF

- Branch range → `245159c..2f4286e`.
- I → superscene realm spine; II → journal/set/reset/write-back; III → HTTP batch contract; IV → running HTTP mutation + pixel artifacts.
- Live sequence → boot 2 scenes/5 entities; runtime green, reset disk-red, dev blue; entropy 1→3; Steiner frames 1→3; PNG center RGBA red→green→blue → `proof/world-core-live/README.md:5-17`.

## GAPS

- None.
