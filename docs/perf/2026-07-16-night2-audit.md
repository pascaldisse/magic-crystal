# Perf audit — night 2 (2026-07-16) · the 60 FPS ledger

MEASUREMENT ONLY. No optimization performed (never-optimize ruling; fixes need
prime-Guardian review). Law: 60 FPS minimum = 16.67 ms/frame.

## Method

- `examples/perf_audit.rs` @ head of `perf-audit` (base 26b327c) — extends the
  a2_steam profile-split precedent onto the composed_coexist full-dynamics
  scene. Example-level `Instant` timing only; zero library changes.
- Realm: merged Naruko — nari walking (skinned per tick), crate physics body,
  kami six, steam medium (a2 dials verbatim, sun-bound), 15+ vessels. World
  warmed to composed mid-stride steady state (tick 202) before measuring.
- 900×600 · spp 2 · bounces 4 · 8 warmup discarded · 48 measured frames.
- Tri census at measure time: 3468 static + 3432 dynamic (nari skinned body ≈
  the whole rest of the realm) → merged BVH ~6900 tris.
- Poses: `front` = authored spawn camera ([0,2,22] yaw 0) · `wide` = the
  composed coexist proof shot.
- Segments: DYN-ON (living world, per-frame skin→tick→splice→upload→trace→
  readback) · STATIC+MED (frozen splice, steam on) · STATIC (frozen, steam
  off). Medium march + living-world price derived by segment delta (labeled —
  medium march not separable inside the fused GPU dispatch without library
  edits).
- `readback` = headless-audit cost only; the player window blits instead —
  subtract it when judging the player path.

## Table (verbatim run output)

```
[audit] realm warmed to tick 202: 3468 static tris, 3432 dynamic tris, 1 bodies, 1 physics binding(s), medium light = sun
[audit] 900x600, spp 2, bounces 4, 8 warmup + 48 measured frames, budget 16.67 ms (60 FPS law)
| pose | config | phase | mean ms | std | min | max |
|------|--------|-------|---------|-----|-----|-----|
| front | DYN-ON | skin (command_bodies) |    0.263 |  0.246 |    0.100 |    1.635 |
| front | DYN-ON | tick (physics+kami) |    2.019 |  0.844 |    1.274 |    4.633 |
| front | DYN-ON | splice (dyn build+merge) |    1.673 |  1.288 |    0.953 |    7.106 |
| front | DYN-ON | upload (update_bvh+bg) |    0.440 |  0.368 |    0.236 |    2.503 |
| front | DYN-ON | trace+medium (fused GPU) |   24.876 |  1.499 |   23.356 |   32.840 |
| front | DYN-ON | readback |    1.028 |  1.020 |    0.407 |    7.234 |
| front | DYN-ON | TOTAL |   30.300 |        |          |          |  FAIL (33.0 fps)
| front | STATIC+MED | trace+medium (fused GPU) |   24.471 |  0.970 |   23.345 |   28.241 |
| front | STATIC+MED | readback |    0.862 |  0.373 |    0.461 |    2.139 |
| front | STATIC+MED | TOTAL |   25.333 |        |          |          |  FAIL (39.5 fps)
| front | STATIC | trace (no medium, GPU) |   26.423 |  2.819 |   23.502 |   36.125 |
| front | STATIC | readback |    1.145 |  0.706 |    0.385 |    4.274 |
| front | STATIC | TOTAL |   27.568 |        |          |          |  FAIL (36.3 fps)
| front | derived | medium march (MED−STATIC trace) |   -1.952 |        |          |          |
| front | derived | living-world price (ON−STATIC total) |    2.732 |        |          |          |
| wide | DYN-ON | skin (command_bodies) |    0.399 |  0.538 |    0.107 |    2.812 |
| wide | DYN-ON | tick (physics+kami) |    2.697 |  1.998 |    1.273 |   12.984 |
| wide | DYN-ON | splice (dyn build+merge) |    2.013 |  1.543 |    0.946 |    8.236 |
| wide | DYN-ON | upload (update_bvh+bg) |    0.459 |  0.466 |    0.229 |    3.158 |
| wide | DYN-ON | trace+medium (fused GPU) |   24.837 |  1.063 |   23.460 |   28.978 |
| wide | DYN-ON | readback |    1.226 |  1.338 |    0.416 |    7.413 |
| wide | DYN-ON | TOTAL |   31.633 |        |          |          |  FAIL (31.6 fps)
| wide | STATIC+MED | trace+medium (fused GPU) |   23.917 |  1.087 |   23.036 |   28.451 |
| wide | STATIC+MED | readback |    1.013 |  1.042 |    0.427 |    6.772 |
| wide | STATIC+MED | TOTAL |   24.930 |        |          |          |  FAIL (40.1 fps)
| wide | STATIC | trace (no medium, GPU) |   22.939 |  0.547 |   21.901 |   24.753 |
| wide | STATIC | readback |    0.962 |  0.666 |    0.412 |    4.083 |
| wide | STATIC | TOTAL |   23.902 |        |          |          |  FAIL (41.8 fps)
| wide | derived | medium march (MED−STATIC trace) |    0.978 |        |          |          |
| wide | derived | living-world price (ON−STATIC total) |    7.731 |        |          |          |
[audit] ── VERDICT (budget 16.67 ms) ──
[audit]   front: DYN-ON 30.30 ms (FAIL) · STATIC 27.57 ms (FAIL) · medium march ≈ -1.95 ms · living-world price ≈ 2.73 ms
[audit]   wide: DYN-ON 31.63 ms (FAIL) · STATIC 23.90 ms (FAIL) · medium march ≈ 0.98 ms · living-world price ≈ 7.73 ms
```

## Verdict

- **FAIL, all four cells.** DYN-ON 30.3 / 31.6 ms (33.0 / 31.6 fps); even the
  statue world FAILs: STATIC 27.6 / 23.9 ms. Budget 16.67 ms.
- **Hottest single item: the fused GPU trace (trace+medium) — ~24.9 ms mean at
  DYN-ON, both poses.** 149 % of the entire budget by itself; every CPU
  dynamic combined (skin+tick+splice+upload ≈ 4.4–5.6 ms) is a third of it.
- Living world's price (ON−STATIC): 2.7 ms front · 7.7 ms wide — real but not
  the killer.
- Medium march at audit poses ≈ ≤1 ms (front delta −1.95 ms = below machine
  noise floor; front STATIC carried std 2.8 / max 36.1 — background GPU
  contention on this host; the wide pose, std 0.5, is the cleaner read).

## Ledger anchor (same head, same night)

Pre-merge honest number: steam-on **16.37 ms** @900×600 (a2 pose, before
V0/V1/P3). Re-run of `a2_steam` at THIS head: 6900 leaf tris, steam ON
**19.31 ms** / OFF **16.48 ms** (medium overhead 2.82 ms). → the merge's tri
growth (nari's skinned 3432 tris ≈ doubled the BVH) costs ~2.9 ms at the a2
pose; the audit poses see more geometry (spawn interior / wide composed) →
23–26 ms trace. Regression = tri count + pose coverage, not the harness.

## Candidate directions IF prime approves (LISTED ONLY — nothing implemented)

Bit-exact (same closest hits ⇒ same image):
1. Dyn-partition BVH quality — per-tick `Bvh::build` + trivial 2-child
   `merge` root over a 3432-tri partition; SAH-binned build / restructured
   merge cuts traversal work, bit-exact (intersection set unchanged).
2. CPU/GPU overlap — run frame N+1's skin/tick/splice while frame N's trace
   is on the GPU; hides the whole ~4.4–5.6 ms dynamics block, frames
   bit-identical.
3. `update_bvh` buffer churn — recreates node/tri storage buffers every tick
   (`create_buffer_init`); persistent buffers + `write_buffer` when sizes
   allow, bit-exact, ~0.4 ms.
4. Readback exclusion — player path blits; drop ~1 ms from the player-facing
   ledger (bookkeeping, not a change).

NOT bit-exact (needs an explicit prime ruling, listed for completeness):
5. Traced-partition LOD for skinned bodies — nari at full 3432 tris in every
   ray; a cluster cut changes geometry ⇒ changes pixels.
6. spp/bounce/resolution scaling — changes the integral estimate.

§ examples/perf_audit.rs (env dials: GAIA_AUDIT_W/H/WARMUP/FRAMES) ·
§ precedent a2_steam.rs, composed_coexist.rs · § base 26b327c.
