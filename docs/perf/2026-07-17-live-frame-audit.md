# Live-frame audit — 2026-07-17 · why the vista window runs ~30 fps

Branch `window-playable`. Harness: `examples/live_loop_audit.rs` (measurement
only — replicates `run_render_loop`'s EXACT per-frame sequence headless at the
live defaults: 640×480 internal, spp 2, NO medium in the surface path, vista
pose (0,1.7,24) yaw 0). Phase-split, median over 80 frames after 10 warmup.

## Verdict — the culprit is CPU physics, NOT the render, NOT serial-vs-overlap

`scene.tick()` → `Physics::step` → `Solver::step` runs a **brute-force
particle-vs-every-collider-triangle** collision pass (`solve_collision_normal`
+ `solve_body_collisions`), no broadphase, ×8 substeps:

```
particles = 108   (the crate's bonded-box lattice)
collider_tris = 11684   (the ENTIRE static world mesh — every static part +
                         terrain is appended into one flat collider triangle list)
substeps = 8
→ 108 × 11684 × 8 = 10,094,976 contact-depth tests PER physics step
```

That is **16.25 ms median, dead stable (std ~0.4, min 15.8 / max 17.6)** — it
alone ≈ the entire 16.67 ms/60-fps budget. `kami::tick_decorative` is 0.02 ms
(noise); pose write-back 0.05 ms.

## Phase table — live loop shape, 640×480, vista pose, spp 2

| phase | median ms | mean | min | max |
|-------|-----------|------|-----|-----|
| skin (`command_bodies_walked`) | 0.536 | 0.540 | 0.485 | 0.689 |
| **tick (physics.step, kami≈0.02)** | **16.250** | 16.260 | 15.808 | 17.620 |
| splice (refit/merge) | 0.225 | 0.234 | 0.196 | 0.520 |
| upload (update_bvh+bg) | 0.457 | 0.464 | 0.413 | 0.585 |
| trace (spp 2, no medium, GPU wait) | 10.948 | 12.478 | 7.167 | 30.157 |
| blit (2× present, GPU wait) | 0.583 | 1.141 | 0.457 | 12.179 |
| **SERIAL TOTAL (median sum)** | **28.998** | | | **34.5 fps** |
| OVERLAP wall (proven lever) | **17.614** | 17.745 | 16.991 | 22.362 | **56.8 fps** |

The SERIAL total **29.0 ms = 34.5 fps** matches the live HUD (27.5–34.4 ms ≈
29–36 fps) to the millisecond — the harness faithfully reproduces the live loop.
(trace's high max is background GPU contention from the live window + a qemu
core; the physics phase is contention-immune, confirming it is real work.)

## Answering the contradiction (each suspect)

1. **Serial vs overlap** — REAL but not sufficient. The live `run_render_loop`
   is serial (submit → present, `poll(Poll)` non-blocking, no per-frame
   readback on the surface path). But wiring the merged CPU/GPU-overlap lever
   in gets only to **17.61 ms — STILL FAIL** (measured, OVERLAP row above),
   because `physics.step` (16.25 ms) is the CPU critical path and overlap can
   only hide GPU behind CPU, not CPU behind CPU. Overlap floor = the CPU sum
   (skin+tick+splice+upload ≈ 17.5 ms) > 16.67. **Wiring overlap does NOT reach
   ≥60 fps here — so it was NOT applied** (a risky loop change for no win).
2. **spp / accum** — identical (spp 2, accum reset every frame both live and
   harness; moving dynamics reset it). Not the cause.
3. **Blocking readback** — the live SURFACE path has NO per-frame readback
   (offscreen copy is `map_buffer_on_submit`, only on the occasional capture
   slot). Not the cause.
4. **Vista heavier than law poses** — NO. The vista (0,1.7,24) ≈ the audit
   `front` (0,2,22); trace at 640×480 is ~11 ms, cheaper than the audit's
   900×600. The pose is not the problem.
5. **HUD/vsync artifact** — the 29 ms serial work is REAL (matches the harness).
   With Fifo present a ~29 ms frame is already past two vsync intervals, so the
   HUD reads genuine work, not a pacing artifact.
6. **kami/dynamic behaviors** — 0.02 ms. Negligible.

## Why the "law harness PASS (11.26/13.23)" no longer holds

That PASS is **STALE**. It predates the realm growing: night-2 census was 3468
static / 3432 dynamic tris, **1** body, tick **2.0 ms**. The current realm is
11684 static / 9132 dynamic tris, **2** bodies + a **108-particle bonded crate**
+ 4 physics bindings, tick **16.25 ms**. Re-running the merged `perf_audit`
TODAY on the current realm at 640×480 gives **front DYN-ON OVERLAP wall 18.12 ms
— FAIL** (tick 16.77 ms inside it). The law audit itself fails on the live
realm; its recorded PASS numbers are from the smaller pre-crate world.

## The exact fix is out of the window lane's sanctioned scope

The only sanctioned window-lane code fix (wiring the overlap lever) does NOT
achieve ≥60 fps (proven above). The actual fix is a **physics collision
broadphase** — give `Solver`'s collision passes a spatial acceleration
structure so a resting 108-particle crate does not test 10 M triangle pairs per
step (or sleep resting bodies). That touches the determinism-critical `elements`
solver (its ordeals + hash-identity), is NOT semantics-preserving by
construction, and is the physics/Architect domain, NOT a window-lane loop lever.
Per the task's own decision tree ("if the culprit is inherent with all exact
levers already live: report honestly with the phase table") this table IS the
Architect's decision data.

## Honest gaps

- No fix committed to the live loop: the sanctioned lever (overlap) cannot reach
  the ≥60 fps acceptance while `physics.step` = 16.25 ms; applying it would be a
  risky, non-improving loop change. The live window was therefore left
  UNTOUCHED (no relaunch — the Architect keeps playing pid 20193).
- Trace/blit medians carry background-contention noise (live window + qemu core
  on the host); the physics number, being contention-immune and std ~0.4 ms, is
  the reliable signal and is what the report rests on.
- The broadphase/sleep fix is UNIMPLEMENTED and UNVERIFIED here — flagged as the
  Architect's physics-domain call.
