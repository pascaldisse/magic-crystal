# ADVERSARY — neural-live (N4, teacher-gated firefly loss): v5 VERDICT — BLACK STANDS (both bars)

## N4 FINISHER (2026-07-19) — VERDICT: BLACK STANDS. v5 fails resid_still +0.00970 AND sparkle_still +8.83 @640×480.
- **The window stays BLACK by law.** v5 (teacher-gated firefly-loss net, sha
  `01b67a4550f8`, warm v3 → resumed ep9 → 90 ep) REAL ordeal @640×480:
  resid_still 0.04470 vs 0.035 → **FAIL +0.00970**; sparkle_still 48.83 vs 40
  → **FAIL +8.83**; tvar/resid_move/ghost PASS w/ margin. No stamp →
  `verify_stamp` false → present black. Gate re-pinned: real_image_gate 2/2
  (unstamped denied). Bars UNTOUCHED.
- **The teacher gate MOVED along the Pareto front, it did not escape it.**
  v4→v5: resid 0.051→0.0447 (real cyan waterline partly recovered, ~0.006 of
  the N3 over-clamp back) BUT sparkle 39→49 (invented dots now survive in
  mid-bright neighbourhoods the gate scored not-dark). Two-of-five fail where
  v4 failed one. Same front, worse point.
- **The seesaw (training curve, § scratch/v5-train-resume.log):** across 90 ep
  val sparkle oscillates 40→330→64→168→185→52→318→197→156 while resid holds
  0.035–0.041 — no joint minimum, `pass=false` all epochs, best saved = ep9
  fallback (lowest sparkle 40.5 / resid 0.0411). A scalar per-channel
  excess-over-cap penalty, gated or not, rides ONE front: crushing an invented
  dot in a mid-bright neighbourhood also dims the real emissive there.
- **Defect read (both eyes @640×480, § proof/neural-live/s24-*.png):** fireflies
  NOT gone — cyan/blue specks scatter the water/waterline; cyan waterline still
  broken dashes, right building-base glow missing (resid climb); lit windows
  CRISP (gate off, MSE rules). Better cyan than v4's smear; sparkle worse.
- **One sentence:** the teacher gate proved the diagnosis exactly — it recovered
  the real cyan v4 over-clamped and paid it straight back in fireflies, sliding
  along the sparkle↔resid front instead of stepping off it, so BLACK stands and
  the next attempt must separate invented energy from real energy structurally
  (temporal flicker / matched high-freq residual), not with a spatial cap.
- Proof: `scratch/v5-train-resume.log`, `scratch/v5-ordeal.log`,
  `proof/neural-live/s24-{still,moving}{,-teacher}.png`, `docs/perf/2026-07-19-neural-live-n4.md`.

---

# ADVERSARY — neural-live (N3, firefly loss): v4 VERDICT — HONEST BLACK (resid)

## N3 FINISHER (2026-07-19) — VERDICT: BLACK STANDS. v4 fails resid_still +0.01599 @640×480.
- **The window stays BLACK by law.** v4 (firefly-loss net, sha 2cc827a) ordeal
  re-run FRESH @640×480: resid_still 0.05099 vs bar 0.035 → **FAIL +0.01599**;
  sparkle 39.06/40 PASS, tvar/resid_move/ghost all PASS w/ margin. No stamp →
  `verify_stamp` false → present black. Gate flip proof: gate did NOT flip to
  allow (real_image_gate 2/2 unstamped-denied, no v4.stamp on disk).
- **The defect (both eyes, defects first):** firefly loss killed v3's sparkle
  (345→39, clears the bar) but OVER-CLAMPED the real emissive it bordered — the
  **cyan waterline is a dim broken blue smear, NOT the teacher's clean cyan
  dashes.** That suppressed emissive is the resid climb 0.0325→0.051. Lit windows
  stay CRISP (high-cap, clamp-free). Motion no-ghost. v4 traded a sparkle-fail
  for a resid-fail: a scalar spatial clamp cannot separate an invented dot from
  the real cyan neon it sits on — killing one kills the other.
- **wip 096c70e:** runaway retrain (surgical margin) killed; weights regression
  reverted to canonical 2cc827a; combined-criterion trainer code kept. Retrain
  log proves non-convergence (sparkle 52→196 as MSE refits). NEXT: edge-aware /
  emissive-preserving loss — a single clamp is on a sparkle↔resid Pareto front.
- Proof: `scratch/v4-ordeal-n3.log`, `proof/neural-live/s21-{still,moving}{,-teacher}.png`.

---

# ADVERSARY — neural-live (N0.g, shift 9): the ONE net presented live

## SHIFT 9 UPDATE (S8 default flip + S9 encode pipeline) — VERDICT: HOLDS parity, STILL FAILS 60fps but HALVES the gap.
- **S8 flipped the default to MPSGraph** (the 1.8× faster path N0.f measured);
  the chain is now the opt-in lab A/B (`GAIA_NATIVE_NET_CHAIN=1`). Parity
  re-gated: MPSGraph vs CPU 1.9e-6, MPSGraph vs chain 4.8e-7. The default is no
  longer the slower path — the shift-8 concordance gap #2 is CLOSED.
- **S9 pipelined the ~14 ms MPSGraph encode onto a background thread** (double-
  buffered sets, 0 latency — the net still reads the frame's own gather; the
  pre-encode records references not data, render commits in-order). Net wall
  **20.5 → 4.16 ms** (encode hidden). TESTED live @640×480, both PNGs read.
- **Honest tax:** trace regressed **6.5 → 12.7 ms (+6 ms)** — the encode thread's
  CPU contends with the render thread's trace submission on the shared Metal
  queue. The chain-pipeline A/B (encode ≈0.4 ms → thread idle) keeps trace at
  6.99 ms, ISOLATING the cause. TOTAL **30.05 → 20.07 ms** (~33 → ~50 fps): real
  tested win, ~6 ms of it eaten by the tax.
- **60 fps: STILL VIOLATED, 3.40 ms short (~50 fps).** The net stage is solved
  (4.16 ms); the sole remaining thief is the trace regression. Recover trace's
  6.99 ms baseline (dedicated encode `MTLCommandQueue` + `MTLSharedEvent` for the
  gather→net dependency) → TOTAL ~12.9 ms → 60 fps MET. Not attacked this shift.
- Source: `docs/perf/2026-07-18-neural-live-n0.md` N0.g; proofs
  `proof/neural-live/n0g-{mpsgraph,chain}-pipeline.log` +
  `s9-pipeline-{mpsgraph,chain}-net.png`.

---

# ADVERSARY — neural-live (N0.f, shift 8): the ONE net presented live

Scope: the `GAIA_NATIVE_NET_PRESENT` live path — trace low radiance + native
AOV → GPU feature gather → net forward (chain default / MPSGraph A/B) → GPU
undo-log-demod → nearest blit to surface. Branch `neural-live`.
Sources cited by line: `packages/scrying-glass/src/{main.rs,rdirect_live.rs,
integrator.wgsl}`, ordeal `tests/rdirect_gather_ordeals.rs`.

## VERDICT: HOLDS on parity, FAILS on budget.
- **Parity — HOLDS.** Chain vs CPU 1.9e-6, chain vs MPSGraph 4.8e-7 (ordeal
  `n0b_gather_and_shared_forward_match_cpu`, live run: GATE A 9.5e-7 / GATE B
  1.9e-6 / n0f 4.8e-7, all under the 1e-3 gate). Frame is coherent, colours
  bounded, demod wired right (both belief PNGs read).
- **Budget — FAILS the 60-fps law.** Neither path hits 16.67 ms @640×480:
  chain TOTAL **53.06 ms** (~19 fps, 3.2× over), MPSGraph **30.05 ms** (~33 fps,
  1.8× over). The cutover cannot claim real-time.
- **The chosen default is the slower path.** S5 makes the raw chain the frame
  default; the honest measurement says the chain is **1.8× slower** than the
  MPSGraph alternative it replaced. Default stands per the S5 charter, but it is
  a perf regression, documented below and in `docs/perf/2026-07-18-neural-live-n0.md`.

## The budget (≥300-frame samples, machine quiet, vs 16.67 ms)

| stage    | CHAIN med/p95 | MPSGraph med/p95 |
|----------|---------------|------------------|
| trace    | 6.65 / 9.25   | 6.53 / 9.22      |
| gather   | 1.06 / 1.99   | 1.01 / 1.99      |
| net wall | 43.15 / 48.29 | 20.51 / 24.19    |
| net GPU  | 42.76 / 47.81 | 6.65 / 10.60     |
| demod    | 0.66 / 1.41   | 0.64 / 1.26      |
| present  | 0.20 / 0.27   | 0.19 / 0.25      |
| **TOTAL**| **53.06 / 58.54** | **30.05 / 34.05** |

Only `net` moves between runs; `trace/gather/demod/present` are within noise, so
the net split is a true GPU-cost difference, not machine load. The chain kills
MPSGraph's ~13.9 ms CPU encode (chain CPU = 43.15−42.76 = 0.39 ms) but its
un-fused per-layer `MPSMatrixMultiplication` dispatches cost **6.4× the GPU**
(42.76 vs 6.65 ms). Trading a 14 ms CPU wall for 36 ms of GPU is the regression.
Real target: a FUSED GPU forward that keeps MPSGraph's ~6.6 ms GPU and sheds its
~14 ms CPU encode (MTL4 tensor / hand-fused compute) — NOT the chain.

## CONCORDANCE — does the code obey the laws? (cite lines)

- **ONE RENDER / output-or-nothing** — HOLDS. The net path presents its frame
  and captures the SAME image; there is no second render. `net_present_frame`
  blits `present_accum` to the offscreen capture (`main.rs:2235`, "net offscreen
  present") AND to the live surface (`main.rs:2246`, "net surface present") from
  one `present_blit_bg` — the screenshot is the frame the window shows. The
  forward leaves radiance ON the GPU, no readback fork (`rdirect_live.rs:745`
  `forward_shared_gpu`; the `Vec` path `forward_shared` is ordeal-only,
  `rdirect_live.rs:774`).

- **640×480 LAW (`0a25530`)** — HOLDS (this shift's fix). Before S4 the net
  target was the WINDOW (`surface_w×surface_h`) — a small trace neurally
  enlarged to the window, the exact thing the law forbids. Now trace == net ==
  present == the canvas: `main.rs:2126-2127` `(low_w,low_h)=(target_w,target_h)=
  (render_width,render_height)` (default 640×480, `main.rs:234-235`); the window
  gets it by a nearest display blit only, `main.rs:2206`
  `blit_uniform.surface=[surface_w,surface_h,1,0]` → the shader's nearest branch
  `integrator.wgsl:882` (`u.surface.z==1u`). Display scaling ≠ rendering, no
  neural enlarge.
  - RESIDUAL: the legacy (non-net) `render` path still defaults
    `upscale_mode=0` (bilinear, `main.rs:236,255`). The net path — the ONE path
    at cutover — is nearest; the bilinear default only survives on the dying
    legacy blit. Flagged, not fixed (out of the net-path scope).

- **NAMING** — HOLDS. `RdirectLive` (the live net), `MatmulChain` (the raw GEMM
  chain), `NetPresent` (the pooled rig), `DemodPass`. The A/B knob is honest:
  `use_mpsgraph` Cell defaults false = chain (`rdirect_live.rs:472`), env
  `GAIA_NATIVE_NET_MPSGRAPH=1` or `set_use_mpsgraph` flips it
  (`rdirect_live.rs:760`); the `[n0e]` line names `net[wall .. gpu ..] demod
  present` — no hidden folding after S3.

- **Absolutes** — 60 FPS minimum: **VIOLATED** (19/33 fps, see budget) — the one
  law this path breaks, and the whole reason the cutover is not yet callable.
  NO LODs / no neural interpolation: HOLDS (one canvas res, nearest display
  blit, no learned upscale in the live path). One light pass, no hardcoded res
  (`GAIA_NATIVE_RENDER_W/H`, default 640×480): HOLDS.

## Gaps carried forward
1. **60 fps unmet** — net stage is the wall on BOTH paths. Chain: 42.8 ms GPU
   (un-fused). MPSGraph: 6.6 ms GPU + 13.9 ms CPU encode. The win is a fused
   forward (MTL4 tensor / hand-fused compute) OR a one-frame pipeline that drops
   the blocking wait (`root.waitUntilCompleted()` in `run_executable`,
   `rdirect_live.rs`). Neither attacked this shift.
2. **Default = slower path.** S5 charter keeps chain default though MPSGraph is
   1.8× faster today. Revisit at cutover with the budget on the table.
3. **Quality** — static 96×64-trained weights run at 307 200 px. God's res
   removed the checkerboard stipple (net at canvas, not window), but fine
   texture is still N1's charter, not N0's.
4. **Legacy bilinear default** — the dying `render` path's `upscale_mode=0`
   (not enforced to nearest); harmless once the net path is the only path.

---

## VERDICT — S11 (SHIFT 11): net-wedge fix over the S12 queue split

**HOLDS parity · FIXES the deadlock · 60 fps STILL UNMET (20.85 ms, 4.18 ms short, ~48 fps).**

- **The wedge — MEASURED, cured.** S12/S12.5's dedicated-net-queue +
  `MTLSharedEvent` fence DEADLOCKED (both eyes black, net GPU 0.00, GPU timeout).
  Instrumented root cause (`GAIA_NATIVE_NET_TRACE`, see perf N0.h): MPSGraph's
  `encodeToCommandBuffer` `commitAndContinue`s, committing the net buffer's
  `encodeWaitForEvent(V)` at ENCODE time, 1–2 frames ahead of the signal; on ONE
  shared net queue, double-buffering lands set-1's V=2 wait on the FIFO ahead of
  set-0's continuation → circular cross-buffer FIFO stall. The event VALUES were
  never wrong (monotonic, paired) — the queue ORDERING was. Fix: one net queue
  PER SET. Verified 646 frames, 0 GPU errors, both eyes render.

- **Parity** — HOLDS. `n0b_gather_and_shared_forward_match_cpu` ok,
  `n0_gate1_live_net_matches_cpu_reference` ok (release). The pipelined-path
  change (per-set queues) does not touch the sync/ordeal path.

- **Absolutes** — 60 FPS: **STILL VIOLATED** (48 fps, 20.85 ms). Honest
  accounting: the split cured the +6 ms trace regression (trace 12.65 → 5.74 ms)
  but the net stage N0.g had hidden on the encode thread (net wall 4.16 ms)
  REAPPEARED on the wall (13.3 ms) — the fenced net GPU no longer overlaps the
  next trace. Cost MOVED, TOTAL flat (20.07 → 20.85). The deadlock is dead; the
  frame is real; 60 fps is not bought this shift. NO LODs / no neural upscale /
  one light pass / no hardcoded res: HOLDS.

## Gaps carried forward (S11)
1. **60 fps unmet, 4.18 ms short.** The standing thief is now net_wall (13.3 ms
   = 4.83 GPU + ~8.5 ms commit/fence serialization at `commit_net`). Next
   target: let the fenced net GPU overlap the next frame's trace (reclaim
   N0.g's encode-hidden net_wall WITHOUT the trace contention). The per-set
   queue split is the right substrate; the blocking `commit_net` wait is what
   to attack.
2. **commitAndContinue is load-bearing but implicit.** The fix depends on MPS
   committing base early; if a future MPS build stops (or `setCommitAndContinueEnabled:`
   becomes available and is set false), the event fence could be simplified to a
   pure CPU-ordered handoff (the gather `device.poll(wait)` already orders
   gather→net). Worth a revisit.
3. Prior gaps 2–4 (default = slower path, static 96×64 weights, legacy bilinear)
   carry forward unchanged.

---

## VERDICT — S13 (SHIFTS 12–13): frame overlap (N0.i) → the outside-9ms hunt (N0.j)

**HOLDS parity · frame overlap is a REAL +37% throughput win · 60 fps STILL
UNMET (~48 fps wall-clock) · the "outside 9 ms" is NAMED: it is world advance.**

Concordance brought current from S11 (stale) through S13:

- **S13 frame overlap (N0.i) — HOLDS, real win, honest ceiling.** Deferring the
  `commit_net` `waitUntilCompleted` one frame downstream drove net_wall
  12.99 → 0.012 ms and moved **wall-clock throughput 35.55 → 48.75 fps (+37%)**
  on the same binary (A/B `GAIA_NATIVE_NET_NOOVERLAP=1`). Output-or-nothing
  intact (each image its own frame's complete evidence; +1 frame display
  latency, 2 frames-in-flight). **[STILL FLAGGED for the Architect: 2 frames in
  flight / +1 frame display latency — his frame-latency ruling is pending; the
  NOOVERLAP toggle is kept alive per that.]** The median stage-sum (11.69 ms →
  "85 fps") was correctly called NOT the throughput — the shift ADDED wall-clock
  fps precisely because ~9 ms lived outside the stage table.

- **S13 outside-9ms hunt (N0.j) — the 9 ms is LOCATED, honestly.** New `/budget`
  `outside` block: **world advance ~7.0 ms is the entire gap**; the other two
  N0.i suspects are cleared — readback ~0, http 0.2 ms. World advance re-splices
  and re-uploads the BVH every animating frame (naruko presence spheres move).
  - **Readback tax KILLED but it was never a thief.** On-demand `capture_presented`
    replaces the per-frame copy; measured render-thread cost of the old copy was
    **0.002 ms** (async GPU/worker), so throughput held flat (48.6 → 48.3, noise).
    Correct hygiene, proven path (both eyes served through it), zero fps.
  - **World-advance overlap TRIED, DOES NOT HELP (47.6 vs 48.4 fps).** Trace is
    synchronous on the render thread (submits+polls the GPU for the AOV feeding
    the gather) → no GPU flight to hide the world CPU under. Serial kept default;
    overlap behind `GAIA_NATIVE_WORLD_OVERLAP=1`. Honest null result, not buried.

- **Parity** — HOLDS. `n0b_gather_and_shared_forward_match_cpu` ok,
  `n0_gate1_live_net_matches_cpu_reference` ok (release, `GAIA_NEURAL_LIVE=1`).
  The overlap loop change + outside instrumentation + on-demand readback do not
  touch the net sync/ordeal path.

- **Absolutes** — 60 FPS: **STILL VIOLATED (~48 fps, ~20.5 ms/frame).** The
  frame overlap is a genuine step (35.55 → 48.75). The remaining wall is now
  fully VISIBLE and honestly attributed: world advance ~7 ms (CPU BVH re-splice/
  upload) + synchronous trace ~6 ms (part GPU-blocking on the render thread) +
  net_gpu ~4.7 ms (single-M1-GPU contention). NO LODs / no neural upscale / one
  light pass / no hardcoded res: HOLDS.

## Gaps carried forward (S13)
1. **60 fps unmet, ~3.8 ms short of 60 at throughput.** Three thieves, all
   named: world advance (cache the BVH harder / skin without full re-splice /
   gate re-splice on real >ε geometry change), trace (make it ASYNC so world
   advance can finally overlap it — the overlap substrate already works, trace's
   synchronous poll is what blocks it), net_gpu (cut it — N1 quality pass).
   Cutting work, not rescheduling — as N0.i and N0.j both conclude.
2. **2 frames-in-flight / +1 display-latency ruling still PENDING** with the
   Architect. NOOVERLAP toggle preserved until he plays the word.
3. Prior gaps (default = slower path is now RESOLVED — S8 flipped MPSGraph to
   default; static 96×64 weights / N1 quality; legacy bilinear) carry forward
   otherwise unchanged.

---

## VERDICT — S15 (N0.k): the "9ms world advance" premise OVERTURNED

**HOLDS parity/determinism · the N0.j blame (BVH re-splice+upload) was WRONG ·
the real 5.36ms thief inside world advance is `elements::Solver::step()`.**

- Instrumented `advance_world`/`tick_with_ops` to the leaf: splice (0.16ms) +
  upload (0.43ms) together are ~0.6ms, NOT the ~7ms N0.j blamed. KAMI decorative
  eval + JSON round-trip are ~0.06ms — also cleared. `elements::Solver::step()`
  is 5.36ms of the 6.97ms world advance — the entire thief.
- One dirty-only cut landed (dirty-skin, static bodies cost zero re-skin) —
  correct, byte-identical (`rite5` 17/17), but only ~0.04ms — skin was already
  cheap. Splice/upload dirty-only correctly NOT attempted (only ~0.6ms surface,
  inside contention noise).
- **60 fps: STILL VIOLATED (~49 fps).** Verdict redirected the charter: next
  shift must attack the solver, not the BVH pipeline. Source:
  `docs/perf/2026-07-18-neural-live-n0.md` N0.k.

## VERDICT — S16 (N0.l): THE SOLVER CHARTER — island sleeping, solver thief KILLED

**HOLDS parity/determinism/motion · solver_step 4.45→0.077ms live (152× on the
isolated settled sub-table) · world advance 6.97→~2.6ms · frame now TRACE-bound
on the offscreen numbers this shift measured (no windowed wall re-measured).**

- Leaf instrument inside `solver_step` found the REAL thief was
  `collision_static` (3.30ms, particle-vs-12368-static-tri broadphase), not the
  O(n²) body pass (0.44ms) — the N0.k lesson ("the sub-table moves the cut")
  held twice. A second hidden cost, `ensure_collision_grid` re-hashing the
  static soup every tick (~1.48ms), was cut with a pointer+len identity cache
  (byte-identical, no possible fingerprint mismatch since colliders are always
  wholesale-replaced, asserted by grep).
- **Island sleeping charter** (union-find over rigid+bond+proximity edges,
  awake-only proximity fan-in so all-asleep unions stay O(bonds)): settled
  naruko (199 particles / 13 islands) drops solver_step 3.88 → 0.026ms
  (152×). Sleep OFF is byte-unchanged (every branch gated). 4 new ordeals green
  (determinism, settled-sleeps, WAKE-TEST — a pushed sleeper travels, no silent
  freeze — rest-pose parity |Δcentroid| < 5e-3m).
- **Projection only, NOT measured this shift:** N0.l projected world≈2.6 +
  trace≈6 + net_gpu≈4.7 ≈ 14-16ms → ~60-70fps, flagged explicitly as a
  PROJECTION (offscreen-only measurement, no windowed wall re-run). **S17 below
  is the verification of that projection.**
- **60 fps: NOT YET MET on any full windowed measurement** (none taken this
  shift). Source: `docs/perf/2026-07-18-neural-live-n0.md` N0.l.

---

## VERDICT — S17 (N0.m, THE CROWN MEASUREMENT): projection REFUTED at wall-clock — 60fps NOT MET, ~53fps

**HOLDS parity/determinism/rite5/motion · sleep charter VERIFIED live (world
advance 5.61→1.75ms median, solver_step 3.95→0.05ms median, ~79× live) · BUT
the N0.l "14-16ms → 60-70fps" projection is REFUTED at the wall-clock mean: only
52.86 fps (18.92 ms/frame mean) reached with sleep ON, 50.23 fps (19.91 ms/frame)
with sleep OFF — a +2.63 fps / ~1.0 ms mean delta, NOT the ~9-11ms the projection
implied.**

- **The world-stage MEDIAN delta reproduces the shift's own prediction closely**
  (5.607→1.747ms = 3.86ms, inside the "~3-4ms/frame" the task asked to
  reproduce) — the sleep charter's CPU win is real and measured twice now
  (N0.l's isolated sub-table, this shift's live A/B).
- **But the wall-clock MEAN barely moves.** Root cause, read from the same
  `/budget`: `demod`'s p95 is **10.46ms vs a 1.66ms median** (both arms) — a
  heavy GPU-contention tail (single-M1-GPU serialization, N0.i's standing
  thief) that solver sleep does not touch. The mean (the metric wall-clock fps
  actually reports) is dominated by that tail; the median table is not. The
  median-level "books balance" the doc has used all shift under-counts this
  tail — **a methodology gap this shift surfaces and flags, not one it
  resolves.**
- **Parity/determinism/motion — ALL HOLD.** `n0b_gather_and_shared_forward_
  match_cpu` ok, `n0_gate1_live_net_matches_cpu_reference` ok, `rite5` 17/17,
  `s16_sleep_ordeals` 4/4, both eyes read coherent (no black/wedge/NaN) in both
  arms, motion gate confirms the presence layer moves under sleep ON.
- **Absolutes** — 60 FPS: **STILL VIOLATED, both arms** (sleep ON 52.86 fps /
  18.92ms mean — 2.25ms short; sleep OFF 50.23 fps / 19.91ms mean). NO LODs /
  no neural upscale / one light pass / no hardcoded res: HOLDS.

## Gaps carried forward (S17)
1. **60fps unmet by 2.25ms mean (sleep ON) despite the solver being killed.**
   The remaining wall is the SAME two thieves N0.i/N0.j already named — trace
   (~5.9-6.0ms, synchronous submit+poll on the render thread) and net_gpu
   (~4.9ms, single-M1-GPU contention) — PLUS a now-visible third: demod's
   heavy p95 tail (10.46ms vs 1.66ms median), which the median-table
   methodology this doc has used since N0.d does not surface. Next shift
   should measure/attack that tail directly (a p99/mean column in `/budget`,
   or GPU timeline tracing across trace/net_gpu/demod to find what's actually
   serializing on the one GPU), not just chase more median-level cuts.
2. **The N0.l projection method (median-sum extrapolation) is now shown
   unreliable for wall-clock prediction** — carry a wall-clock-fps-only
   discipline forward for any future projection (this doc's own N0.i already
   warned "median stage-sum is NOT the throughput"; S17 shows the warning
   applies to inter-shift A/B deltas too, not just single-run stage-sum vs
   wall-fps).
3. 2 frames-in-flight / +1 display-latency ruling — still PENDING with the
   Architect (unchanged).
