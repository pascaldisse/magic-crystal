# NEURAL-LIVE N0 — the ONE net presented in the live window (budget)

Scaffold: `GAIA_NATIVE_NET_PRESENT` (N0.c) · dies at lane cutover.
Chain per live frame: trace low radiance (spp 1) + native AOV → GPU feature
gather (N0.b, pooled zero-copy MTLBuffer) → MPSGraph batched-GEMM forward
(N0.a) → undo log-demod by native albedo → 1:1 nearest blit to surface +
offscreen capture.

## Instance
- binary: `target/release/scrying-glass` (release, optimized)
- env: `GAIA_NATIVE_WORKER_WINDOW=true GAIA_NATIVE_NET_PRESENT=true`
  `GAIA_NATIVE_PORT=8436 GAIA_NATIVE_HUD=false`
- world: `worlds/naruko` (default) · M1 / macOS 26
- trace 640×480 → net 960×640 (614 400 px) · low=`GAIA_NATIVE_RENDER_W/H`,
  target=surface (worker window default 960×640)
- source log: `packages/scrying-glass/proof/neural-live/n0d-run.log`
  (1260+ frames observed; the wall clock stays steady across the run)

## Budget — median / p95 ms per stage (frames=300 sample; vs 16.67 wall)

| stage    | median | p95   | notes                                            |
|----------|--------|-------|--------------------------------------------------|
| trace    | 6.36   | 7.27  | low accum (clear+dispatch) + native AOV, 2 submits+polls |
| gather   | 1.45   | 1.74  | one compute dispatch → pooled shared MTLBuffer    |
| net      | 27.12  | 28.57 | MPSGraph 23→5×64 ReLU→3 forward over 614 400 px    |
| present  | 6.63   | 7.55  | AOV readback + CPU undo-log-demod + upload + blit  |
| **TOTAL**| **41.79** | **47.87** | ~24 fps · **2.5× over the 16.67 ms wall**   |

Steady state (frames=1260) drifts slightly lower — TOTAL median 40.33 /
p95 46.64 — as the OS warms; no upward creep, no leak.

### Reading the numbers
- **net dominates** (27 ms, ~65 % of the frame). The per-pixel MLP runs over
  the full 614 400-px target every frame; this is the cutover's real cost
  centre and where N1 (quality) and any perf pass must land.
- **present is heavier than it should be (6.6 ms)** — it pays a full GPU→CPU
  AOV readback + a per-pixel CPU undo-log-demod + re-upload EVERY frame. That
  is a scaffold shortcut (the demod belongs on the GPU); it also forces a
  pipeline stall. Honest gap, flagged below.
- **trace + gather (7.8 ms)** are already near budget-shaped; gather is
  essentially free (1.4 ms), exactly as N0.b's 0.284 ms @96×64 predicted
  scaled to 614 400 px.

## Pixel words — read the frame
`proof/neural-live/n0d_net_present_960x640.png` (live `/screenshot` off :8436,
the actual net-present surface).

The frame is **coherent and recognizably the naruko scene** — not noise, not a
black void, no NaN holes: brown crates, a translucent green-flecked glass
panel, a central dark tower ringed by concentric halos, several iridescent
spheres, a large glass orb on a green pedestal, a dark industrial block with
chimneys, a pink→mauve sky gradient over a purple ground plane. Colours are
natural, so the undo-log-demod-by-albedo is wired correctly and radiance stays
bounded.

**Rough, as promised** (weights are 96×64-static-trained, run here at a 4×
larger shape): a regular **dithered checkerboard stipple** rides every
surface — crate faces, glass, ground, sphere skins — the tell-tale grid of a
net asked for pixels outside its trained resolution. The spheres carry
**chromatic rainbow speckle** instead of clean reflections, and the sky shows
faint banding. Geometry, shadows/AO and silhouettes read correctly; only the
fine texture is degraded. Truth over pride: presentable as a live embodiment
proof, NOT production quality. **N1 fixes quality.**

## Honest gaps
1. **Not real-time yet** — 41.8 ms median (2.5× the wall). The net stage is
   the wall; the present stage's CPU readback/demod round-trip is a scaffold
   stall that must move to the GPU before cutover.
2. **Per-frame `forward_shared` output `Vec`** (N0.a-owned) still heap-allocs
   3×N floats each frame — outside this shift's scope; the pooled MTLBuffers
   themselves allocate once.
3. **Two compute bind groups rebuilt per frame** (integrator reallocates its
   node/tri storage each dynamic tick) — handle-weight, not buffer churn.
4. **Sized to the boot surface**; the net path self-disables/rebuilds on
   resize (scaffold, dies at cutover).
5. **Static-shape weights at a 4× larger runtime shape** → the stipple/speckle
   above. Quality is N1's charter, not N0's.

---

# N0.e — S1 + S3 killed (SHIFT 6): GPU-only present, GPU-time split of the net

Chain change: net forward now leaves radiance ON the GPU (`forward_shared_gpu`,
NO per-frame `Vec<f32>` readback) → CUT 2 **GPU demod** compute dispatch
(`rdirect_demod.wgsl`: net-out MTLBuffer + AOV albedo → present accum, one
pass, same math as the CPU `undo_log_demod_px`, bit-identical). The AOV
readback + CPU per-pixel demod + re-upload round-trip is GONE. The net forward
now runs through `encodeToCommandBuffer` on a command buffer WE own, so
GPUStartTime/GPUEndTime give the **GPU-only** ms — the number
`runWithMTLCommandQueue` hid.

## S1 — CPU READBACK: KILLED
`present` stage **6.63 → ~1.0 ms** (the GPU demod dispatch). No GPU→CPU AOV
readback, no CPU per-pixel loop, no re-upload. The Vec path (`forward_shared`)
survives for the offline parity ordeal only.

| stage    | n0d median | n0e median | note                                        |
|----------|------------|------------|---------------------------------------------|
| present  | 6.63       | **~1.0**   | CPU readback+demod round-trip → 1 GPU pass  |

## S3 — GPU TIME vs WALL: MEASURED. **The wall is CPU, not GPU.**
Net stage split for the first time:

| net        | GPU-only (median/p95) | WALL (median/p95) | gap (CPU)   |
|------------|-----------------------|-------------------|-------------|
| n0e run    | **~6.0 / ~10.7**      | 40–45 / 47–70     | ~34 ms      |

The **GPU forward is ~6 ms** — probe class (metal4-probe: 4.47 ms GPU @ 614 400
px; the ~1.5 ms delta is live machine contention, see below). The ~34 ms WALL
gap is **CPU-side MPSGraph per-frame cost**: `encodeToCommandBuffer` building
the command buffer + `waitUntilCompleted` blocking. Shift 5's finding is
confirmed with numbers: **pooling the compiled executable did NOT remove the
wall — the wall was never the GPU.** MPSGraph's per-encode CPU overhead + the
blocking wait is the whole gap.

### ⚠ Measurement honesty — heavy contention
This run measured with **9 concurrent `cargo test` lanes** hammering the box
(other worktrees). That inflates every WALL/CPU number and creeps them upward
across the run (trace 8.6→9.9, net wall 40→45, TOTAL 52→62). The **GPU-only
net time is contention-robust and dead flat at ~6.0 ms** the whole run — that
is the trustworthy figure. n0d's clean-machine net wall was 26.7 ms; expect the
n0e net wall to land ~26 ms clean (GPU ~4.5 + ~22 ms CPU encode/wait). Re-run on
a quiet machine to pin the clean wall; GPU number stands as-is.

## Budget shape after S1/S3 (clean-machine projection)
trace ~6.4 · gather ~1.4 · net wall ~26 (GPU ~4.5 + CPU ~22) · present ~1.0 →
TOTAL ~35 ms. Present stage is solved. **The net stage's CPU encode/wait is now
the sole wall** (~22 ms CPU, not GPU).

## Next target (the real wall, for the next shift)
The GPU forward fits (~4.5 ms). The wall is MPSGraph's CPU-side per-frame
encode + the blocking wait. Two independent attacks:
- **S2 one-frame pipeline** — drop `root.waitUntilCompleted()` in
  `run_executable`; present picks up last frame's completed output. Removes the
  wait portion of the gap (standard, documented). Left undone this shift to
  keep GPU timestamps readable.
- **Kill the per-frame MPSGraph encode** — encode the net onto the SAME command
  buffer as trace+demod (one submit, no separate MPS command buffer + wait), or
  move off MPSGraph to a hand-written MTLComputeCommandEncoder / MTL4 tensor
  path whose per-call CPU encode is a fraction of MPSGraph's.

## Proof — both eyes
- presented surface: `proof/neural-live/n0e_net_present_960x640.png` (live
  `/scry` off :8436, the actual GPU-demod net-present surface).
- **Pixel words:** coherent naruko scene — brown crates, translucent mirror
  panel, central lighthouse ringed by a violet halo, chrome orb on a pedestal,
  chimneyed factory silhouette, pink→mauve dusk sky over purple ground. Colours
  natural, radiance bounded → the GPU demod is wired right. **Visibly CLEANER
  than the n0d CPU-demod frame** (n0d's dithered stipple is far reduced here;
  same net, same math, cleaner present path). **Parity vs n0d: HOLDS** — same
  geometry/lighting/silhouettes/demod; the only diff is the animated presence
  spheres (cyan/pink) sitting at a different frame instant.
- accum-belief PNG (raw net output pre-demod): **OWED** — no trivial dump
  endpoint exists; would need a debug readback path (out of this shift's scope,
  the live path is deliberately readback-free now).

## Source
- log: `packages/scrying-glass/proof/neural-live/n0e-run.log` (1260+ frames)
- table tag: `[n0e]` — `net[wall .. gpu ..]` splits GPU from wall live.

---

# N0.f — S5 chain default + S4 God's res (SHIFT 8): the net presents at the 640×480 canvas

Two changes land:
- **S4 GOD'S RES** (law `0a25530`): the net rig now pools + presents at the
  **640×480 canvas**, NOT the window. Trace low == net target == `render_w/h`
  (640×480, 307 200 px). The window gets the canvas by a **nearest/integer
  display blit** only (`blit_uniform.surface=[surf_w,surf_h,1]`, mode 1 =
  nearest). No neural enlarge — the net never runs at the window's pixel count.
  `net_present_frame` derives `target = render_res`; the offscreen capture
  upscales the canvas to the surface exactly as the on-screen blit does.
- **S5 chain default**: the frame's default net is the raw
  `MPSMatrixMultiplication`+bias/ReLU **chain** (parity ordeal `4.8e-7` vs
  MPSGraph). `GAIA_NATIVE_NET_MPSGRAPH=1` flips back to the MPSGraph executable
  (A/B).
- **S3 demod split**: `demod` (GPU undo-log-demod) and `present` (surface blit)
  are now separate `[n0e]` columns.

## Budget — median / p95 ms per stage · 640×480 · ≥300-frame samples · vs 16.67

Machine quieter than n0e (most sibling lanes done). `trace`/`gather`/`demod`/
`present` are near-identical across BOTH runs below — the ONLY stage that moves
is `net`, so its split is a true GPU-cost difference, not load noise.

**CHAIN (S5 default)** — `s3-godres-chain.log`, frames 720:

| stage    | median | p95   | note                                              |
|----------|--------|-------|---------------------------------------------------|
| trace    | 6.65   | 9.25  | low accum (clear+dispatch) + native AOV, 2 submits |
| gather   | 1.06   | 1.99  | one compute dispatch → pooled shared MTLBuffer      |
| net wall | 43.15  | 48.29 | raw GEMM chain, all on one owned command buffer      |
| net GPU  | 42.76  | 47.81 | **6.4× the MPSGraph GPU** — separate per-layer GEMMs |
| demod    | 0.66   | 1.41  | CUT 2 GPU undo-log-demod, one dispatch              |
| present  | 0.20   | 0.27  | surface blit + offscreen capture (nearest)          |
| **TOTAL**| **53.06** | **58.54** | ~19 fps · **3.2× over the 16.67 ms wall**    |

**MPSGRAPH (A/B)** — `s3-godres-mpsgraph.log`, frames 1140:

| stage    | median | p95   | note                                              |
|----------|--------|-------|---------------------------------------------------|
| trace    | 6.53   | 9.22  | (same as chain)                                    |
| gather   | 1.01   | 1.99  | (same)                                             |
| net wall | 20.51  | 24.19 | GPU 6.65 + **~13.9 ms CPU** `encodeToCommandBuffer` |
| net GPU  | 6.65   | 10.60 | fused GEMM+bias+ReLU kernels                        |
| demod    | 0.64   | 1.26  | (same)                                             |
| present  | 0.19   | 0.25  | (same)                                             |
| **TOTAL**| **30.05** | **34.05** | ~33 fps · **1.8× over the wall**              |

## ⚠ THE CHAIN IS A PERF REGRESSION — read the numbers, not the premise
Shift 7's raw-kernel chain was chartered to "kill the MPSGraph per-frame encode
wall". It DID: chain net CPU = wall−GPU = 43.15−42.76 = **0.39 ms** (vs
MPSGraph's ~13.9 ms encode). **But the trade is catastrophic:** the raw
`MPSMatrixMultiplication` chain runs each layer as a separate dispatch with
intermediate buffers and NO fusion, so its **GPU time is 42.76 ms vs MPSGraph's
6.65 ms — 6.4×**. Killing a ~14 ms CPU wall cost ~36 ms of extra GPU. Net:
**chain TOTAL 53.06 ms vs MPSGraph 30.05 ms — the chain is 1.8× SLOWER.**
Parity is green (4.8e-7); SPEED is worse. The dispatcher keeps chain as the
default per the S5 charter, but **the honest budget says MPSGraph wins today**,
and the real target is a fused GPU forward that keeps MPSGraph's ~6.6 ms GPU
while shedding its ~14 ms CPU encode (MTL4 tensor / hand-fused compute), not the
un-fused chain.

## God's res dividend — the frame is CLEANER
The net now runs at 640×480 (the canvas) instead of the n0d 960×640 (the
window). Fewer pixels (307 200 vs 614 400) AND closer to a shape the static
weights tolerate: the n0d **dithered checkerboard stipple is gone**. See
`proof/neural-live/s7-godres-chain-net.png` and `…-mpsgraph-net.png`.

## Proof — both eyes
- chain surface:    `proof/neural-live/s7-godres-chain-net.png` (960×640 PNG =
  the 640×480 canvas nearest-blitted to the worker window, live `/scry` :8436).
- mpsgraph surface: `proof/neural-live/s7-godres-mpsgraph-net.png` (:8437).
- **Pixel words (both):** coherent naruko scene — brown crates, translucent
  green-flecked glass panel, central dark tower ringed by cyan/violet halos,
  iridescent spheres, a large glass orb on a green pedestal, dark chimneyed
  factory block, pink→mauve dusk sky over purple ground. Colours natural,
  radiance bounded → GPU demod wired right. **Clean surfaces — no checkerboard
  stipple** (the God's-res dividend). **Parity chain↔mpsgraph: HOLDS** — same
  geometry/lighting/silhouettes/demod; the only visible diff is the animated
  presence spheres sitting at a different frame instant (live-scene motion +
  spp=1 noise), consistent with the ordeal's numeric 4.8e-7.

## Source
- logs: `s3-godres-chain.log` (:8436), `s3-godres-mpsgraph.log` (:8437) under
  `packages/scrying-glass/proof/neural-live/`.
- env: `GAIA_NATIVE_WORKER_WINDOW=true GAIA_NATIVE_NET_PRESENT=true
  GAIA_NATIVE_HUD=false` (+ `GAIA_NATIVE_NET_MPSGRAPH=1` for the A/B), release
  binary, world `worlds/naruko`, M1 / macOS 26. Worker window = non-activating
  (never key, never pops in front). `[n0e]` tag now carries `demod` + `present`.

---

# N0.g — S8 default flip + S9 encode pipeline (SHIFT 9): the encode leaves the wall

Two changes land, both TESTED live (:8436, `worlds/naruko`, worker window,
release, M1/macOS 26):
- **S8 — MPSGraph is the frame DEFAULT.** N0.f proved the fused MPSGraph GPU
  (6.65 ms) beats the un-fused chain GPU (42.8 ms) 6.4×; the chain was default
  only by S5 charter and lost 1.8×. S8 flips it: `use_mpsgraph` defaults TRUE;
  the raw `MPSMatrixMultiplication` chain stays a lab A/B via
  `GAIA_NATIVE_NET_CHAIN=1` (honest slower measurement, kept). Parity re-gated
  after the flip: MPSGraph(default) vs CPU **1.9e-6**, MPSGraph vs chain
  **4.8e-7** (ordeal `n0b_gather_and_shared_forward_match_cpu`).
- **S9 — the ~14 ms MPSGraph encode rides a dedicated thread.** N0.f's net wall
  = 6.65 GPU + ~13.9 ms CPU `encodeToCommandBuffer`. That CPU encode is
  DATA-INDEPENDENT (fixed shape, pooled buffers), so a background encode thread
  pre-builds the NEXT net command buffer while the render thread does GPU work.
  Double-buffered feature/output/tensor-data SETS (`SET_COUNT=2`): the render
  thread consumes set[frame%2] while the encode thread encodes the other set's
  next buffer. `begin_frame` claims the pre-encoded buffer + names the set the
  gather must fill; `commit_net` commits it (AFTER the gather) + waits the GPU.
  **0 latency — the net still reads THIS frame's own gather** (the pre-encode
  records buffer REFERENCES, not data; the render thread commits in-order so the
  GPU runs gather→net every frame). No design fork: no one-frame-old evidence.

## Budget — median / p95 ms · 640×480 · vs 16.67 wall · quiet machine

**S8+S9 MPSGraph pipeline (the DEFAULT, frames=300)** —
`proof/neural-live/n0g-mpsgraph-pipeline.log`:

| stage    | median | p95   | note                                                 |
|----------|--------|-------|------------------------------------------------------|
| trace    | 12.65  | 19.09 | **+6 ms REGRESSION** — encode-thread CPU contends     |
| gather   | 1.03   | 2.50  | unchanged                                             |
| net wall | 4.16   | 7.16  | **encode HIDDEN** (was 20.5): commit + GPU wait only  |
| net gpu  | 6.42   | 14.04 | fused GEMM+bias+ReLU (unchanged from n0f)             |
| demod    | 0.63   | 1.42  | GPU undo-log-demod, one dispatch                     |
| present  | 0.11   | 0.20  | nearest surface blit + offscreen capture             |
| **TOTAL**| **20.07** | **26.19** | ~50 fps · **1.2× over the wall (3.40 ms short)** |

**Chain pipeline (A/B, `GAIA_NATIVE_NET_CHAIN=1`, frames=720)** —
`proof/neural-live/n0g-chain-pipeline.log`:

| stage    | median | p95   | note                                                 |
|----------|--------|-------|------------------------------------------------------|
| trace    | 6.99   | 9.76  | **baseline** — chain encode 0.4 ms, encode thread idle |
| net wall | 35.55  | 48.19 | chain GPU on the critical path (encode irrelevant)   |
| net gpu  | 35.23  | 47.86 | un-fused per-layer GEMMs                              |
| **TOTAL**| **45.05** | **58.76** | ~22 fps · confirms S8: MPSGraph pipeline wins 2.2× |

### The win, and the honest tax on it
- S9 moved the whole ~14 ms encode off the critical path: **net wall 20.5 → 4.16 ms.**
- BUT **trace regressed 6.5 → 12.7 ms (+6 ms).** The A/B ISOLATES the cause: the
  chain pipeline (encode ≈ 0.4 ms → encode thread ~idle) keeps trace at its
  6.99 ms baseline; only the MPSGraph pipeline (encode thread busy ~14 ms/frame)
  inflates trace. The encode thread's CPU work contends with the render thread's
  trace submission — Metal command-buffer creation on the SHARED wgpu queue +
  raw CPU pressure on the P-cores. So the theoretical 30→~14 ms became **30.05 →
  20.07 ms (−10 ms, ~33→~50 fps)**: a real, tested win, ~6 ms eaten by the tax.
- **Encode-thread occupancy ≈ 69%** (derived: one ~13.9 ms MPSGraph encode per
  ~20.1 ms frame; no hardware counter added — honest label).

## 60 FPS VERDICT — NOT MET, 3.40 ms short (~50 fps)
TOTAL 20.07 ms median vs the 16.67 ms wall. The net stage is SOLVED (4.16 ms
wall). The sole remaining thief is now the **+6 ms trace regression** the encode
thread induces. The chain-pipeline baseline proves trace's natural cost is
6.99 ms; recover that and TOTAL ≈ 6.99+1.03+4.16+0.63+0.11 = **~12.9 ms → 60 fps
MET**. Next target: decouple the encode thread's Metal work from the render
thread's trace — a DEDICATED `MTLCommandQueue` for the net encode + an
`MTLSharedEvent` for the gather→net dependency (the only cross-queue hazard;
demod already waits via `commit_net`). Not attacked this shift (cross-queue
sync is its own ordeal).

## Proof — both eyes
- MPSGraph pipeline surface: `proof/neural-live/s9-pipeline-mpsgraph-net.png`
  (960×640 = the 640×480 canvas nearest-blitted, live `/scry` :8436).
- chain pipeline surface: `proof/neural-live/s9-pipeline-chain-net.png` (A/B).
- **Pixel words (both, READ):** coherent naruko scene — brown crates,
  translucent mirror glass panel, central dark tower ringed by pink/violet
  halos, cyan+pink presence spheres, large glass orb on a cylindrical pedestal,
  dark chimneyed factory block with lit windows, pink→mauve dusk sky over purple
  ground. Colours natural, radiance bounded → GPU demod wired right. **Clean
  surfaces — no checkerboard stipple** (the God's-res dividend holds).
  **Parity MPSGraph-pipeline ↔ chain-pipeline: HOLDS** — same
  geometry/lighting/silhouettes/demod; the only diff is the animated presence
  spheres at a different frame instant (live motion + spp=1 noise), consistent
  with the ordeal's 4.8e-7. **Parity vs s7 (n0f) frames: HOLDS** — same scene,
  same clean surfaces.

## Source
- logs: `n0g-mpsgraph-pipeline.log` (default), `n0g-chain-pipeline.log`
  (`GAIA_NATIVE_NET_CHAIN=1`) under `packages/scrying-glass/proof/neural-live/`.
- env: `GAIA_NATIVE_WORKER_WINDOW=true GAIA_NATIVE_NET_PRESENT=true
  GAIA_NATIVE_HUD=false GAIA_NATIVE_PORT=8436`, release, world `worlds/naruko`,
  M1 / macOS 26. Tag `[n0g]`. Ownership: encode thread `rdirect-net-encode`
  encodes (no commit); render thread commits + waits; `Drop` joins the thread.

---

# N0.h — S12 queue split + S11 net-wedge fix (SHIFT 11): the deadlock, found and cured

The S12/S12.5 cut (dedicated net `MTLCommandQueue` + `MTLSharedEvent`
gather→net fence, meant to kill N0.g's +6 ms trace regression) shipped a
DEADLOCK: both eyes BLACK, net GPU 0.00, `kIOGPUCommandBufferCallbackErrorTimeout`,
all later submissions ignored. S11 diagnoses it with an instrument, fixes it,
and measures the fixed pipeline honestly. Live, offscreen 640×480, release,
M1 / macOS 26, `worlds/naruko`, `GAIA_NATIVE_OFFSCREEN=true`.

## ROOT CAUSE — the stuck value pair (instrumented, `GAIA_NATIVE_NET_TRACE`)
The instrument prints each net buffer's awaited V, each `signal_gather_ready`
V, and each committed buffer's `base.status`/error. It showed:
- Values are PERFECTLY monotonic + paired — encode awaits V=k, signal sets V=k,
  in strict frame order. Suspects (a)/(c)/(d) (wrong/duplicate/non-monotonic
  values, cross-set index mismatch) are FALSE.
- Frame 1 (`set=0`): `base.status=4` (**Completed**) — first net buffer runs.
- Frame 2 (`set=1`): a PRE-commit probe reads `base.status=5` (**Error/timeout**)
  BEFORE the render thread even commits it. And `set=0` pre-commit read
  `status=2` (**Committed**) — proof the buffer was committed at ENCODE time.

Mechanism: **MPSGraph's `encodeToCommandBuffer` internally `commitAndContinue`s**,
so `base` (carrying our `encodeWaitForEvent(V)`) is COMMITTED on the encode
thread at ENCODE time — 1–2 frames AHEAD of `signal_gather_ready`. With
double-buffering (`SET_COUNT=2`) on ONE shared net queue, set-1's `V=2` wait
lands on the FIFO net queue at startup, AHEAD of set-0's continuation buffer.
The GPU drains the queue in commit order and STALLS on set-1's unsignaled V=2;
set-0's continuation is queued behind it, so the render thread's frame-1
`commit_net` wait can never retire, so frame 2 (which would signal V=2) never
runs → **circular cross-buffer FIFO deadlock. Stuck pair: base(set=1) awaits
V=2, times out; V=2's signal is gated behind set-0 work stuck behind that same
wait.** (S12.5's earlier "CPU-side signal fixes it" post-mortem was WRONG about
the cause — the CPU signal is fine; the FIFO ordering across two early-committed
waits on one queue is the wedge.)

## FIX — one dedicated net `MTLCommandQueue` PER SET
`EncodeCtx.net_queues: Vec<_>` (one per set); `encode(set)` builds its buffer on
`net_queues[set]`. Cross-set FIFO coupling is gone — set-1's early-committed
wait can no longer stall set-0's buffers. Within a single set's queue the waits
are strictly increasing and signaled in frame order (no self-block). The event
values + signals stay monotonic and paired. Minimal, sound, keeps the S9
pre-encode pipeline AND the S12 queue split intact. (`setCommitAndContinueEnabled:`
to stop the early commit was tried first — the selector is absent in this MPS
build, ObjC exception → abort; per-set queues are the working path.)

## Budget — median / p95 ms · 640×480 · frames=646 · 0 GPU errors · vs 16.67
`proof/neural-live/s11-offscreen-release.log`, `/budget` `s11-budget.json`:

| stage    | median | p95   | note                                                     |
|----------|--------|-------|----------------------------------------------------------|
| trace    | 5.739  | 7.513 | **+6 ms regression GONE** — split cured the contention   |
| gather   | 0.954  | 1.891 | unchanged                                                |
| net wall | 13.313 | 14.351| **reappears** — net GPU+commit serial on the fenced queue|
| net gpu  | 4.833  | 5.051 | fused GEMM+bias+ReLU                                      |
| demod    | 0.712  | 2.007 | GPU undo-log-demod, one dispatch                         |
| present  | 0.104  | 0.160 | nearest blit + offscreen capture                         |
| **TOTAL**| **20.851** | **24.849** | ~48 fps · **1.25× over the wall (4.18 ms short)** |

### What the split actually bought (honest accounting)
The queue split did exactly what N0.g predicted for trace: **trace 12.65 → 5.74
ms** — the encode-thread contention is cured (matches the crime report's "trace
6.14"). But the net stage that N0.g HID on the encode thread (net wall 4.16 ms)
REAPPEARED on the wall at 13.3 ms: with the dedicated queue + gather→net event
fence, the net GPU can no longer overlap the next frame's trace on a shared
timeline, and `commit_net` serially waits it. So the cost merely MOVED from
trace to net_wall — **TOTAL 20.07 (N0.g) → 20.85 (S11), flat.** The split cures
the deadlock and the trace regression but does NOT advance 60 fps this shift.

## 60 FPS VERDICT — NOT MET, 4.18 ms short (~48 fps)
TOTAL 20.85 ms median vs the 16.67 ms wall. The deadlock is DEAD and the frame
is honest (646 frames, 0 GPU errors, both eyes render). 60 fps is not reached:
the net GPU (4.83 ms) plus its commit/fence serialization (net_wall 13.3 ms) is
the standing thief. Recovering N0.g's encode-hidden net_wall WITHOUT the trace
contention — i.e. letting the fenced net GPU overlap the next trace — is the
next target (the queue split is the right substrate; the serialization at
`commit_net` is what to attack). A working 20.85 ms beats the deadlocked 8.

## Parity ordeal — HOLDS (sync path unaffected)
- `n0b_gather_and_shared_forward_match_cpu` — **ok** (gather + shared forward vs CPU).
- `n0_gate1_live_net_matches_cpu_reference` — **ok** (live net vs CPU reference).
Release, `GAIA_NEURAL_LIVE=1`. The pipelined path change (per-set queues) does
not touch the sync/ordeal path (`pipelined=false`, single queue, no fence).

## Proof — both eyes (READ, pixel words)
- presented: `proof/neural-live/s11-presented.png` (960×640, live `/scry?eye=presented`).
- belief: `proof/neural-live/s11-belief.png` (`?eye=belief`, raw net radiance, no albedo).
- **Pixel words (both, READ — NOT black):** coherent naruko dusk scene — brown
  crates, translucent mirror glass panel (green-tinted), central dark tower
  ringed by cyan/pink/violet concentric halos, pale presence spheres (one at the
  tower mouth, one mid-panel), large glass orb on a green cylindrical pedestal,
  dark chimneyed factory block with lit windows at right, pink→mauve sky over a
  purple ground. Belief eye = same geometry, brighter/desaturated (albedo not
  undone), as designed. Radiance bounded, colours natural → GPU demod wired
  right. **Both eyes render — the wedge is cured.**

## Source
- fix + instrument commit `61b1e4c`; measured on the following commit.
- logs: `s11-offscreen-release.log` (release run), `s11-offscreen.log` (debug
  repro); `s11-budget.json` / `s11-state.json`; instrument repro
  `/tmp/s11-fix2.log` (93/93 status=4).
- env: `GAIA_NEURAL_LIVE=1 GAIA_NATIVE_OFFSCREEN=true GAIA_NATIVE_NET_PRESENT=true
  GAIA_NATIVE_PORT=8438`, release, world `worlds/naruko`, M1 / macOS 26.
  Instrument gate: `GAIA_NATIVE_NET_TRACE=1`. Tag `[n0h]`.
