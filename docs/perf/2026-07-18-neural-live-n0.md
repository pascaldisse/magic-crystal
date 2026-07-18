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

---

# N0.i — S13 FRAME OVERLAP (SHIFT 12): the net wait leaves the critical path

The endgame cut. N0.h left the net stage SOLVED on GPU (net_gpu 4.83 ms) but
`commit_net`'s `waitUntilCompleted` blocked the render thread for the whole net
wall (13.3 ms) — GPU + commit/fence serialization serial per frame, no overlap.
S13 restructures the loop so frame N's committed net forward runs WHILE frame
N+1's trace+gather run, and the wait moves one frame downstream. Live, offscreen
640×480, release, M1/macOS 26, `worlds/naruko`, `GAIA_NATIVE_OFFSCREEN=true`.

## DESIGN CHOICE — present frame N-1's finished image while building N
`commit_net` now: (1) commits THIS frame's pre-encoded buffer WITHOUT blocking
(`commit_prepared_nowait`, its GPU forward starts overlapping the next frame),
(2) stashes it as `pending`, (3) WAITS and returns the PREVIOUS frame's buffer
(`wait_prepared`) — whose net has been running during THIS frame's trace+gather
and is (near-)complete, so the wait is short. The demod+present consume that
FINISHED buffer's set. Chosen over "present N late in frame N" because it keeps
the render thread free of any net-GPU wait on the frame that issued it.

**Output-or-nothing is intact.** Each presented image is the COMPLETE image of
its OWN frame's evidence (its own trace→gather→net→demod); only DISPLAY latency
grows by one frame (frames-in-flight = 2, per-image latency ≈ 2 frames). The
first frame presents nothing (`commit_net` returns `None` — no finished buffer
yet), never a partial image. **[Flagged for the Architect's judgment:** the
image is one frame older on screen than the render thread's current pose; the
image itself is never mixed or partial.**]**

Two correctness pieces:
- **Per-set AOV** (`net_aov: Vec`, one per double-buffer set). The overlap demods
  the PREVIOUS frame's radiance, so its albedo must be THAT frame's, not the one
  trace just wrote. Trace/gather touch `net_aov[set]`; demod reads
  `net_aov[demod_set]` — albedo stays matched to the radiance's frame.
- **Per-queue signal order preserved** (N0.h). The deferral moves only WHEN we
  `waitUntilCompleted`, never the signal↔wait pairing: each set's dedicated net
  queue runs its buffers in commit order (frame N, N+2, …), each awaiting its own
  strictly-increasing V signaled that frame. The N0.h FIFO wedge cannot reopen —
  proven by 3625+/2617+ frames, 0 GPU errors, both eyes render.

## WHERE THE 8.5 ms HID — the WAIT, not the commit (instrumented)
S13 splits the net wall into `commit` (CPU, `commit_prepared_nowait`) and `wait`
(`wait_prepared`), both in the `/budget` JSON:

| path            | net_wall | net_gpu | net_commit | net_wait |
|-----------------|----------|---------|------------|----------|
| blocking (N0.h) | 12.99    | 4.64    | **0.005**  | **12.98**|

The commit is **essentially free (0.005 ms median)**. The entire ~8.5 ms gap
(net_wall − net_gpu) was the render thread BLOCKING at `waitUntilCompleted` on
the net GPU completion + the per-set-queue fence serialization — NOT MPSCommand-
Buffer commit overhead, NOT completion-handler latency, NOT gather poll. Moving
that wait one frame downstream (onto an already-running buffer) collapses it:
net_wait median **12.98 → 0.001 ms**, net_wall **12.99 → 0.012 ms**.

## Budget — median / p95 ms · 640×480 · vs 16.67 wall · SAME BINARY A/B
Toggle: `GAIA_NATIVE_NET_NOOVERLAP=1` forces the old blocking path for an
apples-to-apples wall-clock comparison on one binary.

**OVERLAP (S13 default, frames=3625)** — `s13-offscreen-release.log` / `/budget`:

| stage    | median | p95   | note                                                  |
|----------|--------|-------|-------------------------------------------------------|
| trace    | 5.93   | 7.54  | +0.2 ms — net GPU now contends on the shared M1 GPU    |
| gather   | 0.83   | 1.21  | unchanged                                              |
| net wall | 0.012  | 0.44  | **wait moved downstream** — commit 0.005 + ~0 wait     |
| net gpu  | 4.70   | 5.38  | fused GEMM+bias+ReLU (unchanged)                       |
| demod    | 0.76   | 9.93  | median tiny; **p95 balloons** — blocks behind net GPU  |
| present  | 0.11   | 0.17  | nearest blit + offscreen capture                      |
| **TOTAL**| **11.69** | **18.27** | median stage-sum (NOT the throughput — see below) |

**BLOCKING (`NOOVERLAP=1`, frames=2617)** — `s13-nooverlap.log`:

| stage    | median | p95   | note                                        |
|----------|--------|-------|---------------------------------------------|
| trace    | 5.54   | 6.08  | no net contention (net waited serially)      |
| net wall | 12.99  | 14.39 | **the block reappears** (wait 12.98)         |
| demod    | 0.64   | 1.08  | tight — GPU idle when demod runs             |
| **TOTAL**| **20.24** | **23.15** | N0.h shape reproduced on this binary    |

### WALL-CLOCK fps — the throughput truth (frames / wall-seconds, whole run)
| path                | WALL-FPS | mean ms/frame |
|---------------------|----------|---------------|
| blocking (NOOVERLAP)| **35.55**| 28.1          |
| overlap (S13)       | **48.75**| 20.5          |

**Overlap moves throughput +37 % (35.55 → 48.75 fps).** The median stage-sum
(11.69 ms → "85 fps") is NOT the throughput: ~9 ms of frame-loop work lives
OUTSIDE the per-stage budget (world advance / skin·tick·splice, the offscreen
readback for `/scry`, HTTP) and is present in BOTH A/B arms equally — so the
wall-clock A/B delta (+13.2 fps) is the honest overlap win, and wall-clock fps
is the only number that tells the truth (exactly why this shift added it).

## 60 FPS THROUGHPUT VERDICT — NOT MET at 48.75 fps wall-clock; 20.5 ms/frame, 2 frames-in-flight
`60fps throughput NOT MET at 48.75 fps wall-clock (20.5 ms/frame mean), per-image
latency ≈ 2 frames-in-flight; the frame overlap improved throughput +37 % over
the blocking path (35.55 → 48.75 fps) and drove net_wall 12.99 → 0.012 ms, but
did not reach 60.` Two standing thieves, both now VISIBLE and honest:
1. **Single-GPU serialization tax.** The M1 has ONE GPU: deferring the CPU wait
   does not make the net's 4.70 ms GPU forward run in PARALLEL with trace/demod —
   it converts an explicit render-thread block into implicit GPU contention
   (trace 5.54→5.93, demod p95 0.97→9.93). Real 60 fps needs CUTTING net_gpu or
   trace GPU work, not just rescheduling it (N1 quality pass / a cheaper trace).
2. **~9 ms non-net frame loop** (world advance + offscreen capture + present +
   HTTP) outside the net budget — the next honest target once the GPU work
   itself is cut. `commit`/`wait`/`wall_fps` are now in `/budget` to track both.

## Parity ordeal — HOLDS (sync path untouched)
- `n0b_gather_and_shared_forward_match_cpu` — **ok** (release, `GAIA_NEURAL_LIVE=1`).
- `n0_gate1_live_net_matches_cpu_reference` — **ok**.
The overlap changes only the pipelined render loop; the ordeal's sync path
(`run_set_sync`, single queue+thread, commit_prepared_nowait+wait_prepared
back-to-back) is bit-identical to before.

## Proof — both eyes (READ, pixel words — NOT black)
- presented: `s13-presented.png` (960×640, `/scry?eye=presented`).
- belief: `s13-belief.png` (640×480, `/scry?eye=belief`, raw net radiance).
- **Pixel words (both, READ):** coherent naruko dusk scene — stacked brown
  crates, a translucent green-tinted glass panel, central dark tower ringed by
  cyan/pink/violet concentric halos, pale presence spheres (tower mouth + around
  the panel/platform), a large glass orb on a green cylindrical pedestal, a dark
  chimneyed factory block with lit windows at right, pink→mauve sky over a purple
  ground. Colours natural, radiance bounded → GPU demod wired right; the per-set
  AOV keeps albedo matched to the presented frame. Belief eye = same geometry,
  brighter/desaturated (albedo not undone), as designed. **Both eyes render — the
  overlap is coherent, no wedge, no black.**

## Source
- commits: frame-overlap wip + A/B toggle (this shift), measured on the following.
- logs: `s13-offscreen-release.log` (overlap default), `s13-nooverlap.log`
  (`GAIA_NATIVE_NET_NOOVERLAP=1`) under `packages/scrying-glass/proof/neural-live/`.
- env: `GAIA_NEURAL_LIVE=1 GAIA_NATIVE_OFFSCREEN=true GAIA_NATIVE_NET_PRESENT=true
  GAIA_NATIVE_HUD=false`, release, world `worlds/naruko`, M1/macOS 26. Tag `[n0i]`.
  `/budget` now carries `wall_fps`, `net_commit`, `net_wait`.

---

# N0.j — S13 THE OUTSIDE-9ms HUNT (SHIFT 13): the 9ms named, the tax killed, overlap tried

N0.i left the truth: wall-clock 48.75 fps (20.5 ms/frame) while the stage-sum
median was 11.69 ms — **~9 ms of frame-loop work lived OUTSIDE the per-stage net
budget**, named-but-unmeasured (world advance skin·tick·splice · per-frame
offscreen readback feeding `/scry` · HTTP). S13 INSTRUMENTS that outside-work,
KILLS the readback tax (on-demand), TRIES to overlap the world advance, and
brings back the honest number. Live, offscreen 640×480, release, M1/macOS 26,
`worlds/naruko`, `GAIA_NATIVE_OFFSCREEN=true`.

## (1) INSTRUMENT — where the 9 ms lives (`/budget` `outside` block)
`OutsideBudget` wraps the non-stage frame-loop segments with named timers, one
per frame, spliced into `/budget`. First measurement (on-demand default):

| segment    | median | p95   | note                                              |
|------------|--------|-------|---------------------------------------------------|
| world      | 6.96   | 7.20  | **THE 9 ms — it is ALL world advance**             |
| readback   | 0.000  | 0.000 | on-demand default (per-frame copy gone)            |
| http       | 0.19   | 0.37  | scry drain + /budget + /state JSON on render thread |
| loop_total | 18.19  | 26.06 | whole iteration wall, sans deadline sleep          |

**VERDICT of the hunt: the ~9 ms is `advance_world`** — skin·tick·splice + a
FRESH BVH build/upload EVERY animating frame (naruko's presence spheres move, so
`command_bodies_walked` returns true every tick → `splice.update` +
`integrator.update_bvh` re-run + reset accum). The other two named suspects are
NOT thieves: readback ~0, http 0.2 ms. `stage_sum(10.86) + world(6.96) +
http(0.19) ≈ loop_total(18.19)` — the books balance.

## (2) KILL THE MEASUREMENT TAX — readback is now ON-DEMAND
The per-frame offscreen readback (`copy_texture_to_buffer` + map submit that fed
`latest` EVERY frame) is gone from the render loop. `capture_presented` reads
the current offscreen texture back to the CPU ONLY when a bare `/scry` asks (the
offscreen BLIT still runs every frame, so the texture always holds the latest
presented image). A/B toggle `GAIA_NATIVE_PERFRAME_READBACK=1` restores the old
per-frame copy.

**Honest finding — the readback was NEVER a render-thread thief.** The per-frame
copy costs **0.002 ms** to ENCODE on the render thread (the copy + map callback
run async on the GPU + capture-worker thread), so killing it moves wall-clock
throughput by **noise** (perframe 48.6 → on-demand 48.3 fps). On-demand is still
the right design (no wasted GPU bandwidth/copy when nobody is scrying), and it
**proves the on-demand path** (both eyes below served through `capture_presented`),
but it does NOT buy fps. The N0.i suspect list was wrong about this one.

## (3) OVERLAP THE REAL WORK — TRIED, MEASURED, DOES NOT HELP
Intent: advance the NEXT frame's world AFTER this frame's GPU submit, so the
7 ms world CPU hides under the in-flight GPU trace (`update_bvh` allocs fresh
buffers each tick, in-flight submission retains its own → the reorder is SAFE).
Measured neutral-to-worse: **47.6 vs 48.4 fps**, AND it costs one frame of
world-state latency. **Root cause: `trace` is SYNCHRONOUS on the render thread**
— it submits+POLLS the GPU for the AOV that feeds the gather (N0.d "2
submits+polls"), so by the time the deferred advance runs the GPU is already
idle. There is no GPU flight to hide the world CPU under. Serial is the default;
overlap kept behind `GAIA_NATIVE_WORLD_OVERLAP=1` for the record — a real win
only once trace stops blocking the render thread.

## (4) MEASURE — wall-clock fps A/B (release, offscreen, player-shaped, ≥600 frames)

| arm                                  | WALL-FPS | stage TOTAL med | world med | readback |
|--------------------------------------|----------|-----------------|-----------|----------|
| on-demand + serial (S13 DEFAULT)     | **48.3** | 10.2–11.9       | 7.05      | 0.000    |
| per-frame readback (`PERFRAME=1`, N0.i) | 48.6  | 10.3            | 7.12      | 0.002    |
| world-overlap (`WORLD_OVERLAP=1`)    | 47.6     | 11.5            | 7.32      | 0.000    |

All three within ~2% — contention-band noise (numbers drift with sibling load,
per N0.e's honesty note). The A/B delta from either cut is inside the noise: **no
throughput was bought this shift.** What WAS bought: the 9 ms is now NAMED and in
`/budget`, and one non-thief (readback) is retired from the render loop.

## 60 FPS THROUGHPUT VERDICT — NOT MET at ~48 fps wall-clock; the 9 ms is world advance
`60fps throughput NOT MET at ~48 fps wall-clock (~20.5 ms/frame). The N0.i
"~9 ms outside the stage table" is now LOCATED: it is ~7 ms of world advance
(BVH re-splice + upload every animating frame) — readback (~0) and http (0.2)
are NOT thieves. The per-frame readback tax was killed (on-demand,
capture_presented) but was only ~0.002 ms render-thread cost, so throughput held
flat; world-advance overlap does not help because trace is synchronous on the
render thread (no GPU flight to hide the CPU under).` Remaining-thief table:

| thief             | ms   | why it stands / next attack                                   |
|-------------------|------|---------------------------------------------------------------|
| world advance     | ~7.0 | fresh BVH re-splice+upload EVERY animating frame — cache the   |
|                   |      | static BVH harder / skin without full re-splice / only when    |
|                   |      | geometry actually changes >ε (dynamic partition already split) |
| trace (synchronous)| ~6.0| submits+polls GPU on the render thread for the AOV — make it   |
|                   |      | async so world advance CAN overlap it; also cuts trace GPU     |
| net_gpu contention| ~4.7 | single-M1-GPU serialization (N0.i) — cut net_gpu (N1 quality)  |

The honest sum: ~7 (world) + ~6 (trace, part GPU-blocking) + ~4.7 (net GPU,
contends) ≈ the 20.5 ms wall. 60 fps needs CUTTING work (async trace + a cheaper
world advance + net_gpu), not rescheduling it — as N0.i already warned.

## Parity ordeal — HOLDS
- `n0b_gather_and_shared_forward_match_cpu` — **ok** (release, `GAIA_NEURAL_LIVE=1`).
- `n0_gate1_live_net_matches_cpu_reference` — **ok**.
The outside-work instrumentation + on-demand readback do not touch the net
sync/ordeal path.

## Proof — both eyes (READ, pixel words — served through the ON-DEMAND path)
- presented: `s14-presented.png` (960×640, bare `/scry` → `capture_presented`).
- belief: `s14-belief.png` (640×480, `/scry?eye=belief`).
- **Pixel words (both, READ):** coherent naruko dusk scene — pale green-flecked
  translucent glass panel (reflecting), stacked brown crates + a small orange
  crate on the ground, central dark tower/lighthouse ringed by white/pink
  concentric halos, pale presence spheres (left of the tower, an amber one
  upper-right, two by the platform), a large green-tinted translucent glass orb
  on a dark cylindrical pedestal, a dark chimneyed factory block with lit
  windows at right, pink→mauve sky over a purple-mauve ground. Colours natural,
  radiance bounded → GPU demod wired right. Belief eye = same geometry,
  brighter/desaturated over a cream ground (albedo not undone), as designed.
  **Both eyes render — the on-demand readback path is thereby PROVEN, no black,
  no wedge.**

## Source
- commits: instrument+on-demand `0991c63`, world-overlap toggle (this shift).
- logs: `s14-overlap.log`/`s14-default.log` (on-demand default),
  `s14-perframe.log` (`PERFRAME_READBACK=1`), `s14-worldoverlap.log`
  (`WORLD_OVERLAP=1`), `s14-worldserial.log` under
  `packages/scrying-glass/proof/neural-live/`.
- env: `GAIA_NEURAL_LIVE=1 GAIA_NATIVE_OFFSCREEN=true GAIA_NATIVE_NET_PRESENT=true
  GAIA_NATIVE_HUD=false`, release, world `worlds/naruko`, M1/macOS 26. Tag `[n0i]`.
  `/budget` now carries an `outside` block (world/readback/http/loop_total).
  Toggles: `GAIA_NATIVE_PERFRAME_READBACK=1`, `GAIA_NATIVE_WORLD_OVERLAP=1`,
  `GAIA_NATIVE_NET_NOOVERLAP=1` (N0.i, still live).


# N0.k — SHIFT 15: the 7ms world advance, split to the leaf (premise overturned)

State in: N0.j named the ~7ms world advance as "BVH re-splice + upload EVERY
animating frame" and told S15 to make splice/upload dirty-only. S15's FIRST act
— instrument INSIDE the 7ms — proves that premise WRONG. The whole 7ms is the
physics solver step; splice+upload together are ~0.6ms.

## (1) INSTRUMENT — the 7ms world advance, split to the leaf (`/budget world_stages`)
`advance_world` + `Dynamics::tick_with_ops` now carry per-stage timers, spliced
into `/budget` as a `world_stages` block. Naruko, offscreen 640×480, release,
GAIA_NATIVE_OFFSCREEN=true, player-shaped (POST /walk KeyW bursts), ≥1000 frames,
median/p95 ms:

| stage        | median | p95   | what it is                                        |
|--------------|--------|-------|---------------------------------------------------|
| world        | 6.97   | 7.39  | the whole advance (skin·tick·gather·splice·upload) |
|  ├ command   | 0.539  | 0.617 | command_bodies_walked (SAMA gait + body SKIN)      |
|  ├ tick      | 5.456  | 5.704 | Dynamics::tick_with_ops                             |
|  │  ├ kami   | 0.032  | 0.042 | tick_decorative (KAMI eval → transform ops)        |
|  │  ├ apply  | 0.011  | 0.015 | apply KAMI ops to the ECS                          |
|  │  ├ physics| 5.389  | 5.635 | physics block (solver.step + poll_bonded + write)  |
|  │  │  ├ **solver_step** | **5.362** | **5.603** | **elements::Solver::step() — THE THIEF** |
|  │  │  └ poll | 0.024 | 0.035 | poll_bonded per-tick fracture flood-fill           |
|  │  └ rederive| 0.018 | 0.024 | re-derive every entity model from its transform    |
|  ├ gather    | 0.063  | 0.086 | dynamic_leaf_triangles_for_eye                     |
|  ├ splice    | 0.158  | 0.187 | DynamicSplice::update (refit + CPU merge)          |
|  └ upload    | 0.433  | 0.541 | integrator.update_bvh (GPU node/tri buffers)       |

**VERDICT of the hunt: the 7ms world advance is `elements::Solver::step()`,
5.36ms every tick.** N0.j was wrong TWICE: (a) the BVH re-splice+upload it named
is ~0.6ms (splice 0.16 + upload 0.43), not 7; (b) neither KAMI decorative eval
(0.03) nor the JSON transform round-trip (apply 0.01 + rederive 0.02) is a thief.
The living layer's XPBD rigid+particle solver — `substeps × iterations` of
solve_distance/shape_matching/collision_normal + O(n²) solve_body_collisions over
the 22 declared bodies — owns the entire cost. The books balance:
`command(0.54) + tick(5.46) + gather(0.06) + splice(0.16) + upload(0.43) ≈
world(6.97)`, and inside tick `physics(5.39) + kami/apply/rederive(0.06) ≈ 5.46`.

## (2) THE ONE DIRTY-ONLY CUT IN THE ADVANCE PATH — dirty-only SKIN
`command_bodies_walked` re-skinned ALL bodies every tick even though only the
animating ones change geometry. `BodyInstance::reskin_if_dirty` gates `skin_body`:
an idle, settled body (gait Idle, not blending) whose model is unchanged from the
last skin keeps its `world_tris` verbatim — the engine's own `is_animating`
contract already treats such a body as STATIC geometry, and `skin_body` is pure,
so a kept `world_tris` is byte-identical to a re-skin (determinism intact,
`prev_animating` forces one final skin the tick after motion stops). Toggle
`GAIA_NATIVE_DIRTY_SKIN=0` restores always-skin.

**A/B (command stage, median ms):** dirty-on 0.539 vs dirty-off 0.578 — a ~0.04ms
median / ~0.11ms p95 saving. HONEST: negligible against the 20ms frame, because
`command`/skin was already only ~0.58ms and most naruko presences ARE animating.
It is the RIGHT cut for the advance path (static bodies now cost zero skin) but it
does NOT move 60fps — the thief is solver_step, which this shift did not touch
(a solver rest/sleep charter in the `elements` crate is a separate scope, and the
IRON law + parity/determinism ordeals guard that crate).

## (3) SPLICE/UPLOAD DIRTY-ONLY — NOT DONE, and CORRECTLY not done
N0.j's PRIMARY (only touched subtrees re-splice, only dirty ranges upload) is a
~0.6ms surface (measured above). Cutting it to zero would buy ~0.6ms of a 20ms
frame — inside the contention noise band (N0.e). The instrumentation redirects
the next shift's effort away from a proven non-thief and onto solver_step.

## (4) MEASURE — wall-clock fps A/B (release, offscreen, player-shaped, ≥1000f)

| arm                                  | WALL-FPS | world med | command | solver_step |
|--------------------------------------|----------|-----------|---------|-------------|
| dirty-skin ON (S15 DEFAULT)          | ~49.5    | 6.97–7.05 | 0.539   | 5.362       |
| dirty-skin OFF (`DIRTY_SKIN=0`)      | ~45–49*  | 7.38      | 0.578   | 5.593       |

*the OFF run landed in a more contended moment (sibling load; p95 net_wait rose
to ~7–10ms) so its wall-fps is not comparable — the honest A/B delta is the
`command` stage (0.578→0.539), not wall-fps, which is dominated by contention
noise (N0.e). No meaningful throughput was bought or claimed.

## 60 FPS THROUGHPUT VERDICT — NOT MET at ~49 fps wall-clock; the 7ms is the physics solver
`60fps throughput NOT MET at ~49 fps wall-clock (~20.4 ms/frame). The ~7ms world
advance N0.j blamed on BVH re-splice+upload is, when instrumented to the leaf,
elements::Solver::step() at 5.36ms — splice+upload together are ~0.6ms and are
NOT the thief; KAMI decorative + JSON round-trip are ~0.06ms and are NOT the
thief. The one dirty-only cut available in the advance path (dirty-only skin,
static bodies cost zero) is correct and green but saves only ~0.04ms because skin
was already ~0.58ms. 60fps needs a solver-level rest/sleep charter in the elements
crate — a separate scope guarded by the IRON law and the parity/determinism
ordeals.` Remaining-thief table (updated):

| thief             | ms    | why it stands / next attack                                  |
|-------------------|-------|--------------------------------------------------------------|
| **solver_step**   | ~5.36 | elements XPBD step: substeps×iterations solve + O(n²) body   |
|                   |       | collisions over 22 bodies. CUT: island sleeping (rest bodies |
|                   |       | skip solve), or O(n²)→broadphase body collisions. NEW CHARTER|
| trace (synchronous)| ~6.0 | submits+polls GPU on the render thread (N0.j) — async it     |
| net_gpu contention| ~4.7  | single-M1-GPU serialization (N0.i)                            |
| body skin         | ~0.54 | dirty-only skin landed (static bodies now zero); tail is the |
|                   |       | animating handful — cheap, not worth further cutting          |

Honest sum: solver_step(5.4) + splice/upload/skin/gather(1.2) ≈ world(6.97);
world(6.97) + trace(6) + net_gpu(4.7) ≈ the 20.4ms wall. As N0.i/N0.j warned,
60 needs CUTTING — and the cut that matters is now NAMED to the leaf: the solver.

## Parity + determinism ordeals — HOLD (final build, all timers in)
- `n0b_gather_and_shared_forward_match_cpu` — ok.
- `n0_gate1_live_net_matches_cpu_reference` — ok.
- `rite5` — 17/17 ok, incl. `v0_body_render_is_deterministic`,
  `v1_gait_is_deterministic_byte_identical`, `v2_cat_animation_is_byte_identical`
  (the dirty-skin gate is byte-identical to always-skin, determinism intact).
All added timers are measurement-only (Instant reads), the step-profiled pattern.

## Correctness gate — animation still MOVES (dirty-skin did not freeze it)
Two `/scry` presented frames ~2.5s apart (`s15-motionA.png`, `s15-motionB.png`,
READ): the presence spheres MOVED — the amber sphere travelled from left-of-tower
to right-of-tower, the white spheres shifted, and the tower's concentric halos
sit at a different animation phase. Coherent naruko dusk in both. Dirty-tracking
did NOT silently freeze the living layer.

## Proof — both eyes (READ, pixel words — on-demand path)
- presented `s15-presented.png` (960×640, bare `/scry`): coherent naruko dusk —
  pale green-flecked translucent glass panel, stacked brown crates with a white
  sphere, a small orange crate, central dark tower ringed by white/pink/cyan
  halos, pale presence spheres (amber+white by the tower, one at the platform),
  a large green-tinted glass orb on a dark pedestal, a dark chimneyed factory
  block with lit amber windows, pink→mauve sky over purple-mauve ground. Colours
  natural, radiance bounded.
- belief `s15-belief.png` (640×480): same geometry, brighter/desaturated over a
  CREAM ground (albedo not undone), factory pale beige, glass orb dark olive — as
  designed. Both eyes render, no black, no wedge — the on-demand readback holds.

## Source
- commits: instrument sub-stages + dirty-skin + finer tick/physics split (S15,
  neural-live branch — see `git log`).
- bench: `packages/scrying-glass/proof/neural-live/s15-bench.sh`.
- budgets: `s15-*.budget.json`; logs `s15-*.log`; PNGs `s15-*.png` (same dir).
- env: `GAIA_NEURAL_LIVE=1 GAIA_NATIVE_OFFSCREEN=true GAIA_NATIVE_NET_PRESENT=true
  GAIA_NATIVE_HUD=false`, release, world `worlds/naruko`, M1/macOS 26.
  `/budget` now carries `world_stages`. Toggle `GAIA_NATIVE_DIRTY_SKIN=0`.

# N0.l — SHIFT 16: THE SOLVER CHARTER — island sleeping (settled bodies cost ZERO)

State in: N0.k named `elements::Solver::step()` the 5.36ms thief but stopped at
the solver door. S16's charter: instrument INSIDE the step, then cut with island
sleeping. Both done. Design law kept: the classical XPBD solver is the
teacher/lab equipment (Ananke's learned solve replaces it by measurement later);
this is legal maintenance of the teacher, not a new physics.

## (1) INSTRUMENT — solver_step split to the leaf (`step_profiled`/`PhaseProfile`)
The `PhaseProfile` seam already existed; `naruko_solver_substages` (new example)
loads the REAL settled naruko, warms 120t, then times the resting arrangement.
Median ms/tick, 240 samples, release, single core:

| phase             | BEFORE (sleep off) | note                                        |
|-------------------|--------------------|---------------------------------------------|
| integrate         | 0.003              | symplectic-Euler predict                    |
| solve_distance    | 0.005              | XPBD bond solve (50 bonds)                   |
| shape_matching    | 0.023              | 12 rigids' polar shape-match                 |
| **collision_static** | **3.30**        | **particle-vs-12368-static-tris — THE LEAF THIEF** |
| collision_body    | 0.44               | O(n²) over 199 clustered particles          |
| cluster_floodfill | 0.007              | per-tick fragment components                 |
| velocity_passes   | 0.07               | friction/restitution/strife                  |
| ── TOTAL          | 3.88*              | *after the fp-cache below; was 5.36 (N0.k)   |

**LEAF VERDICT: the thief inside solver_step is `collision_static` (3.30ms) — the
per-substep particle-vs-static-triangle broadphase over naruko's 12368-tri soup,
NOT the O(n²) body pass (0.44ms).** The N0.k lesson held twice: the sub-table
moved the cut from the assumed O(n²) body collisions to the real static pass.

Instrumenting also exposed a SECOND thief hiding in the untimed remainder: N0.k's
5.36ms was 3.88ms of phases + ~1.48ms of `ensure_collision_grid` RE-HASHING the
static 12368-tri soup EVERY tick (`TriangleGrid::fingerprint` ≈ 148k f64 absorbs)
just to check staleness of an immutable collider.

## (2) THE CUTS
- **Collider-fingerprint identity cache** (universal, every tick): colliders are
  static per scene and always REPLACED wholesale, so `(triangles.as_ptr(), len)`
  is a sound identity key — the fingerprint is recomputed only on a genuine
  collider swap. Byte-identical (same fp value either way). Alone: 5.36 → 3.88ms.
- **Island sleeping** (the charter cut): `maintain_sleep` runs once at tick end.
  QUIET counters (speed < `sleep_vel` → +1, else 0). ISLANDS = union-find (root =
  MIN member index — deterministic, no HashMap in outcomes) over rigid
  membership + live bonds + body-vs-body PROXIMITY edges sourced ONLY from AWAKE
  clustered particles (two settled asleep bodies can't newly touch ⇒ all-asleep
  is O(bonds), free). An island sleeps iff no member is wake-flagged AND every
  member's quiet ≥ `sleep_frames`; asleep particles are skipped WHOLE by
  integrate / solve_distance / shape_matching / collision_static / collision_body
  and their velocity is held 0. WAKE on: impulse/op (`apply_impulse*` wake the
  touched island — door pushes), a moving body's proximity union into a sleeping
  island (contact), and fracture (torn bonds wake-flag their endpoints). Sleeping
  is per-ISLAND, never per-body (stacks stay coherent). IRON params:
  `GAIA_NATIVE_SLEEP` (arm), `GAIA_NATIVE_SLEEP_VEL` (def 0.03 m/s),
  `GAIA_NATIVE_SLEEP_FRAMES` (def 24). Sleep OFF = byte-unchanged (every branch
  gated on `sleep_enabled` + a set flag).

**solver_step SUB-TABLE, settled naruko (median ms/tick):**

| phase             | BEFORE | AFTER (sleep on) |
|-------------------|--------|------------------|
| integrate         | 0.003  | 0.002            |
| solve_distance    | 0.005  | 0.0003           |
| shape_matching    | 0.023  | 0.0002           |
| collision_static  | 3.30   | 0.002            |
| collision_body    | 0.44   | 0.008            |
| velocity_passes   | 0.07   | 0.003            |
| ── TOTAL          | 3.88   | **0.026 (152×)** |

199 particles / 13 islands, all asleep at rest. Settled bodies now cost ~ZERO.

## (3) GATES — all hold
- elements ordeals: 40/40 (36 prior + 4 new S16). Prior determinism/broadphase
  byte-identity ordeals unchanged (fp-cache is byte-identical; sleep off by
  default). New `s16_sleep_ordeals`: sleep DETERMINISM (two runs byte-identical
  incl. a mid-run wake), settled-box-SLEEPS (frozen, drift <1e-9), the WAKE-TEST
  (a slept box PUSHED wakes the same instant and travels >0.05m — silent-freeze
  would fail), rest-pose parity (|Δcentroid| slept-vs-always-solve < 5e-3 m).
- `rite5` 17/17 byte-identical. gather + live parity ordeals: ok
  (`n0b_gather_and_shared_forward_match_cpu`, `n0_gate1_live_net_matches_cpu_reference`).
- fracture/building ordeals present + green: `ordeal_building_at_rest_stays_at_rest`,
  `ordeal_building_collapse_replay_byte_identical`, vi2 break ordeals 7/7.
- MOTION gate (render): two `/scry` frames 2.5s apart under sleep ON
  (`s16-motionA/B.png`, READ) — the presence spheres MOVED (pink/cyan pair from
  left-of-tower to right-of-tower and out to the glass panel; halo rings at a
  different phase) while the settled crates/glass-orb/factory stayed put. Sleep
  froze the physics islands, NOT the living animation layer.

## (4) MEASURE — live, offscreen, player-shaped (/walk KeyW bursts), naruko
| arm         | solver_step med | tick med | outside.world med | loop_total med |
|-------------|-----------------|----------|-------------------|----------------|
| sleep OFF   | 4.45 ms         | 4.53 ms  | 5.98 ms           | 6.21 ms        |
| sleep ON    | 0.077 ms        | 0.20 ms  | 2.64 ms           | 3.07 ms        |

solver_step median 4.45 → 0.077 ms (58× live); the world-advance CPU cost
(loop_total) 6.21 → 3.07 ms. p95 solver_step stays ~3.9ms because /walk bursts
wake nearby/own bodies — correct: moving = solving; the median is the settled
town. (Offscreen loop_total is world+http only; it does NOT include the windowed
trace(~6ms) + net_gpu(~4.7ms) N0.k measured as the rest of the 20.4ms wall.)

## 60 FPS THROUGHPUT VERDICT — solver thief ELIMINATED; frame is now TRACE-bound
`60fps NOT YET MET on the full windowed wall, but the SOLVER charter is MET:
elements::Solver::step() is no longer a thief — settled naruko's solver_step
falls 4.45→0.077ms live median (152× on the isolated settled sub-table), and the
world advance drops from N0.k's ~6.97ms to ~2.6ms. The remaining thieves are the
synchronous trace (~6ms, submits+polls GPU on the render thread — N0.j) and
single-GPU net_gpu contention (~4.7ms, N0.i); with world≈2.6 + trace≈6 +
net_gpu≈4.7 the projected wall is ~14-16ms (≈60-70fps) IF those two hold, so the
next cut that matters is async trace, not the solver.` Remaining-thief table:

| thief             | ms    | status                                                    |
|-------------------|-------|-----------------------------------------------------------|
| ~~solver_step~~   | 0.08  | KILLED — island sleeping + fp-cache (settled = zero)      |
| trace (synchronous)| ~6.0 | NEW #1 — async the GPU submit/poll off the render thread  |
| net_gpu contention| ~4.7  | single-M1-GPU serialization (N0.i)                        |

## Adversary — S16 current
The sub-table could still lie about a MOVING town (many awake bodies): the win is
proven only for the settled/mostly-static naruko the charter named — a scene that
keeps 200 bodies tumbling pays the full 3.88ms and gains nothing from sleep (by
design; motion must solve). The p95 3.9ms under /walk is that regime showing
through. The fp-cache assumes no code mutates a collider's triangles in place
without reallocating — true across this engine today (colliders reinstall on
scene change), asserted by grep, but it is an INVARIANT a future in-place
collider editor would silently break. Wake latency is one tick (proximity union
fires at tick end, real overlap the next) — a body moving >~1 cell/tick could
tunnel one frame into a sleeper before it wakes; acceptable at sleep-scale
speeds, unproven for a bullet. The 60fps claim rests on N0.k's trace/net_gpu
numbers being stable; this shift did not re-measure the windowed wall (offscreen
only), so the ~14-16ms projection is a PROJECTION, not a played windowed frame.

## Source
- commits: S16 island sleep + fp-cache + sleep ordeals (neural-live — see git log).
- instrument: `packages/scrying-glass/examples/naruko_solver_substages.rs`.
- ordeals: `packages/elements/tests/s16_sleep_ordeals.rs`.
- PNGs: `/tmp/s16-{off,on}-presented.png`, `/tmp/s16-motion{A,B}.png` (READ above).
- env: `GAIA_NATIVE_OFFSCREEN=true GAIA_NATIVE_SLEEP=1 [GAIA_NATIVE_SLEEP_VEL/_FRAMES]`,
  release, world `worlds/naruko`, M1/macOS 26. Toggle sleep off = omit the env.

---

# N0.m — SHIFT 17: THE CROWN MEASUREMENT — wall-clock fps after the solver cut

State in: N0.l killed `elements::Solver::step()` (settled solver_step 4.45→0.077
ms live median) and PROJECTED (not measured) world≈2.6 + trace≈6 + net_gpu≈4.7 ≈
14-16ms → ~60-70fps. This shift MEASURES the real windowed wall-clock, live,
player-shaped, offscreen, ≥1000 frames both arms. Live, offscreen 640×480,
release, M1/macOS 26, `worlds/naruko`, `GAIA_NATIVE_OFFSCREEN=true`.

## (1) BUILD — release, current HEAD
`cargo build --release -p scrying-glass` under the build-lock token — already
current at `1df1e15` (0.73s, no recompile needed; verified fresh by the lock
acquiring cleanly and the binary timestamp preceding this shift's run).

## (2) BENCH RECIPE — `s17-bench.sh` (new, modeled on `s15-bench.sh`)
Offscreen server, `/walk` POST bursts (`KeyW`, 16 ticks, 70× @0.35s spacing —
long enough for >=1000 frames at ~50fps), then `/budget` + `/state` snapshot +
both eyes via `/scry`. Script:
`packages/scrying-glass/proof/neural-live/s17-bench.sh`.
Env: `GAIA_NEURAL_LIVE=1 GAIA_NATIVE_OFFSCREEN=true GAIA_NATIVE_NET_PRESENT=true
GAIA_NATIVE_HUD=false GAIA_NATIVE_PORT=<port> GAIA_WORLD=.../worlds/naruko`.

## (3) MEASURE — wall-clock fps A/B (release, offscreen, player-shaped, ≥1000f)

| arm       | port | frames | WALL-FPS | mean ms/frame |
|-----------|------|--------|----------|---------------|
| sleep ON (`GAIA_NATIVE_SLEEP=1`, S17 default) | 8442 | 1620 | **52.86** | **18.92** |
| sleep OFF (`GAIA_NATIVE_SLEEP=0`)             | 8443 | 1539 | **50.23** | **19.91** |

**Delta: +2.63 fps / −0.99 ms mean per frame.** This does **NOT** reproduce the
"~3-4 ms/frame" the task context asked to verify — at the wall-clock MEAN level
the win is ~1 ms, not 3-4. See §(5) for why (the tail, not the median, sets the
mean).

## (4) FULL STAGE TABLE — median/p95 ms · `/budget` JSON (verbatim)

**sleep ON** (`s17-sleep-on.budget.json`, frames=1620):

| stage      | median | p95    |
|------------|--------|--------|
| trace      | 5.943  | 9.795  |
| gather     | 0.905  | 1.291  |
| net_wall   | 0.013  | 4.832  |
| net_gpu    | 4.895  | 6.659  |
| net_commit | 0.006  | 0.011  |
| net_wait   | 0.001  | 4.820  |
| demod      | 1.664  | **10.461** |
| present    | 0.086  | 0.131  |
| **total**  | **11.423** | **22.453** |

**sleep OFF** (`s17-sleep-off.budget.json`, frames=1539):

| stage      | median | p95    |
|------------|--------|--------|
| trace      | 5.578  | 9.512  |
| gather     | 0.939  | 1.321  |
| net_wall   | 0.014  | 3.504  |
| net_gpu    | 4.865  | 6.177  |
| net_commit | 0.006  | 0.014  |
| net_wait   | 0.001  | 3.488  |
| demod      | 1.031  | **10.255** |
| present    | 0.089  | 0.136  |
| **total**  | **10.483** | **21.084** |

## (5) OUTSIDE TABLE (world/http/readback/loop_total) — the solver win, VERIFIED live

| segment    | sleep ON median/p95 | sleep OFF median/p95 | note                     |
|------------|----------------------|------------------------|--------------------------|
| world      | **1.747 / 2.187**    | **5.607 / 6.392**      | **−3.86 ms — reproduces the ~3-4ms prediction** |
| readback   | 0.000 / 0.000        | 0.000 / 0.000          | on-demand, unchanged (N0.j) |
| http       | 0.618 / 1.279        | 0.592 / 1.193          | unchanged                |
| loop_total | 13.916 / 24.808      | 16.825 / 27.659        | −2.91 ms (world win partly offset by total +0.94ms) |

`world_stages.solver_step` median: **0.050ms (sleep ON) vs 3.945ms (sleep
OFF) — 79× live** (close to N0.l's 152× isolated sub-table; live regime is
lower because `/walk` wakes the player-proximate island every burst — correct,
moving = solving). `command`(skin)/`splice`/`upload`/`gather` all near-identical
across arms (0.54/0.17/0.48/0.07 vs 0.54/0.16/0.46/0.07) — the fp-cache and
dirty-skin cuts hold, confirming N0.k/N0.l did not regress.

## (6) WHY THE MEAN DIDN'T MOVE LIKE THE MEDIAN — the demod tail
`demod` p95 is **10.46ms (sleep ON) / 10.26ms (sleep OFF) against a ~1-1.7ms
median in both** — an order-of-magnitude GPU-contention tail (single-M1-GPU
serialization, N0.i's standing #2 thief) that solver sleep does not touch (the
solver sits entirely on the CPU side of `world`, not on the GPU timeline `demod`
shares with `trace`/`net_gpu`). Wall-clock fps is `frames / wall-seconds` — it
is a MEAN, and a MEAN is set by the tail, not the median. The world-advance
median win (−3.86ms) is real and reproducible (§5), but the SAME contended GPU
tail that was already eating frames before this shift eats a comparable amount
after it — so the median-level "books balance" arithmetic this doc has used
since N0.d (stage_sum + world + http ≈ loop_total, all at MEDIAN) **does not
extrapolate to wall-clock fps**, exactly as N0.i warned for a single run and as
this shift now shows applies to cross-shift A/B deltas too.

## 60 FPS THROUGHPUT VERDICT — NOT MET at 52.86 fps wall-clock (18.92 ms/frame mean), 2 frames in flight
`60fps throughput NOT MET at 52.86 fps wall-clock (18.92 ms/frame mean, sleep
ON — the S17 default), 2.25 ms short of the 16.67 ms wall; per-image latency ≈ 2
frames-in-flight (N0.i's overlap, unchanged this shift). The N0.l projection
("world≈2.6 + trace≈6 + net_gpu≈4.7 ≈ 14-16ms → 60-70fps") is REFUTED at the
wall-clock mean: the solver cut is real and VERIFIED live (world advance
5.61→1.75ms median, solver_step 3.95→0.05ms median, 79×) but a GPU-contention
tail in demod (p95 10.46ms vs 1.66ms median, unrelated to the solver) absorbs
most of the theoretical gain when measured as frames/wall-seconds rather than
median stage-sum. Sleep OFF, same binary, same walk script: 50.23 fps (19.91
ms/frame mean) — confirms the ON/OFF delta is real (+2.63 fps) but small
relative to the projection.` Remaining-thief table:

| thief                  | ms (median) | ms (tail, p95) | status                                   |
|------------------------|-------------|-----------------|-------------------------------------------|
| ~~solver_step~~        | 0.05        | 0.54            | KILLED, VERIFIED LIVE (was 3.95/4.32)     |
| trace (synchronous)    | 5.94        | 9.80            | UNCHANGED — N0.j's #1, unattacked          |
| net_gpu (contention)   | 4.90        | 6.66            | UNCHANGED — N0.i's #2, unattacked          |
| **demod (GPU tail)**   | 1.66        | **10.46**       | **NEWLY VISIBLE this shift** — the median-table method hid it; it is the actual mean-fps thief, not named as a distinct target before |

Honest sum at MEDIAN (does not predict the mean, see §6): trace(5.94) +
gather(0.91) + net_wall(0.01) + demod(1.66) + present(0.09) = total(8.61,
undercounts vs the measured total 11.42 — encode/submit overhead not itemized)
+ world(1.75) + http(0.62) ≈ loop_total(13.9) — median books still balance
internally; they just don't equal the measured wall-clock mean (18.92ms).

## Parity + determinism + rite5 + motion gates — ALL HOLD (this shift's own runs)
- `n0b_gather_and_shared_forward_match_cpu` — ok (GATE A 9.5e-7, GATE B
  1.9e-6, S8 MPSGraph-vs-chain 4.8e-7; release, `GAIA_NEURAL_LIVE=1`).
- `n0_gate1_live_net_matches_cpu_reference` — ok (live-vs-committed abs
  1.311e-6 rel 5.96e-5).
- `s16_sleep_ordeals` — 4/4: settled-sleeps (drift 0.00e0), WAKE-TEST (pushed
  sleeper travelled 0.5115m, no freeze), rest-pose parity (|Δcentroid|
  9.77e-13m), sleep determinism (byte-identical hash across runs).
- `rite5` — 17/17, incl. `v0_body_render_is_deterministic`,
  `v1_gait_is_deterministic_byte_identical`, `v2_cat_animation_is_byte_identical`.
- MOTION gate (render, sleep ON): two `/scry?eye=presented` frames ~2.5s apart
  under active `/walk` (`s17-motionA-presented.png`, `s17-motionB-presented.png`,
  READ) — the player camera and presence spheres are at visibly different
  positions/framing between A and B (frame A: wide platform view, large
  iridescent sphere near-camera, small sphere upper-left, tower with cyan halo
  rings mid-frame; frame B: under-platform view between two support posts, a
  single small pale sphere near the tower base, dark industrial block at
  right with lit windows) — coherent naruko dusk in both, no black, no wedge,
  no freeze.

## Proof — both eyes (READ, pixel words — NOT black, served on-demand)
- sleep-ON presented/belief: `s17-sleep-on-presented.png` /
  `s17-sleep-on-belief.png` — close-range view of the dark tower base against a
  mauve dusk sky, water reflecting a horizon line below; presented is dark/
  night-toned, belief is the same geometry brighter/desaturated over a cream
  ground (albedo not undone), as designed. (The walk script drove the player
  right up to the tower — a tighter framing than prior shifts' wide establishing
  shots, same scene, same demod correctness.)
- sleep-OFF presented/belief: `s17-sleep-off-presented.png` /
  `s17-sleep-off-belief.png` — **pixel-identical framing to sleep-ON** (same
  deterministic walk script → same player path): confirms the sleep toggle
  changes ONLY solver cost, not render output. Parity HOLDS visually.
- motion gate: `s17-motionA-presented.png` / `s17-motionB-presented.png` (see
  above).

## Source
- bench: `packages/scrying-glass/proof/neural-live/s17-bench.sh` (new).
- logs/budgets/state: `s17-sleep-on.log` / `.budget.json` / `.state.json`,
  `s17-sleep-off.log` / `.budget.json` / `.state.json`, `s17-motion.log`, all
  under `packages/scrying-glass/proof/neural-live/`.
- PNGs: `s17-sleep-{on,off}-{presented,belief}.png`,
  `s17-motion{A,B}-{presented,belief}.png`.
- env: `GAIA_NEURAL_LIVE=1 GAIA_NATIVE_OFFSCREEN=true GAIA_NATIVE_NET_PRESENT=true
  GAIA_NATIVE_HUD=false`, `GAIA_NATIVE_SLEEP=1|0` (A/B), release, world
  `worlds/naruko`, M1/macOS 26. Tag `[n0m]` (reuses `[n0i]` per-frame log lines
  — the print tag was not renamed this shift, an honest cosmetic gap).
- 0 scrying-glass processes running at shift end (verified: every spawned
  server was `kill`ed via bash `trap ... EXIT` and confirmed with `ps aux`).
