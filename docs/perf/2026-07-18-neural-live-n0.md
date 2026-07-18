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
