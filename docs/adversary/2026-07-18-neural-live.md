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
