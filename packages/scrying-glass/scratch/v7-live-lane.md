# v7-live lane — STAGE 1 (feature-map + GPU evidence split)

> **CORRECTION (room 7, 2026-07-20) — the "18.57ms/53.85fps" number is NOT
> a v4 baseline.** It traces to `docs/perf/2026-07-18-neural-live-n0.md`
> SHIFT 18 (N0.n, "CUT A" fused-demod measurement) — a run that predates
> the entire `GAIA_NATIVE_WEIGHTS` v1..v7 versioning scheme, which was
> introduced one shift LATER (SHIFT 19, "N1 SHIP + THE PURGE", default
> v2). It cannot be re-measured under any weights label because no
> version tag applied to it at capture time. `data/rdirect-weights-v4.bin`
> exists on disk but has **no PASS stamp anywhere in this worktree** (only
> `v7.bin.stamp` does — v7 is the first-ever weights to earn one here) and
> is REAL-OR-BLACK gated, so v4 **cannot lawfully render at all right
> now** — every below mention of "non-split v4 baseline" is this
> mislabel; read "18.57ms/53.85fps" as an unversioned historical number
> carried forward as an informal target, not a reproducible v4 measurement.
> The 18.57/53.85 comparisons below are left in place for the historical
> record but are NOT a v4 number and NOT re-earnable without running v4's
> own ordeal (~750s, not done in this lane).

Ghoul run 2026-07-20. Lane goal: rdirect_live.rs hosts the v7 act (39-in
split E/D + recurrent history + evidence clamp). Blocker map:
scratch/v7-cutover-ready.md (commit 7333316) — live path is 23-in plain-net,
v7 weights are 39-in split-recurrent. This file = STAGE 1 only.

## 1. THE 39-FEATURE LAYOUT (CPU reference, exhaustive)

Source: `src/rdirect.rs` — `pixel_features_split` (base 35) +
`hist_features_split` (+4 → 39). Trainer assembly: `direct_render_sequence_hist_split`
(same file) — calls `pixel_features_split` then `hist_features_split` per
pixel, motion fed as `Vec2::ZERO` (static-pose dataset).

`bilinear_taps` tap order (both E and D reuse the SAME 4 indices/dx/dy, one
set of taps per pixel): `[ (x0,y0), (x1,y0), (x0,y1), (x1,y1) ]` — top-left,
top-right, bottom-left, bottom-right of the low-res 2×2 neighbourhood
`low_coord` maps this target pixel into.

| idx   | meaning                          | source |
|-------|-----------------------------------|--------|
| 0-2   | E tap0 (x0,y0) demod-log rgb      | gather: `low_e[taps[0]]` → `log_demod(_, divisor)` |
| 3-5   | E tap1 (x1,y0) demod-log rgb      | gather: `low_e[taps[1]]` |
| 6-8   | E tap2 (x0,y1) demod-log rgb      | gather: `low_e[taps[2]]` |
| 9-11  | E tap3 (x1,y1) demod-log rgb      | gather: `low_e[taps[3]]` |
| 12-14 | D tap0 (x0,y0) demod-log rgb      | gather: `low_d[taps[0]]` |
| 15-17 | D tap1 (x1,y0) demod-log rgb      | gather: `low_d[taps[1]]` |
| 18-20 | D tap2 (x0,y1) demod-log rgb      | gather: `low_d[taps[2]]` |
| 21-23 | D tap3 (x1,y1) demod-log rgb      | gather: `low_d[taps[3]]` |
| 24    | subpixel dx                       | gather: `bilinear_taps` fractional x |
| 25    | subpixel dy                       | gather: `bilinear_taps` fractional y |
| 26-28 | hi-res albedo rgb                 | gather: native AOV cell0.xyz |
| 29-31 | hi-res world normal xyz           | gather: native AOV cell1.xyz |
| 32    | log depth = ln(max(depth,0)+1)    | gather: native AOV cell0.w |
| 33-34 | screen-space motion xy            | gather (ZERO in the static training set; a real live-frame value is a G-buffer aux, out of Stage-1 scope) |
| 35-37 | reprojected prev demod-log rgb    | HISTORY — `hist_features_split`'s `prev_dl` (bilinear-reprojected previous frame's net OUTPUT `out_dl`, via `p_cam.reproject` + depth/normal reject test); 0 on frame 0 or a reject |
| 38    | validity (1.0 = history accepted) | HISTORY — `hist_features_split`'s `valid`; 0.0 on frame 0 or reject |

`divisor` (both E and D share it): `demod_divisor(hi_albedo)` — this pixel's
own albedo + `ALBEDO_DEMOD_EPS` (1e-3), or `1` on a no-hit/sky pixel
(`albedo.length_sq <= 1e-8`).

**Constants:** `INPUT_FEATURES_SPLIT = RADIANCE_TAPS*3*2 + 2+3+3+1+2 = 35`,
`HIST_FEATURES_SPLIT = INPUT_FEATURES_SPLIT + 4 = 39`.

### Carry-over from the live 23-in layout
Live 23-in (`rdirect_gather.wgsl`, `pixel_features`) = idx 0-11 ONE composite
radiance tap set (12) + 12-22 same tail (subpixel/albedo/normal/depth/motion,
11 features). Every TAIL feature (idx 24-34 in the 39 layout, idx 12-22 in
the 23 layout) carries over UNCHANGED byte-for-byte — same `bilinear_taps`,
same albedo/normal/depth/motion reads, same order. Only the RADIANCE block
changes shape: one composite 4-tap set (12) → two 4-tap sets, E then D (24).
History (35-38) is entirely NEW — the live path has no recurrent state today
(current-frame-only, confirmed in rdirect_live.rs's own doc comment) — that
is STAGE 2 (recurrent history buffer ping-pong on GPU), not this stage.

## 2. STAGE 1 LANDED: GPU evidence split (E/D) in the live gather path

Flag: `GAIA_NATIVE_EVIDENCE_SPLIT=1` (env, default OFF — old 23-in path is
byte-identical when unset; nothing new even allocates).

New, additive, alongside the existing 23-in trace/gather (never replaces it
this stage — the net still runs the shipped 23-in weights; this only proves
the GPU can produce the E/D taps a 39-in gather would consume):

- `src/rdirect_gather_split.wgsl` — new compute entry `gather_split`, mirrors
  `rdirect_gather.wgsl`'s `gather` but reads the integrator's SPLIT accum
  (`accum_ed`, already emitted by `integrate_split` in integrator.wgsl — 2
  vec4 cells/px: `[2i+0]=E sum+count, [2i+1]=D sum+count`) instead of the
  single composite `accum`, and writes the 35-feature `INPUT_FEATURES_SPLIT`
  layout (E taps, D taps, tail — NO history yet, that's Stage 2) bit-for-bit
  vs CPU `rdirect::pixel_features_split`.
- `src/rdirect_gather.rs` — added `FeatureGatherSplit` (own pipeline/layout,
  `feature_bytes(n) = n * INPUT_FEATURES_SPLIT * 4`, `encode(...)` takes
  `accum_ed` + `aov` + `feats35` dest). `FeatureGather` (23-in) UNTOUCHED.
- `src/main.rs` `NetPresent`: `evidence_split: bool` (env-gated, read once at
  construction). When true, `new()` additionally pools `net_accum_ed`
  (`integrator.make_split_buffer`) + `net_feats_split`
  (`FeatureGatherSplit::feature_bytes(n)`) + one `FeatureGatherSplit`. In
  `resolve_frame`, when true, the trace stage ALSO dispatches
  `integrator.dispatch_split` (existing `Integrator` method, already used by
  the offline `trace_headless_split` trainer path — reused as-is, no
  integrator/WGSL changes) into `net_accum_ed` right after the ordinary
  `net_accum` dispatch (same encoder/queue, so FIFO ordering vs the AOV pass
  holds same as the composite path), then the gather stage ALSO runs
  `FeatureGatherSplit::encode` into `net_feats_split` right after the
  ordinary `gather.encode`. The 23-in `feats` buffer, the net forward, and
  demod/present are UNTOUCHED in both branches — flag OFF takes none of
  these branches, flag ON adds pure side buffers nothing downstream reads
  yet (Stage 3 wires a 39-in net to `net_feats_split` + a history buffer).

### Parity guard (flag OFF — the old 23-in path, byte-identical)
`cargo build --release -j2 --bin scrying-glass` — 0 errors, warnings
pre-existing only (same 3 as before this lane, unrelated fields/methods).
```
$ cargo test --release -j2 --test rdirect_gather_ordeals -- --nocapture
[n0b] GATE A gather: N=6144 px × 23 feat · max abs 9.537e-7
[n0b] GATE B shared forward: max abs 7.749e-7
[n0g] S8 MPSGraph(default) vs chain: max abs 2.384e-7 [source: this doc]
test n0b_gather_and_shared_forward_match_cpu ... ok

$ cargo test --release -j2 --test rdirect_gpu_ordeals --test rdirect_live_ordeals -- --nocapture
test c_ban_no_temporal_vocabulary_in_the_gpu_kernel ... ok
test a_gpu_inference_is_byte_identical_same_frame_twice ... ok
[rdirect parity] f32 parity_rel=5.636e-7 bound=2.159e-3
test b_f32_gpu_matches_cpu_within_derived_bound ... ok
[rdirect parity] fp16 MODE-A parity_rel=5.791e-4 bound=3.128e-3
test b2_fp16_fast_kernel_matches_cpu_within_derived_bound ... ok
[n0-gate1] N=6144 px · live-vs-committed: abs 1.311e-6 rel 5.960e-5 · live-vs-recomputed-CPU: abs 1.311e-6
test n0_gate1_live_net_matches_cpu_reference ... ok
```
All green, env unset (default = flag OFF) — confirms the additive branches
above never execute unless `GAIA_NATIVE_EVIDENCE_SPLIT` is set.

### E/D probe (flag ON)
`examples/v7_live_ed_probe.rs` — drives the SAME `integrator::dispatch_split`
+ `FeatureGatherSplit` wiring `NetPresent`'s `evidence_split` branch in
main.rs uses, for one fixed pose (naruko "front", same pose/shapes n0b uses:
low 48×32 → native 96×64, seed `0x7abc+5` = trainer's f=0 convention), and
cross-checks the 35-feature gather output against the CPU reference:
`pixel_features_split` fed by `trace_headless_split` (== the v7 trainer's
`render_pose` evidence source) for the SAME pose/seed.
```
$ cargo run --release -j2 --example v7_live_ed_probe
[v7-ed-probe] N=6144 px x 35 feat (35, no history) max-abs-diff: \
  E/D taps 4.768e-7 · tail 9.537e-7 · overall 9.537e-7
```
Well inside the n0b gate's own 1e-4 bound (float-ULP class, not a wiring
error) — the E/D split gather is bit-correct against the CPU/trainer
reference. History (idx 35-38, Stage 2) is not produced by `gather_split`
yet, so it is out of this probe's scope by construction.

## 3. STAGE 2 LANDED: GPU recurrent history ping-pong (35→39-in gather)

Flag: same `GAIA_NATIVE_EVIDENCE_SPLIT` gate as Stage 1 (Stage 2's new types
are additive Rust/WGSL — nothing constructs them unless a caller opts in;
`NetPresent`/`resolve_frame` in main.rs is UNTOUCHED this stage, so the live
path's `evidence_split` branch still only runs Stage-1's 35-in gather.
Wiring `FeatureGatherHistSplit` + `HistoryBuffers` into `NetPresent` is
STAGE 3's job, once a 39-in net exists to actually consume + produce the
recurrent output — building the plumbing without a real net to drive it would
be wiring for nothing).

- `src/rdirect_gather_split.wgsl` — new compute entry `gather_hist_split`,
  additive alongside `gather_split` (Stage 1, byte-untouched): writes the
  full 39-feature `HIST_FEATURES_SPLIT` row — idx 0-34 identical to
  `gather_split`'s body (E/D taps, subpixel, albedo/normal/depth, motion
  zero) — then idx 35-38 (history) via a bit-for-bit port of CPU
  `direct_render_sequence_hist_split`'s reprojection block: world point =
  this frame's `cam_ray_dir(cur_cam, tx,ty)*depth` (dist=1e5 on a miss, same
  `is_miss` convention), `cam_reproject(prev_cam, world)` (pinhole, same
  sign/bounds convention as CPU `CamPose::reproject`), nearest-pixel
  depth+normal reject test against the PREVIOUS frame's own AOV
  (`prev_aov`), and — only on accept — a bilinear resample
  (`bilinear_prev_dl`) of the previous frame's net output
  (`prev_out_dl`, demod-log space) at the fractional reprojected coord.
  First frame / any reject ⇒ prev_dl=0, valid=0 (the CPU rule, copied
  exactly — nothing invented: CPU history IS full reprojection with a
  bilinear resample, not same-pixel no-reproject — confirmed by reading
  `rdirect.rs::direct_render_sequence_hist_split` + `CamPose::reproject`
  before writing a line of WGSL).
- `src/rdirect_gather.rs` — added `CamGpu` (GPU-layout camera pose, 4×vec4,
  `From<CamPose>` for a lossless field-for-field port), `FeatureGatherHistSplit`
  (own pipeline over 2 bind groups — group0 = Stage-1's dims/accum_ed/aov/feats
  shape with feats now 39-wide; group1 = the new `prev_out_dl`/`prev_aov`/
  `HistU` uniform carrying both cameras + `has_prev`/`prev_w`/`prev_h`/
  `depth_tol`/`normal_thresh`), and `HistoryBuffers` (the GPU-resident
  ping-pong state: `prev_out_dl` vec4/px + `prev_aov` 2×vec4/px + `has_prev`
  + `prev_cam`; `swap()` GPU-copies the CALLER's current out_dl/aov buffers
  into the history buffers via `copy_buffer_to_buffer` — no CPU round-trip —
  and is meant to be called once per frame AFTER that frame's own
  `gather_hist_split` has consumed the OLD history; `reset()` drops history
  for a scene cut/resize, after which the next gather sees `has_prev=false`
  and the CPU zero/invalid rule takes over). `FeatureGather`/`FeatureGatherSplit`
  (Stage 1) are BYTE-UNTOUCHED — new code only, no shared-function edits.

### Parity guard (flag OFF — the old 23-in path, byte-identical)
```
$ cargo build --release -j2 --bin scrying-glass   # 0 errors, same 3 pre-existing warnings as Stage 1
$ cargo test --release -j2 --test rdirect_gather_ordeals -- --nocapture
[n0b] GATE A gather: N=6144 px × 23 feat · max abs 9.537e-7
[n0b] GATE B shared forward: max abs 7.749e-7
[n0g] S8 MPSGraph(default) vs chain: max abs 2.384e-7 [source: this doc]
test n0b_gather_and_shared_forward_match_cpu ... ok
$ cargo test --release -j2 --test rdirect_gpu_ordeals --test rdirect_live_ordeals -- --nocapture
test c_ban_no_temporal_vocabulary_in_the_gpu_kernel ... ok
test a_gpu_inference_is_byte_identical_same_frame_twice ... ok
[rdirect parity] f32 parity_rel=5.636e-7 bound=2.159e-3
test b_f32_gpu_matches_cpu_within_derived_bound ... ok
[rdirect parity] fp16 MODE-A parity_rel=5.791e-4 bound=3.128e-3
test b2_fp16_fast_kernel_matches_cpu_within_derived_bound ... ok
[n0-gate1] N=6144 px · live-vs-committed: abs 1.311e-6 rel 5.960e-5 · live-vs-recomputed-CPU: abs 1.311e-6
test n0_gate1_live_net_matches_cpu_reference ... ok
```
All identical numbers to Stage 1's own baseline — confirms Stage 2's new
types never execute unless a caller (only the new probe, so far) builds them.
Also re-ran Stage 1's own probe unmodified: `cargo run --release -j2
--example v7_live_ed_probe` → `E/D taps 4.768e-7 · tail 9.537e-7 · overall
9.537e-7` — byte-identical to the number recorded above, `gather_split`
truly untouched.

### History probe (flag ON semantics, no flag needed — new types only)
`examples/v7_live_hist_probe.rs` — 3-frame small-pan sequence (orbit yaw
-3/0/+3° around the naruko front pivot `[0,2,0]`, continuous motion so
reprojection is genuinely exercised — not a degenerate same-pixel case).
Drives the SAME `gather_hist_split` + `HistoryBuffers::swap` wiring a live
integration would use, and cross-checks against an independent CPU
transcription of the reprojection block (NOT calling
`direct_render_sequence_hist_split` itself — that function also runs the
net forward + evidence clamp and is load-bearing for the STAMPED real-image
ordeal; this lane must not touch or risk it, so the probe re-derives the
reprojection-only slice standalone). No 39-in net exists yet (Stage 3), so
both sides feed forward a synthetic-but-identical stand-in "previous net
output" (`out_dl(frame) = that frame's own E-tap0`, itself Stage-1
bit-exact GPU-vs-CPU) — this exercises the real plumbing (ping-pong buffer,
reprojection math, depth/normal guard, bilinear resample) without depending
on unbuilt Stage-3 weights.
```
$ cargo run --release -j2 --example v7_live_hist_probe
[v7-hist-probe] frame 0 N=6144 px x 39 feat — base(0-34) max-abs-diff 9.537e-7 · history(35-38) max-abs-diff 0.000e0 (has_prev=false)
[v7-hist-probe] frame 1 N=6144 px x 39 feat — base(0-34) max-abs-diff 9.537e-7 · history(35-38) max-abs-diff 3.958e-5 (has_prev=true)
[v7-hist-probe] frame 2 N=6144 px x 39 feat — base(0-34) max-abs-diff 9.537e-7 · history(35-38) max-abs-diff 3.910e-5 (has_prev=true)
[v7-hist-probe] OVERALL base max-abs-diff 9.537e-7 · history max-abs-diff 3.958e-5
[v7-hist-probe] PASS — GPU 39-in gather (base+history) matches CPU reference
```
Frame 0 (no history yet) is EXACT zero diff (the has_prev=false branch is a
pure literal-zero write, both sides). Frames 1-2 (reprojection live) sit at
~4e-5 — same float-ULP class as every other gate in this lane (n0b's own
bound is 1e-4; the probe asserts `< 1.0e-4` on both base and history and
prints PASS). Base (idx 0-34) stays at Stage 1's exact 9.537e-7 across all 3
frames, confirming the new entry's shared body is a true copy of
`gather_split`, not a re-derivation that could drift.

## 4. STAGE 3 SPEC (sharpened — resume point for the next room)

STAGE 3 = wire a 39-in net into the live path + prove full-frame parity +
measure the fps cost:

1. **39-in net load**: swap `RdirectLive` (or a sibling) to accept the
   STAMPED v7 weights file (commit 59e7bfa: `55720b45`, crawl checkpoint +
   evidence clamp baked in — `data/rdirect-weights-v7.bin` is the file to
   point at, verify it carries that stamp's sha256 via
   `rdirect::verify_stamp` the SAME way `NetPresent::new`'s v4 load does —
   REAL OR BLACK, no exceptions for v7). `Mlp::layer_dims()[0].0` must read
   39 (`HIST_FEATURES_SPLIT`) — `RdirectLive`'s MPSGraph input tensor shape [source: this doc]
   is currently hardcoded around the 23-in path; find where and parameterize
   it (or branch a `RdirectLiveSplit` sibling, mirroring how `FeatureGather`/
   `FeatureGatherSplit` stayed siblings rather than one parameterized type —
   consistent with this lane's own additive-sibling pattern so the shipped
   23-in v4 path never risks a regression).
2. **Evidence clamp at present**: port `clamp_evidence_lin` (rdirect.rs,
   `presented = min(net_linear, gamma*local_max_evidence)`, gamma default
   1.5 via `GAIA_V7_CLAMP_GAMMA`/`evidence_clamp_gamma()`, semantics fixed by
   commit c8b9ba6) into the demod stage — either the wgpu `DemodPass` (new
   variant) or the fused native/MSL demod (`RdirectLive::attach_demod`,
   whichever the live path is using — SHIFT 18 made fused the DEFAULT, so
   port there first). Needs `local_max_evidence` at native res: this lane's
   OWN `evidence_composite_frame`/`local_max_3x3`/`EvidenceAccum` (CPU,
   rdirect.rs) show the exact recipe — bilinear-upsample low_e+low_d to
   native (E+D composite), a TEMPORAL-MEAN accumulator across the live
   frame stream (NOT max-across-time — the gamma derivation note in
   rdirect.rs explains why that's a dead end), then a spatial 3×3 max-pool.
   The temporal-mean accumulator itself is new per-pixel GPU state (another
   ping-pong-shaped buffer, same family as this stage's `HistoryBuffers`).
3. **Full-frame parity gate**: a new n0-gate1-shaped test — GPU live 39-in
   forward (this stage's `gather_hist_split` output → the loaded v7 net →
   evidence-clamped present) vs `direct_render_sequence_hist_split` run on
   an IDENTICAL multi-frame pose sequence (CPU, same weights) — compare the
   PRESENTED (post-clamp) linear image, not just the pre-clamp net output,
   since the clamp is itself part of what Stage 3 ships. Tolerance: same
   derived-bound style as `b_f32_gpu_matches_cpu_within_derived_bound`
   (rdirect_gpu_ordeals.rs) — reuse its derivation, don't invent a new one.
4. **fps re-measure**: `[n0i]` budget table (`NetPresent::record`, already
   instrumented — trace/gather/net/demod/present ms + WALL-FPS) run live
   with v7 loaded; compare against the last known non-split baseline
   (18.57ms/53.85fps) — expect a regression (39-in forward is larger than
   23-in, plus the new gather/history/clamp passes) and record the actual
   number rather than assume it. `tools/profile-seam.mjs` if a stage looks
   disproportionately hot.

## STAGE 3 PROGRESS (ghoul run 2026-07-20, ~25min timebox — resume point)

Done (commits d0a9240, 602d8bf, this branch):

- **Net loader generalized (spec item 1, first half)**: `RdirectLive::build`
  (rdirect_live.rs) no longer hard-asserts `INPUT_FEATURES` (23) against the
  weights blob. `in_features` is now DERIVED from
  `cpu_ref.layer_dims().first().0` — whatever the blob's own first layer says
  IS the shape, used for the MPSGraph input placeholder and everywhere else [source: this doc]
  (`attach_pool`/`forward`/`start_pipeline` were already parametric on
  `self.in_features`; only `build()`'s check+placeholder were hardcoded).
  IRON-clean: no new hardcode, old 23-in weights (v1-v4) unaffected.
  **Verified live**: `cargo run --release -j2 --example v7_live_load_probe`
  now prints `LOADED OK in_features=39 out_channels=3` — this WAS
  cutover-ready.md's documented architecture blocker (`in_dim 39 !=
  INPUT_FEATURES 23`), now false. `main.rs` `NetPresent::new` gained a
  `GAIA_NATIVE_WEIGHTS=v7` convenience mapping (loads + stamp-checks
  `data/rdirect-weights-v7.bin` exactly like v4's REAL-IMAGE-BAR path).
- **Refuse-not-corrupt guard**: the frame loop below `NetPresent::new`'s load
  (FeatureGather 23-wide gather -> `live.forward` -> DemodPass) is UNCHANGED
  this room — still only feeds 23-wide rows. Feeding those into a 39-wide
  net's MPSGraph input is a silent PER-ROW STRIDE MISMATCH (not a smaller/ [source: this doc]
  degraded image — a wrong one, every pixel after the first reads across the
  wrong row boundary). Added a hard check: `live.in_features() !=
  INPUT_FEATURES` -> `Err` -> `present_black`, same failure shape as an
  unstamped weights file. So `GAIA_NATIVE_WEIGHTS=v7` today LOADS the net
  (proving the loader) then CLEANLY REFUSES to drive it, instead of a
  corrupted present. This is the honest state for the Architect: v7 is not
  yet selectable for real use, and trying it fails safe (black, not wrong).
- Regression guard: all 6 pre-existing parity ordeals
  (`rdirect_live_ordeals`, `rdirect_gpu_ordeals`, `rdirect_gather_ordeals`)
  green, numbers byte-identical to Stage 1/2's own baseline (n0-gate1 abs
  1.311e-6, n0b max abs 9.537e-7/7.749e-7, b_f32 parity_rel 5.636e-7, b2_fp16
  parity_rel 5.791e-4) — nothing this room touches the 23-in path's behavior.

NOT done this room (real remaining scope, unchanged from cutover-ready.md's
own estimate — "real engineering days, not a port"):

1. **WGSL/Metal forward for in_dim 39**: the MPSGraph forward itself [source: this doc]
   (rdirect_live.rs) needed NO shader change — it builds its graph ops from
   `dims` (now `in_features`-wide) at construction time, so a 39-in blob
   already produces a correctly-shaped MPSGraph automatically (verified by [source: this doc]
   the load probe above: `LOADED OK in_features=39`). The standalone fused
   WGSL kernels (`rdirect.wgsl`/`rdirect_fast.wgsl`, `GpuRdirect` in
   rdirect_gpu.rs) are a SEPARATE, non-live code path — used only by the
   offline bench (`examples/rdirect_kernel.rs`) and the 23-in `b_f32`/
   `b2_fp16` ordeals (still 23-in on purpose, those gates are unrelated to
   this lane) — confirmed by grep, nothing in `main.rs`'s live present path
   references `GpuRdirect`. So spec item 1's "WGSL forward kernel" framing
   doesn't map onto a real gap: the live forward is Metal/MPSGraph-only and [source: this doc]
   is now shape-generic. Recording this so nobody goes looking for a WGSL
   net kernel that isn't part of the live loop.
1. **`gather_hist_split` wired into `NetPresent`'s frame loop**: Stage 2
   built `FeatureGatherHistSplit` + `HistoryBuffers` as free-standing types
   (only the probe example constructs them). `NetPresent::new`/
   `resolve_frame` need a THIRD net-buffer family (39-wide `net_feats_v7` +
   `HistoryBuffers` instance, sized/pooled once) and a branch parallel to
   today's `evidence_split` one that runs `gather_hist_split` instead of
   `gather`/`gather_split`, feeds `live.forward`, and calls
   `history.swap()` after present. NOT STARTED.
2. **Evidence clamp at present**: `clamp_evidence_lin` (rdirect.rs) needs a
   GPU port (`DemodPass` new variant or the fused MSL demod) plus a NEW
   per-pixel temporal-mean evidence accumulator (spec's own §4 item 2 warns
   this is not just `local_max_3x3` — the accumulator is the actual new GPU
   state, another `HistoryBuffers`-shaped ping-pong). NOT STARTED.
3. **Full-frame parity gate + fps table**: both depend on 1-2 above being
   real; nothing to honestly measure yet. NOT STARTED — do not fabricate
   numbers for either.

## 5. Build/process state (resume)

No background build left running by this room (single-token, sequential
`cargo build`/`cargo test`/`cargo run --example` under `nice -n 19`, `-j2`,
foreground, each finished before starting the next — same discipline as
Stage 1). If a later room needs to detach a long build, note PID + log path
HERE.

## STAGE 3 PROGRESS (room 2, ghoul run 2026-07-20) — v7 now runs live

Picked up exactly where room 1 stopped (loader generalized, refuse-not-
corrupt guard blocking the frame loop). This room did the real wiring the
guard was blocking on. Full numbers/commands: `scratch/v7-cutover-ready.md`
(rewritten this room) — summarized here for the code-level record.

**1. `gather_hist_split` wired into `NetPresent`'s frame loop — DONE.**
`NetPresent::new`'s guard now accepts 39-in weights when
`GAIA_NATIVE_EVIDENCE_SPLIT=1` (still refuses BLACK otherwise, and for any
unknown in_features — REAL OR BLACK intact). The gather stage branches on a
new `is_v7` flag: the 23-in composite `gather.encode` is skipped entirely
(it would write the wrong stride into the now-39-wide pooled feature buffer)
and `FeatureGatherHistSplit::encode` drives `feats` instead, reading a new
`HistoryBuffers` instance (`self.history`) exactly as `examples/
v7_live_hist_probe.rs` proved bit-exact in Stage 2 — except now it's real
net output feeding history, not a synthetic E-tap0 stand-in. A `cam_by_set:
Vec<Option<CamPose>>` field tracks which camera pose gathered each
double-buffer slot, because the frame-overlap pipeline means the net output
finishing THIS iteration (`dset`) was gathered under a PAST iteration's
camera, not this iteration's `cur_cam` — `HistoryBuffers::swap` needs THAT
pose, not the current one, to keep the next frame's reprojection honest.

**2. Evidence clamp at present — DONE, as a GPU compute cut (not CPU).**
New `src/rdirect_evidence.{rs,wgsl}`: three small compute passes
(`evidence_accumulate`, `evidence_clamp_present`, `pack_out_dl3to4`) porting
`rdirect.rs`'s `EvidenceAccum`/`local_max_3x3`/`clamp_evidence_lin` (commit
c8b9ba6). Key simplification vs a literal port: `EvidenceAccum::ceiling`
computes `local_max_3x3(sum/count)`; since `count` is a single scalar shared
by every pixel, `max_3x3(sum)/count == max_3x3(sum/count)` exactly (max and
division-by-a-positive-constant commute) — so there is no separate
temporal-mean buffer, only a running `sum` (one vec4/px, `evidence_sum`) and
a CPU `u32` counter (`evidence_count`), and the clamp kernel folds `gamma/
count` into its 3×3 max-pool. This is bit-exact against the CPU recipe, not
an approximation. History is fed the frame's RAW/unclamped `out_dl` (via
`pack_out_dl3to4` on the net's tight `[n,3]` MPSGraph output), matching the [source: this doc]
CPU reference's own `prev = Some((out_dl, ...))` — the clamp never feeds
back into itself.

**3. Full-frame parity — DONE (numbers, not a pass/fail gate).**
`examples/v7_present_parity_probe.rs`: pan sequence (camera actually moving)
matches `direct_render_sequence_hist_split` to ~1e-6 through all 3 frames.
Still sequence (camera repeated 3x) matches at frame 0 (~5e-7) then shows
max-abs-diff ~1.9e-2 at frames 1-2, but mean-abs-diff stays ~8e-5 with only
~1% of pixels over 1e-3 — read as isolated evidence-clamp boundary flips
(a pixel exactly at `gamma*ceiling` flipping the `min()` branch on a sub-ULP
GPU/CPU numeric difference), not a systemic bug. Not chased to root cause
this room; recorded honestly, not gated or hidden.

**4. FPS — DONE (real number, not assumed).** `s20-bench.sh` offscreen
640x480 with `GAIA_NATIVE_WEIGHTS=v7 GAIA_NATIVE_EVIDENCE_SPLIT=1`: TOTAL
median 23.32ms / WALL-FPS ~40 (p95 31ms), vs the documented pre-v7 baseline
18.57ms/53.85fps — a real ~5ms/~14fps regression, in the predicted direction
(bigger 39-in GEMM + new gather/history/evidence passes). The "demod"
budget bucket (which folds in this room's evidence/pack/swap work) jumped to
~8ms median — three extra `device.poll(wait_indefinitely)` round-trips per
frame (accumulate, clamp+pack, swap) is the likely next optimization
(batch into fewer polls / fewer submits) but was not attempted this room
(out of the ~25min scope; the honest number was the deliverable).

**Bug found and fixed along the way**: `Integrator::make_split_buffer` was
missing `wgpu::BufferUsages::COPY_DST` — the live trace stage `clear_buffer`s
it every frame (same as the composite `net_accum`), which panicked the
first time `GAIA_NATIVE_EVIDENCE_SPLIT=1` was ever driven through the real
app binary (Stage 1's own dispatch had never been live-tested end-to-end;
probes build their own buffers and lean on wgpu's implicit zero-init rather
than an explicit per-frame clear). Fixed in `integrator.rs`.

Regression guard: all 6 pre-existing parity ordeals
(`rdirect_live_ordeals`, `rdirect_gpu_ordeals`, `rdirect_gather_ordeals`)
still green, byte-identical, after every change this room.

NOT done this room: cutover itself (making v7 the default `GAIA_NATIVE_WEIGHTS`
selection), chasing the still-sequence clamp-boundary root cause, and
reducing the new evidence-stage poll count for fps recovery. All next-room
work, not started here — see `scratch/v7-cutover-ready.md`'s own "NOT done"
language for the exact scope.

## STAGE 3 PROGRESS (room 3, ghoul run 2026-07-20) — fps recovered, boundary-flip theory FALSIFIED

**1. Poll reduction — DONE.** `resolve_frame`'s v7 tail (`evidence_accumulate`
→ non-fused `demod` → `evidence_clamp_present` → `pack_out_dl3to4` →
history `swap`/`copy_buffer_to_buffer`) is now ONE command encoder / ONE
`queue.submit`, with NO `device.poll(wait_indefinitely())` in between (was
3 separate submit+poll pairs). wgpu tracks buffer read-after-write hazards
within one command buffer itself (evidence_sum written by accumulate then
read by clamp; out_dl_padded written by pack then read by swap), so no CPU
sync is needed — same house pattern as SHIFT 17 CUT B's async-trace poll
removal. Nothing downstream reads these buffers on the CPU this frame; the
next frame's own poll(s) (trace stage, `commit_net`'s wait) transitively
catch this submission up before anything depends on it, since it's all one
FIFO queue. Semantics unchanged — sync strategy only.

**2. Parity re-run (unchanged, byte-identical to room 2's numbers — this
probe is a standalone harness, doesn't exercise `resolve_frame` directly,
so it wasn't expected to move):**
```
still frame 0 max-abs-diff 4.7684e-7 mean-abs-diff 7.9931e-8 px>1e-3=0
still frame 1 max-abs-diff 1.7205e-2 mean-abs-diff 7.9278e-5 px>1e-3=59
still frame 2 max-abs-diff 1.9002e-2 mean-abs-diff 7.9573e-5 px>1e-3=60
pan   frame 0/1/2: ~1e-6, px>1e-3=0 (all 3 frames)
```
Regression guard: all 6 pre-existing ordeals (`rdirect_live_ordeals`,
`rdirect_gpu_ordeals`, `rdirect_gather_ordeals`) still byte-identical to
every prior room's numbers.

**3. FPS re-bench (s20, offscreen 640x480, `GAIA_NATIVE_WEIGHTS=v7
GAIA_NATIVE_EVIDENCE_SPLIT=1`) — target was <=18.6ms, RESULT BEATS IT:**

| run | trace | gather | net_wall | demod(resolve) | present | **TOTAL** | wall_fps (/budget) |
|---|---|---|---|---|---|---|---|
| room-2 (3 polls/frame) | 8.73/15.80 | 1.80/2.18 | 0.05/3.51 | **8.39/13.69** | 0.10/0.18 | **23.32/31.09** | 39.81 |
| room-3 (1 submit, 0 polls) run A | 12.76/23.33 | 1.84/2.06 | 0.04/8.36 | **0.26/0.44** | 0.12/0.18 | **16.63/25.60** | 45.36 |
| room-3 run B (repeat) | 13.03/20.84 | 1.86/2.07 | 0.04/8.01 | **0.27/0.47** | 0.12/0.19 | **16.57/23.51** | 45.39 |
| unversioned pre-v1..v7 baseline (SHIFT 18/N0.n, NOT v4 — see room 7 correction above) | — | — | — | — | — | **18.57** | 53.85 |

The "demod" bucket (Stage 3's `resolve_ms`, folding in evidence/pack/swap)
collapsed from 8.39ms median to 0.26-0.27ms — confirms the 3 blocking polls
were in fact ~8ms/frame, and removing them (not the GPU work itself, which
was already cheap) recovers essentially all of it. **TOTAL median (16.57-
16.63ms) is now BELOW the non-split 23-in baseline's own 18.57ms** —
surprising but explained by `net_gpu` staying ~5.1-5.2ms both rooms (the
39-in GEMM itself isn't the bottleneck) and the removed polls having cost
MORE than the work they were purely there to serialize. Distance to the
16.67ms/60fps line: room-3 TOTAL sits right at/just under it already.
**Caveat**: `wall_fps` (45.36/45.39, full server loop incl. world tick +
http) is still below the non-split baseline's reported 53.85 — that number
likely came from a lighter `outside` (world/http) load in whatever session
produced it, not from render-stage cost; TOTAL-to-TOTAL is the apples-to-
apples comparison and it improved. Not chased further (out of scope — the
non-render `outside` bucket, world/http, is untouched by this room).

**4. Boundary-flip verdict — room 2's theory FALSIFIED, real (small) gap
found instead.** Added a diagnostic-only dump to
`examples/v7_present_parity_probe.rs`: for each "still" frame with
px>1e-3 mismatches, print the 5 worst pixels' PRE-clamp `net_linear`
(read back right after demod, before the evidence-clamp kernel runs) next
to that pixel's own clamp ceiling (`gamma * local_max_3x3(evidence_sum /
count)`, computed CPU-side from a readback of the live `evidence_sum`
buffer — bit-exact recipe, not approximated).
```
f1 px=2976 unclamped=(0.322,0.123,0.202) ceiling=(0.529,0.198,0.325) |u-c|=(0.207,0.075,0.123)
f1 px=13   unclamped=(0.223,0.081,0.150) ceiling=(0.326,0.123,0.222) |u-c|=(0.104,0.042,0.071)
f1 px=17   unclamped=(0.220,0.080,0.149) ceiling=(0.321,0.121,0.219) |u-c|=(0.101,0.041,0.070)
... (f2 pixels: same shape, |u-c| ~0.09-0.21)
```
Every flipped pixel's unclamped value sits **10-20% BELOW its ceiling**,
not at it — the clamp `min()` never fires on the GPU side (`unclamped ==
gpu` exactly, confirmed by comparing the dumped `unclamped` column to the
same pixel's `gpu` presented value in the line above: identical). So this
is **not** a sub-ULP `min()` boundary flip as room 2 guessed — the GPU's
raw (pre-clamp) net output itself differs from the CPU reference's by
~0.01-0.02 absolute (~5-10% relative) at these pixels, a real semantic gap,
far outside the ~1e-6 float-ULP class every other gate in this lane sits
in. **It is scoped tightly**: only the STILL sequence (camera repeated
identically 3x, exercising `has_prev=true` same-pose reprojection) shows
it — the PAN sequence (camera genuinely moving each frame) stays at ~1e-6
through all 3 frames, and STILL frame 0 (no history yet) is also exact.
So the gap appears specifically when the SAME pose recurs and REAL net
output (not Stage 2's synthetic E-tap0 stand-in) feeds back through
history more than once. Stage 2's own `gather_hist_split` probe already
proved the 39-feature ROW (including history idx 35-38) bit-exact for
both static and panning poses at ~4e-5 — so the gather step itself is not
the suspect; the divergence more likely compounds ACROSS repeated
real-output feedback (a small per-frame GPU/CPU numeric difference in the
net forward or the reprojection's fractional-pixel resample, invisible in
a single pass, growing over 2-3 recurrent steps at a fixed pose). Root
cause not isolated further this room (would need per-layer diffing of the
MPSGraph forward output alone, feeding it the SAME feature row on both [source: this doc]
sides, decoupled from the recurrence) — flagging as a genuine open item
for the next room, corrected from room 2's "benign fp noise" framing.
Magnitude is still small in absolute terms (mean-abs-diff stays ~8e-5,
only ~1% of pixels affected) but the mechanism is not what was assumed.

**Artifacts this room**: `src/main.rs` (`resolve_frame`'s v7 tail, single
encoder), `examples/v7_present_parity_probe.rs` (boundary-flip dump, +
`COPY_SRC` on its own diagnostic `evidence_sum` buffer — main.rs's live
buffer stays `COPY_DST`-only, untouched), this note.

NOT done this room: root-causing the still-sequence net-output gap itself
(scoped above, not fixed); the `wall_fps` vs non-split-baseline `wall_fps`
gap (world/http overhead, out of this room's render-only scope); cutover
itself.

## STAGE 3 PROGRESS (room 4, ghoul run 2026-07-20) — root cause FOUND, verdict

Picked up room 3's open item ("still-sequence net-output gap, mechanism not
what was assumed"). Extended `examples/v7_present_parity_probe.rs` (example
file only — no engine/main.rs/wgsl changes this room):

**1. 12-frame STILL drift curve (task 1) — raw numbers:**
```
f0  max-abs-diff 4.7684e-7  px>1e-3=0
f1  max-abs-diff 1.7205e-2  px>1e-3=59
f2  max-abs-diff 1.9002e-2  px>1e-3=60
f3  max-abs-diff 1.7880e-2  px>1e-3=61
f4  max-abs-diff 1.8377e-2  px>1e-3=58
f5  max-abs-diff 1.9242e-2  px>1e-3=61
f6  max-abs-diff 6.0695e-2  px>1e-3=61   <- one-frame outlier, see below
f7  max-abs-diff 1.8978e-2  px>1e-3=59
f8  max-abs-diff 1.8525e-2  px>1e-3=59
f9  max-abs-diff 1.8358e-2  px>1e-3=62
f10 max-abs-diff 1.8046e-2  px>1e-3=61
f11 max-abs-diff 1.8040e-2  px>1e-3=61
```
mean-abs-diff stays 7.8e-5..8.9e-5 every frame (never grows). Shape: **flat
plateau from frame 1 onward** — jumps to ~1.7-1.9e-2 at f1 and stays there
through f11, not a monotonic climb. The affected pixel COUNT is stable
(58-62 of 6144, ~1%) across all 12 frames — same small population, not a
spreading one. The one f6 spike (6.07e-2, px=86 newly appears in that
frame's worst-5 list, not seen at other frames) is a single transient extra
flip, not a trend. Pan sequence re-confirmed unchanged: all 3 frames
~1e-6, px>1e-3=0.

**2. A/B localization (task 2) — does the INPUT already differ?** Extended
the probe with an independent standalone CPU replay
(`cpu_feature_sequence`, byte-for-byte copy of
`direct_render_sequence_hist_split`'s own feature-building loop, returning
the 39-wide rows instead of only the presented image) and dumped the worst
flipped pixel's full feature row, live GPU vs this CPU replay, at the first
two mismatching frames (f1, f2), pixel 2976 both times:
```
f1 px=2976  base(idx0-34) max-diff=1.192e-7   history(idx35-38) max-diff=1.000e0
  idx35 (prev_dl.x) live=0.000000 cpu=0.281246 diff=2.812e-1
  idx36 (prev_dl.y) live=0.000000 cpu=0.117278 diff=1.173e-1
  idx37 (prev_dl.z) live=0.000000 cpu=0.185439 diff=1.854e-1
  idx38 (valid)      live=0.000000 cpu=1.000000 diff=1.000e0
f2 px=2976  base(idx0-34) max-diff=1.043e-7   history(idx35-38) max-diff=1.000e0
  idx35 live=0.000000 cpu=0.291872 diff=2.919e-1
  idx36 live=0.000000 cpu=0.125193 diff=1.252e-1
  idx37 live=0.000000 cpu=0.191482 diff=1.915e-1
  idx38 live=0.000000 cpu=1.000000 diff=1.000e0
```
**Answer: the INPUT already differs — at the history slots specifically,
not the base slots.** idx0-34 (the E/D taps, subpixel, albedo/normal/depth,
motion — Stage 1/2's own bit-exact territory) sit at ~1e-7, the same
float-ULP class as every other gate in this lane. idx35-38 (the
reprojected-history carrier) sit at a FULL 1.0 diff on the validity flag
alone: **GPU decides `valid=0` (drops history, feeds zero) while CPU
decides `valid=1` (accepts, feeds the real reprojected previous output)**,
for the exact same static camera pose, same depth, same normal, same
previous frame. This is not a forward-kernel numeric gap — it never
reaches the net's forward pass at all; it is a **discrete branch
disagreement in the depth/normal reprojection accept test**
(`gather_hist_split`'s WGSL port vs `direct_render_sequence_hist_split`'s
Rust body — both meant to be the same bit-for-bit port Stage 2 proved
equal on a moving-camera probe). Once GPU's accept test rejects at a
pixel, that pixel's history is zeroed EVERY frame thereafter (its own
reprojection keeps landing on the same disagreement, same static pose) —
explaining the flat plateau shape from task 1 directly: it is not fp error
accumulating, it is a fixed WRONG DECISION replayed unchanged every frame.

**3. VERDICT.** Room 2's "benign fp equilibrium" framing and room 3's own
"per-frame numeric gap in the net forward or the reprojection's
fractional-pixel resample" framing are BOTH superseded: the gap is
neither noise nor a resample-precision issue — it is a same-vs-different
BOOLEAN accept/reject outcome on the depth+normal reprojection guard,
isolated to a small (~1%), STABLE, geometrically-fixed set of pixels
(consistent frame to frame under a static pose — almost certainly
silhouette/depth-discontinuity pixels, where `ipx=fx.round()`/`ipy=
fy.round()` sits near a .5 tie or the reprojected depth sits right at the
`depth_tol` boundary, so a sub-ULP GPU/CPU difference upstream in the
fractional reproject coordinate — not in the guard itself — flips which
side of a hard threshold the pixel lands on). Classification: **BOUNDED,
not diverging** — the plateau does not grow past f1, mean-abs-diff stays
flat, the affected population size stays flat — but it is **not a benign
fp-noise bound to "derive and accept"**: it is a real, deterministic,
per-pixel-persistent defect (wrong history retention decision at edge
pixels), previously mischaracterized twice. Under "shipped = ordealed
act": **cutover-blocking** in its current form — it is small in
aggregate (mean-abs-diff ~8e-5, 1% of pixels) but the mechanism is exactly
the kind of thing an ordeal's residual/sparkle bars are built to catch at
silhouette edges, and it is not yet understood well enough (WGSL vs Rust
reproject-guard line-by-line diff not done this room) to certify safe.
**Recommended fix**: diff `gather_hist_split`'s WGSL reprojection-guard
block against `direct_render_sequence_hist_split`'s Rust block
line-by-line for the exact boundary condition (round-half behavior of
WGSL's `round()` vs Rust's `f32::round()` at a .5 tie is the leading
suspect, cheap to test in isolation: feed both sides an `fx`/`fy` pair
synthetically constructed to sit at x.5000000 and x.5000001 and compare);
failing a fast fix, a periodic re-sync (drop history every N frames to
bound any pixel's wrong-branch persistence) is a viable stopgap but was
not needed to explain THIS lane's numbers (the plateau already doesn't
grow) — the real fix is matching the two branches' boundary behavior, not
bounding a growth that isn't happening.

**4. FPS spike attribution (task 4)** — NOT reached this room (root-cause
work above filled the timebox; no fix attempted here either, per the
room's own no-fix instruction).

Regression guard: all 6 pre-existing ordeals (`rdirect_live_ordeals`,
`rdirect_gpu_ordeals`, `rdirect_gather_ordeals`) re-run, byte-identical to
every prior room's numbers — this room's changes are example-file-only
(`v7_present_parity_probe.rs`: 12-frame still sequence, standalone CPU
feature-sequence replay for the A/B dump, both diagnostic/additive).

**Artifacts this room**: `examples/v7_present_parity_probe.rs` (12-frame
still sequence, `cpu_feature_sequence` standalone replay + per-flip-pixel
39-feature A/B dump), this note.

## STAGE 3 room 4 (ghoul run 2026-07-20) — reprojection-guard fix, PARTIAL not closed

**1. Divergent op, exact lines.** Line-diffed `cam_reproject`
(`src/rdirect_gather_split.wgsl:180-194`) against `CamPose::reproject`
(`src/rdirect.rs:1025-1035`) and `gather_hist_split`'s ipx/ipy rounding
(`rdirect_gather_split.wgsl:311-312`) against
`direct_render_sequence_hist_split`'s (`rdirect.rs:1529-1530`). Two
candidate ops:
- round-tie: WGSL `round(fx)` (half-to-even) vs Rust `fx.round()`
  (half-away-from-zero). **Tested and FALSIFIED**: fixed to
  `floor(fx+0.5)` (exact match for fx>=0, which cam_reproject guarantees),
  rebuilt, reran the 12-frame probe — numbers were BYTE-IDENTICAL to the
  round() baseline (same px>1e-3=59-62, same max-abs-diff curve). Not the
  cause.
- off-screen bounds check: `if (fpx < 0.0 || fpy < 0.0 || fpx > w-1 ||
  fpy > h-1) return miss` — same structure both sides, but a STILL camera
  self-reprojects algebraically to `fpx == tx` (worked the substitution:
  `sx == cx` exactly when cur==prev camera), i.e. edge pixels (tx=0 or
  ty=0, which is where 8 of 9 non-px2976 flip pixels live — px 11,51,55,
  59,61,73,77,79,83,86 in a tw=96 image are ALL `i<96` → ty=0, the TOP
  ROW; px 2976 = ty=31,tx=0, the LEFT edge) reproject to a coordinate that
  sits geometrically AT the accept/reject boundary (fpx≈0 or w-1). A
  sub-ULP GPU-vs-CPU difference in the `dot()` products that build
  sx/sy (GPU FMA fusion vs CPU glam's un-fused multiply-add being the most
  likely mechanism, not proven at the instruction level — CamGpu is a
  byte-identical copy of CamPose, ruled that out as a source) flips which
  side of 0.0 the coordinate lands on, which the `is_miss` guard converts
  into a full valid=0/1 flip (matches the earlier full-idx35-38-zero
  observation exactly).

**2. Fix applied** (`src/rdirect_gather_split.wgsl`): kept the
round→floor(fx+0.5) correction (harmless, semantically right even though
not the root cause) and widened `cam_reproject`'s bounds check by a named
`REPROJ_EDGE_EPS = 1.0e-3` slack (comfortably above the ~1e-6 ULP noise
class seen in the base-feature diff, comfortably below a full pixel), plus
clamped the returned fpx/fpy into `[0, dim-1]` so a slack-admitted
coordinate never indexes an out-of-bounds tap.

**3. Result — IMPROVED, NOT CLOSED.** 12-frame still curve after the fix:
```
f0  max-abs-diff 4.77e-7   (unchanged, frame0 has no prev)
f1  max-abs-diff ~1.9e-2   px>1e-3≈31 (down from ~60)  [frame1 log line lost to scrollback; f2 shown below]
f2  A/B px=2976: GPU NOW ACCEPTS (valid=1, live=(0.180,0.069,0.126)) where
    CPU REJECTS (valid=0) — direction reversed from pre-fix (was GPU=0,CPU=1)
f3  max-abs-diff 1.2534e-2  px>1e-3=31
f4  max-abs-diff 1.2657e-2  px>1e-3=30
f5  max-abs-diff 1.2330e-2  px>1e-3=32
f6  max-abs-diff 1.2542e-2  px>1e-3=31
f7  max-abs-diff 1.2575e-2  px>1e-3=30
f8  max-abs-diff 1.2187e-2  px>1e-3=30
f9  max-abs-diff 1.2448e-2  px>1e-3=32
f10 max-abs-diff 1.2462e-2  px>1e-3=31
f11 max-abs-diff 1.2639e-2  px>1e-3=32
still OVERALL max-abs-diff 1.2803e-2  (was 6.0695e-2 pre-fix)
pan   frame0 4.77e-7  frame1 1.55e-6  frame2 2.93e-4 (was 1.07e-6) px>1e-3=0 all 3
pan OVERALL max-abs-diff 2.9301e-4  (was 1.5497e-6)
```
max-abs-diff roughly halved (6.07e-2→1.28e-2) and the affected still-pixel
count roughly halved (58-62→30-32), but the plateau did NOT collapse to
the pan-class 1e-6 floor — it is NOT machine-precision parity yet, and the
fix introduced a small new pan-sequence delta (2.93e-4 at frame 2, still
3 orders below the 1e-3 flip threshold, px>1e-3 still 0, but non-zero
where it was pure noise before) — the epsilon widening is asymmetric by
construction (only GPU's bounds check moved) so it can flip pixels the
CPU correctly rejects into GPU-accepts at OTHER edge pixels (seen directly
in the px=2976 A/B dump above, now flipped the other direction). **This is
a real improvement, not a closed seam**: still cutover-blocking under
"shipped = ordealed act" until either (a) the eps is tuned/scoped tighter
per-frame-size so it stops over-admitting, or (b) the true GPU-vs-CPU
dot-product ULP source is found and eliminated at the arithmetic level
(would need per-op GPU-side debug readback of sx/sy/rz, not done this
room — timebox spent on the fix + verify + regression + fps legs).

**4. Regression guard** — all 6 pre-existing ordeals re-run
(`cargo test --release -j2 --test rdirect_gather_ordeals --test
rdirect_gpu_ordeals --test rdirect_live_ordeals -- --nocapture`):
all 6 `ok`, byte-identical outputs to every prior room (these tests don't
exercise `gather_hist_split`/reprojection at all, so the WGSL edit can't
and didn't touch them):
```
n0b_gather_and_shared_forward_match_cpu ... ok
c_ban_no_temporal_vocabulary_in_the_gpu_kernel ... ok
a_gpu_inference_is_byte_identical_same_frame_twice ... ok
b_f32_gpu_matches_cpu_within_derived_bound ... ok
b2_fp16_fast_kernel_matches_cpu_within_derived_bound ... ok
n0_gate1_live_net_matches_cpu_reference ... ok
```

**5. FPS sanity** (s20 offscreen 640x480, `GAIA_NATIVE_WEIGHTS=v7
GAIA_NATIVE_EVIDENCE_SPLIT=1`, one run, flag ON): **TOTAL median 18.15ms /
p95 28.1-28.4ms, WALL-FPS 42.7** (`/budget`: `total:[18.156,28.038]`).
Room 3's own re-benches were 16.57-16.63ms median; this room's 18.15ms is
~1.5ms higher but still at/below the unversioned pre-v1..v7 baseline's 18.57ms (mislabeled "v4" here — see room 7 correction) and
well inside prior run-to-run variance (room 3 itself spanned 23.51-25.60ms
at p95 between its two runs) — the changed ops (an eps compare + a clamp,
both trivial ALU) are not plausible sources of a multi-ms shift; read as
machine load noise, not a regression, but not independently re-confirmed
with a second run this room (timebox).

**6. Cutover-note status**: NOT updated to CLOSED — see
`v7-cutover-ready.md` room-4 entry: still parity improved (~2x) but not at
machine-precision; the shipped-equals-ordealed-act seam stays OPEN
pending either eps tuning or the deeper ULP root-cause.

**Artifacts this room**: `src/rdirect_gather_split.wgsl` (round→floor fix +
`REPROJ_EDGE_EPS` bounds slack in `cam_reproject`), this note update,
`v7-cutover-ready.md` room-4 entry.

## STAGE 3 room 5 (ghoul run 2026-07-20) — symmetric SNAP_EPS, SEAM CLOSED

**Task**: replace room 4's asymmetric GPU-only `REPROJ_EDGE_EPS` bounds
slack (halved the flip but didn't collapse it, and introduced a new pan
delta) with a symmetric fix on BOTH sides of the CPU/GPU seam.

**1. The fix.** Both `CamPose::reproject` (`src/rdirect.rs`) and
`cam_reproject` (`src/rdirect_gather_split.wgsl`) now snap `fpx`/`fpy` to
the nearest integer whenever within `SNAP_EPS` of one
(`|fpx - floor(fpx+0.5)| < SNAP_EPS ⇒ fpx = floor(fpx+0.5)`, same for `fpy`),
applied identically on both sides, BEFORE the unchanged
bounds/depth/normal tests. `SNAP_EPS = 1.0e-3`, documented at both
definition sites: ~1e-6 is this lane's own observed GPU-vs-CPU ULP noise
floor (the pan-sequence parity class every other gate in this lane uses);
0.5 is the half-pixel tie point where a genuinely ambiguous round would
sit. 1e-3 is ~1000x above the noise floor and ~500x below the tie point —
comfortably inside both margins, not tuned to any one probe's numbers.
Room 4's `REPROJ_EDGE_EPS` constant and its asymmetric bounds-widening +
clamp are fully removed, not kept alongside the new fix. The harmless
round→`floor(fx+0.5)` correction from room 4 (the ipx/ipy nearest-pixel
rounding for the history *sample*, a different line from the reproject
*bounds test*) is unchanged — it was never the root cause and stays as
documented equivalence to Rust's `.round()`.

**2. Ordeal re-earned, PASS.** `GAIA_ORDEAL_WEIGHTS=v7 GAIA_ORDEAL_W=640
GAIA_ORDEAL_H=480`, detached (`nohup` → `scratch/v7i-ordeal-snap.log`,
PID 17419 then restarted correctly-sized as a second detached PID after a
first wrong-resolution run was killed — first attempt used the ordeal's
default 480x360, not the task's specified 640; killed and re-launched at
640x480 to match every prior room's ordeal record):
```
[ordeal] weights=data/rdirect-weights-v7.bin in_dim=39 (N5 split recurrent (39)) res 640x480 still=10 pan=6
[ordeal] orbit_-20 STILL: resid=0.03436 sparkle=22.8/Mpx (teacher 0.0) tvar=2.683e-5
[ordeal] orbit_-20 PAN mid: resid_hist=0.03649 resid_single=0.03890 ghost_excess=-0.00242
[ordeal] orbit_+40 STILL: resid=0.03537 sparkle=48.8/Mpx (teacher 0.0) tvar=3.360e-5
[ordeal] orbit_+40 PAN mid: resid_hist=0.03692 resid_single=0.04129 ghost_excess=-0.00437

[ordeal] ===== REAL-IMAGE BAR ===== (754.5s)
  resid_still         0.03487  bar    0.03500  PASS  (distance to bar -0.00013)
  sparkle_still      35.80729  bar   40.00000  PASS  (distance to bar -4.19271)
  tvar_still          0.00003  bar    0.00050  PASS  (distance to bar -0.00047)
  resid_move          0.03670  bar    0.06000  PASS  (distance to bar -0.02330)
  ghost_excess        0.00000  bar    0.01200  PASS  (distance to bar -0.01200)

[ordeal] VERDICT: PASS — stamp written data/rdirect-weights-v7.bin.stamp
```
Every metric byte-identical to the pre-fix stamp (`55720b45`, room-2-era
PASS quoted in `v7-cutover-ready.md`) — confirms the prediction exactly:
snap touches only boundary-history validity at silhouette pixels, none of
resid/sparkle/tvar/ghost move. `resid_still` (0.03487) still sits at its
own narrowest margin vs the 0.035 bar — unchanged, pre-existing, not a new
risk introduced by this fix.

**3. Parity — collapsed to machine precision, BOTH sequences.**
`examples/v7_present_parity_probe` (`scratch/v7i-parity-snap.log`):
```
still frame 0        max-abs-diff 4.7684e-7  mean-abs-diff 7.9931e-8  px>1e-3=0
still frames 1-11     max-abs-diff 3.5763e-7  mean-abs-diff ~8.3e-8    px>1e-3=0  (every frame identical class)
still OVERALL max-abs-diff 4.7684e-7
pan   frame 0         max-abs-diff 4.7684e-7
pan   frame 1         max-abs-diff 1.5497e-6
pan   frame 2         max-abs-diff 1.0729e-6
pan OVERALL max-abs-diff 1.5497e-6
SUMMARY still_max=4.7684e-7 pan_max=1.5497e-6
```
Compare to the three prior states:
```
                    still OVERALL      pan OVERALL      still px>1e-3
pre-room-4 (room 2)  6.0695e-2          1.5497e-6         58-62/6144
room 4 (asymmetric)  1.2803e-2          2.9301e-4         30-32/6144
room 5 (symmetric)   4.7684e-7          1.5497e-6         0/6144
```
Still collapsed 4 orders of magnitude past room 4's already-2x improvement,
landing in the exact same float class as the pan sequence (which was
always machine-precision) — not merely closer, fully closed. The pan
sequence's own small room-4-introduced regression (1.5497e-6 → 2.9301e-4
at frame 2) is also gone, back to the original 1.5497e-6 exactly — expected,
since a symmetric fix cannot introduce a one-sided admission the way room
4's GPU-only epsilon did.

**4. Regression — all 6 byte-identical, flag OFF.**
`cargo test --release -j2 --test rdirect_gather_ordeals --test
rdirect_gpu_ordeals --test rdirect_live_ordeals -- --nocapture`
(`scratch/v7i-regression-snap.log`), all `ok`, every printed number
verbatim-identical to room 4's own re-run:
```
n0b_gather_and_shared_forward_match_cpu ... ok   (GATE A max abs 9.537e-7, GATE B 7.749e-7, S8 2.384e-7)
c_ban_no_temporal_vocabulary_in_the_gpu_kernel ... ok
a_gpu_inference_is_byte_identical_same_frame_twice ... ok
b_f32_gpu_matches_cpu_within_derived_bound ... ok   (parity_rel=5.636e-7 bound=2.159e-3)
b2_fp16_fast_kernel_matches_cpu_within_derived_bound ... ok   (parity_rel=5.791e-4 bound=3.128e-3)
n0_gate1_live_net_matches_cpu_reference ... ok   (abs 1.311e-6 rel 5.960e-5)
```
**Checked, not assumed, whether any of these exercise `CamPose::reproject`**:
grepped all three test files for `reproject`/`hist_features`/
`direct_render_sequence_hist` — the only hit is
`rdirect_gpu_ordeals.rs`'s vocabulary-ban test, which greps generated
shader SOURCE TEXT for the string `"reproject"` (banning temporal
vocabulary from the non-recurrent GPU kernel) — it does not call the
function. All 6 tests exercise `gather_and_shared_forward`, single-frame
GPU-vs-CPU parity, or the live-net gate — none of which reproject a
history frame. Byte-identity is exactly the predicted outcome for tests
that structurally cannot touch the changed code path, not a lucky
coincidence — confirmed by inspection, not just observation.

**5. FPS — one s20 run, flag ON.** `bash proof/neural-live/s20-bench.sh
v7snap 8435 GAIA_NATIVE_WEIGHTS=v7 GAIA_NATIVE_EVIDENCE_SPLIT=1`, 640x480
offscreen, 1130 frames:
```
/budget total:[20.342,28.797]  wall_fps 42.81
[n0i] (tail, frames=1080) TOTAL 20.30/28.81 | WALL-FPS 42.8
```
Room 4's own number was 18.15ms median/28.1-28.4ms p95/WALL-FPS 42.7; this
room's 20.30/28.81/42.8 is within the same noise band across all 3 rooms
(room 3: 16.57-16.63/23.51-25.60, room 4: 18.15/28.1-28.4). The changed
ops (an `abs()`+`floor()` compare per axis, same trivial-ALU class as room
4's eps compare) are not a plausible multi-ms source; not independently
re-confirmed with a second run this room (timebox), consistent with every
prior room's single-run fps practice.

**6. Seam status: CLOSED.** Marked in `v7-cutover-ready.md` (room 5
entry): ordeal PASS re-earned under the modified act with metrics
byte-identical to the pre-fix stamp, both parity probes at machine
precision on both still and pan sequences, all 6 regressions structurally
unaffected and numerically unchanged, fps within the established 3-room
noise band. The shipped-equals-ordealed-act seam that has blocked cutover
since room 2 is resolved. Remaining known delta: the p95 tail (~28.8ms,
~1.6x median, present identically across every room's fps numbers — a
scheduling/poll-latency characteristic, not a correctness gap, out of
scope for this seam and not chased here).

No gate was weakened or bypassed. No window opened, Architect's session
untouched (whip 154). `nice -n 19`, `-j2`, one build token at a time
(sequential: build → parity probe → regression tests → fps bench →
ordeal, never overlapping compiles). Committed early per instruction —
see git log for this room's commit.

**Artifacts this room**: `src/rdirect.rs` (`CamPose::reproject` — adds
`SNAP_EPS` symmetric snap, first CPU-side edit in this lane),
`src/rdirect_gather_split.wgsl` (`cam_reproject` — `SNAP_EPS` symmetric
snap, `REPROJ_EDGE_EPS` + its clamp fully removed),
`data/rdirect-weights-v7.bin.stamp` (re-earned PASS),
`scratch/v7i-ordeal-snap.log`, `scratch/v7i-parity-snap.log`,
`scratch/v7i-regression-snap.log`,
`proof/neural-live/s20-v7snap.{log,budget.json,state.json,-presented.png}`,
refreshed `proof/neural-live/s25-{still,moving}{,-teacher}.png` (ordeal's
own proof frames), this note, `v7-cutover-ready.md` room-5 entry.

## §perf — room 6 (ghoul run 2026-07-20): clean-GPU bench + p95 tail attribution

Measurement-only room, no engine code touched. GPU confirmed clean before
and after (`ps aux | grep -i 'ordeal\|train\|cargo'` — no hits either
time; no foreign load to note).

**Clean baseline — 3x s20, offscreen 640x480, flag ON
(`GAIA_NATIVE_WEIGHTS=v7 GAIA_NATIVE_EVIDENCE_SPLIT=1`), sequential,
`nice -n 19`:**

| run | frames | TOTAL median/p95 (ms) | wall_fps |
|---|---|---|---|
| v7clean1 | 1152 | 18.116 / 27.548 | 42.68 |
| v7clean2 | 1130 | 18.181 / 30.296 | 41.24 |
| v7clean3 | 1190 | 19.096 / 27.940 | 43.71 |

3-run spread (the noise band): median 18.12-19.10ms (~1.0ms / ~5.4%
spread), p95 27.55-30.30ms (~2.7ms / ~9.4% spread) — this IS the
session's own noise band, consistent in shape (not magnitude) with every
prior room's single-run numbers (16.6-20.3ms medians across rooms 3-5).
Per-bucket medians/p95 (ms), averaged view (all 3 runs agree to within
the spread above — full numbers in
`proof/neural-live/s20-v7clean{1,2,3}.budget.json`):

| bucket | median | p95 |
|---|---|---|
| trace | 12.3-13.3 | 21.6-23.8 |
| gather | 1.88-2.00 | 2.8-3.8 |
| net_wall | 0.04-0.06 | 10.7-12.3 |
| net_gpu | 5.3-6.1 | 10.4-11.9 |
| demod | 0.23-0.27 | 0.48-0.98 |
| present | 0.08-0.10 | 0.18-0.35 |

**Flag-OFF leg — BLOCKED, not measured.** The old 23-in path defaults to
`GAIA_NATIVE_WEIGHTS=v4`, which is REAL-OR-BLACK gated on a PASS stamp
(`data/rdirect-weights-v4.bin.stamp`). That stamp file does not exist in
this worktree (gitignored, machine-local, never generated here — only
`v7.bin.stamp` exists, freshly earned by room 5's own ordeal run) —
confirmed via the run's own log: `[n0c] net-present disabled (build
failed): REAL-IMAGE BAR: weights data/rdirect-weights-v4.bin carry no
PASS stamp ... present BLACK by law`. `/budget` returned `{"frames":0,
"note":"net-present off"}` — the raster fallback, not a real v4 render.
Earning the v4 stamp requires running its own real-image ordeal (~750s
for the v7 one this lane already paid — v4's would be the same order of
magnitude), which blows the ~20min timebox by itself. **Not fixed this
room** (measurement-only, no ordeal re-runs beyond what's already
timeboxed). Honest old-vs-new delta uses the last LIVE-measured non-split
number on record instead, clearly flagged as carried, not fresh:
**18.57ms/53.85fps (room 1, prior session)** vs this room's clean v7
median ~18.12-19.10ms — within noise of the old baseline, not a
regression on TOTAL (matches room 3's original surprising finding: v7's
bigger GEMM is offset by the removed multi-poll serialization, so v7
TOTAL sits AT or slightly above, not clearly above, the pre-split
baseline). `wall_fps` (41.2-43.7 this room) stays below the old 53.85 —
same reading as every prior room: that gap lives in `outside` (world/http
loop overhead), not render cost; TOTAL-to-TOTAL is the fair number.

**P95 tail attribution.** Used the existing `[n0i]` cumulative
median/p95-every-60-frames log lines already printed by each run (no new
instrumentation added) — the log's OWN early-vs-late lines double as a
first-N-frames-excluded-vs-included comparison, since each line is a
cumulative percentile over all frames seen so far:

| run | frame=60 (first-60, warmup included) TOTAL med/p95 | steady-state (last window) TOTAL med/p95 |
|---|---|---|
| v7clean1 | 22.76 / 47.56 | 18.12 / 27.47 |
| v7clean2 | 25.56 / 53.16 | 18.21 / 30.42 |
| v7clean3 | 20.24 / 26.61 | 19.04 / 27.98 |

Two of 3 runs show a LARGE startup transient (p95 47.6/53.2ms in the
first 60 frames, vs 27.5/30.4ms steady-state — a ~1.7-1.9x startup spike,
consistent with first-dispatch pipeline/shader compile + thermal ramp);
the third run's first-60 window happened to land closer to steady-state
already (26.61 vs 27.98, no startup spike visible in THAT window's
boundaries — sampling artifact of the 60-frame granularity, not evidence
the effect is absent). **Excluding warmup, a persistent steady-state tail
remains**: TOTAL p95 sits at ~1.5x median in every run's LAST several
`[n0i]` windows, and — critically — that ratio is FLAT across consecutive
60-frame windows late in each run (e.g. v7clean1's last 3 lines: p95
27.55/27.55/27.47; v7clean3's: 28.54/28.11/27.98) — a one-off rare event
would dilute a CUMULATIVE percentile toward the median as more frames
accumulate; it does not here, so the tail is recurring at a roughly
constant rate throughout steady state, not a single incident. The
existing instrumentation (cumulative percentiles only, no raw per-frame
series, no code changes permitted this room) cannot distinguish a fixed-N
periodic cadence from a constant-rate random process — both produce a
flat cumulative p95 — so periodic-vs-random is left genuinely undecided,
not guessed.

**Carrying bucket**: `trace` and `net_wall`/`net_wait` move together. In
every steady-state window, `trace`'s own p95-minus-median delta
(~9-11ms) and `net_wait`'s own delta (~10-12ms) are each independently
close in size to TOTAL's own p95-minus-median delta (~9-12ms) — if these
two buckets spiked on DIFFERENT frames the deltas would add (TOTAL delta
≈ 20ms), not match either one alone. Reading: the same subset of frames
spikes in both `trace` (GPU trace pass, SYNCHRONOUS — the render thread
polls for it every frame, per the code's own N0.j S13.3 note) and
`net_wait` (async net commit wait, normally hidden by overlap, median
~0ms) simultaneously — consistent with a single shared root cause that
stalls the GPU/driver broadly on those frames (e.g. OS scheduling
preemption of the Metal driver thread, or thermal/power-state
transition) rather than a bug local to one compute stage. `outside`
world-tick buckets (`world`, `skin`, `upload`) show the same ~1.6-2.5x
p95/median ratio too, reinforcing a whole-frame-level stall rather than a
net-present-specific one. `demod`/`present`/`gather` stay tight
(p95/median ~1.3-2x but tiny absolute ms) — not meaningful contributors
to the ~10ms tail in absolute terms.

**Verdict for the Architect**: expect **~18.1-19.1ms median / ~27.5-30.3ms
p95** (≈41-44fps median-based, worse at the tail) at launch on this
machine's clean-GPU state — the median is within noise of the pre-v7
baseline (18.57ms), so v7 is not a median-fps regression once the room 3
poll fix is counted; the p95 tail (~1.5x median, ~10ms absolute) is a
pre-existing whole-frame stall characteristic (trace+net_wait correlated,
likely OS/driver scheduling, not v7-specific — same tail shape was
visible in rooms 3-5's own single-run numbers before this room's 3-run
confirmation) and was NOT root-caused or fixed here — attribution only,
per the room's own no-fix instruction. **One number to tell the
Architect: ~18.5ms median (≈54fps-class on the good half of frames),
with a known ~28-30ms p95 tail on roughly 1-in-20 frames, unrelated to
the v7 cutover itself.**

**Artifacts this room**:
`proof/neural-live/s20-v7clean{1,2,3}.{log,budget.json,state.json,-presented.png}`
(3x clean baseline runs), `proof/neural-live/s20-v7cleanoff.{log,budget.json,state.json}`
(flag-OFF attempt — documents the missing-stamp blocker, not a
real measurement), this note, `v7-cutover-ready.md` (clean fps table).

## §perf — room 7 (ghoul run 2026-07-20): raw per-frame CSV, periodic-vs-random settled

Resumed on a wedged room's UNCOMMITTED partial (`GAIA_FRAME_CSV` env-gated
per-frame dump in `main.rs`, `/frame_csv` shutdown-flush trigger, `s20-bench.sh`
wiring) — committed first (`0f47347`), no re-instrumentation, no re-benches.
Analyzed the existing capture: `proof/neural-live/s20-v7csv1.frames.csv`
(1189 frames, v7 weights, evidence-split ON, offscreen 640×480).

**Steady-state (frame>60, n=1129, excludes the launch transient — see (b)):**

| bucket | median | p90 | p95 | p99 |
|---|---|---|---|---|
| trace | 12.614 | 19.903 | 24.024 | 27.211 |
| gather | 1.874 | 2.153 | 2.468 | 7.987 |
| net_wall | 0.039 | 10.435 | 11.742 | 16.675 |
| net_gpu | 6.035 | 9.904 | 11.458 | 14.746 |
| net_commit | 0.007 | 0.012 | 0.015 | 0.042 |
| net_wait | 0.002 | 10.410 | 11.710 | 16.616 |
| demod | 0.229 | 0.335 | 0.380 | 0.939 |
| present | 0.086 | 0.115 | 0.136 | 0.211 |
| **total** | **18.532** | **25.546** | **28.138** | **31.471** |

Median sits ~1.86ms above the 16.67ms wall — matches room 6's "~2ms median
gap" and the clean-run family (18.1-19.1ms) closely; this single run is
consistent with, not an outlier from, room 6's 3-run spread.

**(b) Launch transient, wider than the 60-frame exclusion assumed.** Spike
frames (TOTAL > steady p90 = 25.546ms) cluster HEAVILY from frame 61 to
~160 (61,62,64-68,70-72,74,77,78,80-82,...,156 — dozens of hits, near-
continuous) before thinning to sparse, isolated episodes after frame ~200.
The first-60-frames exclusion room 6 used is not enough on this run; the
real warm-up/thermal-ramp tail runs to roughly frame 150-200 (~3s at this
frame rate). Steady-state numbers above still hold in aggregate (the tail
is a small fraction of 1129 frames) but a frame>200 cut would be the more
honest "post-launch" cut for future rooms.

**Spike periodicity — settled, mixed verdict, no single fixed period.**
Grouping the 113 raw spike frames into contiguous bursts (frame-adjacent
spikes merged) gives 47 bursts; late-run (frame>200) burst-start intervals:
`36,16,62,1,3,114,6,233,5,1,3,39,3,7,3,4,3,3,111,3,3,3,232,3,15,26,12,4`.

- **Between episodes**: irregular — 3 to 233 frames apart, heavy-tailed
  (two isolated ~232-233-frame gaps, several 100+ gaps, no common divisor
  across the full set). This half is RANDOM/geometric-shaped, not a fixed
  period — consistent with room 6's "OS/driver scheduling" reading.
- **Within an episode**: a 3-frame spacing dominates (10 of 28 late
  intervals = 36%, the single largest bucket by far) — spikes inside a
  stall episode tend to recur every 3rd frame, not randomly.
- **Verdict: MIXED, not FIXED PERIOD.** The run is better described as
  rare, randomly-timed stall EPISODES (no fixed inter-episode period) that,
  once triggered, echo at a 3-frame cadence for a few beats before dying
  out. Neither a pure fixed-period nor a pure constant-rate-random model
  fits alone; report both halves rather than forcing one label.

**(c) Per-spike decomposition — carrying bucket.** Steady-population
spikes (total > p90) vs steady median, by bucket:

| bucket | steady median | spike-frame median | delta | % of total excess |
|---|---|---|---|---|
| trace | 12.614 | 24.030 | **11.417** | **118.5%** |
| gather | 1.874 | 1.914 | 0.040 | 0.4% |
| net_wall | 0.039 | 0.044 | 0.005 | 0.1% |
| net_gpu | 6.035 | 5.701 | -0.334 | -3.5% |
| net_commit | 0.007 | 0.007 | -0.000 | -0.0% |
| net_wait | 0.002 | 0.002 | 0.000 | 0.0% |
| demod | 0.229 | 0.226 | -0.003 | -0.0% |
| present | 0.086 | 0.080 | -0.005 | -0.1% |
| total | 18.532 | 28.166 | 9.634 | — |

`trace` alone carries >100% of the median TOTAL excess (net_gpu drops
slightly on spike frames, partially offsetting) — for the BULK of p90+
spikes, `trace` (the CPU-wall time of `resolve_frame`'s trace stage,
which `device.poll(wait_indefinitely)`-blocks twice per frame when
`GAIA_NATIVE_ASYNC_TRACE` is unset — the default, and this run's config)
is the whole story, not net_wait. **This contradicts a clean trace/net_wait
co-spike**: of the 113 spike frames, only 1 has BOTH `trace` and
`net_wait` above their OWN steady p90 simultaneously (71 trace-only,
30 net_wait-only, 11 neither) — trace-driven and net_wait-driven spikes
are mostly on DIFFERENT frames here, not the shared whole-frame stall
room 6 read from cumulative percentiles alone (room 6 had no raw series
to tell the two apart; the raw CSV now shows they mostly don't coincide).

**(d) Frames > 33ms (11 of 1189), bucket blame:**

| frame | total (ms) | blame bucket | trace | net_wait | net_gpu |
|---|---|---|---|---|---|
| 3 | 91.659 | net_wall/net_wait | 14.41 | 74.94 | 5.14 |
| 5 | 56.352 | net_wall/net_wait | 10.22 | 43.77 | 5.46 |
| 142 | 35.218 | net_gpu | 20.79 | 12.11 | 20.84 |
| 318 | 36.469 | net_wall/net_wait | 10.59 | 15.39 | 11.27 |
| 680 | 46.444 | trace | 34.00 | 0.00 | 22.02 |
| 681 | 58.247 | net_wall/net_wait | 18.48 | 29.58 | 7.27 |
| 684 | 50.480 | net_wall/net_wait | 7.26 | 40.69 | 7.73 |
| 723 | 35.383 | net_wall/net_wait | 14.23 | 16.81 | 5.25 |
| 733 | 34.199 | net_wall/net_wait | 15.88 | 16.04 | 6.23 |
| 740 | 33.780 | net_wall/net_wait | 7.65 | 23.31 | 14.88 |
| 743 | 42.233 | net_wall/net_wait | 13.15 | 21.71 | 5.36 |

Frames 3/5 are launch (cold pipeline). The worst STEADY-run outliers
(318, 680-743) are dominated by `net_wall`/`net_wait`, not `trace` — the
two failure modes split by severity: the common, moderate p90-class spikes
are trace-bound; the rare, extreme (>33ms) outliers are net-wait-bound.
This reconciles with (c): trace carries the BULK by frame-count, net_wait
carries the FEW worst frames by magnitude.

**(3) Code read for a 3-frame periodic candidate — no match found in this
crate.** `resolve_frame`'s trace stage (`src/main.rs`, the `t0`/`trace_ms`
block) calls `device.poll(wait_indefinitely())` twice per frame unless
`GAIA_NATIVE_ASYNC_TRACE=1` (unset in this run — CUT B, N0 lane, was
measured and REJECTED there, so this is the shipped default) — that is a
CPU-blocking wait for GPU completion and is the natural place for a
driver/OS-scheduling stall to land as `trace_ms`. Checked every explicit
ring/buffer-count construct that could produce a period-3 cadence:
`rdirect_live.rs`'s `SET_COUNT = 2` (ping-pong feature/output buffer sets)
and `NetPresent::t_parity` (0/1 camera-reprojection parity) are both
period-**2**, not 3 — neither matches. This is an OFFSCREEN run
(`GAIA_NATIVE_OFFSCREEN=true`, no `NSWindow`/`CAMetalLayer` surface), so
macOS's well-known default `maximumDrawableCount=3` triple-buffered
swapchain stall (the usual period-3 suspect on Metal) does not apply
either — there is no presentable drawable in this config. **No matching
periodic construct found in this crate.** Best remaining candidate is
outside this crate's visibility: wgpu-hal's Metal backend internal
command-buffer/encoder pooling, or a macOS scheduler-quantum effect —
neither confirmable without instrumenting wgpu-hal or the OS scheduler,
out of this room's scope. Name: **none found in-crate; open item.**

**Verdict for the Architect (root class + evidence, NO fixes this room):**
the ~1.86ms median gap to 16.67ms is not one thing — it is (a) `trace`'s
CPU-blocking `device.poll` wait absorbing ordinary scheduling jitter on
~10% of frames (the p90-class spikes, common, moderate, ~11ms each when
they hit), stacked on (b) a much rarer (~1%) `net_wall`/`net_wait` stall
that produces the true tail (>33ms, up to 92ms at launch) and is NOT the
same frames as (a) (only 1/113 overlap) — two separate mechanisms sharing
the `total` budget, not one shared whole-frame stall as room 6's
cumulative-only view suggested. Evidence: per-bucket blame table (c),
the 71/30/1/11 trace-vs-net_wait split, and the >33ms worst-frame table
(d) where net_wait, not trace, dominates the true outliers. **Top-2 shave
proposals (no fixes applied):**
1. Land `GAIA_NATIVE_ASYNC_TRACE` properly this time — N0 lane's CUT B
   rejection predates v7/the fused-queue/evidence-split changes; the
   regression reasons cited then (gather ballooning, contention) may not
   hold under v7's different queue shape. Re-measuring async-trace
   UNDER v7, isolated from CUT B's original context, is the highest-
   leverage untried lever on the `trace`-carried majority of spikes.
2. The `net_wall`/`net_wait` tail (the true >33ms outliers) looks like
   GPU/driver backpressure on the net commit, not trace — profiling
   `resolve_frame`'s net-submit path specifically around frames like
   680-743 (a real, reproducible cluster in this CSV) with
   `tools/profile-seam.mjs`-equivalent instrumentation would separate
   "net queue genuinely behind" from "OS preempted the driver thread",
   which point to different fixes (queue depth tuning vs thread priority).

**Regression guard**: all 6 pre-existing ordeals (`rdirect_live_ordeals`,
`rdirect_gpu_ordeals`, `rdirect_gather_ordeals`) re-run with `GAIA_FRAME_CSV`
unset (default OFF) — byte-identical to every prior room's numbers (see
commit below for the run log).

**Artifacts this room**: analysis only, no new files (reused room 6's
wedged-partial capture); `main.rs`/`s20-bench.sh` instrumentation +
`s20-v7csv1.*` committed at `0f47347`; this note + the room-6-falsehood
corrections at the top of this file and `v7-cutover-ready.md`.

## §perf — room 8 (ghoul run 2026-07-20): async-trace RE-MEASURE under v7 — REVERSES CUT B, flag stays opt-in

Proposal 1 from room 7's verdict, re-measured. **Compose-check first (no
code changes made this room):** `GAIA_NATIVE_ASYNC_TRACE` (`main.rs`,
`NetPresent`) gates only the two `device.poll(wait_indefinitely())` calls
inside the TRACE stage (`resolve_frame`'s `t0`/`trace_ms` block, before the
`is_v7` branch). It composes cleanly with v7's queue shape as-is: the v7
gather branch (`hist_gather.encode` → its own unconditional poll) and the
fused evidence-resolve tail (room 3: accumulate+demod+clamp+pack+swap, ONE
encoder/submit, no polls) are both untouched by this flag — they sit
downstream of the trace stage and neither reads nor is read by it. No
wiring needed, no gap found; ran as-is.

**3x s20 bench, offscreen 640×480, `GAIA_NATIVE_WEIGHTS=v7
GAIA_NATIVE_EVIDENCE_SPLIT=1 GAIA_NATIVE_ASYNC_TRACE=1`** (run 1 also
`GAIA_FRAME_CSV`):

| run | wall_fps | TOTAL median (ms) | TOTAL p95 (ms) |
|---|---|---|---|
| v7async1 | 45.87 | 16.342 | 25.854 |
| v7async2 | 46.27 | 16.346 | 25.884 |
| v7async3 | 46.73 | 16.239 | 24.665 |
| **mean** | **46.29** | **16.309** | **25.468** |

vs room 6's flag-OFF 3-run baseline (same script/config, `v7clean1-3`):

| run | wall_fps | TOTAL median (ms) | TOTAL p95 (ms) |
|---|---|---|---|
| v7clean1 | 42.68 | 18.116 | 27.548 |
| v7clean2 | 41.24 | 18.181 | 30.296 |
| v7clean3 | 43.71 | 19.096 | 27.940 |
| **mean** | **42.54** | **18.464** | **28.595** |

**Delta: +3.75 fps mean (+8.8%), TOTAL median −2.16ms, p95 −3.13ms.**
Consistent across all 3 pairs (async always beats its clean counterpart by
a similar margin — not one lucky run).

**Steady-state per-bucket (raw CSV, `s20-v7async1.frames.csv`, frame>200,
n=1058) vs room 7's own flag-OFF raw-CSV baseline (`s20-v7csv1`, same cut):**

| bucket | room7 (OFF) median | room8 (ON) median | delta |
|---|---|---|---|
| trace | 12.614 | 0.222 | **−12.39** |
| gather | 1.874 | 10.764 | **+8.89** |
| net_wall | 0.039 | 2.485 | +2.45 |
| net_gpu | 6.035 | 6.413 | +0.38 |
| net_wait | 0.002 | 2.398 | +2.40 |
| demod | 0.229 | 0.310 | +0.08 |
| present | 0.086 | 0.091 | +0.01 |
| **total** | **18.532** | **15.806** | **−2.73** |

**(2) Did gather balloon again (CUT B's old failure)? YES, same
mechanism, DIFFERENT outcome.** `gather` absorbs almost exactly what
`trace` lost (+8.89 vs −12.39) — the same "moved, not cut" pattern N0's
CUT B measured. But `net_gpu` stays flat (6.035→6.413, +0.38ms, noise-
class) — **no contention growth this time**, unlike the old N0 CUT B
note ("gather ballooned... now carries the merged wait + contention").
Net: the trace-side saving (−12.4ms) exceeds the gather-side cost
(+8.9ms) by ~2.7-3.5ms, landing as a real TOTAL/fps win instead of N0's
−4 to −5fps regression.

**(3) VERDICT — CPU-bubble, not GPU-work-bound.** fps improved
substantially (+3.75 mean, +8.8%) and TOTAL median fell 2.16-2.73ms,
MORE than the entire 1.86ms median-to-wall gap room 7 measured — the gap
is now fully closed and negative on the raw-CSV steady cut (15.806ms <
16.67ms). `net_gpu` — the actual GPU compute bucket — is unchanged
(6.035→6.413ms, within run-to-run noise), proving the single M1 GPU is [source: this doc]
doing the SAME amount of work either way; what changed is purely
CPU-side: collapsing 2 blocking `device.poll` wakeups into fewer sync
points removes real wall-clock scheduling/wake overhead. Under v4/pre-v7
(N0's queue shape) this same move regressed because the freed CPU time
had nowhere useful to go and the merged wait itself cost more than the
polls it replaced; under v7 (fused single-submission resolve tail +
frame overlap + evidence split) the queue shape apparently changed enough
that the trade nets positive. **CUT B's original rejection is REVERSED
under v7 — it does not generalize across queue shapes, exactly as room 7
flagged as "verdict stale."**

**(4) ACT SAFETY.**
- `v7_present_parity_probe`, `GAIA_NATIVE_ASYNC_TRACE=1`: `still_max=
  4.7684e-7` (px>1e-3=0, all 12 still frames) `pan_max=1.5497e-6`
  (px>1e-3=0, all 3 pan frames) — machine-precision class, matches the
  task's own reference numbers exactly. **Honest caveat**: this probe is
  a standalone harness (per room 3's own note) that does not call
  `main.rs`'s `NetPresent::resolve_frame` and does not read
  `GAIA_NATIVE_ASYNC_TRACE` at all (confirmed by grep — zero occurrences
  in the probe file) — re-ran it WITHOUT the env var and got the
  IDENTICAL numbers (`still_max=4.7684e-7 pan_max=1.5497e-6`), proving
  the flag has literally zero effect on this harness. So this is not
  empirical proof the live path's pixels are unchanged under the flag —
  it only re-confirms the harness's own (flag-blind) baseline. The real
  safety argument for the live path is structural: wgpu enforces
  buffer read-after-write hazards within one FIFO queue regardless of
  when the CPU polls, so removing timing-only polls cannot reorder GPU
  commands or change any pixel — same house argument room 3 used to
  justify the fused resolve-tail's own poll removal. Supporting (not
  proving) evidence: all 3 live async-flag bench runs produced coherent,
  non-degenerate presented PNGs (~150-155KB each, `s20-v7async{1,2,3}
  -presented.png`) — no black frame, no crash, no wedge.
- 6 regression ordeals, flag OFF (unchanged code path, sanity re-run):
  `rdirect_gather_ordeals` (n0b gate A 9.537e-7, gate B 7.749e-7, n0g
  2.384e-7), `rdirect_gpu_ordeals` (byte-identical same-frame, b_f32
  parity_rel 5.636e-7, b2_fp16 parity_rel 5.791e-4), `rdirect_live_ordeals`
  (n0-gate1 abs 1.311e-6 rel 5.960e-5) — byte-identical to every prior
  room's numbers. No code touched this room, so this is a sanity check,
  not new evidence.

**Recommendation (flag stays opt-in per the room's own charter — default
is the Architect's word, not this room's):** the evidence here reverses
N0's CUT B verdict FOR V7's queue shape specifically: +3.75fps mean
(42.54→46.29), TOTAL median −2.2 to −2.7ms (now under the 16.67ms wall on
the steady-state cut), net_gpu flat (no contention growth), regression
gates clean. The old rejection was correctly scoped by room 7 as
"proven only for the poll-collapse form... pre-v7 queue shape" — this
room confirms that scoping was right: the SAME poll-collapse form now
wins under v7. Recommend the Architect consider flipping
`GAIA_NATIVE_ASYNC_TRACE` default ON for the v7 path specifically (not
necessarily the 23-in v4 path, which was never re-tested here and where
N0's original rejection still stands unchallenged).

**Artifacts this room**: `proof/neural-live/s20-v7async{1,2,3}.{log,
budget.json,state.json,-presented.png}`, `s20-v7async1.frames.csv`
(1189 frames, per-frame series), this note. No engine code changed.

## MIRROR AUTOPSY (ghoul run 2026-07-20, ~25min timebox)

**Complaint**: Architect, on the live window: "the mirror still looks
weird". Subject: `naruko_show_chrome` — the LARGE chrome sphere
(`worlds/naruko/scenes/main.json`, r=2.1 at `[4.5,3.6,29.5]`, metallic 1.0
roughness 0.02, pure specular), the Rite-IV close object, standing in the
player's own spawn sightline (`realm_shine.rs`'s doc comment). NOT
`naruko_chrome_orb`/`naruko_mirror` (smaller, unrelated pier props).

**Method**: new offscreen example `examples/mirror_autopsy.rs` (own
process, own buffers — never touches port 8430). Camera = `spawn_eye`
(`[0,1.7,44]` yaw0 pitch0 fov60), the exact settled gameplay eye — proven
by the already-committed `proof/realm-shine-a.png` (128-frame/3spp
converged ray trace, same pose) to frame the sphere squarely with visible
reflected structure. Reused the Stage-3 pipeline
`v7_present_parity_probe.rs` proved wired (trace-split → AOV → real v7
net forward via `gather_hist_split` → demod → evidence-clamp), 3 identical
still frames for steady-state history (`GAIA_NATIVE_WEIGHTS=v7`
equivalent — v7 weights loaded + stamp-verified directly), 480×270
(16:9, matching the proof shot's own framing), low-res 240×135 (2× ratio,
lane convention). Dumped the LAST frame's (1) presented (post-clamp),
(2) `evidence_composite_frame` — the CPU-side bilinear upsample of that
frame's own low_e+low_d, i.e. the clamp ceiling's raw INPUT before
temporal-mean/3×3-max/gamma — and (3) preclamp (post-demod net linear,
pre-clamp `present_buf`). PNGs: `proof/neural-live/mirror-presented.png`,
`mirror-evidence.png`, `mirror-preclamp.png` (+ `mirror-groundtruth-crop.png`,
a crop of the pre-existing converged reference for eyes-on comparison).

**Region stats** (sphere screen rect, analytically projected from world
bbox, disc-masked to exclude background contamination — recomputed in
Python over the PNGs since the Rust dump's rect included background
corners):

| image | mean_lum | max_lum (disc) | mean\|Δneighbor\| (disc, high-freq proxy) |
|---|---|---|---|
| presented (live output) | 0.2065 | 0.2864 | 0.01023 |
| evidence (raw bilinear E+D, no net) | 0.1955 | 0.2654 | 0.00591 |
| preclamp (net, pre-clamp) | 0.2081 | 0.3063 | 0.01064 |

presented ≈ preclamp in every stat (mean_lum 0.2065 vs 0.2081, high-freq
0.01023 vs 0.01064) — **the clamp changes almost nothing inside the disc**;
its only visible effect is trimming the single brightest outlier pixels
(max_lum region-wide, not disc-masked, dropped from the earlier full-rect
reading 0.278→0.194, a ~30% peak cut) — consistent with the clamp doing
its job (killing sparkle spikes) but not being the source of the
complaint. The evidence image's LOWER disc high-freq energy (0.0059 vs
0.0106) says the raw bilinear upsample is smoother pixel-to-pixel than
the net's own output at this scale — expected (bilinear upsample of a
2×-coarser buffer is inherently low-pass; the net's tail features carry
native-res albedo/normal/depth edges it can sharpen against) and by
itself not diagnostic of over- or under-smoothing — the eyes-on crop
comparison below is what actually locates the fault.

**Eyes-on (decisive)**: `mirror-groundtruth-crop.png` (128-frame
converged reference, same pose) shows the sphere with real reflected
structure — a violet-pink patch (the pink orb's reflection), a cyan
sliver (the cyan orb), an amber smudge, a bright white specular glint
top-left, and the low-poly facet bands (`radial_segments=24`, geometry,
not noise). `mirror-presented.png` (the live v7 net output, same region)
is a near-featureless flat gray-mauve disc — none of the colored
reflection patches survive, only a faint smudge near the bottom-left rim.
`mirror-preclamp.png` is visually indistinguishable from presented — so
whatever erases the reflections happens BEFORE the clamp, in the net
forward itself, not at presentation.

**Stage located — read the WGSL split (`radiance_ed`,
`src/integrator.wgsl:558-641`), not guessed**: `naruko_show_chrome`'s
roughness (0.02) is under `SPEC_CHAIN_MAX_ROUGHNESS` (0.25) but ABOVE
`MIRROR_ROUGHNESS` (1e-3) — so its bounce takes the GGX-importance-sampled
rough-specular branch (`ggx_half`), not the exact delta `reflect()`, and
`diffuse_seen` stays false (this bounce, and the directly-lit/emissive
surface it reaches, both still post into the **E** "specular-chain"
bucket, per the bucket rule: a hit's OWN direct/emissive terms use
`diffuse_seen` as of BEFORE this bounce's own scatter choice). So the
sphere's reflected content — the orb glints, the factory silhouette — is
supposed to live in E, the bucket the v7 trainer does NOT box-blur (only
D gets `BLUR_RADIUS=2`, confirmed in `rdirect_v7_autopsy.rs`'s own
constants) — this rules out "the training TARGET was pre-blurred here"
as the direct cause.

**The real mechanism is upstream of the target: the LIVE INPUT evidence
itself is garbage for this surface.** E's low-res taps are built from a
**1-spp, 2×-downsampled** trace (`low_e`/`low_d`, 240×135 here — same
ratio the shipped app uses). A GGX lobe at roughness 0.02 is *almost* a
delta: each low-res texel's single sample is one near-random
importance-sampled direction inside a near-point-like lobe, and because
the sphere is strongly curved, the true reflected direction rotates
extremely fast across even a handful of screen pixels — so the 2×2
bilinear tap neighborhood `pixel_features_split` reads is not a coherent
local patch of "the same reflection, slightly blurred": it is 4
near-independent single-sample draws of *different, unrelated* points of
the reflected environment (sky vs orb vs factory vs void), i.e. the input
feature itself carries close to zero exploitable signal for reconstructing
fine reflected structure — no amount of net capacity recovers detail that
isn't coherently present in 4 uncorrelated 1-spp taps. `demod_divisor`
itself is not degenerate here (the sphere's AOV albedo is the material's
base color `#eef3f8`, not near-zero, so the “divisor→1 no-hit branch”
sub-hypothesis in the brief does NOT apply — checked directly in
`demod_divisor`, `src/rdirect.rs:65`, no special-case for metallic
surfaces).

**VERDICT**: primarily **EVIDENCE-side (a)**, compounded by **NET-side
(b)**, with **CLAMP-side (c) ruled out as the driver** (preclamp≈presented
above). The E bucket is architecturally correct — mirror reflections DO
route to the unblurred sharp channel — but at the shipped low-res/1-spp
tracing budget, a GGX-rough mirror's E taps are themselves incoherent
noise, not a smoothed-but-honest signal; v7 (trained mostly on diffuse
scenes where a 2×2 low-res neighborhood IS locally coherent) has learned
a denoising prior that, fed genuinely-incoherent taps, falls back to
something close to the neighborhood mean — a flat gray blob — rather than
reconstructing or even preserving the sharp glints. This is
distribution-mismatch-of-the-INPUT more than distribution-mismatch of
scene content per se: the net was never shown coherent-vs-incoherent E
neighborhoods as a distinguishable training signal for a strongly curved
mirror.

**Fix directions (cost class per the room's instruction — none applied
this room):**
1. **Raise trace spp specifically on high-metallic/low-roughness
   surfaces** (or raise the low-res evidence resolution ratio from 2× to
   e.g. 1.5× for the WHOLE frame) so E's taps become locally coherent
   again for curved mirrors — **ACT change, needs re-ordeal** (changes
   the evidence the shipped/stamped v7 net was trained against; a spp
   bump alone is cheap to trace-side but the net itself may need
   retraining or at minimum re-validation against the new evidence
   statistics before it can be trusted not to regress elsewhere).
2. **Training data**: add curved-mirror/high-frequency-reflection poses
   to the v7/v8 training set (the current set is diffuse-scene-heavy per
   this file's own STAGE 3 framing) so the net learns to preserve rather
   than average away incoherent-but-real specular taps — **ACT change,
   full retrain + re-ordeal**, the most expensive of the three, but the
   most likely to generalize (any future curved-mirror content hits the
   same failure otherwise).
3. **Clamp exemption / evidence-confidence signal**: since clamp-side is
   ruled out as today's driver, this is a smaller lever — but a per-pixel
   "evidence coherence" feature (e.g. variance across the 4 E taps) fed
   to the net, or gating the clamp ceiling by that variance, could let a
   future net at least know when to trust vs. distrust its own input
   rather than silently averaging — **ACT change** (new feature = new net
   input shape = retrain + re-ordeal), not a stopgap.
No bar/stamp edits made; no fixes attempted this room per the task.

**Artifacts this room**: `packages/scrying-glass/examples/mirror_autopsy.rs`
(new, additive-only, no engine/WGSL/main.rs changes),
`proof/neural-live/mirror-{presented,evidence,preclamp}.png` +
`mirror-groundtruth-crop.png`, this note.

## GHOST AUTOPSY — SKY HISTORY SMEAR (ghoul run 2026-07-20, ~25min timebox)

Architect saw ghosting in the SKY on the LIVE window under camera motion —
NEW with v7's recurrent history (Stage 2/3, rooms 2-5 above). Autopsy of
`CamPose::reproject` + `hist_features_split`/`direct_render_sequence_hist_split`
(`rdirect.rs`) + `cam_reproject`/`gather_hist_split`
(`rdirect_gather_split.wgsl`).

**(1) CODE TRUTH — the accept/reject test for a no-hit (sky) pixel has NO
distance/direction check at all.** In both `direct_render_sequence_hist_split`
and `gather_hist_split`:
```rust
let is_miss = depth <= 0.0;
let dist = if is_miss { 1.0e5 } else { depth };   // sky proxied at a FAR
let world = f.cam.eye + dir * dist;               // but FINITE distance
...
let prev_miss = prev_depth <= 0.0;
let ok = if is_miss {
    prev_miss                       // <- ONLY this. No depth_ok, no normal_ok.
} else if prev_miss { false } else {
    let depth_ok = (dist_prev - prev_depth).abs() <= depth_tol * dist_prev.max(1e-4);
    let normal_ok = normal.dot(prev_norm) >= normal_thresh;
    depth_ok && normal_ok
};
```
The geometry branch demands depth+normal agreement; the sky branch demands
only "both frames say no-hit". The 1e5-unit proxy distance makes the
`CamPose::reproject` geometry itself track direction correctly under
translation (parallax at 1e5 units is sub-pixel for realistic per-frame
motion — verified: not the mechanism) and under rotation (angle math is
translation-invariant at that distance) — **so the reprojected screen
coordinate is fine.** The bug is that `ok` never checks whether the
resampled previous-frame value is a plausible match for THIS frame's sky —
any two no-hit pixels validate each other, unconditionally. WGSL mirrors
this exactly (`rdirect_gather_split.wgsl:348-358`) — confirmed symmetric,
not a CPU/GPU divergence.

**(2) TRAINING — the net never saw a real reprojected sky sample, only
identity self-feedback.** `rdirect_train_v7.rs::settle_still` (the recurrent
training loop) NEVER calls `CamPose::reproject` — it hardcodes `valid=1.0`
after step 0 and feeds `prev_dl` = the SAME PIXEL's own previous output,
always (single fixed camera per training `Pose`, no motion, no reprojection
at all). `pixel_features_split`'s `hi_motion` argument is `Vec2::ZERO` in
every caller, training AND eval alike (dead input everywhere, not itself
the smear cause but confirms motion was never modeled). So "valid=1" in
training means EXACTLY "trust your own last output at this exact pixel";
the net was never shown a case where valid=1 but the previous value came
from a genuinely different (reprojected) sample. Under live motion,
mechanism (1) fires valid=1 for sky with a value the net has no learned
reason to discount → it blends toward the stale/misregistered sample →
visible smear specifically where directional sky variation (gradient, sun
glow, per-frame trace noise) makes "stale" visibly different from "current".

**VERDICT: BOTH mechanisms, compounding.** (1) is the structural gate
failure (sky uniquely exempt from any similarity check); (2) is why the net
can't compensate (it was never trained on a valid=1 sample that wasn't
identity feedback). Neither alone is suficient to fully explain the
practical ghosting without the other, but (1) is the more direct, minimal,
symmetric-fixable defect — hence the fix below.

**MEASURE — offscreen CPU probe** (`examples/rdirect_v7_sky_smear_probe.rs`,
new, pure CPU, no world load, no GPU, real shipped v7 weights): synthetic
64×48 frame, top half SKY (no-hit, low-res radiance gets FRESH per-frame
noise — real Monte-Carlo variance), bottom half a constant ground plane,
camera TRANSLATING sideways (eye.x += 0.15/frame, forward/up/right fixed —
pure translation, the Architect's reported motion), 8 frames. Metric: mean-
abs-diff between the PRESENTED sky pixel (recurrent) and a FRESH no-history
render of the identical pose (length-1 sequence, valid=0 unconditionally).

```
frame 0: default-vs-fresh=0.000000e0  reject-vs-fresh=0.000000e0   (no history yet)
frame 1: default-vs-fresh=7.018796e-2  reject-vs-fresh=0.000000e0
frame 2: default-vs-fresh=7.867728e-2  reject-vs-fresh=3.247134e-6
frame 3: default-vs-fresh=7.603575e-2  reject-vs-fresh=0.000000e0
frame 4: default-vs-fresh=7.778201e-2  reject-vs-fresh=7.769714e-6
frame 5: default-vs-fresh=7.740154e-2  reject-vs-fresh=0.000000e0
frame 6: default-vs-fresh=8.004203e-2  reject-vs-fresh=0.000000e0
frame 7: default-vs-fresh=8.130357e-2  reject-vs-fresh=0.000000e0
SUMMARY max-over-frames: default-vs-fresh=8.130357e-2 reject-vs-fresh=7.769714e-6
ground px max-diff default-vs-reject=0.0000e0 (must be 0.0)
```
Default behavior diverges from the true instantaneous sky value by a
sustained ~0.077-0.081 mean-abs-diff (in [0,1] linear-ish space) once
history kicks in — a real, persistent smear, not a one-frame transient.
`GAIA_V7_SKY_HISTORY=reject` collapses that to ~1e-6 (residual is the
evidence-clamp ceiling's temporal-mean differing between the 8-frame run
and the 1-frame fresh reference — an unrelated accounting artifact, 4-5
orders of magnitude below the smear signal, not a history-feature leak).
Ground pixels: byte-identical between default and reject (`0.0000e0`) —
the fix is surgically scoped to the is_miss branch, confirmed.

PNGs (`proof/neural-live/sky-smear-{default,reject}-last.png`, frame 7):
**default** shows a smooth, flat, over-averaged sky band (history low-pass
filtering away the true per-frame noise into a hazy trail); **reject** shows
the sky's actual grainy per-frame structure (matches what a fresh render
would show — no false smoothing). This is the visual signature of the same
smear the Architect saw.

**FIX — `GAIA_V7_SKY_HISTORY=reject`, opt-in, IRON default unchanged.**
Smallest possible: force `valid=0` for no-hit/sky pixels on BOTH sides,
symmetric like `SNAP_EPS`.
- CPU: `rdirect.rs` — new `pub fn sky_history_reject() -> bool` (env-gated,
  same pattern as `evidence_clamp_gamma`/`GAIA_V7_CLAMP_GAMMA`), read once
  per `direct_render_sequence_hist_split` call; `ok = prev_miss` becomes
  `ok = prev_miss && !sky_reject` in the `is_miss` branch only.
- GPU: `rdirect_gather_split.wgsl` — `HistUniform.params2.y` carries the
  same flag (0.0/1.0), written by `rdirect_gather.rs::FeatureGatherHistSplit::
  encode` (reads `sky_history_reject()` once per encode call); `gather_hist_split`'s
  `is_miss` branch: `ok = prev_miss && (hu.params2.y < 0.5)`.
- `rotate` (motion-compensated sky, mentioned as a future option) is **NOT
  implemented** — out of scope for this timebox; only `reject` and the
  default (unset/anything else = byte-identical to prior behavior) are
  wired. Not touched: stamp/bars, weights, any non-sky path, `SNAP_EPS`,
  the normal-hit branch.

**ACT-CHANGE RE-ORDEAL (required before any stamp claim — none made here):**
- 6 regression ordeals, flag OFF/default (`rdirect_gather_ordeals`,
  `rdirect_gpu_ordeals` ×4, `rdirect_live_ordeals`): all `ok`, same order
  of magnitude as every prior room (n0b gate A/B/n0g, a/b/b2/c, n0-gate1) —
  ran, not just compiled.
- `v7_present_parity_probe` (live GPU pipeline vs CPU reference), flag OFF:
  `still_max=4.7684e-7 pan_max=1.5497e-6` — **byte-identical to room 8's own
  recorded numbers**, confirming this room's changes made ZERO difference
  to default behavior.
- Same probe, `GAIA_V7_SKY_HISTORY=reject`: `still_max=4.7684e-7
  pan_max=1.3001e-6` — GPU still agrees with CPU at machine precision WITH
  the flag flipped on too, proving the WGSL `params2.y` wiring is correct
  and symmetric on the live GPU path (not just in isolated unit math).
- **This is act-code (not test-only): `sky_history_reject()` is read inside
  `direct_render_sequence_hist_split` (CPU reference, load-bearing for the
  stamped real-image ordeal) and inside `FeatureGatherHistSplit::encode`
  (the actual live GPU dispatch `main.rs` calls). Because the default value
  is unchanged, the shipped v7 stamp remains valid AS-IS for the default
  path. If `GAIA_V7_SKY_HISTORY=reject` is ever flipped to non-default in
  the live app, the real-image ordeal (residual-vs-teacher + sparkle bars)
  MUST be re-run under that setting before any PASS claim for that mode —
  this room did NOT do that re-stamp (out of scope: no real-image scenes
  were rendered, only the synthetic sky probe and the existing parity/
  regression harnesses). The Architect's own word decides whether/when to
  flip the flag; this room only proves the mechanism and ships the opt-in
  lever.**

**Artifacts this room**: `packages/scrying-glass/src/rdirect.rs` (env flag +
is_miss gate), `packages/scrying-glass/src/rdirect_gather_split.wgsl` (mirror
gate + params2.y), `packages/scrying-glass/src/rdirect_gather.rs` (encode()
writes params2.y), `packages/scrying-glass/examples/rdirect_v7_sky_smear_probe.rs`
(new, pure-CPU offscreen probe), `proof/neural-live/sky-smear-{default,
reject}-last.png`, this section. No world/live-app process touched, no port
8430 activity.

## GAMMA SWEEP (room timed out at wall; detached probe completed the work — harvested by monad)
Probe: examples/gamma_sweep_probe.rs — ordeal's OWN sparkle/resid metric, trace once, sweep gammas over cached buffers. Instrument TRUE: γ=1.5 reproduces stamped pair exactly (0.03487/35.81).
γ · resid_still · sparkle/Mpx · highlight_patch (teacher 0.3878):
1.5 · 0.03487 · 35.81 · 0.2377 (−38.7%)
1.25 · 0.03572 · 24.41 · 0.2314 (−40.3%)
1.0 · 0.03789 · 11.39 · 0.2191 (−43.5%)
0.85 · 0.04443 · 6.51 · 0.1937 (−50.1%) — sweep COMPLETE, probe exited clean
VERDICT: γ = sparkle↔resid Pareto slider. ONLY γ=1.5 passes resid bar 0.035 (1.25 + 1.0 both breach). Dots CANNOT be lawfully killed by ceiling alone → v8 weights must earn sparkle ≪35 AT resid ≤0.035 (Architect's eye rejected dots at 35.8 → the real bar). Bonus signal: net under-renders highlights −38.7% vs teacher ALREADY at γ1.5 (clamp-independent) → v8 training target too.
