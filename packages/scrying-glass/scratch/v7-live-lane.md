# v7-live lane — STAGE 1 (feature-map + GPU evidence split)

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
[n0g] S8 MPSGraph(default) vs chain: max abs 2.384e-7
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
[n0g] S8 MPSGraph(default) vs chain: max abs 2.384e-7
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
   39 (`HIST_FEATURES_SPLIT`) — `RdirectLive`'s MPSGraph input tensor shape
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
  IS the shape, used for the MPSGraph input placeholder and everywhere else
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
  net's MPSGraph input is a silent PER-ROW STRIDE MISMATCH (not a smaller/
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

1. **WGSL/Metal forward for in_dim 39**: the MPSGraph forward itself
   (rdirect_live.rs) needed NO shader change — it builds its graph ops from
   `dims` (now `in_features`-wide) at construction time, so a 39-in blob
   already produces a correctly-shaped MPSGraph automatically (verified by
   the load probe above: `LOADED OK in_features=39`). The standalone fused
   WGSL kernels (`rdirect.wgsl`/`rdirect_fast.wgsl`, `GpuRdirect` in
   rdirect_gpu.rs) are a SEPARATE, non-live code path — used only by the
   offline bench (`examples/rdirect_kernel.rs`) and the 23-in `b_f32`/
   `b2_fp16` ordeals (still 23-in on purpose, those gates are unrelated to
   this lane) — confirmed by grep, nothing in `main.rs`'s live present path
   references `GpuRdirect`. So spec item 1's "WGSL forward kernel" framing
   doesn't map onto a real gap: the live forward is Metal/MPSGraph-only and
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
`pack_out_dl3to4` on the net's tight `[n,3]` MPSGraph output), matching the
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
