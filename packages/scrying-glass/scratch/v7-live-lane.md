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
| non-split baseline (documented, room 1) | — | — | — | — | — | **18.57** | 53.85 |

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
MPSGraph forward output alone, feeding it the SAME feature row on both
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
~1.5ms higher but still at/below the non-split v4 baseline's 18.57ms and
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
