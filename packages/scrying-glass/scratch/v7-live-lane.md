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

## 3. NEXT STAGE (resume point for the next room)

STAGE 2 = recurrent history buffer ping-pong on GPU: a per-pixel GPU-resident
`prev_dl`(vec3)+`valid`(f32) buffer that survives frame-to-frame, written by
a new pass mirroring the CPU sequence loop's reprojection (`p_cam.reproject`
+ depth/normal reject test in `direct_render_sequence_hist_split`) — needs
previous frame's camera pose + depth + normal kept alive one frame (currently
`net_aov` is per-SET, not per-frame-history; check whether the existing
double-buffer sets can serve this or a dedicated ping-pong buffer is needed).
Then `HIST_FEATURES_SPLIT` (39) gather = Stage-1's 35-feature `gather_split`
+ append prev_dl/valid (4 features) from that buffer.

STAGE 3 = 39-in net load (swap `RdirectLive` to accept v7's
`data/rdirect-weights-v7.bin`, in_dim 39) + port `clamp_evidence_lin` into
the demod stage (WGSL or fused MSL, mirroring `EvidenceAccum`/
`local_max_3x3`/`evidence_composite_frame` on GPU) + full v7 parity gate
(new n0-gate1-shaped test over the 39-in fixture) + fps re-measure (last
known: 18.57ms/53.85fps pre-v7, non-split net — expect regression, unmeasured
for 39-in).

## 4. Build/process state (resume)

No background build left running by this room (single-token, sequential
`cargo test`/`cargo run --example` under `nice -n 19`, `-j2`, foreground,
each finished before starting the next). If a later room needs to detach a
long build, note PID + log path HERE.
