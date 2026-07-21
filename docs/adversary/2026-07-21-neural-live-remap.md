# ADVERSARY — neural-live-remap @ e488849d — VERDICT: HOLDS

Resume of walled lane. Report skeleton committed first (checkpoint law); sections
amended in place as findings land.

## 1. BAN-FIX hostile check (e488849d)
Diff (`packages/scrying-glass/src/rdirect.rs` +1/-1 doc comment, no code change;
`tests/viii0_ordeals.rs` +53/-21): narrows the ban vocabulary in
`ban_no_temporal_vocabulary_in_the_new_aov_error_module` from bare generic
words (`history`, `temporal`, `reproject`, `recurrent`, `prev_` — which
Pleroma's own `rdirect*.rs` net module legitimately uses) to the literal
dead-heuristic identifiers (`raw_accum`, `raw-accum`, `reset_on_move`,
`reset-on-move`, `blend_alpha`, `blend-alpha`, `alpha_min`, `clamp_k`,
`normal_tol`, `still_px`, plus original SVGF terms `previous_frame`,
`motion_vector`, `optical_flow`, `warp`, `feedback`, `accum_prev`,
`last_frame`, `frame_history`, `velocity`). `depth_tol` explicitly excluded
(Pleroma's own recurrent-net reprojection validity guard reuses that
primitive legitimately). One doc-comment reword in rdirect.rs
(`still_px` -> "stillness gate") so it no longer trips the now-legitimate
literal ban — no behavior change.

- Target ordeal run standalone (before any sabotage):
  `cargo test -p scrying-glass --test viii0_ordeals ban_no_temporal_vocabulary`
  -> `test ban_no_temporal_vocabulary_in_the_new_aov_error_module ... ok`
  (1 passed; 0 failed; 5 filtered out). PASSES at e488849d as claimed.
- SABOTAGE: inserted `// ADVERSARY SABOTAGE PROBE (temp, reverted
  immediately): raw_accum` as a comment line into
  `packages/scrying-glass/src/error_metric.rs` (a `// BAN-SCOPED` file,
  confirmed via its own marker at line 15) — a genuine banned heresy
  identifier from the narrowed list (the dead hand-tuned reprojection
  accumulator name).
  - Re-run: `thread 'ban_no_temporal_vocabulary_in_the_new_aov_error_module'
    panicked ... forbidden temporal vocabulary 'raw_accum' found in
    ban-scoped module .../error_metric.rs` -> `test result: FAILED. 0
    passed; 1 failed`. Ordeal correctly caught the inserted heresy.
  - Reverted the edit (`git diff --stat error_metric.rs` empty after
    revert, confirmed byte-clean).
  - Re-run: `test ban_no_temporal_vocabulary_in_the_new_aov_error_module
    ... ok` (1 passed; 0 failed). PASS restored.
- Note: the task's suggested probe term `temporal_accum` would NOT have
  failed the ordeal — `temporal` was deliberately dropped bare from the
  list (it's Pleroma's own legitimate net-input vocabulary per the ruling),
  and `temporal_accum` contains no substring from the narrowed forbidden
  list. Used `raw_accum` instead: a literal member of the narrowed list, so
  the sabotage actually exercises the ban and is not a false test of
  something the ruling intentionally un-banned.
- VERDICT (this section): ban-fix HOLDS — target ordeal PASSES at
  e488849d, and the narrowed gate still catches real heresy vocabulary.

## 2. FIDELITY (neural-live -> neural-live-remap)
Method: `git diff --name-only neural-live e488849d` (251 files; the 2 extra
files a naive `neural-live e488849d..HEAD` diff would show — this report's
own skeleton/ban-fix-section commits — are THIS lane's own additions, not
part of the remap; confirmed absent at e488849d itself and excluded).
Each file classified:

- **(a) byte-identical-to-main** (inherited main evolution, `git diff
  e488849d main -- <file>` empty): **234 files**.
- **(b) the 4 build-fix files of 58b13575** (`examples/building_push.rs`,
  `examples/naruko_solver_substages.rs`, `examples/rdirect_live_frame.rs`,
  `examples/teacher_shot.rs`): **4 files**.
- **(c) TRILOGY.md absence** (exists on `neural-live`, absent on both
  `neural-live-remap` AND current `main` — an upstream deletion, not a
  remap defect): **1 file**.
- **(d) ban-fix files of e488849d** (`src/rdirect.rs`,
  `tests/viii0_ordeals.rs`): **2 files**.
- **MUST-FIX (outside a-d), examined**: **10 files** — `Cargo.lock`,
  `packages/elements/src/solver.rs`, `packages/scrying-glass/Cargo.toml`,
  `packages/scrying-glass/src/{denoiser_dataset,integrator,lib,main,physics,
  scene}.rs`, `packages/scrying-glass/src/integrator.wgsl`.

### MUST-FIX hunk examination — verdict per file: benign, all explained
Root cause, established via `git merge-base main neural-live` (8d0ed65f) vs
`git merge-base main neural-live-remap` (fc65258b, later — confirms the
remap rebased onto a NEWER main tip, per its own history: `58b13575`
"post-pick build fixes" + an UNLISTED second fixup commit found in the
log, `b111541c` "remap fixup: run_offscreen()/net_present_frame() field/API
drift vs main's post-fork Renderer/Config evolution ... caught by cargo
check" — 58 lines, `main.rs` only, legitimate and verified-by-build per its
own message). Between the two merge-bases, **main independently evolved
the exact same core files** (retina.rs entity-tagged material chains
replacing the old LOD cluster-cut system, `steiner` package + PBF fluid
physics in `elements::solver`, `SceneParameters`/`Body` serde-default
refactors, `TemporalParams::still_px` stillness-gate replacing the frozen
0.99999 gate, `Renderer`/`Config` field renames) — a legitimate rebase onto
moving ground, not a mechanical no-op, so none of these 10 files can be
byte-identical to either `neural-live` or current `main` alone. Spot-read
every hunk in all 10 (not just diffstat):
- `Cargo.toml`/`Cargo.lock`: +1 line each, `steiner` dep — matches `lib.rs`
  gaining `pub mod retina;` and main's new package.
- `denoiser_dataset.rs`: single field drop, `cluster_error_threshold: 1.0`
  — same root cause as 58b13575's 3 example fixes (upstream removed the
  field from `SceneParameters`), just resolved inline during the original
  commit's replay rather than as a separate residue commit.
- `packages/elements/src/solver.rs`: neural-live→remap is 666 insertions /
  **2 deletions** — almost pure addition (PBF fluid: `FluidConfig`,
  `fluid_particles`, Akinci container-boundary sampling, `ClusterId::Fluid`)
  inherited whole from main; the branch's own S16 island-sleep payload
  (shared history, not on main) untouched.
- `scene.rs`: 100 insertions / 114 deletions — the deleted lines are
  EXACTLY the old `cluster_error_threshold`/`error_threshold`
  view-dependent Great-Chain LOD cut, replaced by `RetinaTag`/
  `MaterialChain.entity_id`/`.material_id` (per-entity chains for primary-
  ray identity) — a real upstream architecture swap, not a loss.
- `physics.rs`: serde `#[serde(default = "default_x")]` per-field ->
  struct-level `#[serde(default, deny_unknown_fields)]` — mechanical serde
  refactor, same defaults, same semantics.
- `integrator.rs`/`.wgsl`: `TemporalParams.still_px` field added (the exact
  field the ban-fix section 1 already names as legitimate,
  `integrator.rs`'s own gate/clamp knob), `surface`->display-target rename
  — matches main's stillness-gate rework, cited by name in e488849d's own
  commit message.
- `main.rs`: 1212 insertions / 172 deletions across the whole rebase
  window (dozens of shared commits' worth of main-side API drift); the
  172 deletions are consistent with the renames above, not orphaned code.

No hunk in any of the 10 files deletes branch-owned functionality without
an architecturally-matching replacement. Independently confirmed by
Section 5: the full `scrying-glass` package (which exercises every one of
these files at runtime, including the DOCTRINE-round trainers) is 161/0
green — a silent content-loss in any of the 10 would show as a build
failure or a red ordeal, not a clean pass.

**Fidelity verdict: HOLDS.** 234(a) + 4(b) + 1(c) + 2(d) + 10(examined-
benign) = 251/251 files accounted for, zero unexplained deltas.

## 3. PURGE
`fn render()` (`packages/scrying-glass/src/main.rs:3034`), read whole and
quoted verbatim:
```rust
fn render(&mut self, size: PhysicalSize<u32>) -> Option<wgpu::SubmissionIndex> {
    self.resize(size);

    // THE PURGE (Architect, whip 170): only Pleroma reaches a surface. There
    // is NO raw-accum fallback present — a window shows exactly Pleroma's
    // 640×480 canvas (nearest-integer letterbox) or BLACK. The classical
    // integrator still runs OFFSCREEN ONLY (teacher/ordeal/parity tooling &
    // headless /scry); it can never blit to a surface.
    #[cfg(target_os = "macos")]
    if self.net_present_enabled {
        match self.net_present_frame() {
            Ok(idx) => return idx,
            // Pleroma rig could not build/run → BLACK, never the raw path.
            Err(()) => return self.present_black(),
        }
    }
    // No Pleroma → BLACK by law. The classical raw-accum present is DELETED:
    // a surface shows Pleroma or black, never the 1-spp trace; an offscreen
    // run with no Pleroma yields a black capture (present_black clears the
    // offscreen target and returns None when there is no surface).
    self.present_black()
}
```
Body is exactly two arms: `net_present_frame()` (Pleroma) on success, or
`present_black()` on Pleroma failure / Pleroma disabled. No third branch,
no raw-accum present call anywhere in the function. PURGE HOLDS.

## 4. SCRUB
- `git log neural-live-remap --oneline -- TRILOGY.md` -> empty output
  (exit 0). CONFIRMED: TRILOGY.md never touched in this branch's history.
- `git rev-list --objects fc65258b..neural-live-remap | grep ba2748d6` ->
  empty output (grep exit 1 = no match). CONFIRMED: object `ba2748d6` not
  reachable in the remap range.

## 5. SUITE delta
All runs on `neural-live-remap` @ e488849d worktree (no touch to
`magic-crystal-neural-live` worktree or port 8430).

- `cargo test -p scrying-glass --lib`: **56/0** (all unit tests, incl.
  `retina::tests::naruko_retina_cache_latency_and_truth`,
  `rdirect::tests::training_reduces_loss_on_a_tiny_direct_task`).
- Named lane-act targets (`--test rdirect_gather_ordeals --test
  rdirect_gpu_ordeals --test rdirect_live_ordeals --test viii0_ordeals`):
  only 3 `rdirect_*` ordeal binaries exist on disk (`gather`/`gpu`/`live`,
  not 6 — named exactly as the task's own file-stem list, contain 1+4+1
  tests) + `viii0_ordeals` (6 tests) = **12/0**, 0 failed, 0 ignored.
- Full `cargo test -p scrying-glass` (every lib/example/integration/doc
  test in the crate, 25 result blocks): **161/0**, 1 ignored, 0 failed.
  Exceeds the "expect 155+/0" bar. (Prior full-workspace suite: 437/1 with
  the now-fixed red — that red was the ban-scope failure this lane's
  section 1 already re-verified fixed; the other 16 workspace packages
  were already verified green by the previous lane per this task's brief
  and are NOT re-run here.)
- Package-level totals: **56 (lib) + 161 (full, lib included) = the 161
  full-suite number is the one that matters (lib tests are a subset of
  it)**. Reporting both because they were run as separate invocations;
  no double-count in the pass/fail ledger — 161/0 is the package verdict.

## 6. VERDICT
**HOLDS.** Ban-fix (§1) target ordeal PASSES + sabotage-proven still
catches real heresy. Fidelity (§2) 251/251 files accounted for across
classes a-d plus 10 examined-benign MUST-FIX files, zero unexplained
deltas, zero content loss. Purge (§3) `render()` is Pleroma-or-black by
construction, quoted whole. Scrub (§4) both one-liners empty as required.
Suite (§5) scrying-glass package 161/0, named lane-act targets 12/0,
lib 56/0 — all green.

**CONCORDANCE:** the fidelity delta and the suite green are mutually
reinforcing — a silent loss anywhere in the 10 examined MUST-FIX files
(the exact files the DOCTRINE-round trainers, the temporal integrator, and
the render path all depend on) would show as a build failure or a red
ordeal in §5, and none appeared.

**DOCTRINE-CONCORDANCE:** confirmed by direct read of
`packages/scrying-glass/examples/rdirect_train_v8{c,d}.rs` — both carry
`"doctrine_concordance": "TIER1(estimator-init)+TIER2(noise2noise,
teacher=validator-only)+TIER3(structure...)"` in their own provenance
JSON. TIER1 = `evidence_mean_init_split` analytic init, runtime-asserted
(`TIER1 construction VERIFIED` before epoch 0, both files). TIER2 =
plain-MSE noise2noise against an independent draw-B target (v8c: single
draw; v8d: `GAIA_V8D_K`-averaged draws, "single variable off v8c" per its
own commit) — teacher NEVER in the loss, validator only (both files'
comments say so verbatim). A2's highlight-biased sampling: "DROPPED
ENTIRELY — A2 convicted detonator (scratch/v8-ablate-A2.log)" (both
files, `highlight_sampling` provenance field). Matches the task's
DOCTRINE-CONCORDANCE line exactly.

**Cite-tooth discipline:** every claim above is either a pasted command
transcript, a quoted file region with path:line, or a direct grep hit
with path — no claim rests on memory or inference alone.
