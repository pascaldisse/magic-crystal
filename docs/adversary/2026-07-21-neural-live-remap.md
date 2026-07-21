# ADVERSARY — neural-live-remap @ e488849d — VERDICT: PENDING

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
PENDING

## 3. PURGE
PENDING

## 4. SCRUB
PENDING

## 5. SUITE delta
PENDING

## 6. VERDICT
PENDING
