# v7 autopsy — evidence-clamp gamma derivation addendum (2026-07-19)

§ prior: scratch/v7d-autopsy.log — diagnosed net overshoot of genuinely-
bright E-structure at sparkle-outlier px, ratio net_lum/(E_lum+D_raw_lum)
1.149x-4.119x, measured against ref_frames-converged (96 rays/px) E_full/
D_full. That ceiling source is unavailable at runtime (96 rays/px defeats
the 1-spp-net point) — diagnosis only, not a gate.

§ runtime-honest ceiling: built from what the net ACTUALLY reads —
low_e/low_d 1-spp taps, bilinear-upsampled to native res
(evidence_composite_frame, src/rdirect.rs), same taps pixel_features_split
samples.

§ max-across-time tried FIRST -> DEAD END: a single noisy 1-spp specular
sample spikes far above the converged value, so outlier-population ratio
against a max-across-time ceiling fell BELOW 1 at every gamma>=1 (the
"ceiling" already exceeded the net's overshoot before any clamp applied —
no room to clamp).

§ fix: temporal MEAN over the K settle taps (approximates what the net's
own recurrent averaging estimates, variance ~1/K) -> spatial 3x3 max-pool
(local_max_3x3) per the task's window -> EvidenceAccum::ceiling().

§ measured (scratch/v7e-gamma-derive.log, tool
examples/rdirect_v7e_gamma_derive.rs, val pose orbit_-20, K=8,
ref_frames=96):

  res      non-outlier p99.9   outlier p50 (median)
  480x360  1.5052              1.6097
  640x480  1.4353              1.9854

§ GAMMA = 1.5 — set just above the non-outlier p99.9 ceiling (~1.44-1.51,
full headroom for real bright detail so genuine highlights never clip)
while sitting below the outlier median (~1.6-2.0, clamps the bulk of the
overshoot mass). Encoded as EVIDENCE_CLAMP_GAMMA_DEFAULT in src/rdirect.rs
(env override GAIA_V7_CLAMP_GAMMA).

§ result (real_image_ordeal, 640x480, zero retrain — clamp applied at the
presented act only, checkpoint 636c8743 unchanged):
  sparkle_still  52.08 -> 27.67/Mpx  (bar 40)   FAIL -> PASS
  resid_still    0.03666 -> 0.03704 (bar 0.035)  FAIL -> FAIL (worse, still
                                                    short — clamp trades a
                                                    hair of resid for a
                                                    large sparkle win)
  tvar/resid_move/ghost_excess: PASS both before and after (unaffected)

§ open bar: resid_still. Clamp only fixes presentation of existing net
output — closing resid needs the net itself to learn tighter (fine-tune
lane, see v7f/resume-notes for the training-side attempt).
