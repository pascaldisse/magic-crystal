#!/bin/bash
# v8 ablation matrix — isolate the sparkle detonator among v8's three new
# components: (a) moving-camera history (pan_step), (b) mirror pose,
# (c) highlight-targeted sampling. 4 runs, 25 epochs each, SERIAL (one GPU),
# fresh init every run (trainer has no resume path), same INIT_SEED (compile-
# time const, shared across all runs by construction). Each run gets its own
# GAIA_V8_TAG so checkpoints/floors/provenance never collide with each other
# or with the real v8 checkpoint (data/rdirect-weights-v8.bin, untouched by
# this script). Run detached:
#   nohup scratch/v8-ablate.sh > scratch/v8-ablate-driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

common_env="GAIA_V7_SKY_HISTORY=reject GAIA_V8_EPOCHS=25"

run_one() {
  local tag="$1" pan="$2" mirror="$3" hifrac="$4" logf="$5"
  echo "=== $(date -u +%FT%TZ) launching $tag pan=$pan mirror=$mirror hifrac=$hifrac -> $logf ==="
  env GAIA_V7_SKY_HISTORY=reject GAIA_V8_EPOCHS=25 GAIA_V8_TAG="$tag" \
      GAIA_V8_PANSTEP="$pan" GAIA_V8_MIRROR_POSE="$mirror" GAIA_V8_HIGHLIGHT_FRAC="$hifrac" \
      nice -n 19 cargo run --release -j2 --example rdirect_train_v8 > "$logf" 2>&1
  local rc=$?
  echo "=== $(date -u +%FT%TZ) $tag finished rc=$rc ==="
  return $rc
}

# A0: v7e-parity baseline inside v8 code — all three components off.
run_one v8ablA0 0     0 0    scratch/v8-ablate-A0.log

# A1: moving-camera history ONLY (default pan_step 0.004).
run_one v8ablA1 0.004 0 0    scratch/v8-ablate-A1.log

# A2: highlight-targeted sampling ONLY (default frac 0.30).
run_one v8ablA2 0     0 0.30 scratch/v8-ablate-A2.log

# A3: mirror pose ONLY.
run_one v8ablA3 0     1 0    scratch/v8-ablate-A3.log

echo "=== $(date -u +%FT%TZ) ablation matrix complete ==="
