#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
out="proof/PLAY-d-suite/full"
lock="$HOME/projects/magic-crystal/.build-lock"
run() {
  local label="$1" seconds="$2"; shift 2
  local log="$out/${label}.log" rc
  : > "$log"
  while ! mkdir "$lock" 2>/dev/null; do sleep 2; done
  trap 'rmdir "$lock" 2>/dev/null || true' EXIT
  printf '[build-lock] %s acquired\n' "$label" >> "$log"
  timeout "$seconds" nice -19 cargo test -j2 "$@" >> "$log" 2>&1
  rc=$?
  rmdir "$lock" 2>/dev/null || true
  trap - EXIT
  printf '%s\trc=%s\n' "$label" "$rc"
}
run oracle-rest_pose_canon 300 -p oracle --test rest_pose_canon
run oracle-vi2_fracture_canon 300 -p oracle --test vi2_fracture_canon
run oracle-vii0b_gaze 300 -p oracle --test vii0b_gaze
run scrying-glass-lib 900 -p scrying-glass --lib
run scrying-glass-bins 300 -p scrying-glass --bins
run scrying-glass-doc 300 -p scrying-glass --doc
