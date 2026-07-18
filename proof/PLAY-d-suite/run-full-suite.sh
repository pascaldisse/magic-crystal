#!/usr/bin/env bash
# PLAY.d: named package suite; scrying-glass integration binaries isolated.
set -u
cd "$(dirname "$0")/../.."
out="proof/PLAY-d-suite/full"
summary="proof/PLAY-d-suite/full-suite.summary"
lock="$HOME/projects/magic-crystal/.build-lock"
mkdir -p "$out"
: > "$summary"

run() {
  local label="$1" seconds="$2"; shift 2
  local log="$out/${label}.log" start end rc total passed failed ignored
  start=$(date +%s)
  while ! mkdir "$lock" 2>/dev/null; do sleep 2; done
  trap 'rmdir "$lock" 2>/dev/null || true' EXIT
  printf '[build-lock] %s acquired\n' "$label" >> "$log"
  timeout "$seconds" nice -19 cargo test -j2 "$@" >> "$log" 2>&1
  rc=$?
  rmdir "$lock" 2>/dev/null || true
  trap - EXIT
  end=$(date +%s)
  total=$(awk '/^running [0-9]+ test/{n += $2} END{print n+0}' "$log")
  passed=$(awk '/test result: ok\./{for(i=1;i<=NF;i++) if($i=="passed;") n+=$(i-1)} END{print n+0}' "$log")
  failed=$(awk '/test result: FAILED\./{for(i=1;i<=NF;i++) if($i=="failed;") n+=$(i-1)} END{print n+0}' "$log")
  ignored=$(awk '/test result: (ok|FAILED)\./{for(i=1;i<=NF;i++) if($i=="ignored;") n+=$(i-1)} END{print n+0}' "$log")
  printf '%s\trc=%s\tseconds=%s\ttotal=%s\tpassed=%s\tfailed=%s\tignored=%s\n' "$label" "$rc" "$((end-start))" "$total" "$passed" "$failed" "$ignored" | tee -a "$summary"
  return 0
}

# Cargo metadata defines this list; isolated explicitly so each scrying test binary
# gets its own timeout/log. All get the soak ceiling (900 s).
for package in crystal aether char-editor homunculus vessel sama elements fracture transmutation kami seed pleroma wired jormungandr oracle steiner; do
  run "package-${package}" 300 -p "$package"
done
for binary in bloodbend_ordeals body_sigil floor_fallthrough horizon light_temporal medium_parity ordeals patch_gate physics pose_trace rite5 seam_step vii0b_terrain viii0_ordeals viii1_ordeals viii2_ordeals viii3_ordeals viii3b_ordeals; do
  run "scrying-glass-${binary}" 900 -p scrying-glass --test "$binary"
done
# Unit/bin/doc targets complete the package surface; lib receives soak ceiling.
run "scrying-glass-lib" 900 -p scrying-glass --lib
run "scrying-glass-bins" 300 -p scrying-glass --bins
run "scrying-glass-doc" 300 -p scrying-glass --doc
awk -F'\t' '
  { total += substr($4,7); passed += substr($5,8); failed += substr($6,8); ignored += substr($7,9); runs++ }
  END { printf "GRAND\truns=%d\ttotal=%d\tpassed=%d\tfailed=%d\tignored=%d\n",runs,total,passed,failed,ignored }
' "$summary" | tee -a "$summary"
