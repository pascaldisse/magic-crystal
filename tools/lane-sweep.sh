#!/usr/bin/env bash
# lane-sweep.sh — find & (optionally) clean idle magic-crystal-* worktree
# cargo targets. LANE-CLOSE HYGIENE (CLAUDE.md): a lane that ends cleans its
# worktree target/ the same hour. This is the sweeper.
#
# Safety model (no hardcoded exceptions — IRON):
#   a worktree is "busy" iff `pgrep -f "<its own absolute path>"` finds a
#   live process whose argv references something inside that worktree
#   (e.g. a built binary running from <worktree>/target/...). Busy
#   worktrees are NEVER touched, regardless of size.
#
# Usage:
#   tools/lane-sweep.sh                 # dry-run report (default, safe)
#   tools/lane-sweep.sh --clean         # actually delete idle target/ dirs > threshold
#   PROJECTS_DIR=/other tools/lane-sweep.sh
#   LANE_PREFIX=magic-crystal- THRESHOLD_GB=2 tools/lane-sweep.sh --clean
#
# All paths/thresholds are env-parameterized (IRON: never hardcode).

set -euo pipefail

PROJECTS_DIR="${PROJECTS_DIR:-/Users/pascaldisse/projects}"
LANE_PREFIX="${LANE_PREFIX:-magic-crystal-}"
THRESHOLD_GB="${THRESHOLD_GB:-1}"
TARGET_SUBDIR="${TARGET_SUBDIR:-target}"
CLEAN=0

for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --help|-h)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

if [ ! -d "$PROJECTS_DIR" ]; then
  echo "PROJECTS_DIR not found: $PROJECTS_DIR" >&2
  exit 1
fi

# bytes -> GB (integer-ish, one decimal) using du -sk for portability (macOS bash 3.2 has no bc by default reliably, use awk)
kb_to_gb() {
  awk -v kb="$1" 'BEGIN { printf "%.2f", kb/1024/1024 }'
}

echo "=== lane-sweep: $PROJECTS_DIR/${LANE_PREFIX}*/${TARGET_SUBDIR} (threshold ${THRESHOLD_GB}GB) ==="
printf '%-45s %10s %8s %s\n' "WORKTREE" "SIZE" "BUSY?" "ACTION"

any_idle=0
idle_list=()

shopt -s nullglob
for dir in "$PROJECTS_DIR"/${LANE_PREFIX}*; do
  [ -d "$dir" ] || continue
  name="$(basename "$dir")"
  tdir="$dir/$TARGET_SUBDIR"
  [ -d "$tdir" ] || continue

  size_kb=$(du -sk "$tdir" 2>/dev/null | cut -f1)
  size_gb=$(kb_to_gb "$size_kb")

  # busy check: any live process whose command line references this worktree's own path
  if pgrep -f "$dir" >/dev/null 2>&1; then
    busy="yes"
  else
    busy="no"
  fi

  over_threshold=$(awk -v g="$size_gb" -v t="$THRESHOLD_GB" 'BEGIN { print (g > t) ? 1 : 0 }')

  action="-"
  if [ "$over_threshold" = "1" ] && [ "$busy" = "no" ]; then
    action="FLAG (idle, >${THRESHOLD_GB}GB)"
    any_idle=1
    idle_list+=("$tdir|${size_gb}G")
  elif [ "$busy" = "yes" ]; then
    action="skip (busy)"
  fi

  printf '%-45s %9sG %8s %s\n' "$name" "$size_gb" "$busy" "$action"
done
shopt -u nullglob

echo
if [ "$any_idle" = "0" ]; then
  echo "No idle worktree target/ dirs over ${THRESHOLD_GB}GB. Nothing to do."
  exit 0
fi

echo "Idle candidates:"
for entry in "${idle_list[@]}"; do
  echo "  ${entry%%|*}  (${entry##*|})"
done

if [ "$CLEAN" = "1" ]; then
  echo
  echo "=== --clean: deleting idle candidates ==="
  for entry in "${idle_list[@]}"; do
    tdir="${entry%%|*}"
    dir="$(dirname "$tdir")"
    # re-check busy right before delete (race guard)
    if pgrep -f "$dir" >/dev/null 2>&1; then
      echo "SKIP (became busy): $tdir"
      continue
    fi
    echo "rm -rf $tdir"
    rm -rf "$tdir"
  done
  echo "done."
else
  echo
  echo "(dry-run — pass --clean to delete the above)"
fi
