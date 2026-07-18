#!/bin/bash
# Metal package-builder reachability → build token + exact exit status.
set -o pipefail
ROOT=/Users/pascaldisse/projects/magic-crystal
WORKTREE=/Users/pascaldisse/projects/magic-crystal-ane
LOCK="$ROOT/.build-lock"
while ! mkdir "$LOCK" 2>/dev/null; do sleep 1; done
trap 'rmdir "$LOCK"' EXIT
cd "$WORKTREE"
rm -rf tools/silicon-race-2/work/package.mtlpackage
set +e
nice -n 19 xcrun metal-package-builder -ml \
  ane-spike/f_16384.mlmodelc \
  -o tools/silicon-race-2/work/package.mtlpackage 2>&1 \
  | sed -E 's/^[^:]+: error:/metal-package-builder: error:/'
tool_status=${PIPESTATUS[0]}
if [ -e tools/silicon-race-2/work/package.mtlpackage ]; then
  output_exists=yes
  probe_status=$tool_status
else
  output_exists=no
  probe_status=1
fi
printf 'tool_exit_status=%d output_exists=%s probe_exit_status=%d\n' \
  "$tool_status" "$output_exists" "$probe_status"
exit "$probe_status"
