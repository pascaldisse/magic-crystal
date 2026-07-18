#!/usr/bin/env bash
# Enforce the adversary charter on merge commits before they leave this repository.
set -euo pipefail

law='ADVERSARY CHARTER: every merge requires cross-model adversary + spec-concordance.'

if [[ -n "${GAIA_JAGD_SKIP:-}" ]]; then
  printf 'WILDE JAGD BYPASS — GAIA_JAGD_SKIP=%s\n' "$GAIA_JAGD_SKIP" >&2
  printf 'WILDE JAGD BYPASS — %s\n' "$law" >&2
  exit 0
fi

commits=( "$@" )
if (( ${#commits[@]} == 0 )); then
  commits=( HEAD )
fi

status=0
for commit in "${commits[@]}"; do
  if ! git rev-parse --verify --quiet "${commit}^{commit}" >/dev/null; then
    printf 'WILDE JAGD REFUSED — %s is not a commit.\n' "$commit" >&2
    status=1
    continue
  fi

  if (( $(git rev-list --parents -n 1 "$commit" | awk '{ print NF - 1 }') < 2 )); then
    printf 'WILDE JAGD REFUSED — %s is not a merge commit.\n' "$commit" >&2
    status=1
    continue
  fi

  message=$(git log -1 --format=%B "$commit")
  adversary=$(printf '%s\n' "$message" | git interpret-trailers --parse | grep -E '^Adversary: .+ HOLDS$' || true)
  concordance=$(printf '%s\n' "$message" | git interpret-trailers --parse | grep -Fx 'Concordance: checked' || true)
  if [[ -z "$adversary" || -z "$concordance" ]]; then
    printf 'WILDE JAGD REFUSED — merge %s lacks required trailers.\n' "$commit" >&2
    printf 'Required: Adversary: <agent> HOLDS\n' >&2
    printf 'Required: Concordance: checked\n' >&2
    printf '%s\n' "$law" >&2
    status=1
  else
    printf 'WILDE JAGD HOLDS — merge %s\n' "$commit" >&2
  fi
done

exit "$status"
