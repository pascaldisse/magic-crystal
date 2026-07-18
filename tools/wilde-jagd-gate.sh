#!/usr/bin/env bash
# Enforce the adversary charter on merge commits before they leave this repository.
set -euo pipefail

law='ADVERSARY CHARTER: every merge requires cross-model adversary + spec-concordance.'
cite_law='CITE TOOTH: newly added Markdown silicon/host claims require [source: ...] or UNVERIFIED on the same line.'
cite_pattern='ANE|CoreML|MPSGraph|MTLTensor|Metal [0-9]|macOS [0-9]|M[0-9] (Pro|Max)?|neural cores'

if [[ -n "${GAIA_JAGD_SKIP:-}" ]]; then
  printf 'WILDE JAGD BYPASS — GAIA_JAGD_SKIP=%s\n' "$GAIA_JAGD_SKIP" >&2
  printf 'WILDE JAGD BYPASS — %s\n' "$law" >&2
  exit 0
fi

commits=()
cite_ranges=()
while (( $# > 0 )); do
  case "$1" in
    --cite-range)
      [[ $# -ge 2 ]] || { printf 'WILDE JAGD REFUSED — --cite-range needs a range.\n' >&2; exit 1; }
      cite_ranges+=( "$2" )
      shift 2
      ;;
    *)
      commits+=( "$1" )
      shift
      ;;
  esac
done

status=0
if (( ${#cite_ranges[@]} > 0 )); then
for range in "${cite_ranges[@]}"; do
  violations=()
  while IFS= read -r line; do
    [[ "$line" == +++* || "$line" != +* ]] && continue
    added=${line:1}
    if grep -Eq "$cite_pattern" <<<"$added" && ! grep -Eq '\[source:|UNVERIFIED' <<<"$added"; then
      violations+=( "$added" )
    fi
  done < <(git diff --no-ext-diff --unified=0 "$range" -- '*.md')
  if (( ${#violations[@]} > 0 )); then
    printf 'WILDE JAGD REFUSED — cite tooth found uncited newly added Markdown claim(s) in %s.\n' "$range" >&2
    printf '%s\n' "$cite_law" >&2
    printf 'Required: [source: <doc/hash>] or UNVERIFIED on the same line.\n' >&2
    printf '  + %s\n' "${violations[@]}" >&2
    status=1
  else
    printf 'WILDE JAGD CITE TOOTH HOLDS — %s\n' "$range" >&2
  fi
done
fi

if (( ${#commits[@]} == 0 && ${#cite_ranges[@]} == 0 )); then
  commits=( HEAD )
fi

if (( ${#commits[@]} > 0 )); then
for commit in "${commits[@]}"; do
  if ! git rev-parse --verify --quiet "${commit}^{commit}" >/dev/null; then
    printf 'WILDE JAGD REFUSED — %s is not a commit.\n' "$commit" >&2
    status=1
    continue
  fi

  # BASELINE EXEMPTION: merges already in history at the tooth's birth are not
  # policed retroactively (param GAIA_JAGD_BASELINE, default = the tooth commit).
  baseline="${GAIA_JAGD_BASELINE:-bf2779a}"
  if git merge-base --is-ancestor "$commit" "$baseline" 2>/dev/null; then
    printf 'WILDE JAGD BASELINE — %s predates the tooth; exempt.\n' "$commit" >&2
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
  # THE TOOTH (07-18, his 4th order): a trailer is words; the gate demands the
  # ARTIFACT. Adversary-Report: <path> must name a file that EXISTS in the
  # merge commit's own tree and carries a HOLDS verdict + a CONCORDANCE section.
  report_path=$(printf '%s\n' "$message" | git interpret-trailers --parse | sed -n 's/^Adversary-Report: //p' | head -1)
  report_ok=""
  if [[ -n "$report_path" ]]; then
    report_blob=$(git show "${commit}:${report_path}" 2>/dev/null || true)
    if [[ -n "$report_blob" ]] && grep -q 'VERDICT: HOLDS' <<<"$report_blob" && grep -qi 'CONCORDANCE' <<<"$report_blob"; then
      report_ok=1
    fi
  fi
  if [[ -z "$adversary" || -z "$concordance" || -z "$report_ok" ]]; then
    if [[ -z "$report_ok" ]]; then
      printf 'WILDE JAGD REFUSED — merge %s lacks a real adversary ARTIFACT.\n' "$commit" >&2
      printf 'Required trailer: Adversary-Report: <path committed IN this merge>\n' >&2
      printf 'The file must exist in the merge tree and contain "VERDICT: HOLDS" + a CONCORDANCE section.\n' >&2
    fi
    printf 'WILDE JAGD REFUSED — merge %s lacks required trailers.\n' "$commit" >&2
    printf 'Required: Adversary: <agent> HOLDS\n' >&2
    printf 'Required: Concordance: checked\n' >&2
    printf '%s\n' "$law" >&2
    status=1
  else
    printf 'WILDE JAGD HOLDS — merge %s\n' "$commit" >&2
  fi
done
fi

exit "$status"
