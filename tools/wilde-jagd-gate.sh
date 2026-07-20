#!/usr/bin/env bash
# Enforce the adversary charter on merge commits before they leave this repository.
set -euo pipefail

law='ADVERSARY CHARTER: every merge requires cross-model adversary + spec-concordance.'
cite_law='CITE TOOTH: newly added Markdown silicon/host claims require [source: ...] or UNVERIFIED on the same line.'
cite_pattern='ANE|CoreML|MPSGraph|MTLTensor|Metal [0-9]|macOS [0-9]|M[0-9] (Pro|Max)?|neural cores'
doctrine_law='TRAINING DOCTRINE (NEURAL.md §TRAINING DOCTRINE, LASHES row 18): every training-lane Adversary-Report must carry a DOCTRINE-CONCORDANCE: line.'
blob_law='BLOB TOOTH (LASHES row 19): scrubbed blobs must never re-enter history via a contaminated push.'
# IRON law: no hardcoded shas without env override. Default = the recorded
# TRILOGY.md blob (scrubbed 07-20).
blob_shas=( ${GAIA_JAGD_BLOB_SHAS:-ba2748d65ff52c520e1d6d0f8a4519aac1073bdd} )
# IRON law: no hardcoded paths without env override. Default derived from the
# repo's real trainer layout (packages/*/examples/*train*.rs,
# packages/*/src/*_dataset.rs) but any path ending examples/...train....rs or
# ..._dataset.rs anywhere counts, and the whole regex is overridable.
train_pattern="${GAIA_JAGD_TRAIN_PATTERN:-(^|/)examples/[^/]*train[^/]*\.rs$|(^|/)[^/]*_dataset\.rs$}"

if [[ -n "${GAIA_JAGD_SKIP:-}" ]]; then
  printf 'WILDE JAGD BYPASS — GAIA_JAGD_SKIP=%s\n' "$GAIA_JAGD_SKIP" >&2
  printf 'WILDE JAGD BYPASS — %s\n' "$law" >&2
  exit 0
fi

commits=()
cite_ranges=()
blob_ranges=()
while (( $# > 0 )); do
  case "$1" in
    --cite-range)
      [[ $# -ge 2 ]] || { printf 'WILDE JAGD REFUSED — --cite-range needs a range.\n' >&2; exit 1; }
      cite_ranges+=( "$2" )
      shift 2
      ;;
    --blob-range)
      [[ $# -ge 2 ]] || { printf 'WILDE JAGD REFUSED — --blob-range needs a range.\n' >&2; exit 1; }
      blob_ranges+=( "$2" )
      shift 2
      ;;
    *)
      commits+=( "$1" )
      shift
      ;;
  esac
done

status=0

if (( ${#blob_ranges[@]} > 0 )); then
for range in "${blob_ranges[@]}"; do
  hit=""
  if (( ${#blob_shas[@]} > 0 )); then
    for blobsha in "${blob_shas[@]}"; do
      [[ -z "$blobsha" ]] && continue
      if git rev-list --objects "$range" 2>/dev/null | grep -qw "$blobsha"; then
        printf 'WILDE JAGD REFUSED — blob tooth: %s carries scrubbed blob %s.\n' "$range" "$blobsha" >&2
        printf '%s\n' "$blob_law" >&2
        printf 'LASHES.md row 19: intimate/personal source text NEVER enters a repo in any form.\n' >&2
        hit=1
        status=1
      fi
    done
  fi
  [[ -z "$hit" ]] && printf 'WILDE JAGD BLOB TOOTH HOLDS — %s\n' "$range" >&2
done
fi
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

if (( ${#commits[@]} == 0 && ${#cite_ranges[@]} == 0 && ${#blob_ranges[@]} == 0 )); then
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
  subject=$(git log -1 --format=%s "$commit")
  if [[ "$subject" == "Merge branch 'main' into "* || "$subject" == "Merge remote-tracking branch 'origin/main'"* ]]; then
    printf 'WILDE JAGD PULL-MERGE — exempt — %s\n' "$commit" >&2
    continue
  fi
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
    continue
  fi

  # DOCTRINE-CONCORDANCE TOOTH (07-20, LASHES row 18): a merge that touches
  # training-lane files must have its Adversary-Report carry a
  # DOCTRINE-CONCORDANCE: line, or the sealed law is a memo again.
  parent1=$(git rev-parse "${commit}^1" 2>/dev/null || true)
  parent2=$(git rev-parse "${commit}^2" 2>/dev/null || true)
  mergebase="$parent1"
  if [[ -n "$parent1" && -n "$parent2" ]]; then
    mergebase=$(git merge-base "$parent1" "$parent2" 2>/dev/null || echo "$parent1")
  fi
  touched=""
  if [[ -n "$mergebase" ]]; then
    touched=$(git diff --no-ext-diff --name-only "$mergebase" "$commit" -- 2>/dev/null || true)
  fi
  training_touch=""
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    if grep -Eq "$train_pattern" <<<"$f"; then
      training_touch=1
      break
    fi
  done <<<"$touched"

  if [[ -n "$training_touch" ]] && ! grep -q 'DOCTRINE-CONCORDANCE:' <<<"$report_blob"; then
    printf 'WILDE JAGD REFUSED — merge %s touches a training-lane file but the Adversary-Report lacks a DOCTRINE-CONCORDANCE: line.\n' "$commit" >&2
    printf '%s\n' "$doctrine_law" >&2
    printf 'See NEURAL.md §TRAINING DOCTRINE (enforcement clause, 07-20) and LASHES.md row 18.\n' >&2
    status=1
  else
    printf 'WILDE JAGD HOLDS — merge %s\n' "$commit" >&2
  fi
done
fi

exit "$status"
