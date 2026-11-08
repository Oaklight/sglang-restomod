#!/usr/bin/env bash
# Move origin/upstream to the pin, or check that it is already there.
#
#   scripts/move_upstream.sh --check   compare the remote with UPSTREAM_PIN.toml
#   scripts/move_upstream.sh           perform the move
#
# The move cannot run in CI. The branch holds the upstream SGLang tree, which
# carries 114 .github/workflows/ files, and a GITHUB_TOKEN may not create or
# update a workflow file:
#
#   ! [remote rejected] ... (refusing to allow a GitHub App to create or update
#   workflow `.github/workflows/_npu-pr-test-stage.yml` without `workflows`
#   permission)
#
# So a maintainer runs this with a credential that can push workflow files, and
# restomod-pin-sync runs --check and fails loudly when the branch is stale.
#
# The fetch is deliberately not --depth 1: a shallow pack leaves out the
# boundary commit's parent, and the receiver's connectivity check rejects the
# push ("did not receive expected object <parent>"). Filling in the history one
# commit at a time does not help, because the boundary only moves up.
set -euo pipefail

SELF="$(readlink -f "$0")"
cd "$(dirname "$SELF")/.."

REMOTE="${REMOTE:-origin}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/sgl-project/sglang.git}"

MODE=move
case "${1:-}" in
  --check) MODE=check ;;
  "") ;;
  -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
  *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

read -r kind ref commit < <(python3 - <<'PY'
import pathlib, tomllib
pin = tomllib.loads(pathlib.Path("UPSTREAM_PIN.toml").read_text())["upstream"]
print(pin["kind"], pin["ref"], pin["commit"])
PY
)

label="${ref:-${commit:0:12}}"
tag="upstream-${label}"

branch_sha="$(git ls-remote "$REMOTE" refs/heads/upstream | cut -f1)"
# An annotated tag resolves under ^{}; a lightweight one only under its name.
tag_sha="$(git ls-remote "$REMOTE" "refs/tags/${tag}^{}" | cut -f1)"
if [[ -z "$tag_sha" ]]; then
  tag_sha="$(git ls-remote "$REMOTE" "refs/tags/${tag}" | cut -f1)"
fi

report_stale() {
  [[ "$branch_sha" == "$commit" ]] || \
    echo "origin/upstream is ${branch_sha:-missing}; pin is ${commit}" >&2
  [[ "$tag_sha" == "$commit" ]] || \
    echo "origin refs/tags/${tag} is ${tag_sha:-missing}; pin is ${commit}" >&2
}

if [[ "$MODE" == check ]]; then
  if [[ "$branch_sha" == "$commit" && "$tag_sha" == "$commit" ]]; then
    echo "origin/upstream and ${tag} are at the pin ${commit}"
    exit 0
  fi
  report_stale
  echo "move it with scripts/move_upstream.sh from a maintainer checkout;" >&2
  echo "CI cannot, because a GITHUB_TOKEN may not push workflow files" >&2
  exit 1
fi

if [[ "$branch_sha" == "$commit" && "$tag_sha" == "$commit" ]]; then
  echo "origin/upstream and ${tag} are already at ${commit}; nothing to do"
  exit 0
fi

# Fetch into a scratch repository: the pin's history is large and does not
# belong in the working checkout.
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
git init -q "$scratch"
git -C "$scratch" remote add upstream "$UPSTREAM_URL"

if [[ "$kind" == "tag" ]]; then
  git -C "$scratch" fetch --no-tags --force upstream "refs/tags/${ref}:refs/tags/${ref}"
else
  git -C "$scratch" fetch --no-tags --force upstream "$commit"
fi
git -C "$scratch" cat-file -e "${commit}^{commit}"

# REMOTE is normally a remote name; accept a URL or a path as well, so the
# script can be pointed at a scratch receiver when testing.
remote_url="$(git remote get-url "$REMOTE" 2>/dev/null || printf '%s' "$REMOTE")"
if [[ "$branch_sha" != "$commit" ]]; then
  git -C "$scratch" push "$remote_url" "${commit}:refs/heads/upstream" --force
fi
if [[ "$tag_sha" != "$commit" ]]; then
  git -C "$scratch" tag -f "$tag" "$commit"
  git -C "$scratch" push "$remote_url" "refs/tags/${tag}" --force
fi

# Read both refs back; the move has happened when they say so, not when the
# pushes return.
"$SELF" --check
