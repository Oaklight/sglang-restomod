#!/usr/bin/env bash
# Merge a new upstream tag into a bump branch and run the pin/delta checks.
#
# Usage: scripts/upstream_sync.sh [tag-or-commit]
# Defaults to the newest upstream v* tag.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

git fetch upstream --tags --force
ref="${1:-$(git tag --list 'v[0-9]*' --sort=-v:refname | head -1)}"
commit="$(git rev-parse "$ref^{commit}")"
branch="bump/upstream-$(printf '%s' "$ref" | tr '/' '-')"

git checkout -B "$branch" main
if ! git merge --no-ff "$ref"; then
  cat <<EOF
Merge conflicts in $branch.
Resolve them, then run:
  python scripts/upstream_bump.py --ref $commit --tag $ref
  python scripts/check_pin.py
  python v100_plus/check-core-diff.py --ref HEAD
EOF
  exit 1
fi

python scripts/upstream_bump.py --ref "$commit" --tag "$ref"
python scripts/check_pin.py
python v100_plus/check-core-diff.py --ref HEAD

echo "Clean. Review, then: git push -u origin $branch"
