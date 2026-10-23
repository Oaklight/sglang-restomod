#!/usr/bin/env bash
# Apply patches/*.patch to the fetched source (in lexical order).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

src="${1:-.build/sglang}"
if [[ ! -d "$src/.git" ]]; then
  echo "no source at $src; run scripts/fetch_upstream.sh first" >&2
  exit 1
fi

shopt -s nullglob
patches=(patches/*.patch)
if (( ${#patches[@]} == 0 )); then
  echo "no patches to apply"
  exit 0
fi

for patch in "${patches[@]}"; do
  echo "applying $patch"
  git -C "$src" apply --whitespace=nowarn "$PWD/$patch"
done
echo "applied ${#patches[@]} patches to $src"
