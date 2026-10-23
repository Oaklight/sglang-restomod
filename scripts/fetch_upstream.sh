#!/usr/bin/env bash
# Fetch the pinned SGLang source into .build/sglang.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

read -r ref commit < <(
  python - <<'PY'
import pathlib, tomllib
pin = tomllib.loads(pathlib.Path("UPSTREAM_PIN.toml").read_text())["upstream"]
print(pin["ref"], pin["commit"])
PY
)

dest="${1:-.build/sglang}"
mkdir -p "$(dirname "$dest")"

if [[ ! -d "$dest/.git" ]]; then
  rm -rf "$dest"
  git init -q "$dest"
  git -C "$dest" remote add origin https://github.com/sgl-project/sglang.git
fi

git -C "$dest" fetch --depth 1 --force origin "$ref"
git -C "$dest" checkout -q --detach FETCH_HEAD

actual="$(git -C "$dest" rev-parse HEAD)"
if [[ "$actual" != "$commit" ]]; then
  echo "pin mismatch: expected $commit, got $actual" >&2
  exit 1
fi
echo "fetched sglang $ref ($actual) into $dest"
