#!/usr/bin/env bash
# Build the layer. With no patches this is just the pinned wheel + this package.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

version="$(
  python - <<'PY'
import pathlib, tomllib
print(tomllib.loads(pathlib.Path("UPSTREAM_PIN.toml").read_text())["upstream"]["pypi_version"])
PY
)"

shopt -s nullglob
patches=(patches/*.patch)

if (( ${#patches[@]} > 0 )); then
  scripts/fetch_upstream.sh
  scripts/apply_patches.sh
  python scripts/check_patches.py
  python -m pip install .build/sglang
else
  python -m pip install "sglang==${version}"
fi

python -m pip install -e .
echo "installed sglang-restomod (sglang ${version}, ${#patches[@]} patches)"
