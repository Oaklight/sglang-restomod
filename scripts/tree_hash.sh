#!/usr/bin/env bash
# Content hash over the files that define the engine at a pin.
#
# Coverage is `UPSTREAM_PIN.toml`, `patches/` and `sglang_restomod/`: the pin,
# the patch queue applied to it, and the plugin that rides on PYTHONPATH. A
# change in any of them moves the digest, which is what makes it usable as a
# gate on a receipt or on a measurement arm.
#
# The digest is composite over the sorted `sha256sum` listing, so it covers the
# file set, the relative paths and the order. It is a content hash, not a path
# hash: an export may live anywhere with the same relative layout. `__pycache__`
# is excluded and the sort is locale-pinned here, not in the caller's shell --
# a hash that depends on the caller's environment is not a gate.
#
# usage: tree_hash.sh [<tree-root>] [--expect <digest>]
#
#   <tree-root>  directory holding UPSTREAM_PIN.toml; default: this repository
#   --expect     exit 1 unless the digest equals <digest>
#
# Record the printed digest together with the commit it belongs to. The script
# prints its own sha256 as well: a stale copy of a gate explaining a mismatch is
# indistinguishable from a content change unless you can tell them apart.
set -euo pipefail

SELF="$(readlink -f "$0")"
DEFAULT_ROOT="$(dirname "$SELF")/.."

root="$DEFAULT_ROOT"
expect=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --expect) expect="${2:?--expect needs a digest}"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$SELF"; exit 0 ;;
    -*) echo "unknown argument: $1" >&2; exit 2 ;;
    *) root="$1"; shift ;;
  esac
done

if [[ ! -f "$root/UPSTREAM_PIN.toml" ]]; then
  echo "not a pin tree: $root has no UPSTREAM_PIN.toml" >&2
  exit 2
fi

# Reading the pin needs tomllib (Python 3.11+). Host pythons on the lab machines
# can be older, and a gate that silently uses a different interpreter than the
# one you think you ran is not a gate, so say which one was chosen.
py="${PYTHON:-}"
if [[ -z "$py" ]]; then
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import tomllib' >/dev/null 2>&1; then
      py="$candidate"
      break
    fi
  done
fi
if [[ -z "$py" ]] || ! "$py" -c 'import tomllib' >/dev/null 2>&1; then
  echo "no interpreter with tomllib (Python 3.11+); set PYTHON to one" >&2
  exit 2
fi

read -r ref commit < <("$py" - "$root/UPSTREAM_PIN.toml" <<'PY'
import pathlib, sys, tomllib
pin = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["upstream"]
print(pin["ref"], pin["commit"])
PY
)

digest="$(
  cd "$root"
  find UPSTREAM_PIN.toml patches sglang_restomod -type f \
       -not -path '*/__pycache__/*' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | cut -d' ' -f1
)"

echo "script $(sha256sum "$SELF" | cut -d' ' -f1)"
echo "pin    ${commit} (${ref})"
echo "tree   $(cd "$root" && pwd)"
echo "digest ${digest}"

if [[ -n "$expect" && "$digest" != "$expect" ]]; then
  echo "expected ${expect}; the tree differs from what that digest was taken on" >&2
  exit 1
fi
