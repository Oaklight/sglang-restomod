#!/usr/bin/env bash
# Produce a reproducible evidence bundle for a GPU run.
#
# Run from the repository checkout at the commit under test. This script never
# signs; it writes evidence/ and packs it. Signing happens on the operator host
# (see docs/gpu-receipts.md).
#
# Usage:
#   scripts/gpu_suite.sh --out evidence
#   scripts/gpu_suite.sh --out evidence \
#       --bench-host 127.0.0.1:11435 --model qwen38next-nvfp4 \
#       --model-dir /path/to/checkpoint --contexts 8192,32768
#
# Environment:
#   OOB      path to oob (default: oob from PATH)
#   PYTHON   interpreter with torch (default: python3)
#   GPUS     CUDA_VISIBLE_DEVICES for the probe (default: 0)
set -euo pipefail

OUT=evidence
BENCH_HOST=""
MODEL=""
MODEL_DIR=""
CONTEXTS="8192,32768"
OUTPUT_TOKENS=128
SKIP_GPU_PROBE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --bench-host) BENCH_HOST="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --contexts) CONTEXTS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --skip-gpu-probe) SKIP_GPU_PROBE=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

OOB="${OOB:-oob}"
PYTHON="${PYTHON:-python3}"
GPUS="${GPUS:-0}"

command -v git >/dev/null 2>&1 || {
  echo "error: git is required; the runtime image may not ship it" >&2
  exit 2
}
git rev-parse --git-dir >/dev/null 2>&1 || {
  echo "error: run this script from a git checkout" >&2
  exit 2
}

# Measure dirt before creating the output directory. Otherwise a default
# in-repo --out marks the run dirty and every receipt needs --allow-dirty.
DIRTY="$(git status --porcelain | wc -l)"
mkdir -p "$OUT"

# Bind the evidence to the exact commit and tree, not the branch.
{
  echo "repo=$(git config --get remote.origin.url 2>/dev/null || true)"
  echo "commit=$(git rev-parse HEAD)"
  echo "tree=$(git rev-parse HEAD^{tree})"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "dirty_files=$DIRTY"
  echo "host=$(hostname)"
  echo "date=$(date -Is)"
  uname -a
} > "$OUT/commit.txt"

if [[ "$SKIP_GPU_PROBE" != "1" ]]; then
  nvidia-smi -q > "$OUT/nvidia-smi-q.txt" 2>&1 || true
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version,clocks.current.sm,clocks.max.sm,memory.total,temperature.gpu --format=csv > "$OUT/gpus.csv" 2>&1 || true
  nvidia-smi topo -m > "$OUT/topo.txt" 2>&1 || true
  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" - > "$OUT/runtime.txt" 2>&1 <<'PY'
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("gpu", i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
print("matmul_sum", float((x @ y).float().sum()))
PY
fi

{
  "$PYTHON" -c 'import sglang; print("sglang", sglang.__version__)' 2>&1 || true
  "$PYTHON" -c 'import torch; print("torch", torch.__version__)' 2>&1 || true
  "$OOB" --version 2>&1 || true
  # The launch configuration of the server under test. /server_info covers the
  # topology and the speculative settings; the component list exists only in
  # the launcher's environment, so the operator declares it here.
  echo "plugins=${SGLANG_PLUGINS:-<unset>}"
  echo "components=${SGLANG_RESTOMOD_COMPONENTS:-<unset>}"
} > "$OUT/versions.txt"

if [[ -n "$BENCH_HOST" ]]; then
  if [[ -z "$MODEL" || -z "$MODEL_DIR" ]]; then
    echo "--model and --model-dir are required with --bench-host" >&2
    exit 2
  fi
  # bench.py builds its URL as http://<host>, so accept a scheme here and strip
  # it once for both the probe and the bench.
  SERVER_HOST="${BENCH_HOST#http://}"
  SERVER_HOST="${SERVER_HOST#https://}"
  SERVER_HOST="${SERVER_HOST%/}"
  # Record what the endpoint is actually serving. The bench numbers are only
  # meaningful next to the spec-decoding setting: MTP off and MTP 3/4 differ by
  # 3-4x at 32k, and a bundle without this cannot tell them apart.
  # Write to a dotfile first: a plain redirect would leave a 0-byte
  # server_info.json in the bundle when both endpoints refuse, which reads as
  # "recorded" to anyone listing the archive.
  for endpoint in /server_info /get_server_info; do
    if "$PYTHON" -c 'import json, sys, urllib.request
host, endpoint = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"http://{host}{endpoint}", timeout=30) as r:
    json.dump(json.load(r), sys.stdout, indent=1, sort_keys=True)
' "$SERVER_HOST" "$endpoint" > "$OUT/.server_info.tmp" 2>"$OUT/.server_info.err"; then
      mv "$OUT/.server_info.tmp" "$OUT/server_info.json"
      rm -f "$OUT/.server_info.err"
      break
    fi
  done
  if [[ ! -f "$OUT/server_info.json" ]]; then
    {
      echo "server_info unavailable from $SERVER_HOST"
      echo "tried: /server_info, /get_server_info"
      head -3 "$OUT/.server_info.err" 2>/dev/null || true
    } > "$OUT/server_info.unavailable.txt"
    rm -f "$OUT/.server_info.tmp" "$OUT/.server_info.err"
  fi

  # A component that fails to register is skipped silently by the plugin
  # loader, and the launch then serves with the engine's own backend under the
  # same flag. The endpoint's own view is the only thing that can tell the two
  # apart, so record it and fail when the selection is not the one that served.
  if [[ -f "$OUT/server_info.json" ]]; then
    check_rc=0
    "$PYTHON" - "$OUT/server_info.json" "${SGLANG_RESTOMOD_COMPONENTS:-}" \
      > "$OUT/launch_check.txt" 2>&1 <<'PY' || check_rc=$?
import json
import sys

path, components = sys.argv[1], sys.argv[2]
with open(path) as fh:
    info = json.load(fh)
selected = {c.strip() for c in components.split(",") if c.strip()}
expected = {}
if "attention_v100" in selected:
    expected["attention_backend"] = "tilelang_fa_v100"
if "spec_sm70" in selected:
    expected["speculative_draft_attention_backend"] = "tilelang_fa_v100"
mismatched = 0
for key, want in expected.items():
    got = info.get(key)
    status = "ok" if got == want else "MISMATCH"
    mismatched += got != want
    print(f"{status}\t{key}\twant={want}\tgot={got}")
if not expected:
    print("ok\tno component in the selection maps to a server_info field")
sys.exit(1 if mismatched else 0)
PY
    if [[ "$check_rc" -ne 0 ]]; then
      echo "error: the serving configuration does not match SGLANG_RESTOMOD_COMPONENTS; see $OUT/launch_check.txt" >&2
      exit 1
    fi
  fi

  "$PYTHON" scripts/bench.py \
    --host "$SERVER_HOST" \
    --model "$MODEL" \
    --model-dir "$MODEL_DIR" \
    --contexts "$CONTEXTS" \
    --output-tokens "$OUTPUT_TOKENS" \
    --chat --no-think --fresh --reps 2 --warmups 1 \
    > "$OUT/bench.log" 2>&1 || echo "bench failed; see bench.log" >> "$OUT/bench.log"
fi

PACK="${OUT%/}.tgz"
"$OOB" pack --evidence-dir "$OUT" --out "$PACK"
echo "evidence: $PACK"
if [[ "$DIRTY" != "0" ]]; then
  echo "warning: worktree is dirty ($DIRTY files); sign with --allow-dirty or clean it first" >&2
fi
