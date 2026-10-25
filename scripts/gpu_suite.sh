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
} > "$OUT/versions.txt"

if [[ -n "$BENCH_HOST" ]]; then
  if [[ -z "$MODEL" || -z "$MODEL_DIR" ]]; then
    echo "--model and --model-dir are required with --bench-host" >&2
    exit 2
  fi
  "$PYTHON" scripts/bench.py \
    --host "$BENCH_HOST" \
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
