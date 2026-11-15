#!/usr/bin/env bash
# Issue #227 A/B. Same launch and request; the only difference is the
# sglang_restomod tree:
#   baseline = 1c11d10150 (the #219 tree)
#   fixed    = 8f821b25a7 (admit RowParallelLinear by class)
# Each arm records the admission report, one decode trace, a cold-prefix
# prefill bench, and one greedy completion.
#
# Usage: GPUS=0,1,2,3 ./issue227_run.sh
set -euo pipefail

SITE=/nfs/lambda_stor_01/homes/pding
SIF=$SITE/qlab/env/cuda_12.8.0-devel-ubuntu24.04.sif
ENV_BIN=$SITE/qlab/env/conda/sglang-v100-py312/bin
PY=$ENV_BIN/python
MODEL_DIR=$SITE/qlab/models/Qwen3.8-Flash-Next-NVFP4
GPUS=${GPUS:?set GPUS, for example 0,1,2,3}
PORT=${PORT:-11436}
ARMS=${ARMS:-baseline fixed}
RUN=$SITE/issue227/run-$(date +%Y%m%d-%H%M%S)
# Reuse a warm JIT cache when one is supplied; the compiled sources are the
# same as #219's, so this only saves the ~15 min rebuild.
CACHE=${CACHE_ROOT:-$RUN/cache}
mkdir -p "$RUN" "$CACHE" "$CACHE/tmp" "$CACHE/home" "$CACHE/torch_extensions" "$CACHE/triton"

tree_for() {
  case "$1" in
    baseline) echo "$SITE/issue219-main" ;;
    fixed) echo "$SITE/issue227-main" ;;
    *) return 1 ;;
  esac
}

{
  echo "run_dir=$RUN"
  echo "host=$(hostname)"
  echo "gpus=$GPUS"
  echo "arms=$ARMS"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
  df -h / "$SITE" 2>/dev/null || df -h /
} > "$RUN/host.txt" 2>&1
cat "$RUN/host.txt"

SERVER_PID=""
stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 2
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

run_arm() {
  local name=$1
  local tree
  tree=$(tree_for "$name") || { echo "unknown arm: $name" >&2; return 2; }
  local arm=$RUN/$name
  mkdir -p "$arm"

  echo "=== arm $name: $tree ==="
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
    --format=csv,noheader > "$arm/gpus.before.csv"
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv,noheader > "$arm/clocks.before.csv" 2>/dev/null || true
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader > "$arm/apps.before.csv" 2>/dev/null || true

  GPUS=$GPUS PORT=$PORT TREE="$tree" CACHE_ROOT=$CACHE \
    CONFIG_OUT=$arm/launch_config.txt \
    "$SITE/issue227/issue227_launch.sh" > "$arm/server.log" 2>&1 &
  SERVER_PID=$!

  local ready=0
  for _ in $(seq 1 900); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited during startup ($name); tail of the log:"
      tail -80 "$arm/server.log"
      return 1
    fi
    if grep -q "The server is fired up and ready to roll" "$arm/server.log" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    echo "server did not become ready in 30 min ($name); tail of the log:"
    tail -80 "$arm/server.log"
    return 1
  fi
  echo "server ready ($name)"

  nvidia-smi -i "$GPUS" --query-gpu=index,memory.free,memory.used \
    --format=csv,noheader > "$arm/memfree.after_ready.csv" 2>/dev/null || true
  local free_mb
  free_mb=$(nvidia-smi -i "$GPUS" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
  if [ -n "$free_mb" ] && [ "$free_mb" -lt 1500 ]; then
    echo "abort: only ${free_mb} MiB free on at least one card ($name)"
    tail -40 "$arm/server.log"
    return 1
  fi

  grep "sm70_fp8_dense_quant" "$arm/server.log" > "$arm/admission.log" || true

  # Decode trace + anchor.
  singularity exec --nv --tmpdir "$CACHE/tmp" --bind /nfs:/nfs "$SIF" env \
    HOME="$CACHE/home" \
    TMPDIR="$CACHE/tmp" \
    PYTHONPATH="$tree" \
    CUDA_VISIBLE_DEVICES="$GPUS" \
    "$PY" "$SITE/issue219/trace_decode_budget.py" collect \
    --host "127.0.0.1:$PORT" \
    --model qwen38-flashnext-nvfp4 \
    --model-dir "$MODEL_DIR" \
    --out-dir "$arm/trace" \
    --context-tokens 4096 \
    --output-tokens 400 \
    --steps 16 \
    --json-out "$arm/budget.json" 2>&1 | tee "$arm/trace.log"

  # Cold-prefix prefill: --fresh salts every request, so TTFT is a cold prefill.
  singularity exec --tmpdir "$CACHE/tmp" --bind /nfs:/nfs "$SIF" env \
    HOME="$CACHE/home" \
    TMPDIR="$CACHE/tmp" \
    "$PY" "$SITE/issue219-main/scripts/bench.py" \
    --host "127.0.0.1:$PORT" \
    --model "$MODEL_DIR" \
    --model-dir "$MODEL_DIR" \
    --contexts 4096 \
    --output-tokens 16 \
    --warmups 1 --reps 2 --fresh --no-spec-accept \
    2>&1 | tee "$arm/prefill.jsonl"

  # One greedy completion for the output check.
  singularity exec --tmpdir "$CACHE/tmp" --bind /nfs:/nfs "$SIF" env \
    HOME="$CACHE/home" \
    TMPDIR="$CACHE/tmp" \
    "$PY" "$SITE/issue227/issue227_probe.py" \
    --host "127.0.0.1:$PORT" \
    --out "$arm/output.json" 2>&1 | tee "$arm/output.txt"

  stop_server
  sleep 5
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
    --format=csv,noheader > "$arm/gpus.after.csv" 2>/dev/null || true
  cp "$tree/COMMIT" "$arm/commit.txt" 2>/dev/null || true
}

for arm in $ARMS; do
  run_arm "$arm"
done

echo "done: $RUN"
