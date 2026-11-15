#!/usr/bin/env bash
# Issue #219 measurement runbook. Launch only after the coordinator confirms
# the slot. Two arms in one slot:
#   custom-on  : the recipe launch (CustomAllReduceV2)
#   custom-off : the same launch with --disable-custom-all-reduce (NCCL)
# Each arm runs two collect passes, so the allreduce comparison has a variance
# estimate on both sides. Set ARMS=custom-on to run one arm if the slot is short.
#
# Usage: GPUS=4,5,6,7 PORT=11436 ./run_issue219.sh
set -euo pipefail

SITE=/nfs/lambda_stor_01/homes/pding
SIF=$SITE/qlab/env/cuda_12.8.0-devel-ubuntu24.04.sif
ENV_BIN=$SITE/qlab/env/conda/sglang-v100-py312/bin
PY=$ENV_BIN/python
GPUS=${GPUS:?set GPUS, for example 4,5,6,7}
PORT=${PORT:-11436}
ARMS=${ARMS:-custom-on custom-off}
RUN=$SITE/issue219/run-$(date +%Y%m%d-%H%M%S)
CACHE=$RUN/cache
mkdir -p "$RUN" "$CACHE" "$CACHE/tmp" "$CACHE/home" "$CACHE/torch_extensions" "$CACHE/triton"

{
  echo "run_dir=$RUN"
  echo "commit=$(cat "$SITE/issue219-main/COMMIT")"
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

collect() {
  local arm=$1
  local i=$2
  singularity exec --nv --tmpdir "$CACHE/tmp" --bind /nfs:/nfs "$SIF" env \
    HOME="$CACHE/home" \
    TMPDIR="$CACHE/tmp" \
    TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions" \
    TRITON_CACHE_DIR="$CACHE/triton" \
    PYTHONPATH="$SITE/issue219-main" \
    CUDA_VISIBLE_DEVICES="$GPUS" \
    "$PY" "$SITE/issue219/trace_decode_budget.py" collect \
    --host "127.0.0.1:$PORT" \
    --model qwen38-flashnext-nvfp4 \
    --model-dir "$SITE/qlab/models/Qwen3.8-Flash-Next-NVFP4" \
    --out-dir "$arm/trace$i" \
    --context-tokens 4096 \
    --output-tokens 400 \
    --steps 16 \
    --json-out "$arm/budget$i.json" 2>&1 | tee "$arm/collect$i.log"
}

run_arm() {
  local name=$1
  local disable=$2
  local arm=$RUN/$name
  mkdir -p "$arm"

  echo "=== arm $name: DISABLE_CUSTOM_AR=$disable ==="
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
    --format=csv,noheader > "$arm/gpus.before.csv"
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv,noheader > "$arm/clocks.before.csv" 2>/dev/null || true
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader > "$arm/apps.before.csv" 2>/dev/null || true

  GPUS=$GPUS PORT=$PORT DISABLE_CUSTOM_AR=$disable CACHE_ROOT=$CACHE \
    CONFIG_OUT=$arm/launch_config.txt \
    "$SITE/issue219/issue219_launch_flashnext_nvfp4_tp4.sh" \
    > "$arm/server.log" 2>&1 &
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
  free_mb=$(nvidia-smi -i "$GPUS" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
  if [ -n "$free_mb" ] && [ "$free_mb" -lt 1500 ]; then
    echo "abort: only ${free_mb} MiB free on at least one card ($name)"
    tail -40 "$arm/server.log"
    return 1
  fi
  grep -iE "custom.?all.?reduce|CustomAllReduce|all-reduce|allreduce" \
    "$arm/server.log" > "$arm/ar_init.log" || true

  collect "$arm" 1
  collect "$arm" 2

  stop_server
  sleep 5
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
    --format=csv,noheader > "$arm/gpus.after.csv" 2>/dev/null || true
  cp "$SITE/issue219-main/COMMIT" "$arm/commit.txt"
}

for arm in $ARMS; do
  case "$arm" in
    custom-on) run_arm "$arm" 0 ;;
    custom-off) run_arm "$arm" 1 ;;
    *) echo "unknown arm: $arm" >&2; exit 2 ;;
  esac
done

echo "done: $RUN"
