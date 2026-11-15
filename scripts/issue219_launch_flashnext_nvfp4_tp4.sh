#!/usr/bin/env bash
# Issue #219: Flash-Next NVFP4, TP4, current main (1c11d10150).
# Launch parameters follow recipes/flash-next-nvfp4-v100-tp4-mtp.md and the
# README row measured by receipt evidence-07c26a2 (temperature 0, 400 tokens).
# The only delta from that receipt's launcher is moe_align_sm70, which the
# recipe has selected since 2026-11-05 (ef2662a8de).
#
# Usage: GPUS=4,5,6,7 PORT=11436 ./issue219_launch.sh
#        DISABLE_CUSTOM_AR=1 ...  # NCCL arm for the allreduce A/B
#        CONFIG_OUT=/path/config.txt ...  # record the resolved launch
set -euo pipefail

SITE=/nfs/lambda_stor_01/homes/pding
SIF=$SITE/qlab/env/cuda_12.8.0-devel-ubuntu24.04.sif
ENV_BIN=$SITE/qlab/env/conda/sglang-v100-py312/bin
PY=$ENV_BIN/python
GPUS=${GPUS:?set GPUS, for example 4,5,6,7}
PORT=${PORT:-11436}
COMPONENTS=attention_v100,fp4_moe,moe_align_sm70,ple_sm70,spec_sm70,fp8_dense,fp8_lm_head,fp8_dense_quant
DISABLE_CUSTOM_AR=${DISABLE_CUSTOM_AR:-0}
MEM_FRACTION=${MEM_FRACTION:-0.82}
# lambda6's root filesystem is small; keep every JIT cache on the NFS tree.
CACHE=${CACHE_ROOT:-$SITE/issue219/cache}
mkdir -p "$CACHE/home" "$CACHE/tmp" "$CACHE/torch_extensions" "$CACHE/triton"

SERVER_ARGS=(
  --model-path "$SITE/qlab/models/Qwen3.8-Flash-Next-NVFP4"
  --dtype float16 --quantization modelopt_fp4
  --moe-runner-backend marlin
  --tp-size 4 --ep-size 1
  --attention-backend qsa_v100
  --linear-attn-prefill-backend triton
  --linear-attn-decode-backend triton
  --sampling-backend pytorch
  --ple-offload-embedding
  --context-length 8192 --page-size 64
  --mem-fraction-static "$MEM_FRACTION"
  --max-running-requests 1 --chunked-prefill-size 4096
  --speculative-algorithm NEXTN
  --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4 --speculative-use-rejection-sampling
  --cuda-graph-backend-prefill=disabled
  --port "$PORT"
)
if [ "$DISABLE_CUSTOM_AR" = "1" ]; then
  SERVER_ARGS+=(--disable-custom-all-reduce)
fi

if [ -n "${CONFIG_OUT:-}" ]; then
  {
    echo "commit=$(cat "$SITE/issue219-main/COMMIT")"
    echo "host=$(hostname)"
    echo "gpus=$GPUS"
    echo "port=$PORT"
    echo "disable_custom_ar=$DISABLE_CUSTOM_AR"
    echo "mem_fraction=$MEM_FRACTION"
    echo "components=$COMPONENTS"
    echo "pythonpath=$SITE/pin-site:$SITE/issue219-main"
    echo "cache=$CACHE"
    printf 'server_arg=%s\n' "${SERVER_ARGS[@]}"
    echo "env SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=${SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2:-<unset, default true>}"
    echo "env SM70_MOE_DECODE_GEMV=${SM70_MOE_DECODE_GEMV:-<unset, default 1>}"
    echo "env SGLANG_SM70_FP4_MOE_SPLIT_K=${SGLANG_SM70_FP4_MOE_SPLIT_K:-<unset, default per-shape>}"
    echo "env SGLANG_SM70_FP4_MOE=${SGLANG_SM70_FP4_MOE:-<unset, default auto>}"
  } > "$CONFIG_OUT"
fi

exec singularity exec --nv --tmpdir "$CACHE/tmp" --bind /nfs:/nfs "$SIF" env \
  HOME="$CACHE/home" \
  TMPDIR="$CACHE/tmp" \
  TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions" \
  TRITON_CACHE_DIR="$CACHE/triton" \
  PYTHONPATH="$SITE/pin-site:$SITE/issue219-main" \
  SGLANG_PLUGINS=restomod \
  SGLANG_RESTOMOD_COMPONENTS="$COMPONENTS" \
  SM70_MARLIN_DIR="$SITE/sm70_marlin" \
  SGLANG_SM70_TURBOMIND_OPS_PATH="$SITE/_sm70_turbomind_v100.so" \
  SM70_TURBOMIND_SO="$SITE/_sm70_turbomind_v100.so" \
  TORCH_CUDA_ARCH_LIST=7.0 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  SGLANG_MAMBA_CONV_DTYPE=float16 \
  SGLANG_MAMBA_SSM_DTYPE=float16 \
  CUDA_HOME=/usr/local/cuda \
  TVM_NVCC=/usr/local/cuda/bin/nvcc \
  PATH="/usr/local/cuda/bin:$ENV_BIN:$PATH" \
  CUDA_VISIBLE_DEVICES="$GPUS" \
  "$PY" -m sglang.launch_server "${SERVER_ARGS[@]}"
