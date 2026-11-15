#!/usr/bin/env bash
# Issue #227 A/B: baseline main vs the fp8_dense_quant admission fix.
# Launch parameters are the #219 recipe launch, so only the tree differs.
#
# Usage: GPUS=0,1,2,3 TREE=/nfs/.../issue227-main ./issue227_launch.sh
set -euo pipefail

SITE=/nfs/lambda_stor_01/homes/pding
SIF=$SITE/qlab/env/cuda_12.8.0-devel-ubuntu24.04.sif
ENV_BIN=$SITE/qlab/env/conda/sglang-v100-py312/bin
PY=$ENV_BIN/python
GPUS=${GPUS:?set GPUS, for example 0,1,2,3}
PORT=${PORT:-11436}
TREE=${TREE:?set TREE to the restomod checkout under test}
MEM_FRACTION=${MEM_FRACTION:-0.78}
COMPONENTS=attention_v100,fp4_moe,moe_align_sm70,ple_sm70,spec_sm70,fp8_dense,fp8_lm_head,fp8_dense_quant
CACHE=${CACHE_ROOT:-$SITE/issue227/cache}
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

if [ -n "${CONFIG_OUT:-}" ]; then
  {
    echo "tree=$TREE"
    echo "commit=$(cat "$TREE/COMMIT" 2>/dev/null || echo unknown)"
    echo "host=$(hostname)"
    echo "gpus=$GPUS"
    echo "port=$PORT"
    echo "mem_fraction=$MEM_FRACTION"
    echo "components=$COMPONENTS"
    echo "pythonpath=$SITE/pin-site:$TREE"
    echo "cache=$CACHE"
    printf 'server_arg=%s\n' "${SERVER_ARGS[@]}"
  } > "$CONFIG_OUT"
fi

exec singularity exec --nv --tmpdir "$CACHE/tmp" --bind /nfs:/nfs "$SIF" env \
  HOME="$CACHE/home" \
  TMPDIR="$CACHE/tmp" \
  TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions" \
  TRITON_CACHE_DIR="$CACHE/triton" \
  PYTHONPATH="$SITE/pin-site:$TREE" \
  SGLANG_PLUGINS=restomod \
  SGLANG_RESTOMOD_COMPONENTS="$COMPONENTS" \
  SGLANG_SM70_FP8_DENSE_QUANT_DETAILS=1 \
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
