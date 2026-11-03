# Provenance: quant/sm70_fp4_moe

## Source

| Field | Value |
|---|---|
| Repository | `zhinianqin/marlin_v100` |
| Commit | `3f16d442cdb4c24dd225bbec196c982a54d9a31c` |
| License | Apache-2.0 by lineage (vLLM and Marlin); the repository ships no `LICENSE` file |
| Built artifacts | `_C.abi3.so`, `_moe_C.abi3.so` (sm_70) |

Patches applied before the build, taken from `heislera763/sglang-v100-plus`:

- `marlin-v100-nvfp4.patch` — reduced NVFP4 build profile.
- `marlin-v100-prefill-gather.patch` — cached gather for the prefill tile.
- `marlin-v100-block-fp8-prefill.patch` — not used by NVFP4; applied for a future block-FP8 selection.

Build:

The shipped `.so` files were built inside `cuda_12.8.0-devel-ubuntu24.04.sif`
with torch 2.9.1+cu128 and CUTLASS v4.2.1, by this command — the same one the
lab keeps as `marlin_build.sh`, reproduced here so the recipe is not one host's
private script:

```bash
singularity exec --nv cuda_12.8.0-devel-ubuntu24.04.sif env \
  CUDA_HOME=/usr/local/cuda \
  TORCH_CUDA_ARCH_LIST=7.0 \
  CUTLASS_DIR=/path/to/cutlass \
  MAX_JOBS=32 CMAKE_BUILD_PARALLEL_LEVEL=32 \
  CMAKE_ARGS="-DMARLIN_V100_NVFP4_ONLY=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_CUDA_FLAGS=-gencode=arch=compute_70,code=sm_70" \
  PATH="$ENV_BIN:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
  "$ENV_BIN/python3.12" setup.py build_ext --inplace
```

The three patches listed above are applied to the source tree before the build;
`git apply --check --reverse` against the built tree confirms it, and the same
check reports that the fourth patch in that directory,
`marlin-v100-sm70-mxfp8.patch`, is **not** applied in the tree that produced the
shipped binaries.

## What this component uses

- `_C.gptq_marlin_repack` at load time.
- `_moe_C.moe_wna16_marlin_gemm` for decode and prefill.

The conversion code (expert repack, scale encoding, call bridge) is ported from
`sglang-v100-plus` (`v100_plus/sglang_v100_plus/quantization.py`), which is
Apache-2.0.

## Not vendored

The extension source and its build output stay outside this repository. The
component only requires `SM70_MARLIN_DIR` to point at the built libraries.
