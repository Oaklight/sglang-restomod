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

```bash
CUTLASS_DIR=/homes/pding/cutlass TORCH_CUDA_ARCH_LIST=7.0 \
  python setup.py build_ext --inplace
```

inside `cuda_12.8.0-devel-ubuntu24.04.sif` with torch 2.9.1+cu128 and CUTLASS v4.2.1.

## What this component uses

- `_C.gptq_marlin_repack` at load time.
- `_moe_C.moe_wna16_marlin_gemm` for decode and prefill.

The conversion code (expert repack, scale encoding, call bridge) is ported from
`sglang-v100-plus` (`v100_plus/sglang_v100_plus/quantization.py`), which is
Apache-2.0.

## Not vendored

The extension source and its build output stay outside this repository. The
component only requires `SM70_MARLIN_DIR` to point at the built libraries.

## WNA16 capability gate (2026-11-02)

`CompressedTensorsWNA16.get_min_capability()` returns 80 upstream, so a W4A16
checkpoint is rejected on Volta before the Marlin backend is consulted.
`wna16_capability.py` wraps the classmethod and answers 70 when the device is
sm70 and this component's extension is loadable; every other device keeps the
upstream 80, so nothing changes on Ampere and up.

Measured by unit test only: no W4A16 checkpoint has been served through this
path on a V100 yet, so `COMPATIBILITY.md` is unchanged. Issue #39.
