# Quantization dispatch on SM70

This note fixes the contract between a checkpoint format and the kernel that
runs it on V100. It exists so that FP8 and NVFP4 support can grow side by side
without either path special-casing model code.

## Principle

One model implementation, several weight formats. The checkpoint's
`quantization_config` selects a quant method; the quant method selects a kernel
behind a device predicate. The predicate always validates the **weight layout**,
not only the quant method name.

## Format to kernel matrix

| Checkpoint format | Dense kernel | MoE kernel | Status |
|---|---|---|---|
| `fp8`, `weight_block_size=[128,128]`, dynamic activations | `sm70_turbomind_fp8` (W8A16) | none yet, tracked in #19 | Dense validated on 27B-FP8 (#17) |
| `modelopt_fp4` (NVFP4) | Marlin v100 (W4A16) | SM70 NVFP4 MoE decode | Flash-Next NVFP4 measured 85-100 tok/s |
| `compressed-tensors` WNA16 (INT4 W4A16) | Marlin v100 after the capability patch | `moe_wna16` / Marlin MoE | 27B W4A16 measured 8.6 tok/s |
| unquantized FP16 | cuBLAS | `SM70FP16MoEMethod` | Baseline fallback |

KV dtype is orthogonal: the attention backend accepts fp16, and the tilelang
FA-V100 decode kernels have an E5M2 bridge. E5M2 is **not usable yet**: the
partial-decode kernel it enables corrupts output (#51). fp16 is the validated
choice.

## Selection rules

1. **FP8 dense.** `can_use_sm70_turbomind_fp8(weight_block_size)` accepts
   exactly `[128,128]` on compute capability 7.0 and selects the TurboMind
   W8A16 kernel. Any other block size falls back to the Marlin FP8 path.
   `SGLANG_SM70_FP8_BACKEND=auto|turbomind|marlin` pins or forbids the choice;
   `turbomind` raises instead of silently falling back.
2. **FP8 MoE.** `Fp8MoEMethod` must dispatch to the SM70 block-FP8 expert
   kernel. Until that kernel is bound (#19) and wired (#20), the current
   behaviour is a generic-runner fallthrough, which is neither correct nor
   attributable; the target contract is an explicit error rather than silent
   dequantization.
3. **NVFP4.** `modelopt_fp4` selects Marlin v100 for dense layers on SM70
   (`--fp4-gemm-backend marlin`). The SM70 NVFP4 MoE decode kernel is used when
   the checkpoint layout matches; otherwise the Marlin MoE path applies.
4. **INT4 W4A16.** `compressed-tensors` WNA16 is admitted at capability 7.0
   only when the Marlin v100 backend is available; the stock floor of 8.0 is a
   portability bug, not a hardware limit.
5. **Unquantized.** On SM70, `FusedMoE` swaps in `SM70FP16MoEMethod` when
   `params_dtype` is fp16 and no quant method was selected.

## Fallback policy

- **Never dequantize a large checkpoint at load.** Doubling the expert
  footprint does not fit 8xV100-32GB for Flash-Next; the kernel must read the
  packed format.
- **A missing kernel is an error.** The message must name the format and the
  environment variable that can pin the path.
- **Fallbacks are explicit and logged once.** Ambient fallbacks make
  performance results unattributable.

## Priority

| Path | Role | Why |
|---|---|---|
| FP8 block | Throughput and official checkpoints | Both target models publish `[128,128]` FP8; one dense kernel covers both |
| NVFP4 | Memory footprint | W4A16 on V100; no official Qwen checkpoint |
| INT4 W4A16 | Compatibility | Works, but the dense path is not competitive |

Weight format was not the decode bottleneck: after the CUDA graph metadata fix
the same FP8 run reached 30-35 tok/s (#31). Format work is still needed on the
MoE path, where no SM70 kernel exists.

## FP8 specifics

Both target checkpoints are block FP8:

| Model | Block | Excluded from quantization |
|---|---|---|
| Qwen3.8-27B-FP8 | `[128,128]` | visual blocks, `lm_head`, embeddings, norms |
| Qwen3.8-Flash-Next-FP8 | `[128,128]` | `lm_head`, embeddings, hyper-connections, every `linear_attn` projection, `mlp.gate`, shared expert |

Environment:

| Variable | Values | Effect |
|---|---|---|
| `SGLANG_SM70_FP8_BACKEND` | `auto`, `turbomind`, `marlin` | Dense kernel choice; `turbomind` raises instead of falling back |
| `SGLANG_SM70_FP8_MOE` | `auto`, `on`, `off` (planned, #20) | Expert kernel choice |
| `SGLANG_SM70_FP16_MOE` | `1`, `0` | `SM70FP16MoEMethod` for unquantized FP16 experts |
| `SGLANG_V100_DECODE_CUDA` | `1`, `0` | E5M2 partial decode kernel; leave off until #51 is fixed |

Known limits:

- The shared expert and router stay BF16 and run without tensor cores; whether
  to requantize them is #33.
- CUDA graph capture needs the fork patch
  `patches/sglang-sxm2/graph-metadata.patch`; without it capture fails on
  `swa_out_cache_loc`. See the graph run record.
- E5M2 KV corrupts output with the default partial kernel (#51).
- Flash-Next PLE is offloaded to host memory by default; it is not a weight
  format concern.

Receipts: see `docs/gpu-receipts.md`. The two recorded runs are
`docs/runs/2026-10-25-qwen3.8-27b-fp8-lambda12.md` and
`docs/runs/2026-10-25-qwen3.8-27b-fp8-cuda-graph-lambda12.md`.

## Registration

Kernels enter through the additive plugin in `sglang_restomod/`. A change to an
upstream registration point becomes a patch with a manifest entry. Vendoring a
whole project tree is out of scope; select the kernel and its call site.

## Test matrix

Every cell below needs a run and a receipt before it is called validated.

| Model | FP8 | NVFP4 | INT4 W4A16 |
|---|---|---|---|
| Qwen3.8-27B | validated (#17); 30-35 tok/s with the graph fix (#31) | open (#24) | measured, not competitive |
| Qwen3.8-Flash-Next | blocked on #19 | measured (85-100 tok/s) | n/a |

## Open gaps

- Block-FP8 W8A16 MoE kernel and its binding (#19).
- 1Cat NVFP4 dense kernel extraction (#37) for the 27B memory path.
- Shared expert and router stay BF16 in the FP8 checkpoint; whether to requantize
  them is #33.

## Why `[128,128]`

Both official checkpoints declare `weight_block_size=[128,128]`, and the SM70
TurboMind kernel accepts exactly that granularity. A per-tensor FP8 checkpoint
would need a different kernel; requantizing to block FP8 offline is cheaper than
adding one.
