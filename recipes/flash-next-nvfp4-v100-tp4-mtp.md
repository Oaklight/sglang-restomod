# Qwen3.8-Flash-Next-NVFP4 on 4x V100-SXM2, TP4 + MTP

| | |
|---|---|
| Model | `Qwen3.8-Flash-Next-NVFP4` (512 experts, NVFP4, QSA + GDN) |
| Hardware | 4x Tesla V100-SXM2-32GB (sm70), one NVLink island (0-3) |
| Topology | TP4, `--ep-size 1` |
| Context | 4,096 |
| Engine | upstream `v0.5.22 @ baa9befa` + `sglang_restomod` |
| Components | `attention_v100`, `fp4_moe`, `ple_sm70` |
| External | `marlin_v100` extension (`SM70_MARLIN_DIR`) |
| Date | 2026-11-01 |

## Why this shape

The FP8 checkpoint is about 120 GB, so it needs eight cards and spans both
NVLink islands. The NVFP4 checkpoint is about 126 GB on disk but 4-bit on the
device, so four cards fit, and four cards 0-3 are one island.

Expert parallelism is not usable here: the SM70 Marlin kernel indexes this
rank's weights by global expert id, and EP marks remote routes with -1. Use
`--ep-size 1`.

## Prerequisites

1. Build the extension from `zhinianqin/marlin_v100@3f16d442` with the patches
   listed in `sglang_restomod/quant/sm70_fp4_moe/PROVENANCE.md`, inside the
   CUDA 12.8 image, with CUTLASS v4.2.1:

   ```sh
   CUTLASS_DIR=/path/to/cutlass TORCH_CUDA_ARCH_LIST=7.0 \
     CMAKE_ARGS="-DMARLIN_V100_NVFP4_ONLY=ON -DCMAKE_CUDA_ARCHITECTURES=70" \
     python setup.py build_ext --inplace
   ```

2. Put `_C*.so` and `_moe_C*.so` in one directory and set
   `SM70_MARLIN_DIR` to it.

## Flags

```
--dtype float16 --quantization modelopt_fp4
--moe-runner-backend marlin
--tp-size 4 --ep-size 1
--attention-backend qsa_v100
--linear-attn-prefill-backend triton --linear-attn-decode-backend triton
--sampling-backend pytorch
--ple-offload-embedding
--context-length 4096 --page-size 64
--mem-fraction-static 0.82
--max-running-requests 1 --chunked-prefill-size 4096
--speculative-algorithm NEXTN
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--cuda-graph-backend-prefill=disabled
```

`--sampling-backend pytorch` is required for non-greedy requests. The default
FlashInfer sampling backend refuses to JIT below sm75:

```
RuntimeError: FlashInfer requires GPUs with sm75 or higher.
Detected TARGET_CUDA_ARCHS=[(7, '0')].
```

Greedy requests (temperature 0, top-k 1) never reach that kernel, so a
temperature-0 benchmark passes while a default-sampling request crashes the
server.

**MTP with non-greedy sampling still fails.** The flag is necessary but not
sufficient: the EAGLE verify path calls FlashInfer directly, regardless of the
sampling backend:

```
eagle_worker_v2.py:1908 verify
eagle_worker_common.py:698 run_eagle_verify
eagle_utils.py:984 eagle_sample -> top_k_renorm_prob
flashinfer/sampling.py:2095 -> get_sampling_module
RuntimeError: FlashInfer requires GPUs with sm75 or higher.
```

So the measured MTP numbers are temperature 0. Serving MTP with sampling needs
a non-FlashInfer `top_k_renorm_prob` on SM70; that is not implemented yet.

`--ple-offload-embedding` is required: the n-gram table does not fit on four
cards. The offloaded table is fp8, and the pinned gather kernel cannot compile
on SM70; the `ple_sm70` component replaces it.

## Measured

`scripts/bench.py`, prompt 74 tokens, 128 output, temperature 0, three reps:

| Config | Decode | accept len |
|---|---|---|
| no MTP | 44.05 tok/s | — |
| MTP3/4 | **88.80 tok/s** | 3.12—3.20 |

Per-step time is about 35.2 ms, versus 32.5 ms for the FP8 TP8/EP8 run at
accept 3.48. The remaining throughput gap is acceptance, not kernel time.

Memory: about 23 GiB resident per card; `--mem-fraction-static 0.85` leaves
only 0.89 GiB free, so MTP runs at 0.82.

## Known gaps

- The first forward pass JIT-compiles the `moe_sum_reduce` replacement; the
  log warns `device-loaded after serving started`.
- The checkpoint's FP4 nibble order is validated only indirectly, by
  generation quality. No independent dequantization comparison exists yet.
- Accept length is 3.12, below the FP8 run's 3.48. Cause not confirmed.
- Contexts above 4k are untested.
