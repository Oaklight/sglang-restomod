# The two DSV4 torch fallbacks: what a kernel has to beat

Microbenchmark, no serve. Measures the fallbacks a DSV4 backend on SM70 would
sit on, so the kernel question has a target rather than an adjective.

**Arm.** Tesla V100-SXM2-32GB, cap (7,0), torch 2.9.1+cu128, CUDA 12.8, one
card, no CUDA graphs, 10 warm-up iterations discarded, 50 repeats, median of
per-iteration CUDA events. Script: `bench_fallbacks.py` (scratch, lambda4
`/tmp/mxfp4-verify/work`). No claim about bluewhale throughput: this measures
the fallback, not the model.

## `hc_pre_torch_impl` at hc_hidden 20480 (4 x 5120), mix 24

Its four lines reproduced verbatim: `x.flatten(1).float()`, `rsqrt` of the row
mean square, `F.linear(x_flat, hc_fn) * rsqrt`. Floor = `torch.mm(x_fp32,
fn.T)` with the fp32 copy made outside the timed region, i.e. cuBLAS alone.

| M | impl ms | cuBLAS floor ms | impl/floor | impl GB/s | floor GB/s | bound |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1157 | 0.0266 | 4.35 | 1.8 | 3.1 | launch |
| 4 | 0.1198 | 0.0287 | 4.18 | 6.8 | 11.4 | launch |
| 32 | 0.1178 | 0.0266 | 4.42 | 55.7 | 98.5 | launch |
| 256 | 0.2120 | 0.0645 | 3.29 | 247.3 | 325.1 | bandwidth |
| 1024 | 0.7373 | 0.1843 | 4.00 | 284.4 | 455.1 | bandwidth |
| 4096 | 2.6173 | 0.6415 | 4.08 | 320.5 | 523.0 | bandwidth |

Reading: below M=64 the cost is flat at ~0.117 ms regardless of M — that is
four kernel launches, not arithmetic. Above M=256 it scales linearly at
320 GB/s against a ~900 GB/s part, so it is 2.8x off the memory floor; the
cuBLAS-only floor reaches 523 GB/s and is itself short of peak because a
24-output GEMM reads a large matrix for very little math.

**The number that matters for a serve:** `hc_mult` is global in the config
(`hc_mult 4`), so the prenorm runs on **all 40 layers**. At decode shapes
(M <= 48) 0.117 ms x 40 = **~4.7 ms per step** in torch (pre step only; the
post step is a separate call not measured here), all of it launch overhead.
One fused kernel could plausibly reach the floor's 0.027 ms, i.e. ~1.1
ms/step.

## `fp8_paged_mqa_logits_torch`

Contract shapes: head_dim 128, block_size 64, cache `[pages, 64, 1, 132]`,
num_heads 32, weight `[1, 32]`, page_table contiguous.

| seq_len | pages | ms | GB/s | bound |
| --- | --- | --- | --- | --- |
| 2048 | 32 | 0.2872 | 0.9 | launch/overhead |
| 8192 | 128 | 0.2898 | 3.7 | launch/overhead |

4x the cache pages, 0.7% more time: the cost is per call, not per byte. The
fallback reads at 0.9-3.7 GB/s, three orders below the part's bandwidth.

**It is not 40 calls per step.** V4.1-Flash's config names
`index_source_layer_ids [2, 8, 14, 20, 24, 28, 32, 36]` — **8 index layers**
out of 40 (`kv_source_layer_ids` names 4, and `compress_ratios` is 20x ratio 1,
18x ratio 2, 5x ratio 0 including the head/tail). So the indexer is
8 x 0.287 = **~2.3 ms/step**, against **~4.7 ms/step** for the prenorm on all
40 layers. An earlier version of this page said 11.6 ms/step by multiplying by
40; that was wrong — it assumed every layer runs the indexer. Retracted here
and in the plan: the prenorm is the larger per-step cost in torch, and both
are still overhead-bound.

## What this does not say

- Not a bluewhale number: no serve, one card, synthetic tensors, and the
  fallbacks are not what bluewhale would run if a kernel existed.
- The 8192-page case allocates and reads a cache built from zeros; the
  arithmetic is real but the values are not, which is fine for timing and
  says nothing about correctness.
- `deep_gemm_metadata` is passed as None, as the fallback ignores it
  (`_ = deep_gemm_metadata`).
