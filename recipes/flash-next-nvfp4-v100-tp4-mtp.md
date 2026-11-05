# Qwen3.8-Flash-Next-NVFP4 on 4x V100-SXM2, TP4 + MTP

| | |
|---|---|
| Model | `Qwen3.8-Flash-Next-NVFP4` (512 experts, NVFP4, QSA + GDN) |
| Hardware | 4x Tesla V100-SXM2-32GB (sm70), one NVLink island (0-3) |
| Topology | TP4, `--ep-size 1` |
| Context | 4,096 |
| Engine | upstream `v0.5.22 @ baa9befa` + `sglang_restomod` |
| Components | `attention_v100`, `fp4_moe`, `ple_sm70`, `spec_sm70` |
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
--speculative-use-rejection-sampling
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

**MTP with non-greedy sampling needs two changes.** The sampling backend flag is
necessary but not sufficient.

1. The EAGLE verify path calls FlashInfer directly, regardless of
   `--sampling-backend`:

   ```
   eagle_worker_v2.py:1908 verify
   eagle_worker_common.py:698 run_eagle_verify
   eagle_utils.py:984 eagle_sample -> top_k_renorm_prob
   flashinfer/sampling.py:2095 -> get_sampling_module
   RuntimeError: FlashInfer requires GPUs with sm75 or higher.
   ```

   The `spec_sm70` component replaces `top_k_renorm_probs` and
   `top_p_renorm_probs` with torch implementations on SM70.

2. With that fixed, the next call is the tree sampler, whose JIT build fails
   against this image's CCCL:

   ```
   flashinfer/data/include/flashinfer/sampling.cuh(40): error:
   namespace "cuda" has no member "maximum"
   ```

   Launch with `--speculative-use-rejection-sampling`. That selects the Triton
   chain sampler (`chain_speculative_sampling_triton`) instead of the JIT tree
   sampler. With `--speculative-eagle-topk 1` the tree is a chain, so this is
   the natural equivalent.

   The flag is also load-bearing for a second reason. The tree sampler
   (`tree_speculative_sampling_target_only`) validates `target_probs` as
   float32; the chain sampler reads whatever dtype it is given. Before
   `sm70_sampling` mirrored FlashInfer's per-function dtypes, a chained
   `top_k` -> `top_p` could hand it fp16. Both are fp32 now, but the chain
   sampler is still the path that was measured.

Non-greedy measurement (temperature 1, top-p 0.95, top-k 20, 128 output):
100.89 / 103.91 / 78.24 tok/s, accept len 3.27—3.83.

`--ple-offload-embedding` is required: the n-gram table does not fit on four
cards. The offloaded table is fp8, and the pinned gather kernel cannot compile
on SM70; the `ple_sm70` component replaces it.

## Split-K

The two expert GEMMs of a layer have different K, and split-K helps one while
hurting the other. `sglang_restomod` picks it per launch through
`SGLANG_SM70_FP4_MOE_SPLIT_K`:

| value | behaviour |
|---|---|
| unset, `per-shape` | 2 for a decode call (`size_m` at or below 8) whose K is at least 2560 and 64-aligned, 1 otherwise; prefill keeps 1 |
| `off` | leave `SM70_MARLIN_MOE_SPLIT_K` as the extension finds it |
| `1`, `2`, `4`, `8` | pin that value for every call |

The decode gate is the cut `_record_moe_stats` uses. Split-K buys back a
half-empty wave, and a prefill chunk has none to fill. Nothing here measures
prefill; the gate is what keeps the rule off the unmeasured case.

Trace measurement, ten decode steps at 4,096 context, GPUs 0-3 with the other
cards idle, device time per launch:

| call | N | K | split-K 1 | split-K 2 | split-K 4 |
|---|---:|---:|---:|---:|---:|
| gate_up | 320 | 2560 | 130.7 us | 102.7 us | 102.1 us |
| down | 2560 | 160 | 44.7 us | 76.8 us | 113.3 us |

gate_up plateaus at 2: 400 CTAs fill the machine, so a fourth share buys
nothing, and a per-split K of 640 is no slower than 1280. down has the opposite
slope at every step. At 48 calls each per step the MoE term is 6.98-7.21 ms for
the rule against 8.42-8.46 at split-K 1 and 8.59-8.62 at split-K 2, and the
step follows: 29.76 and 30.10 ms against 31.08 and 31.14.

These are trace numbers. Decode tok/s for this configuration has not been taken
under the receipt protocol.

**Numerics.** Only gate_up runs at split-K 2, so the difference enters before
the activation and every expert output downstream carries it. On the
`sm70_fp4_moe.check` case (K=256, N=256, output magnitude 56), split-K 1
matches the dequantized reference bit-exactly and split-K 2 differs from it by
at most 3.125e-2, which is 1.1 fp16 ulp at that magnitude. Two identical
prompts can therefore diverge at the first token where the two accumulation
orders round differently; a diff between runs is not by itself a bug report.

One measurement to not repeat: a single earlier trace showed the TP all-reduce
behind the down call at 2.81 ms/step under split-K 2 against 0.59 at split-K 1,
and that was written up as a second cost of splitting down. A repeat of the
same configuration put it back at 0.68, so it was a transient, not the
mechanism. The rule rests on the GEMM numbers only.

## Measured

4× V100-SXM2, TP4 on one NVLink island (0-3), `--ple-offload-embedding`,
`--mem-fraction-static 0.82`, MTP3/4. `scripts/bench.py`, 128 output tokens,
two reps each. These are the runs the signed receipt
`evidence-6294562-nvfp4` covers:

| Sampling | Prompt tokens | Decode |
|---|---|---|
| temperature 0 | 993 | 87.27 / 87.24 tok/s |
| temperature 0 | 3473 | 72.43 / 90.68 tok/s |
| temp 1, top-p 0.95, top-k 20 | 74 | 81.83 / 88.06 tok/s |

Accept length comes from the server log, not from `bench.py`: 3.12—3.20 on the
greedy runs, 3.27—3.83 on the sampled ones.

Two earlier figures are **not** in the receipt and must not be quoted as if
they were: 44.05 tok/s with MTP off, and 88.80 tok/s at a 74-token prompt.

Compare against FP8 TP8/EP8 at the same sampling mode. Greedy FP8 on 8 cards is
98.3—106.2 tok/s at accept 3.48—3.70, which is 34.8—35.4 ms per step. NVFP4
greedy is about 35.2 ms per step. Per-step time is at parity; the throughput
difference tracks acceptance, not kernel time.

The 73.9 tok/s / accept 2.83 row in
`recipes/flash-next-fp8-v100-tp8ep8-mtp.md` is the default-sampling run. It is
not comparable to these temperature-0 numbers, and mixing the two reads as a
larger gap than exists.

Memory: about 23 GiB resident per card; `--mem-fraction-static 0.85` leaves
only 0.89 GiB free, so MTP runs at 0.82.

## The dense path in block FP8 (2026-11-04)

Add `fp8_dense,fp8_lm_head,fp8_dense_quant` to `SGLANG_RESTOMOD_COMPONENTS`.
The checkpoint stores every non-expert weight in BF16, and at M=1..4 those
GEMMs are bandwidth-bound, so the byte count is the cost. `fp8_lm_head`
quantizes the head after load; `fp8_dense_quant` does the same for every dense
linear whose shape is a multiple of the 128 block.

Measured with `profile_fn_decode.py`, 4k prefix, 10 decode steps, TP-0/TP-1
traces:

| build | step | cutlass 16x16 | cutlass 32x32 | TurboMind GEMM |
|---|---:|---:|---:|---:|
| base | 32.79 ms | 8.45 ms | 2.40 ms | 0 |
| `fp8_lm_head` | 31.54 ms | 6.57 ms | 2.41 ms | 0.84 ms |
| + `fp8_dense_quant` | **31.08 ms** | 4.56 ms | 2.44 ms | 2.29 ms |

That is -1.71 ms per step, -5.2%. The bench cannot resolve it on this box
(82.75 / 97.79 / 84.28 tok/s against an 82-84 baseline across three fresh
reps), so the trace is the evidence.

Only 49 linears per rank are converted: the rest of the dense weights have an N
that is not a multiple of 128 (864, 1076, 320, 24, 1), which the TurboMind
kernel's contract does not admit. `--quantization modelopt_fp4` keeps the
experts out of this path; they stay on the marlin NVFP4 GEMM.


## Known gaps

- The first forward pass JIT-compiles the `moe_sum_reduce` replacement; the
  log warns `device-loaded after serving started`.
- The checkpoint's FP4 nibble order is validated only indirectly, by
  generation quality. No independent dequantization comparison exists yet.
- Accept length is 3.12, below the FP8 greedy run's 3.48—3.70. Cause not
  confirmed.
- Contexts above 4k are untested.
