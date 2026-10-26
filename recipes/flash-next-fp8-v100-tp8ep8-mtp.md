# Qwen3.8-Flash-Next-FP8 on 8x V100-SXM2, TP8 + EP8, CUDA graph + MTP

| | |
|---|---|
| Model | `Qwen3.8-Flash-Next-FP8` (512 experts, block FP8 `[128,128]`, QSA + GDN) |
| Hardware | 8x Tesla V100-SXM2-32GB (sm70) |
| Topology | TP8 + EP8 (64 local experts per rank) |
| Context | 8,192 |
| Engine | `dg1kjd/sglang-sxm2` @ `73abfe1` + `sm70-fixes.patch` + `graph-metadata.patch` + `qsa-decode-h3.patch` |
| Components | `fp8_moe` |
| Date | 2026-10-26 |

## Flags

```
--dtype float16 --quantization fp8
--tp-size 8 --ep-size 8
--attention-backend tilelang_fa_v100
--linear-attn-prefill-backend tilelang --linear-attn-decode-backend triton
--kv-cache-dtype auto --context-length 8192
--mem-fraction-static 0.80 --max-running-requests 1
--chunked-prefill-size 4096 --max-prefill-tokens 8192
--speculative-algorithm NEXTN
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

## Environment

Complete list, as the launcher sets them. The Mamba dtypes are not optional:
the GDN layers fail or drift without them.

```
SGLANG_PLUGINS=restomod
SGLANG_RESTOMOD_COMPONENTS=fp8_moe
TORCH_CUDA_ARCH_LIST=7.0
FLASHINFER_DISABLE_VERSION_CHECK=1
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage
SGLANG_MAMBA_CONV_DTYPE=float16
SGLANG_MAMBA_SSM_DTYPE=float16
SGLANG_SM70_FORCE_FP16=1
SGLANG_SM70_DENSE_GEMV=1
SGLANG_SM70_QWEN_FUSIONS=1
SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
```

`SGLANG_CUSTOM_ALLREDUCE_ALGO` also accepts `2stage`; measured at 87.9 vs 86.5
tok/s on a short prompt and 80.7 vs 84.0 at 4.8k, i.e. no difference.

## Measured

| Case | Prompt | Decode | Accept len |
|---|---|---|---|
| explain | 37 tok | **82.9-86.5 tok/s** | 2.78-2.88 |
| memo | 45 tok | **67.9 tok/s** | 2.33 |
| long | 4,856 tok | **83.2 tok/s** | 2.69 |

Against the same graphed configuration without MTP (30.8-36.9 short, 25.96-36.65
at 3k) that is 2.2-2.7x. Cold 4,856-token prefill: 0.96 s, about 5,061 tok/s.

## Notes

- **EP is required.** block-128 experts cannot be TP-sharded (640/8 = 80), so
  TP8 alone leaves each rank with a partial expert.
- **The draft chain is capped at 4 tokens.** `--speculative-num-draft-tokens 6`
  fails at startup: `Qwen QSA requires speculative_num_draft_tokens <= the QSA
  compress ratio (4): the pending index-key ring holds one group`. This is
  structural, not a tuning choice.
- QSA decode needs `qsa-decode-h3.patch`, which templates the SM70 kernel over
  the query-head group so the 3-head layout at TP8 works.
- The first request after startup takes about 85 s (JIT and warmup after graph
  capture). Benchmark with a warmup request.
- Under MTP the all-reduce becomes the top cost: 12.2 of 29.1 ms per step, in
  305 collectives above 500 us. They never overlap compute. See #64.

Launcher and full measurement note: `benchmarks/flash-next-fp8-sglang-lambda4-tp8.md`
in the lab repository.
