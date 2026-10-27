# Qwen3.8-27B-FP8 on 2x V100-SXM2, TP2, CUDA graph + MTP

| | |
|---|---|
| Model | `Qwen3.8-27B-FP8` (block FP8, hybrid dense + GDN) |
| Hardware | 2x Tesla V100-SXM2-32GB (sm70) |
| Topology | TP2, no EP |
| Context | 32,768 |
| Engine | `dg1kjd/sglang-sxm2` @ `73abfe1` + `sm70-fixes.patch` + `graph-metadata.patch` |
| Components | none (the dense path needs no restomod component) |
| Date | 2026-10-26 |

## Flags

```
--dtype float16 --quantization fp8
--attention-backend tilelang_fa_v100
--linear-attn-prefill-backend tilelang --linear-attn-decode-backend triton
--kv-cache-dtype auto                 # fp16; e5m2 is not validated, see #51
--tp-size 2 --context-length 32768
--mem-fraction-static 0.85 --max-running-requests 4
--chunked-prefill-size 4096 --max-prefill-tokens 8192
--disable-prefill-cuda-graph          # decode graph stays on
--speculative-algorithm NEXTN
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

## Environment

Complete list, as the launcher sets them. The Mamba dtypes are not optional:
the GDN layers fail or drift without them.

```
TORCH_CUDA_ARCH_LIST=7.0
FLASHINFER_DISABLE_VERSION_CHECK=1
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage
SGLANG_MAMBA_CONV_DTYPE=float16
SGLANG_MAMBA_SSM_DTYPE=float16
SGLANG_SM70_FORCE_FP16=1
SGLANG_SM70_DENSE_GEMV=1
SGLANG_SM70_QWEN_FUSIONS=1
SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
SGLANG_NUMA_BIND_V2=0
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
```

The launcher also passes `--reasoning-parser auto --tool-call-parser auto
--sleep-on-idle`, which are serving preferences rather than performance flags.

## Measured

Decode tok/s, graph on, same harness for every row:

| Context | MTP off | MTP on | Accept len |
|---|---|---|---|
| 380 tok | 34.70 | **67.6-78.8** | 2.61-3.05 |
| 2.7k tok | 33.67 | **66.9-74.5** | 3.12-3.46 |
| 8.2k tok | 34.4 | **45.7** | 3.05 |
| 12.7k tok | 33.5 | 34.8 | 2.91 |
| 17.0k tok | **32.8** | 30.7 | 3.05 |
| 25.6k tok | **31.4** | 23.3 | 3.05 |
| 32.4k tok | **30.4** | 20.5 | 3.20 |

Prefill, warm process: 1,680-1,708 tok/s at 2k, 1,331-1,364 at 8k.

## Pick the spec setting by context

**MTP is now the right setting at every context**, once the draft rows are
served through the 1-row split-KV decoder
(`patches/sglang-sxm2/draft-extend-native.patch` in the lab repository, enabled
with `SGLANG_V100_DRAFT_NATIVE=1`).

| context | no MTP | MTP, draft-extend kernel | MTP, per-row decoder |
|---:|---:|---:|---:|
| 2k | 34.5 | 59.2 / 63.3 | 66.4 / 71.9 |
| 8k | 34.1 | 38.1 / 43.3 | 56.2 / 67.5 |
| 16k | 33.7 | 25.3 / 37.6 | 55.3 / 58.8 |
| 24k | 33.3 | 21.4 / 20.6 | 51.0 / 54.3 |
| 32k | 32.8 | 15.4 / 16.3 | 47.7 / 49.8 |

The old crossover was the draft-extend kernel, not MTP: it re-read the prefix
once per draft row, so its cost grew with context while the accepted tokens per
verify stayed flat. 1Cat vLLM, measured on the same checkpoint and contexts,
never showed the collapse (51.6 tok/s at 2k, 41-46 at 32k), which is what
identified it as a bug in our stack.

```
# every context: draft chain on, and the per-row decoder
SGLANG_V100_DRAFT_NATIVE=1
--speculative-algorithm NEXTN
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

The per-row decoder needs **fp16 KV**. It calls the CUDA decode partial, and
with an e5m2 cache that partial corrupts short contexts (#51), so do not pair
`--kv-cache-dtype fp8_e5m2` with this setting. e5m2 is the wrong choice here
anyway: with MTP it measured 4.9 tok/s at 32k against 20.5 for fp16, and without
MTP the fp16 partial is the faster decode path.

## Notes

- The checkpoint carries its own MTP head (`mtp_num_hidden_layers=1`), so no
  draft model is needed. There is no QSA ring here, so the draft chain is not
  capped at 4 tokens the way it is on Flash-Next.
- Decode graph capture needs `graph-metadata.patch`; without it the graph runner
  calls `init_forward_metadata_out_graph`, the base class leaves it a no-op, and
  capture dies on `swa_out_cache_loc`.
- The first request after startup is much slower while prefill kernels JIT. A
  benchmark has to warm up or it quotes the warmup.

Run records: [docs/runs/2026-10-26-qwen3.8-27b-mtp-lambda12.md](../docs/runs/2026-10-26-qwen3.8-27b-mtp-lambda12.md),
[docs/runs/2026-10-26-qwen3.8-27b-kv-dtype-mtp-crossover.md](../docs/runs/2026-10-26-qwen3.8-27b-kv-dtype-mtp-crossover.md).
Launcher: `scripts/serve_sglang_27b_v100.sh` in the lab repository.
