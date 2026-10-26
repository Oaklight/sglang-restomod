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

MTP crosses over between 12.7k and 17k: **leave it on up to ~12k, turn it off
above ~16k**. At 32k it costs 33% (20.5 versus 30.4 tok/s).

The accept length is flat across that range (2.91-3.20), so this is not an
acceptance effect. Each MTP step runs one target forward over `1 + draft`
positions, which pays the attention and the weight read once per verify instead
of once per accepted token; as KV length grows the verify step's cost grows with
it while the accepted tokens per verify stay the same, so the win shrinks and
then inverts.

Two launches, same flags otherwise:

```
# <= ~12k: draft chain on
--speculative-algorithm NEXTN
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4

# > ~16k: drop the four --speculative-* flags entirely
```

Do not use `--kv-cache-dtype fp8_e5m2` with MTP: it is corrupt (#51) and, because
MTP bypasses the decode partial that is the only reason to choose it, it is also
slower. Measured at 32k with MTP on: 4.9 tok/s versus 20.5 for fp16.

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
