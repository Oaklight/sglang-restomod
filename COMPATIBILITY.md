# Compatibility matrix

Two components ship (`fp8_moe`, `attention_v100`); both are opt-in and the
patch queue is empty. A row becomes "validated" only after a recorded run.

| Hardware | Architecture | Path | Status |
|---|---|---|---|
| Tesla V100-SXM2 (2x) | sm70 | `dg1kjd/sglang-sxm2` @ `73abfe1` + 3 local patches, block FP8 | Validated 2026-10-25: Qwen3.8-27B-FP8, TP2, 28K — [run](docs/runs/2026-10-25-qwen3.8-27b-fp8-lambda12.md) |
| Tesla V100-SXM2 (2x) | sm70 | same, CUDA graph + MTP | Validated 2026-10-26: 27B decode 67-79 tok/s at 1-2.7K, 42-45 at 8K — [run](docs/runs/2026-10-26-qwen3.8-27b-mtp-lambda12.md) |
| Tesla V100-SXM2 (8x) | sm70 | same + `fp8_moe`, TP8/EP8, CUDA graph + MTP | Validated 2026-10-26: Qwen3.8-Flash-Next-FP8 end to end, 83-86 tok/s decode — [lab record](https://github.com/Oaklight/lambda-v100-lab/blob/main/benchmarks/flash-next-fp8-sglang-lambda4-tp8.md) |
| Tesla T4 | sm75 | upstream SGLang 0.5.22 | Planned |
| RTX 4070 Super | sm89 | upstream SGLang 0.5.22 | Planned |
| H100 | sm90 | upstream SGLang 0.5.22 | Planned |
| DGX Spark | sm121 / aarch64 | upstream SGLang 0.5.22 | Planned |

Notes:

- Quantization dispatch and FP8 specifics: `docs/quantization-dispatch.md`.
- Upstream SGLang supports sm75 and newer; it does not target sm70. V100 needs
  selected components from this layer.
- A validated row records: hardware, upstream pin, component versions, model,
  topology, context, date.
- The 2026-10-25 row is a fork run recorded for evidence; the same path is not
  yet available through this layer's components.
- With the CUDA graph metadata fix in the fork, the same run reaches 30-35
  tok/s decode: [graph run](docs/runs/2026-10-25-qwen3.8-27b-fp8-cuda-graph-lambda12.md).
- The `fp8_moe` component (block-FP8 expert GEMM) was validated numerically
  on 2026-10-25 and loads the full Flash-Next checkpoint on 8x V100 (17.29 GB
  per rank, TP8/EP8). See the
  [expert GEMM run](docs/runs/2026-10-25-qwen3.8-flash-next-fp8-sm70-moe-lambda4.md).
- The QSA decode blocker recorded earlier is **resolved**: the SM70 kernels were
  templated over the query-head group, so the 3-head layout at TP8 no longer
  falls back to a FlashAttention build the V100 image lacks. Flash-Next now
  generates end to end, and with CUDA graph and MTP reaches 83-86 tok/s.
- MTP works on both target models from the checkpoint's own head, with no draft
  model: `--speculative-algorithm NEXTN`. On Flash-Next the draft token count
  cannot exceed 4, because QSA's pending index-key ring holds one compress
  group; the 27B has no such ring.
- `fp8_e5m2` and `fp8_e4m3` KV are **validated**, and e4m3 is the recommended
  one. The corruption recorded on 2026-10-25 was two uninitialised reads in the
  decode partial (#51), not the format; after that fix and #79, e5m2 carries the
  MTP target verify and reaches 256k. e4m3 is the same size and speed with a
  smaller perturbation of the model's own next-token distribution — see the
  recipe's "KV dtype" section. The
  [E5M2 run](docs/runs/2026-10-25-qwen3.8-27b-fp8-e5m2-kv-lambda12.md) is the
  record of the pre-fix failure.
