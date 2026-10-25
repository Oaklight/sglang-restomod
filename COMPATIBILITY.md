# Compatibility matrix

The baseline layer is empty: it pins SGLang and ships no V100 components. V100
support is added component by component, and a row becomes "validated" only
after a recorded run.

| Hardware | Architecture | Path | Status |
|---|---|---|---|
| Tesla V100-SXM2 | sm70 | this layer + selected components | Baseline only; no components selected |
| Tesla V100-SXM2 (2x) | sm70 | `dg1kjd/sglang-sxm2` @ `73abfe1` + 3 local patches, block FP8 | Validated 2026-10-25: Qwen3.8-27B-FP8, TP2, 28K — [run](docs/runs/2026-10-25-qwen3.8-27b-fp8-lambda12.md) |
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
  per rank, TP8/EP8). End-to-end generation is blocked in the QSA decode path,
  whose SM70 kernels only cover a `(6, 256)` query-head layout: at TP8 the
  model has 3 query heads and falls back to a FlashAttention build that the
  V100 image does not ship. See the
  [expert GEMM run](docs/runs/2026-10-25-qwen3.8-flash-next-fp8-sm70-moe-lambda4.md).
- `fp8_e5m2` KV is **not validated**: with the default fast decode kernel it
  returns corrupted output, and without that kernel it does not improve
  throughput. See the [E5M2 run](docs/runs/2026-10-25-qwen3.8-27b-fp8-e5m2-kv-lambda12.md)
  and #51.
