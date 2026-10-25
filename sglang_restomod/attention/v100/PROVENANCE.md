# Provenance: attention/v100

## Source

| Field | Value |
|---|---|
| Repository | `dg1kjd/sglang-sxm2` |
| Commit | `73abfe1dd7e7aae7a62b222a201e48f532fb298e` |
| File | `python/sglang/srt/layers/attention/flash_attn_v100_backend.py` |
| License | Apache-2.0 |
| Related | `.../attention/tilelang_fa_v100/` (decode/prefill kernels) |

## What will be selected

Only the pieces needed to run the two target models on SM70:

- the full-attention decode/extend delegation to the Triton backend;
- CUDA graph capture/replay metadata hooks (`init_forward_metadata_out_graph`);
- E5M2 KV detection used by the grouped decode path.

The QSA sparse-attention path is a separate selection decision and is not part
of this scaffold.

## CUDA graph fix

The fork's backend did not implement `init_forward_metadata_out_graph`, which
the graph runner calls instead of the legacy capture/replay hooks. Capture
failed on `swa_out_cache_loc`. The validated delegation is recorded in the lab
repository at `patches/sglang-sxm2/graph-metadata.patch`, and its effect is
measured in `docs/runs/2026-10-25-qwen3.8-27b-fp8-cuda-graph-lambda12.md`
(10.3 -> 30-35 tok/s decode).

That method becomes component code here; it does not need a patch against
upstream, because the upstream pin does not contain this backend at all.

## Status

Scaffold only. No backend code is imported yet; selecting the component logs a
no-op registration. E5.1 fills in the files and the plugin registration point.
