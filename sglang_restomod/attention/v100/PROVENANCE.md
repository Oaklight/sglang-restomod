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

The graph metadata delegation is component code (`graph_metadata.py`). It is
injected at import time through `FlashAttnV100Backend` and skips itself when
the engine already defines the method, so a patched engine and a stock one
behave the same.

The rest of the backend (decode/extend delegation, E5M2 KV detection) is still
read from the engine checkout; extracting it is the remaining part of E5.1.

## What is vendored now (2026-10-28)

The backend itself is in the component, not read from the engine checkout:

| file | source |
|---|---|
| `flash_attn_v100_backend.py` | fork `73abfe1` + lab patches |
| `qwen_sparse_attn_backend.py` | same |
| `tilelang/` (11 modules) | same, with the package path rewritten to `sglang_restomod.attention.v100.tilelang` |
| `kernels/sm70_longctx_decode.cu` | same, **re-synced** to include `sm70-decode-occ2.patch` |
| `kernels/sm70_nvfp4_moe_decode.cu` | same |

`sm70_longctx_decode.cu` was first vendored before the two-CTA decode patch
landed in the lab tree, so the component would have shipped the slower kernel.
It now carries `sm70-decode-occ2.patch`: the batched tile loader, the
warp-specialized pipeline and the `kCtas` template, selectable at runtime with
`SGLANG_SM70_DECODE_VARIANT=serial|occ2|pipelined` (default `occ2`). That is
worth -28% per layer at 256k on e5m2 and +3.7% end-to-end decode.

Still read from the engine checkout:

- `sglang.kernels.ops.attention.sm70_qsa_combine` — the Flash-Next sparse path
  needs it. Vendoring it is what makes the component independent of the fork.
- `sglang.srt.environ.envs` for `SGLANG_SM70_QSA_COMBINE`.

Not yet done: the component has not been run on a GPU through its own module
path. Every V100 number in `recipes/` and `docs/runs/` was produced by the fork.
