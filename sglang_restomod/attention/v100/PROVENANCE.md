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
measured in `docs/runs/2026-10-25-qwen3.8-27b-fp8-cuda-graph-node-b.md`
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
| `qwen_sparse_attn_backend.py` | sglang `v0.5.23` file + the fork's SM70 additions (2026-11-06, see below) |
| `tilelang/` (11 modules) | same, with the package path rewritten to `sglang_restomod.attention.v100.tilelang` |
| `tilelang/_kernels_paged.py` | **not vendored by hand**: synced from `sm70-kernels` at the pin (`[[kernels.files]]`). The dim-256 tile holds Q in a fragment and runs `block_N=64`; the source of truth is `sm70_kernels/tilelang_fa/_kernels_paged.py` |
| `kernels/sm70_longctx_decode.cu` | same, **re-synced** to include `sm70-decode-occ2.patch` |

`kernels/sm70_nvfp4_moe_decode.cu` used to sit in this directory. Nothing loads
it -- the NVFP4 expert GEMM goes through the out-of-tree `marlin_v100`
extension (`sglang_restomod/quant/sm70_fp4_moe/PROVENANCE.md`) -- and it was
never covered by `UPSTREAM_PIN.toml`, so its hash in `SHA256SUMS` guarded a
file with no pin. Removed on 2026-11-01.

`sm70_longctx_decode.cu` was first vendored before the two-CTA decode patch
landed in the lab tree, so the component would have shipped the slower kernel.
It now carries `sm70-decode-occ2.patch`: the batched tile loader, the
warp-specialized pipeline and the `kCtas` template, selectable at runtime with
`SGLANG_SM70_DECODE_VARIANT=serial|occ2|pipelined` (default `occ2`). That is
worth -28% per layer at 256k on e5m2 and +3.7% end-to-end decode.

## The QSA backend now follows the pin (2026-11-06)

`qwen_sparse_attn_backend.py` is no longer fork-lineage code with pin ports
bolted on. It is the pin's own `v0.5.23` file with the fork's SM70 additions
re-applied on top:

- the SM70 dense prefill route (`SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS`) and
  the TileLang D256 kernel it calls;
- the SM70 sparse chunk-prefill route (E5M2 K/V read straight from the pool);
- the SM70 dense and sparse decode routes into `tilelang/_decode_cuda.py`.

Taking the pin file as the base also adopts upstream #45752 (prefill indexer
gather) and #45791 (long cached prefixes), which the fork predates. Nothing is
read from the engine for the combine: `tilelang/_decode_cuda.py` binds the
TileLang combine, and neither `sglang.kernels.ops.attention.sm70_qsa_combine`
nor `SGLANG_SM70_QSA_COMBINE` exists at the pin.

The pin also imports the external `flash_mla` wheel at module scope on the
decode-graph path: `eagle_worker_v2` imports `flashmla_backend`, which imports
`kernels.ops.attention.flash_mla_interface`. That wheel has no SM70 build, and
without it graph capture aborted before the QSA backend was asked to capture.
`flash_mla_compat.py` installs a stub when that import fails, keyed on the
import rather than on a device name, so a host with the wheel is left alone and
a launch that really runs FlashMLA still fails with an install instruction.

Validated through its own module path. Since 2026-11-02 every 27B measurement in
this repository runs this component rather than the fork: the launchers set
`SGLANG_RESTOMOD_COMPONENTS=attention_v100,fp8_dense,spec_sm70,gdn_tilelang_sm70`
with `PYTHONPATH` pointing at the component tree, `--attention-backend
tilelang_fa_v100` is a name the pin does not define on its own, and the
receipts' `versions.txt` records that component list next to `sglang 0.5.22`.

Receipts: `evidence-2649742-gdn-tilelang`, `evidence-153a4c2-kv-splits`,
`evidence-4a1ea84-pin-kv`, `evidence-e9ef57f-spec-depth`. What they cover is the
TP2 27B-FP8 path at up to 32k, with MTP and the four components.

Still fork runs: the 128k and 256k rows in `recipes/` and `docs/runs/`, and every
Flash-Next number. Those need a 4-card or 256k launch through this component and
are tracked in #92, #93 and #95.

