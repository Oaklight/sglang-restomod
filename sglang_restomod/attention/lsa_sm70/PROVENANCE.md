# Provenance: attention/lsa_sm70

## Source

| Field | Value |
|---|---|
| Repository | `Prefrontal-AI/latent-sparse-attention` |
| Commit | `97b82d695b74ca341965a16bbc2a951c8b0c567a` |
| License | Apache-2.0 |
| Paper | arXiv 2610.42147, *Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention* |

## What is selected

The portable selection core, adapted for this layer:

| file here | source | change |
|---|---|---|
| `constants.py` | `src/lsa/constants.py` | `PAD` and `NORM_EPS` only |
| `budget.py` | `src/lsa/budget.py` | imports made package-local |
| `index_space.py` | `src/lsa/core/index_space.py` | the grouped-query route only |
| `ward.py` | `src/lsa/core/ward.py` | imports made package-local; members built without a per-bucket loop |
| `reference.py` | `src/lsa/core/reference.py` | flat design, no bucket shapes; tensors built on the input's device |

The serving path, written here against those semantics:

| file here | what it does |
|---|---|
| `design.py` | the flat design's knobs and a request's bucket table |
| `selection.py` | `reference.read` vectorized a chunk of rows at a time |
| `read.py` | the sparse read through the pinned `decode_attention_fwd` |
| `backend.py` | `LsaSm70Backend(AttentionBackend)` |
| `backend_registry.py` | registers `lsa_sm70` with the pinned engine |

## What is not selected

- `src/lsa/core/kernels/cutedsl/` - Hopper WGMMA and Blackwell `tcgen05`
  kernels. The reference dispatches them only on H200/B300, so they are dead
  weight on sm70.
- `src/lsa/core/kernels/triton/` - the fused scan, top-k and expansion
  kernels. The first serving path scores every visible bucket with a matmul
  (the reference's own path below its fused-scan gate) and selects with
  `torch.argsort`; the fused kernels are a later optimization.
- `src/lsa/core/shape.py` - bucket shapes. The target design scores centroids
  (`bucket_shape: null`, `node_bucket_shape: null`).
- `src/lsa/core/hierarchy.py` - the fixed/tree hierarchy. The first target is
  the flat design (`top_level: 0`).
- `LatentMetric` and `regroup_latent` - the MLA route. Qwen3.8-27B is a
  grouped-query model; GLM-5.3-Flash is out of scope for V100.
- `src/lsa/training/` - native and distilled training. This layer only serves.

## Integration

The component is selected with `SGLANG_RESTOMOD_COMPONENTS=lsa_sm70`. It is
additive: no upstream file is edited and no patch is added. The backend is
registered as `--attention-backend lsa_sm70`, replaces the full-attention
layers of a hybrid model, and leaves the linear-attention layers to the
engine's GDN backend.

The read is not vendored: it calls the pin's
`sglang.kernels.ops.attention.decode_attention.decode_attention_fwd` with a
per-query `kv_indptr`/`kv_indices` list at `page_size=1`, the same interface the
reference uses when its CuTeDSL kernels are unavailable.

## Status

| part | status |
|---|---|
| index space, Ward, flat selection | ported, CPU-tested |
| vectorized selection | written here, CPU- and GPU-tested against the oracle |
| sparse read | written here, GPU-tested against a dense gather |
| `LsaSm70Backend` and Qwen wiring | written here, end-to-end on 4x V100 |
| fused Triton scan/top-k/expansion | not started |
| CUDA graph capture | refused; the selection is host-orchestrated |

No numbers are claimed in `COMPATIBILITY.md` until a receipted run exists.
