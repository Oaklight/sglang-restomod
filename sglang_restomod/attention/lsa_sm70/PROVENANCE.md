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
| `ward.py` | `src/lsa/core/ward.py` | imports made package-local |
| `reference.py` | `src/lsa/core/reference.py` | flat design, no bucket shapes |

## What is not selected

- `src/lsa/core/kernels/cutedsl/` - Hopper WGMMA and Blackwell `tcgen05`
  kernels. The reference dispatches them only on H200/B300, so they are dead
  weight on sm70.
- `src/lsa/core/shape.py` - bucket shapes. The target design scores centroids
  (`bucket_shape: null`, `node_bucket_shape: null`).
- `src/lsa/core/hierarchy.py` - the fixed/tree hierarchy. The first target is
  the flat design (`top_level: 0`).
- `LatentMetric` and `regroup_latent` - the MLA route. Qwen3.8-27B is a
  grouped-query model; GLM-5.3-Flash is out of scope for V100.
- `src/lsa/training/` - native and distilled training. This layer only serves.

## Integration

The component is selected with `SGLANG_RESTOMOD_COMPONENTS=lsa_sm70`. It is
additive: no upstream file is edited and no patch is added. The selection core
is plain PyTorch and runs on CPU or CUDA; the serving backend and its sm70
kernels are staged in the tracking issue.

## Status

| part | status |
|---|---|
| index space, Ward, flat selection | ported, CPU-tested |
| sm70 scoring/top-k/expansion kernels | not started |
| `LsaSm70Backend` and Qwen wiring | not started |
| 4x V100 end-to-end validation | not started |

No numbers are claimed in `COMPATIBILITY.md` until a receipted run exists.
