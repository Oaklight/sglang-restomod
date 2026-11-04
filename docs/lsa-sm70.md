# LSA on SM70

Latent Sparse Attention (LSA) is the next long-context lever for the V100
stack. It changes which keys the attention reads, not the attention kernel:
keys are clustered in a learned latent space, each query scores bucket
centroids instead of every key, and the read sees the selected keys plus a
local window and sinks.

- Paper: arXiv 2610.42147, *Latent Sparse Attention: Towards Subquadratic
  Ultra-Long Context Attention*.
- Reference: `Prefrontal-AI/latent-sparse-attention` at
  `97b82d695b74ca341965a16bbc2a951c8b0c567a`, Apache-2.0.
- Target: Qwen3.8-27B, the D-LSA (flat) design, 4x V100 in tensor parallel.
  GLM-5.3-Flash is out of scope: it stores fp8 weights and an MLA latent, and
  V100 has neither fp8 tensor cores nor the reference's Hopper read.

## What is ported

`attention/lsa_sm70/` carries the portable selection core:

| module | what it does |
|---|---|
| `index_space.py` | position-free dims, grouped-query collapse, learned projection |
| `ward.py` | bottom-up Ward clustering of a cluster block's pools |
| `reference.py` | flat design: build buckets, score, top-p, expand to top-k |
| `budget.py` | the smaller budget of the early queries |
| `constants.py` | `PAD` and the norm floor |

The QA path is the key and query input: every KV head's position-free dims side
by side (4 x 192 = 768), and each group's six queries summed beside them, through
a `[768, 64]` map. The rows are unit length, so a dot product is a cosine.

## Why the reference's servers do not run here

The reference serves on H200/B300:

- CuTeDSL kernels use Hopper WGMMA and Blackwell `tcgen05`; the dispatcher
  selects them only on compute capability 9.0 and 10.x.
- The grouped-query read calls FA3 with a page table of single slots, and FA3
  is Hopper-only. The sparse read otherwise falls back to SGLang's Triton
  decode kernel.
- The index space is bf16, the Ward default is `tf32x3`, and node centroids may
  be fp8. bf16 and tf32 tensor cores are Ampere and later; fp8 is sm89.

Every selection step still has a Triton implementation, and Triton 3.5.1 on the
SM70 environment lowers `tl.dot` for fp16, bf16, fp32-IEEE, tf32 and tf32x3.
bf16 and tf32 lower to slow paths, so the sm70 kernels use fp16 operands or
fp32 FMA, and Ward uses `input_precision="ieee"`.

## The read

The pinned SGLang exposes
`sglang.kernels.ops.attention.decode_attention.decode_attention_fwd` with
`kv_indptr`/`kv_indices` and `page_size=1`, with no architecture gate. That is
the interface LSA needs: one query row, a list of KV slots, one softmax over the
list. The component's V100 backend already delegates decode to this Triton path
and carries QSA kernels that read a per-query list.

`page_size=1` is a requirement of the list read, not of the KV pool: the pool
may keep its V100 page size and the selection is packed into a slot list at read
time.

## Phases

1. **Portable core** (this change). Index space, Ward, flat selection, early
   budget; CPU tests against the reference's invariants, including the exactness
   anchor: at a budget at least the context, the selected set is the causal
   prefix.
2. **SM70 selection kernels.** A chunked bucket scan in fp16/fp32, top-k and
   expansion in Triton, each tested against the core oracle on an SM70 device.
   The reference's fused scan is gated at 32,768 gathered buckets; below the
   gate it uses a dense matmul, which is the first path here.
3. **Backend.** `LsaSm70Backend(AttentionBackend)`: build the per-layer index
   from q/k, store the index keys where the read can find them, maintain the
   per-request bucket cache at page ends, select for prefill and decode, and
   issue `decode_attention_fwd` over the lists. Under tensor parallel the index
   maps are split by head and the partials all-reduced, as the reference's
   `HeadShard` does for 24 query and 4 KV heads at TP4.
4. **End to end.** Qwen3.8-27B at TP4 on 4x V100, bf16 KV cache (the key and
   query input reads bf16 keys), D-LSA map loaded from the release, a smoke
   request, accuracy against the dense baseline at a short and a long context,
   and a signed receipt.
5. **Docs and compatibility.** A recipe, a `COMPATIBILITY.md` row only after a
   receipted run, and the README component table.

## Risks and stop conditions

- **KV dtype.** The index reads bf16 keys. An fp8 KV cache is not usable without
  a separate index-key buffer; the first run keeps `kv_cache_dtype auto`.
- **Memory.** 27B fp8 weights plus a bf16 KV cache and the bucket cache must fit
  4x 32 GB. TP4 is the minimum; TP2 is not assumed.
- **Numerics.** bf16/tf32 tensor-core paths are slow on sm70. If the fp16/fp32
  scan misses the prefill target, the fallback is the dense matmul below the
  reference's fused-scan gate, not a new algorithm.
- **Graphs.** Prefill selection is host-orchestrated in the reference and is not
  CUDA-graph capturable; decode selection is. Do not promise a captured prefill.
- **Pinned internals.** The component imports pinned SGLang symbols lazily, as
  the QSA backend does. A pin bump can break them; the component must fail at
  startup, not on the first forward.

Stop and report rather than merge if: the core oracle cannot reproduce the
reference's selection on the same tensors; the SM70 kernels cannot match the
oracle within the test tolerance; or the 27B end-to-end run cannot hold a bf16
KV cache at the target context.
