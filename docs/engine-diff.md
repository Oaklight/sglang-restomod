# What `sglang-sxm2` adds over upstream, and what our models actually need

This note answers one question: to run the two target models from this layer,
which parts of the fork are necessary, and which of those can live here?

## How it was measured

- Fork: `dg1kjd/sglang-sxm2` @ `73abfe1dd7e7aae7a62b222a201e48f532fb298e`
  (a dev snapshot, 2026-10-07).
- Upstream: `sgl-project/sglang` tag `v0.5.22` (the pin in `UPSTREAM_PIN.toml`).
- Every file in the fork's `python/sglang` mentioning `sm70`, `SM70`, `_v100`
  or `V100` was checked against the same path upstream:

| Class | Count | Meaning |
|---|---|---|
| fork-only | 82 | file does not exist upstream |
| modified | 107 | file exists upstream but the fork's copy carries SM70 code the tag does not have |

The fork is not a thin patch on `v0.5.22`: it also carries upstream drift, so
the 107 is an upper bound on SM70 edits, not a changelog.

## What the target models need

### Qwen3.8-27B-FP8 (dense, full attention)

| Piece | Where it lives | Fork-only? | Needed for |
|---|---|---|---|
| `tilelang_fa_v100/*` (10 files, ~2.8k lines) | `srt/layers/attention/` | yes | prefill + paged decode/prefill kernels |
| `flash_attn_v100_backend.py` | `srt/layers/attention/` | yes | the SM70 full-attention backend |
| `attention_registry.py` (+38) | `srt/layers/attention/` | modified | registers the `tilelang_fa_v100` name |
| `triton_backend.py` (+247) | `srt/layers/attention/` | modified | decode delegation, graph metadata, SM70 guards |
| `sm70_turbomind_fp8.py` (512 lines) | `srt/layers/quantization/` | yes | dense block-FP8 W8A16 |
| `fp8.py` (+161) | `srt/layers/quantization/` | modified | selects that kernel on SM70 |
| `qwen3_5.py` (+413) | `srt/models/` | modified | SM70 fusions/hooks for this architecture |
| `sm70_turbomind_bindings.cpp` + `_sm70_turbomind_v100.so` | `kernels/` | yes | the TurboMind ops themselves |

### Qwen3.8-Flash-Next-FP8 (hybrid linear + QSA, 512 experts)

Everything above, plus:

| Piece | Where | Fork-only? | Needed for |
|---|---|---|---|
| `qwen_sparse_attn_backend.py` (2700 lines, +651 vs upstream) | `srt/layers/attention/` | modified | QSA sparse prefill/decode |
| `qsa/*` (`mqa.py`, `qsa_indexer.py`) | `srt/layers/attention/` | modified | sparse selection + indexer |
| `tilelang_fa_v100/_kernels_dense_d256*.py` | `srt/layers/attention/` | yes | QSA dense/D256 kernels |
| `sm70_longctx_decode.cu`, `sm70_qsa_combine.py`, `mqa_logits_sm70.py` | `kernels/` | yes | QSA decode + combine + indexer logits |
| `linear/gdn_backend.py` (+121), `gdn_tilelang.py`, `gdn_chunked_tilelang.py` | `srt/layers/attention/` | mixed | the 3-of-4 linear-attention layers |
| `qwen4_exp.py` (+345), `qwen4_exp_mtp.py` (+69), `qwen4_ple.py` (336) | `srt/models/`, `srt/layers/` | mixed | model + MTP + PLE |
| `sm70_shared_expert_fold.py` | `srt/layers/moe/` | yes | shared-expert handling |
| FP8 experts | — | — | already here (`fp8_moe`) |

Not needed for either target, despite being a large part of the fork: DeepSeek
V4 (`dsv4/*`, `sm70_csa2*`), GLM NVFP4, KDA, DSA, multimodal (`multimodal_gen/*`).

## What can live here

Four portability classes:

| Class | Meaning | Example | Component? |
|---|---|---|---|
| A | pure Python, no upstream counterpart | `tilelang_fa_v100/*`, `sm70_turbomind_fp8.py`, `qwen4_ple.py` | vendored module + `register()` |
| B | Python wrapper around a JIT `.cu` | `_decode_cuda.py` + `sm70_longctx_decode.cu` | vendored, built at first use (we already do this for `shim.cpp`) |
| C | upstream file edited in place | `fp8.py` (+161), `triton_backend.py` (+247), `qwen_sparse_attn_backend.py` (+651), `qwen3_5.py` (+413) | **not** a component: either a pinned patch or a narrow runtime monkeypatch |
| D | prebuilt binary | `_sm70_turbomind_v100.so` | loaded by path (already done) |

Class C is the honest limit of the "additive component" idea. A 40-line hook
(the CUDA graph metadata delegation) is fine as a runtime patch — that is what
`attention_v100` does today. A 651-line diff inside one 2700-line backend is a
fork, and pretending otherwise would produce a component that silently drifts
from the engine.

## Staged plan

| Stage | Scope | Unlocks | Rough size |
|---|---|---|---|
| done | `fp8_moe`, graph metadata | Flash-Next experts; 27B graph speedup | 2 PRs |
| 1 | `fp8_dense`: vendor `sm70_turbomind_fp8.py`, hook `Fp8Config` | 27B block-FP8 quant from this layer | 1 file + 1 hook |
| 2 | `attention_v100` stage 2: vendor `flash_attn_v100_backend.py` + `tilelang_fa_v100/`, register the backend name | 27B attention from this layer | ~3.9k lines vendored |
| 3 | `qsa_v100`: QSA backend + `qsa/*` + the `.cu` | Flash-Next attention from this layer | ~3k lines + 651-line patch |
| 4 | model hooks (`qwen3_5`, `qwen4_exp*`) | remove the last fork patches for our models | ~800 changed lines |

After stage 2 the 27B is served by `engine pin + components`; after stage 3/4
the same holds for Flash-Next. Until then, Flash-Next runs as
"fork @ pin + `qsa-decode-h3.patch` + `fp8_moe`", which is what the recorded
runs measure.

## Recommendation

Keep the engine as the fork, and shrink the *patch* surface stage by stage
instead of reimplementing the engine here. Each stage is independently
validated and receipted, and none of them requires the previous one.

Stage 1 is next: it is one vendored file plus the same plugin hook `fp8_moe`
already uses, and it makes the 27B's quantization a first-class citizen of this
layer rather than a fork detail.
