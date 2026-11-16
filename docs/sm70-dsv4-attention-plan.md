# DSV4 attention on SM70: plan

Status: plan, not implementation. Everything below the "Measured" heading was
read from the pin (`297e54f`, v0.5.23); the rest is proposed.

## Why a component and not a flag

- `create_dsv4_backend` (`srt/layers/attention/attention_registry.py:199`)
  branches on platform: NPU → `DeepseekV4AscendAttnBackend`, HIP →
  `DeepseekV4HipRadixBackend`, else CUDA →
  `deepseek_v4_trtllm_backend.create_deepseek_v4_attn_backend`.
- That CUDA factory picks by `--dsv4-attn-backend` (`trtllm`,
  `flashmla_mega`, default) between `DeepseekV4TrtllmAttnBackend`,
  `DeepseekV4MegaAttnBackend` and `DeepseekV4AttnBackend` — and **the first
  two subclass the third** (`deepseek_v4_mega_backend.py:43`,
  `deepseek_v4_trtllm_backend.py:142`). All three run
  `DeepseekV4AttnBackend.__init__`, which does
  `from sglang.kernels.ops.attention import flash_mla_interface` under
  `if is_cuda():` (`deepseek_v4_backend.py:1263`). No flag avoids it.
- The HIP branch is not an alternative: it imports `hip_flash_mla` (`:98`)
  for the paged decode, alongside Gluon/Triton pieces for glue, sparse
  prefill and swap-AB.
- `--attention-backend triton` cannot serve this model class
  structurally: `triton_backend.py` has no DSV4 or compression support, and
  the DSV4 backend asserts `page_size == 256`, `head_dim == 512`, a
  `DeepSeekV4TokenToKVPool` with `present_ratios`/`swa_page_size`, and builds
  `make_candidate_indexer`/`make_full_topk_indexer`.

Conclusion: on CUDA the only in-tree DSV4 attention needs `flash_mla`, which
is not available or valid on Volta. Producing a token on this hardware needs
a new implementation, which is a component beside `attention_v100`.

## Semantics the implementation must satisfy

All of these are asserted or constructed by the existing backend and are not
negotiable by the new one:

| Requirement | Source |
| --- | --- |
| `page_size == 256`, hardcoded by the system | `deepseek_v4_backend.py:1251` |
| MQA `head_dim == 512` (448 nope + 64 rope) | `:1247` |
| `DeepSeekV4TokenToKVPool` with `present_ratios`, `swa_page_size`, `swa_key_layout` | `:1265-1284` |
| compress ratios in play: 1, 2, 4, 128 (`low_ratios`, `has_c4`, `has_c128`) | `:1283-1287` |
| SWA window 128, page-aligned (`swa_page_size % SWA_WINDOW == 0`) | `:1275` |
| sparse indexer top-k: `make_candidate_indexer`, `make_full_topk_indexer`, `index_topk` | `:1291-1296` |
| `deepseek_v41` flag changes the indexer path | `:1289` |

## Interface it plugs into

- Register a backend for the `dsv4` attention backend name, or hook
  `create_dsv4_backend`'s CUDA branch from a component; either way the object
  must satisfy the subset of `AttentionBackend`
  (`base_attn_backend.py:38`) that the runner calls for our launch profile:
  `init_forward_metadata`, `init_forward_metadata_decode`,
  `init_forward_metadata_prefill`, `forward_decode`, `forward_extend`, and the
  capability flags that turn off CUDA graphs and speculative decoding.
- Stage 1 supports exactly one profile: TP1, no spec decode, no CUDA graph,
  no DP attention, batch sizes 1..N decode with chunked prefill ≤
  `max_prefill_tokens`. Anything outside it should raise with the profile
  name, not fall back.

## Staged sequence

**Stage 0 — skeleton.** Component loads only when selected; registering it
must not push it into `auto`. If the backend cannot be constructed it raises
at startup, naming the profile it does not support.

**Stage 1 — torch reference: paged decode and sparse prefill.** Correctness
only. There is almost no in-tree torch material to borrow: the only torch
code in `deepseek_v4_backend.py` is `_low_ratio_compress_torch` (`:3761`),
one sub-step of the compression. The paged KV addressing, the sparse top-k
selection, the SWA window and the ratio-1/2 paths all have to be written.

**Stage 1 must force `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0`.** The sparse
prefill path is on by default, and `_use_sparse_prefill`
(`deepseek_v4_backend.py:2917`) is consulted at `:2950`, `:4043` and `:4263`
before any dense fallback. A backend that supplies no sparse implementation
would take the sparse branch and never reach its own code, so the switch is
part of stage 1, not a tuning knob.

Acceptance, both required:

1. **One token.** `models/bluewhale/mini-real` serves a prompt and returns
   tokens (the checkpoint loads today under pin + 0004 + 0005 + 0006 + the
   `fp4_moe` overlay).
2. **Prefill and decode agree.** The same prefix computed as one prefill
   chunk and the same prefix fed token-by-token as decode must produce the
   same next-token logits within a tolerance fixed after the first
   measurement. This is the check that catches paged-KV addressing errors,
   which is the class of bug a "it runs" milestone hides. A tolerance that
   has to be widened per run is a failure, not a result.

**Stage 2 — Triton hot kernels.** Only after stage 1 passes both acceptance
checks. Triton on sm70 is FMA-only (see
`docs/sm70-tilelang-dtypes.md`), so the target is a defensible fraction of the
memory floor, not parity with the CUDA path.

**Stage 3 — real kernels** only if stage 2's numbers justify a window.

## Cost, and how it will be reported

Stage 1 will be slow, and the number decides whether stage 2 earns a window.
Report, for the stage-1 backend on `mini-real`, TP1: decode step time (elapsed
/ steps, per `docs/measurement-prerequisites.md`), prefill time for one 2k
prompt, and the share of the step spent in attention. A correct-but-slow
token is a milestone; it is not a result until those three numbers exist.

## Non-goals, stated so they are not rediscovered

- **No `flash_mla` of any kind, including an import shim.** A stub that lets
  `__init__` pass and then takes a different branch produces a server that
  starts and is silently wrong.
- No speculative decoding, no CUDA graphs, no DP attention in stage 1.
- No change to `patches/0007` (the dense MXFP8 guard): that gates a load-time
  stage and is off this critical path.
