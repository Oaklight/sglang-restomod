# Compatibility matrix

Eight components ship (`fp8_moe`, `fp8_dense`, `fp4_moe`, `attention_v100`,
`ple_sm70`, `spec_sm70`, `gdn_tilelang_sm70`, `lsa_sm70`); all are opt-in. A row becomes "validated" only after
a recorded run.

Some rows carry decode figures from 64-output-token runs. On one unchanged
launch at 31K the decode rate spanned 38.09 to 49.21 tok/s across four reps
(`sglang_restomod/linear_attn_sm70/PROVENANCE.md`), so read decode in those rows
as indicative, and do not compare it across rows. Prefill and the receipt are
what the row claims.

| Hardware | Architecture | Path | Status |
|---|---|---|---|
| Tesla V100-SXM2 (2x) | sm70 | fork `dg1kjd/sglang-sxm2` @ `73abfe1` + 3 local patches, block FP8 | Validated 2026-10-25: Qwen3.8-27B-FP8, TP2, 28K — [run](docs/runs/2026-10-25-qwen3.8-27b-fp8-node-b.md) |
| Tesla V100-SXM2 (2x) | sm70 | fork, same, CUDA graph + MTP | Validated 2026-10-26: 27B decode 67-79 tok/s at 1-2.7K, 42-45 at 8K — [run](docs/runs/2026-10-26-qwen3.8-27b-mtp-node-b.md) |
| Tesla V100-SXM2 (8x) | sm70 | fork, same + `fp8_moe`, TP8/EP8, CUDA graph + MTP | Validated 2026-10-26: Qwen3.8-Flash-Next-FP8 end to end, 83-86 tok/s decode — private lab benchmark record |
| Tesla V100-SXM2 (2x) | sm70 | pin v0.5.22 + `attention_v100`, `fp8_dense`, `spec_sm70`, TP2, MTP 3/4 | Validated 2026-11-02: Qwen3.8-27B-FP8, 60.40-64.50 tok/s at 8.2K and 41.48-44.71 at 31.7K — receipt `evidence-e5a5740-draft-extend` |
| Tesla V100-SXM2 (2x) | sm70 | same, fp16 KV | Validated 2026-11-02: the same model without the draft-extend fix measures 17.89-17.86 tok/s at 31.7K — issue #75 |
| Tesla V100-SXM2 (2x) | sm70 | pin v0.5.22 + `attention_v100`, `fp8_dense`, `spec_sm70`, `gdn_tilelang_sm70`, TP2, MTP 3/4, 40 KV splits | Validated 2026-11-02: Qwen3.8-27B-FP8, prefill 1,933.5 / 1,925.7 tok/s at 8.0K and 1,658.7 / 1,651.5 at 31.1K; decode 73.12 / 63.40 and 45.98 / 46.07, indicative only — receipt `evidence-e9ef57f-spec-depth`, whose `server_info.json` records `tp_size` 2, `speculative_num_steps` 3, `speculative_num_draft_tokens` 4 and `triton_attention_num_kv_splits` 40, and whose `versions.txt` records the component list |
| Tesla V100-SXM2 (2x) | sm70 | pin v0.5.22 + `attention_v100`, `fp8_dense`, `spec_sm70`, `gdn_tilelang_sm70`, TP2, MTP 3/4, 40 KV splits, `fp8_e5m2` KV, 262k context, mem fraction 0.82 | Validated 2026-11-03: Qwen3.8-27B-FP8, prefill 1,029.3 / 1,028.4 tok/s and decode 28.71 / 30.06 at 127,463 prompt tokens, prefill 688.6 / 688.7 and decode 17.60 / 18.84 at 254,914 — receipt `evidence-dccc907-long-context`, 10 files including `gpu_tests.txt` (6 passed) |
| Tesla V100-SXM2 (2x) | sm70 | pin v0.5.22 + `attention_v100`, `fp8_dense`, `fp8_lm_head`, `spec_sm70`, `gdn_tilelang_sm70`, TP2, DFLASH2 draft, block 8 | Lab record 2026-11-04: Qwen3.8-27B-FP8 with the `Qwen3.8-27B-DFlash2-fp16` draft, accept rate 0.22-0.47, 62.2 / 62.4 / 49.9 tok/s at 1.7K / 6.7K / 27.4K prompts (256 outputs, CUDA graphs), 14-17% under the MTP head on the same bench; the bench numbers are not receipted — [lab record](https://github.com/Oaklight/lambda-v100-lab/blob/main/notes/dflash-sm70-blockers.md) |
| Tesla V100-SXM2 (8x) | sm70 | pin v0.5.22 + `attention_v100`, `fp8_dense`, `fp8_moe`, TP8/EP8, `qsa_v100`, MTP 3/4 | Lab record 2026-11-04: Qwen3.8-Flash-Next-FP8 serves at 8,192 context; prefill 2,424.7 / 2,427.1 tok/s and decode 80.72 / 90.21 at 4,108 prompt tokens, prefill 2,420.5 / 2,436.7 and TTFT 3.310 / 3.288 s at 8,012, with every card at 1530 MHz SM (equal to `clocks.max.sm`) and 35-43 C; no receipt for the bench numbers yet -- issue #93 closed on this run; [lab record](https://github.com/Oaklight/lambda-v100-lab/blob/main/notes/flashnext-at-the-pin-2026-11-04.md) |
| Tesla V100-SXM2 (4x) | sm70 | pin v0.5.22 + `fp4_moe`, `attention_v100`, `ple_sm70`, TP4, MTP 3/4 | Validated 2026-11-02: Qwen3.8-Flash-Next-NVFP4, decode 84.69-99.43 tok/s at 31.7K, prefill 1,599-1,600 tok/s with a cold prefix (`bench.py --fresh`; a warm prefix cache reports 12-14k as prefill, #95) — no receipt yet (#95); [lab record](https://github.com/Oaklight/lambda-v100-lab/blob/main/notes/flash-next-nvfp4-context-ladder.md) |
| Tesla V100-SXM2 (4x) | sm70 | pin v0.5.22 + `fp8_dense`, `lsa_sm70`, TP4, fp16 KV, page size 1, no CUDA graph | Recorded 2026-11-04: Qwen3.8-27B-FP8, greedy output identical to `tilelang_fa_v100` on 32 tokens at 2,048 and 16,384 prompt tokens (mean absolute log-probability shift 6.7e-4 and 1.7e-4 nats); prefill 445 / 259.5 tok/s and decode 7.7 / 8.0 tok/s at 2k / 16k; signed receipt posted on PR #182 (`gpu-receipt`) — [recipe](recipes/lsa-qwen3.8-27b-v100-tp4.md) |
| Tesla T4 | sm75 | upstream SGLang 0.5.22 | Planned |
| RTX 4070 Super | sm89 | upstream SGLang 0.5.22 | Planned |
| H100 | sm90 | upstream SGLang 0.5.22 | Planned |

Every row that runs the pin carries a receipt, whose release is named in the
row. The fork rows name a run log or a lab record instead: the receipt gate
binds an `Oaklight/sglang-restomod` commit and a signed evidence bundle, and
those runs predate both, so they cannot be receipted after the fact. The
Flash-Next TP4 row is the one pin cell still on a lab record, because a bundle
for it needs four idle cards (#95). The `lsa_sm70` row is the one pin cell whose
bundle is packed but not yet signed.

The `lsa_sm70` component does not carry the implementation: it registers the
standalone `Oaklight/lsa-sm70` package at the commit pinned in
`UPSTREAM_PIN.toml` (`[lsa]`). The backend name, the launch flags and the
`launch_check` comparison are unchanged.

## Known failures

A cell that does not run is a result. Each row names the stage it stops at and
where the log excerpt lives.

| cell | stage | error | where |
|---|---|---|---|
| Flash-Next FP8, 2x V100, pin | first forward: the dense FP8 compute path | recorded in the issue | #92 |
| 27B, `--kv-cache-dtype fp8_e4m3`, pin | grouped decode-attention kernel compile, before the model loads | `ValueError: type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')` | #112 |
| DGX Spark | sm121 / aarch64 | upstream SGLang 0.5.22 | Planned |

Notes:

- Quantization dispatch and FP8 specifics: `docs/quantization-dispatch.md`.
- The speculative tree-sampling JIT needs `spec_sm70` on a CUDA 12.x toolkit:
  the pinned `flashinfer/sampling.cuh` uses `cuda::maximum` / `cuda::minimum`,
  which only CCCL 3.x (CUDA 13) provides, and the component puts a shim on the
  JIT include path. Without it the build stops at
  `error: namespace "cuda" has no member "maximum"` (#179).
- Upstream SGLang supports sm75 and newer; it does not target sm70. V100 needs
  selected components from this layer.
- A validated row records: hardware, upstream pin, component versions, model,
  topology, context, date.
- The 2026-10-25 row is a fork run recorded for evidence; the same path is not
  yet available through this layer's components.
- With the CUDA graph metadata fix in the fork, the same run reaches 30-35
  tok/s decode: [graph run](docs/runs/2026-10-25-qwen3.8-27b-fp8-cuda-graph-node-b.md).
- The `fp8_moe` component (block-FP8 expert GEMM) was validated numerically
  on 2026-10-25 and loads the full Flash-Next checkpoint on 8x V100 (17.29 GB
  per rank, TP8/EP8). See the
  [expert GEMM run](docs/runs/2026-10-25-qwen3.8-flash-next-fp8-sm70-moe-node-a.md).
- The QSA decode blocker recorded earlier is **resolved**: the SM70 kernels were
  templated over the query-head group, so the 3-head layout at TP8 no longer
  falls back to a FlashAttention build the V100 image lacks. Flash-Next now
  generates end to end, and with CUDA graph and MTP reaches 83-86 tok/s.
- MTP works on both target models from the checkpoint's own head, with no draft
  model: `--speculative-algorithm NEXTN`. On Flash-Next the draft token count
  cannot exceed 4, because QSA's pending index-key ring holds one compress
  group; the 27B has no such ring.
- `fp8_e5m2` and `fp8_e4m3` KV are **validated on the fork**, where e4m3 is the
  better of the two. The corruption recorded on 2026-10-25 was two uninitialised
  reads in the decode partial (#51), not the format; after that fix and #79,
  e5m2 carries the MTP target verify and reaches 256k. e4m3 has a smaller
  perturbation of the model's own next-token distribution — see the recipe's
  "KV dtype" section. The
  [E5M2 run](docs/runs/2026-10-25-qwen3.8-27b-fp8-e5m2-kv-node-b.md) is the
  record of the pre-fix failure.
- **At the pin, e4m3 does not start and e5m2 does.** `fp8_e4m3` fails because
  the grouped decode kernel compiles for `fp8e4nv`, which Volta does not
  implement (#112). `fp8_e5m2` cost 1.38x decode at 32k until #129: the fp8
  dispatch clauses sent the draft extend to the Triton extend kernel. With the
  routing fixed it measures 58.81 against 53.43 tok/s at 32k, greedy, and is
  what the receipted 128k and 256k cells use. fp16 stays the pin default
  because it is exact and faster at 8k.
