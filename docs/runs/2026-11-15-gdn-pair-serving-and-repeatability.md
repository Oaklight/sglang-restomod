# 2026-11-15 — GDN pair in serving, and why token equality could not gate it

Status: **the pair serves and the prefill gain reproduces. Token-level
equivalence was not established, and the control arm shows it cannot be
established on this stack by token equality.**

## What was run

lambda4, GPUs 4-7 (one island, TP4), Flash-Next checkpoint at
`/nfs/ml_lab/.../flashnext/model-sm70`, commit `241a17747d`, `IMPL=pair`,
`--mem-fraction-static 0.82`, CUDA graphs enabled, `--linear-attn-prefill-backend
triton`, SPEC off. Component presence was read from `/proc/<pid>/environ`, not
from the launcher: an earlier launch silently omitted `gdn_tilelang_sm70` from
`SGLANG_RESTOMOD_COMPONENTS` and produced the same trace as pure Triton.

## Acceptance: the kernels are reached in a served prefill

Cold 3072-token prompt, one warm-up request discarded, TP-0 trace:

```
kernels=3314  triton_chunk_o=0  triton_chunk_h=0  main_kernel=72
  36 x main_kernel grid=[2, 46, 12]   chunk_o (V//64, chunks, B*H)
  36 x main_kernel grid=[4, 12, 1]    chunk_h (V//32, B*H)
```

The grids identify the kernels; a bare `main_kernel` count would not, because
the engine's other TileLang kernels share the name.

Two defects were found by this run and fixed: `cache_indices` arrive as int32
and `index_select`/`index_copy_` require int64 (chunk_h stayed on Triton on a
caught exception), and the launcher above. Both have regression tests that fail
without the fix.

## Prefill

Same arm, `bench.py` at 4096 tokens, three runs:

```
1018 / 1004 / 1007 ms at 4121 prompt tokens -> 4048 / 4106 / 4094 tok/s
median 1007 ms, 4069 tok/s, spread 1.4%
```

Against the SPEC=0 control on the same box and the same launcher (2495 ms
median, measured at an earlier commit) that is **2.48x**. The arms differ in
commit; the delta between them is a kernel-pin docstring bump and the int64
fix, neither of which touches the control path.

This is the **pair-only** arm. The component as selected (`auto`) additionally
installs the `whole` wrapper, so this number is not the component's.

## Token equality is not a usable criterion on this stack

Three configurations failed to repeat themselves under greedy decoding with
identical prompts:

| arm | prompt | protocol | result |
|---|---|---|---|
| Triton (control) | synthetic ids | one server, two requests | diverged at token 7 |
| pair | realistic text | one server, cold then cache hit | diverged at token 86 |
| pair | realistic text | `/flush_cache` between runs | diverged at token 4 |
| Triton (control) | realistic text | `/flush_cache` between runs | diverged at token 86 |

The last row is the one that matters: the **control** — no GDN substitution at
all — diverges cold-to-cold. Top-2 logits at the split, from
`output_top_logprobs`:

```
run a: 16:-2.094272  15:-2.109897  17:-2.234897   gap=1.562e-02
run b: 16:-2.095664  15:-2.095664  17:-2.236289   gap=0.000e+00
```

The same computation on the same input moved a full 1.6e-2 nats between runs,
and in run b the top two are bit-identical, so the argmax broke the exact tie
one way and run a broke a near-tie the other way. A token-equality gate on this
stack therefore measures the baseline's run-to-run variation, not the kernel
under test. Reporting either outcome as evidence for the pair would be wrong.

`tie_word_embeddings` is false for this checkpoint, so the exact tie is not a
shared embedding row. The two entries carry **distinct ids** (`15` and `16`),
which rules out a reporting path that repeats one value under one id, and the
bf16 `lm_head` rows behind them differ by up to 0.423 — so it is not identical
weights either. That leaves two candidates: a collapse introduced after load
(the head is re-quantized by `fp8_lm_head` in this component list, so the
level count matters and was not read back from the loaded module), or a
reporting path that duplicates a value across distinct ids. Unresolved; the
decisive measurement is to dump the raw logits for both ids at that step and
compare the bits, which was not done.

The variation is **per call, not fixed at start**: both arms diverged
cold-to-cold *within one server process*, so anything chosen once at init
(kernel selection, a split-k value, custom-allreduce presence) cannot explain
it. That leaves per-call state or a race. Atomic accumulation of split-k parts
is compatible with this and remains untested.

## Open, and not established here

- **Run-to-run numeric variation in the baseline.** 1.6e-2 nats on final logits
  is larger than fp32 reassociation in one GEMM would give. A candidate is the
  marlin split-k path: if `split_k` were derived from anything other than the
  shape, the accumulation order would change between runs. This was **not
  tested**; logging `split_k` per shape across two runs would rule it in or out.
  It is independent of GDN and belongs to its own investigation.
- **Correctness of the pair.** It rests on the kernel-level comparison in
  `tests/test_linear_attn_sm70_gpu.py`, not on this gate.
