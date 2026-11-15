## What

`fp8_dense_quant` refused every `RowParallelLinear` on Flash-Next. The
predicate required `output_size_per_partition`, which only
`ColumnParallelLinear` sets; `RowParallelLinear` is a `LinearBase` without that
attribute. Test `LinearBase` membership instead, with the old duck-typed test
kept as a fallback when the class cannot be imported.

`ReplicatedLinear` stays excluded on purpose: it carries the router, whose
forward is a fused kernel and needs its own numeric A/B.

The fix also adds an env-gated details log (`SGLANG_SM70_FP8_DENSE_QUANT_DETAILS`)
naming every 2-D weight and its refusal reason, so a budget can list what
stayed on bf16. The recipe's dense-path section now records the resulting
count and the remaining refusals.

## Admission, same launch

| arm | converted | Row refused by predicate | Row refused by shape |
|---|---:|---:|---:|
| `1c11d10150` | 49 | 151 | — |
| this commit | 98 | 0 | 102 |

The 49 newly admitted are the 36 `linear_attn.out_proj` and 12 `o_proj`
(all `[2560, 1536]`), plus one vision merger. The 102 still refused are
27 `1152x288`, 27 `1152x1076`, 48 `2560x160` — unchanged shapes, still not
128-aligned.

## Decode A/B

Flash-Next NVFP4, TP4 on one V100 island, 4,096-token prompt, 16 profiled
steps, `--mem-fraction-static 0.78`, otherwise identical launch.

| term | baseline | this commit | delta |
|---|---:|---:|---:|
| **dense GEMM total** | 10.372 ms | 10.106 ms | **-0.266 ms (-2.6%)** |
| cutlass 16x16 | 3.993 ms / 250 calls / 16.0 us | 3.221 / 201 / 16.0 | -0.772 |
| TurboMind fp8 GEMM | 1.557 / 57 / 27.3 us | 2.851 / 111 / 25.7 | +1.294 |
| split-K reduce (half) | 0.568 / 198 | 0.447 / 149 | -0.121 |
| cutlass 32x32 | 1.807 / 99 | 1.823 / 99 | +0.016 |
| big TurboMind | 0.879 / 4 | 0.222 / 1 | -0.657 |
| anchor | 156.1 tok/s | 157.76 tok/s | +1.06% |

The dense term and the unprofiled anchor are the evidence. The two traces
differ in the allreduce term for unrelated reasons, so per-step totals are not
compared here.

## Prefill and output

Cold prefix, 4,011 tokens, two reps:

| arm | rep1 | rep2 |
|---|---|---|
| baseline | 1366.1 tok/s, TTFT 2.935 s | 595.0 tok/s (host-state outlier, already documented for cold prefill) |
| this commit | 1387.4 tok/s, TTFT 2.891 s | 1388.6 tok/s, TTFT 2.888 s |

No regression. A 64-token greedy completion is coherent `<think>` text.

## Scope

No block-64 kernel; the 320-wide hyper-connection group and the other
unaligned shapes need a separate change. The router (`ReplicatedLinear`) is
deliberately still on bf16.

## Tests

`tests/test_sm70_fp8_dense_quant.py`: 13/13 in the lab environment (CPU).
Adds cases for `RowParallelLinear` admitted without
`output_size_per_partition` and `ReplicatedLinear` refused as `replicated`.
Locally this box has no torch, so the module skips here.
