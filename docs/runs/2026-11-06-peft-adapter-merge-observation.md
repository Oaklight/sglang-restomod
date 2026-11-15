# 2026-11-06 — PEFT adapter merge on a hybrid attention backbone

Status: **observation from a sibling project, no run here.** Recorded because
it bears on serving: whether an adapter is merged into the base weights or kept
as a wrapper changes latency by far more than the adapter itself. Recorded from
issue #211.

## The finding

One model, per item, prefix cache warm:

| scope | merged into weights | frozen, no adapter | kept as a wrapper |
|---|---|---|---|
| narrow: q and v only, 16 modules, 231 items | 89.1 / 78.0 ms | 89.5 / 78.3 ms | 3.2% slower than frozen |
| wide: plus `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, 136 modules | 88.8 / 77.7 ms | 89.0 / 77.9 ms | 27.3% slower than frozen |
| prefix not shared, re-encoded per option | — | — | 1.25–1.41 s per item, 16–18x the merged cost |

The merge is not a small optimisation on top of the adapter. It is the
difference between the adapter being nearly free and the adapter being a
different serving regime.

## Why the merge target set is the point

A serving stack that merges only the attention projections accepts a LoRA that
adapts the linear-attention projections and silently ignores those modules,
because the adapter was applied through a wrapper at training time and the merge
walks a different list. The failure is quiet: the model serves, the weights are
slightly wrong, and nothing errors.

The numerical cost of merging is real but small, and it is rounding rather than
logic. Merging in float and casting to bf16 for storage changed 1 of 231 top-1
predictions at the narrow scope and 4 of 231 at the wide scope. Every flip was a
top-two tie with a margin under 0.02. Bit-identical predictions need a
higher-precision merge or a comparison that tolerates ties.

## What probably transfers to serving

- Merge, do not wrap, and treat the merged and wrapped paths as separate
  deployments that need separate latency numbers.
- On a hybrid backbone the merge list must include the linear-attention
  projections, not only attention.
- The prefix or radix cache has to survive the merge. If merging invalidates
  the cache key, the prefix sharing is lost and the wide-scope wrapper number is
  the honest one.
- One sanity number for any implementation: the merged path should land within
  noise of the frozen path. A merged path measurably slower than frozen means
  the merge did not happen and the wrapper is still in the graph.

## What does not transfer

The measured benchmark, the model, and the two option-count regimes are
specific to the sibling project. The 89 ms figure is not a target; the ratio
between merged, wrapped and unshared is what carries.

## Proposed first step

On the target hardware, with a model that has linear-attention projections,
measure the same four configurations at batch 1 and at the intended serving
batch with the prefix cache warm, and check that the merged path lands within
noise of frozen. If it does not, the merge is not reaching every adapted
module, and the module list is the first place to look.
