# 2026-10-26 — Flash-Next decode: removing the per-layer host reads

Status: **decode 2.9 → 5.5–6.3 tok/s.** The remaining step time is launch
overhead and the expert-parallel all-reduce, not the expert GEMM.

## What was measured

Torch profiler on TP0 (`/start_profile`, one 32-token decode), aggregated over
the `step[DECODE bs=1]` windows:

| | before | after |
|---|---|---|
| CPU time per step | 294 ms | 218 ms |
| GPU busy per step | 84.5 ms | 57.2 ms |
| … of which EP all-reduce (49×/step) | 69.9 ms | 48.4 ms |
| decode throughput (57-token prompt) | 2.86 tok/s | 5.55–6.27 tok/s |
| decode throughput (3069-token prompt) | 1.67 tok/s | 5.28–5.74 tok/s |

The profiler itself costs roughly 15%, so the honest range is the two numbers
per cell. GPU busy never exceeded 30% of the step: the rest was the CPU waiting.

`aten::item` / `_local_scalar_dense` / `is_nonzero` accounted for ~47 s of
blocked CPU across the trace — one pair per MoE layer per step, each landing
behind that layer's all-reduce.

## What changed

`SM70Fp8MoEMethod.apply` used to branch on `bool(valid.all())` and then take
`flat_ids.numel()`. Both read a value back from the device, twice per layer,
48 layers per step. Decode now uses a static-shape path:

- dropped expert-parallel routes (`-1`) are clamped onto expert 0 and their
  router weight is zeroed, so every tensor keeps the shape `[tokens * top_k]`;
- `argsort` / `bincount` / `index_select` / `index_add` all run on static
  shapes, with no host read anywhere.

The compaction path stays for prefill above `COMPACT_THRESHOLD = 64` tokens,
where its two host reads amortise over a large forward and the wasted rows
would not.

Numerics are unchanged (`dense rel=0.00099`, `ep rel=0.00072`,
`empty_nonzero=0`).

## What is left

1. **CPU launch overhead, ~100 ms/step.** ~1,500 kernel launches per step with
   no CUDA graph. The method is now graphable (static shapes, no host reads);
   the question is whether the rest of the Flash-Next stack (QSA decode, GDN)
   captures.
2. **The EP all-reduce, 48.4 ms/step.** One per MoE layer, 49 per step, about
   1 ms each for a 5 KB payload. This is the dominant GPU cost and it is
   inherent to expert parallelism without an all-to-all backend.
3. Prefill still takes the compaction path; its cost is dominated by the same
   launch overhead, not by the MoE.
