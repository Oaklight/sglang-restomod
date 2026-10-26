# 2026-10-26 — Flash-Next decode under CUDA graph

Status: **25.9–36.9 tok/s** across runs, up from 5.5–6.3 without a graph, on
8x V100 (TP8/EP8). Output is deterministic and identical to the un-graphed run.
The spread is page-cache warmth and profiler attachment, not run-to-run noise
in the graph itself.

## Where it started

The previous record left decode at 218 ms of CPU per step against 57 ms of GPU
work: ~1,500 kernel launches per step and no graph. Capture was impossible
because `apply` read values back from the device (`bool(valid.all())`,
`numel()`); that was removed first, leaving a static-shape path.

## What capture then rejected

Three separate blockers, each one a different general rule:

| Symptom | Cause | Fix |
|---|---|---|
| `cudaErrorStreamCaptureUnsupported` at `bincount` | bincount reads values back internally | `scatter_add_` into a zeroed counter |
| same error at `offsets[0] = 0` | scalar assignment is a host-to-device copy | `offsets[:1].zero_()` + `offsets[1:].copy_(...)` |
| `Expected index [570] ... src [320]` at first request | `ones` came from the persistent workspace, sized for 32 tokens, while a 57-token prefill used the transient workspace | take `ones` from the workspace tuple like every other buffer |

The third one is worth recording because the number is misleading: 570 was
`rows = 57 * 10`, not an expert id, and it only appeared once capture put the
two workspace paths side by side.

## Result

| | no graph | graph |
|---|---|---|
| decode @57-token prompt | 5.55–6.27 tok/s | **30.75–36.93** |
| decode @3069-token prompt | 5.28–5.74 tok/s | **25.96–36.65** |
| two greedy runs identical | yes | yes |
| text matches the un-graphed run | — | yes (same prefix, same tokens) |

That is a 5x step. The remaining per-step cost is the expert-parallel
all-reduce and whatever kernel efficiency is left; both are now visible
instead of hidden behind launch overhead.

## Rules this leaves behind

1. No host reads inside the forward. They are both a pipeline stall and a
   capture error.
2. Every buffer an op touches must come from the same workspace decision for
   the current token count — persistent and transient paths must not be mixed.
3. Prefer kernels over host-side value reductions for shape bookkeeping
   (`scatter_add_` over `bincount`, `copy_` over scalar assignment).

## Per-token breakdown once the graph is in place

Measured with the torch profiler over one 48-token request, normalised by
token (23.7 ms of kernel time against a 27 ms wall):

| ms/token | calls | kernel |
|---|---|---|
| 3.54 | 100 | NCCL all-reduce (`RING_LL`) |
| 2.44 | 96 | TurboMind dense GEMM (two variants) |
| 2.29 | 194 | `sm70_hc::up_kernel` / `down_kernel` |
| 1.62 | 96 | TurboMind GEMV (two variants) |
| 0.67 | 36 | GDN packed decode |
| ~2.1 | ~470 | gathers, copies, elementwise |

Two corrections to the earlier reading:

- **The all-reduce is not pathological.** A standalone 8-rank benchmark puts a
  `[1, 2560]` fp16 all-reduce at 37.9 µs; 100 calls per token therefore cost
  ~3.8 ms, which is what the profile shows. The 48 ms figure from the
  un-graphed run was wait time between launches, not kernel time.
- **The cost is spread across many small kernels.** A 10-row expert GEMM costs
  79 µs against a ~6 µs memory floor, and grows only ~3.7 µs per additional
  routed expert, so the kernel is launch/setup bound rather than reading
  unrouted weights.

The applied-side gathers (`index_select`, `index_add`, the mask multiply, the
`zero_`) are a few hundred of those calls; TurboMind's own
`moe_permute_with_scratch` cannot be used because it does not honour the `-1`
drop sentinel, so the next step is a small fused permute/scale/unpermute kernel
of our own.
