# 2026-11-15 — Attribution and TP4 placement

Status: **measured. The decode win was not the GDN pair's, and a TP4 deployment
belongs on one host.**

## Why this note exists

Three notes from today already record the GDN prefill A/B
([kernel-level](2026-11-15-gdn-tilelang-prefill-ab.md)), the pair in serving
with the failed token-equality gate
([pair-serving](2026-11-15-gdn-pair-serving-and-repeatability.md)), and the
Flash-Next context sweep
([context](2026-11-15-flash-next-nvfp4-tp4-context-cold-warm.md)). None of them
answers two questions that reconcile the day:

1. With several changes on the box, how much of each number belonged to which
   change -- in particular, how much of the decode win was never the pair's?
2. May TP4 be split across hosts?

This note answers those and points at the other three for everything else.

## Setup

| item | value |
|---|---|
| model | Qwen3.8-Flash-Next, NVFP4, our re-export with `mtp.*` excluded from quantization |
| hardware | V100-SXM2-32GB (NVLink, two islands of four per host) |
| topology | single-node TP4 on one island; dual-node TP4 across two hosts |
| trees | A/B on `ff2d0dc`; C on `81b3b7ea` (before the tail-padding change) |
| spec decode | SPEC=0 for every number here |
| component line | `SGLANG_RESTOMOD_COMPONENTS` includes `gdn_tilelang_sm70`; arm B sets `SM70_GDN_TILELANG=0` |
| prefill protocol | `/flush_cache` plus a nonce per request, `cached_tokens=0`, first request discarded |
| prompt | 4096 requested, 4121 actual |
| measurement | `bench.py` prefill wall, `ms/step` decode, median of independent runs |

## The A/B/C attribution

One box, same prompt:

| arm | tree | GDN pair | prefill | decode |
|---|---|---|---|---|
| A | `ff2d0dc` | on | 3450-3565 tok/s | 17.0-17.4 ms/step |
| B | `ff2d0dc` | off (`SM70_GDN_TILELANG=0`) | 1476 tok/s | 16.94 ms/step |
| C | `81b3b7ea` | n/a, tail refused | 2714 tok/s | 40.35 ms/step |

A against B changes one component on one tree: **the GDN TileLang pair is
prefill 2.4x and decode 0.** The decode step does not move.

The TurboMind discovery (#286) is **decode 2.4x and prefill -46%**. It was
measured on the open branch (`fix/turbomind-discovery`), which is not in `main`.
So the decode improvement that the day's earlier results could have been read as
the pair's belongs to a different change; the pair's decode effect is zero.

C is context, not a control. It differs from B in two things at once -- the tail
gate (#287) and the discovery -- so the -46% prefill is recorded as the branch's
number and stays confounded until a same-tree A/B exists. The regression is an
open defect, and its size is not settled. It is not the price of the decode win
until someone finds the mechanism.

## TP4 belongs on one host

Trace of one 3741-token prefill across two hosts:

| quantity | value |
|---|---|
| cross-host NCCL all-reduce | 65.9% of GPU kernel time |
| GDN kernels | 0.1% |
| `main_kernel` duration | 74.67 ms and 74.64 ms in the two arms |
| kernel busy | 1.9 s against ~2.0 s end to end |

The same kernel takes the same time in both arms, and two thirds of kernel time
is the cross-host collective. **Inference:** a TP4 deployment on one island will
beat a split one until the interconnect changes, and a kernel A/B across hosts
answers a different question -- the all-reduce hides it.

## The tail gate was a decode fix

`#287` (`ff2d0dc`) removed the `%64` refusal in `linear_attn_sm70`. With
`--chunked-prefill-size 4096` most real prompts end on a partial chunk, so after
the pair began serving, most real traffic still fell back to Triton; the gate
also cost a Triton `chunk_o` on every decode step. That is why a change made for
prefill moved decode.

Padding is exact, not approximate: `g` repeats its last real value, because
`glast` is read from the chunk's last row; `v=0` and `beta=0` make each padded
row's `v_new` zero so it adds nothing to the carried state; `w` and `u` are
zero-padded. The GPU tests compare both wrappers against the pinned Triton call
at 300 tokens, a length that is neither a multiple of 64 nor a whole number of
chunks.

## Receipt and pin, briefly

The signed block on PR #275 names commit `3880f8e058`, job `gpu-bench`,
**`result: fail`**, 9 evidence files and digest
`sha256:a4894beff97bcbe2b8abcc5e0b29564d23ba695941eddad73a3eca6317b1d6b1`.
Its reds include three third-party ones: flashinfer's `find_loaded_library`
substring-matches TileLang's `libcudart_stub.so`. Removing the stub breaks
TileLang -- 6 failed / 2 passed, and 8 passed after the revert -- so there is no
environment-only fix. The runbook is [gpu-receipts.md](../gpu-receipts.md).

The marlin pin is a correctness issue, not a formality. With
`sglang_restomod.quant.sm70_fp4_dense.check`, three shapes, threshold 0.05:
`3f16d44` (the `PROVENANCE.md` pin, clean worktree, container CUDA 12.8) gives
0.0 / 0.0 / 0.0, reproduced twice; the build the site actually loaded,
`6d72a49`, gave 0.265 / 0.067 / 0.505 and then 0.356 / 0.138 / 0.981, with the
suite once seeing 2.795. **Wrong and nondeterministic**, so a spot check can
pass by luck. Hash-check the site `.so` against the pin before any measurement.

## Measurement rules

Each of these cost a wrong number or a wrong conclusion today.

1. **`flush_cache` plus a nonce, and `cached_tokens=0`,** or you measure
   radix-cache hits: 12k-15k tok/s of nothing.
2. **Discard the first request.** It pays about 1 s of Triton compile.
3. **Read `/proc/<pid>/environ`, not the launch script.** A launcher silently
   omitted `gdn_tilelang_sm70`, and the server then produced the same trace as
   pure Triton.
4. **`register()`-phase logging never reaches the log file.** A missing
   "routed" line proves nothing about whether the component loaded.
5. **Pin the serving tree to a SHA.** A stale export copy made a live component
   look inert for hours.
6. **Never cite prefill without the component list.** The 2.4x rides on one
   environment line.

## Not established

- A clean-island re-derivation of the A/B/C. The arms ran with other services
  resident; the numbers are recorded as measured.
- The size and mechanism of the #286 prefill regression, until a same-tree A/B
  separates it from the tail gate.
- The component-level ratio. Every pair number here is the pair alone; the
  component as a user selects it (`auto`) additionally installs the `whole`
  wrapper.
- Token equality. It cannot gate the pair on this stack; the control diverges
  cold-to-cold. See the [pair-serving note](2026-11-15-gdn-pair-serving-and-repeatability.md).

## Related

- [GDN prefill kernels: Triton vs TileLang](2026-11-15-gdn-tilelang-prefill-ab.md)
  -- the kernel-level ratios and the four retracted figures.
- [GDN pair in serving, and token equality](2026-11-15-gdn-pair-serving-and-repeatability.md)
  -- acceptance trace, the prefill gain, and why token equality failed.
- [Flash-Next NVFP4 TP4, 8k/16k/64k, cold vs warm](2026-11-15-flash-next-nvfp4-tp4-context-cold-warm.md)
  -- the context sweep and the flat-prefill reading.
- [PR #288](https://github.com/Oaklight/sglang-restomod/pull/288) -- competitor
  numbers by card count. Its tables are not repeated here.
