# 2026-11-16 — The decode GEMM gap: where it is, and what removing it is worth

Status: **measured.** The per-card decode deficit is a kernel gap, not a
capacity gap: two GEMM families hold 52-62% of our verify step and run 5-8x
slower per call than the reference's, and the expert half of it localises to
the projection our fast kernel cannot read. One arm was served for this page
and one decode window was traced; the capacity model is a separate page
([`2026-11-16-capacity-model.md`](2026-11-16-capacity-model.md)) and nothing
here restates it.

## The metric, because two are in circulation and only one is the question

Measured on this box: **31.6 tok/s per card** (126.2 tok/s on four cards) against
the capacity reference's **55.9 tok/s on one card** — a **1.77x** per-card
deficit. Per *model* the same two runs give us 2.26x ahead, because the
reference pays no cross-card work for the model it serves from one card.

Both are true; they are different metrics. An earlier **2.3-2.8x per-card band
is retired**: it sits above both measured points and is reproduced by neither.
Every number below is anchored to the measured 31.6 per card.

## Arm: dense precision off (`fp8_dense_quant`, `fp8_lm_head` removed)

One launch, lambda4 GPUs 4-7, SPEC=1, branch tree, one request, `bench.py`,
4121-token prompt, three runs per column. The only delta against the
fp8-on baseline arm (same host, same launcher, same day) is the component
list: the checkpoint's 98 fp16 projections per rank that `fp8_dense_quant`
converts, and the fp8 LM head, stay in fp16.

| column | fp8 dense on | fp8 dense off | delta |
|---|---:|---:|---|
| prefill, median of 3 | 1053 ms / 3912 tok/s (spread 1.5%) | 1034 ms / 3985 tok/s (spread 0.5%) | -19 ms, -1.8%, inside the two arms' spread |
| decode step | 24.74 ms, accept 3.12 | 27.05 ms, accept 2.72 | **+2.31 ms, +9.3%** |
| ms per output token | 7.93 | 9.94 | **+25%** |
| tok/s | 126.2 | 100.7 | -20% (carries the accept difference) |

**Reading.** Block FP8 on the dense projections is worth ~9% of a decode step
and nothing measurable at prefill. The earlier suspicion that we lose prefill
by running fp8 where the reference runs fp16 **does not hold on this arm**: the
prefill columns agree. So the dense bucket's 5x is a kernel-rate question, not a
precision choice, and turning the conversion off costs decode. Greedy decode
carries a content-dependent accept (3.12 vs 2.72), so the step and per-token
rows are the comparison; the tok/s row is not.

## The decode-window census

`profile_decode.py` (`/start_profile` with `profile_stages=["decode"]`, warm
4096-token prompt with a full cache hit) on the same arm; one TP-0 trace,
34,107 kernel instances, 1389.5 ms of kernel time.

| family | instances | ms | % |
|---|---:|---:|---:|
| collective | 1316 | 343.4 | 24.7 |
| dense GEMM | 7576 | 322.8 | 23.2 |
| **expert GEMM (Marlin)** | 624 | **299.6** | **21.6** |
| other | 21428 | 279.6 | 20.1 |
| expert GEMM (other) | 1777 | 104.3 | 7.5 |
| linear-attn (GDN) | 625 | 28.9 | 2.1 |
| attention | 761 | 10.8 | 0.8 |

The expert families, by kernel name:

| kernel | calls | ms | ms/call |
|---|---:|---:|---:|
| `sm70_marlin_moe_gemm_kernel` (two instantiations) | 48 + 576 | 156.9 + 142.7 | 0.48 |
| `decode_moe_gemv_kernel` (ours, no M tile) | 528 | 32.3 | **0.061** |
| `fused_moe_kernel` (Triton, draft model) | 64 | 51.4 | 0.80 |
| `moe_sum_reduce_kernel` | 577 | 17.1 | 0.03 |
| `moe_align_block_size_kernel` | 608 | 3.6 | 0.01 |

**Both paths run, and Marlin carries 85% of the expert time** (299.6 of
352.5 ms), at **8x the per-call cost of our own kernel**. The window held 40
verify steps (`step[VERIFY bs=1]`, 13.1 ms span each); the trace also holds two
`step[EXTEND bs=1 toks=4096]` spans totalling 1536 ms, so it is **not
decode-pure** despite the staging and those spans are excluded from every
statement above.

### Which shape each path serves — inference, stated as such

Per rank after TP4 the routed shapes are **w13 n=320, k=2560** and **down
n=160, k=2560**. Our kernel's admission (`decode_gemv.supports`) requires
`n % 64 == 0` and an odd `n/64`, so w13 passes (5) and down fails
(`160 % 64 = 32`); and it requires `size_m <= MAX_M = 8`, which at MTP4 means
batch sizes up to 2.

The served log settles the capture side: at bs=8 capture,
`decode gemv skipped: unsupported m=32 n=320 k=2560 topk=10 block=8` — the w13
call, refused by the M cut, **at capture time**. At runtime in this window
bs=1, m=4, so w13 is admitted. The instance counts agree with that reading
(528 = 48 layers x 11 windows for one shape; Marlin's two instantiations cover
the other), but **the census counts kernels, not shapes**: without a
correlation-to-shape join the split is inferred from eligibility plus counts.
That join, or a counter inside the kernel wrapper, is the one measurement that
would close it.

## Attribution: the M tile, not split-K, not the repack

Our own module documents the mechanism
(`sglang_restomod/quant/sm70_fp4_moe/decode_gemv.py:1-8`):

> The pinned Marlin kernel runs a 64-row M tile whatever M is, so at decode one
> expert's 64-row tile carries about one real row and the call reads its
> weights at 16% of the card's bandwidth.

`1 / 0.16 = 6.25`, against 5-8x measured in two windows. Split-K is already
present (`K_SPLIT = 4`) and the repack layout is decoded by hand — it is an
eligibility constraint, not a rate. The reference's decode expert path has no M
tile, which is why its per-call cost is a fraction of ours.

**Where the fix is cheap and where it is not.** w13 already runs the no-tile
kernel at bs<=2, so raising `MAX_M` (with the workspace check that
`decode_gemv._workspace_for` exists for — a resize inside a CUDA-graph capture
is an illegal memory access) is the cheap half. Down cannot reach the kernel at
all: `160 % 64 != 0`, and the module records a measured wash for the even-N
shape (49.3 us against Marlin's 48.3 us) from before the sharding made that
case moot. A down fix is a different kernel or a padded/repacked 192-wide tile.

## What closing the buckets is worth

Per verify step, from the traced arm and the clean-box dense measurement:
expert 5.8-6.8 ms (17-20%) and dense 10.5-12.3 ms (31-37%); on the clean box
the dense cluster is 9.5-11.3 ms of a 26.8-28.6 ms step (35-42%). Replacing
both rates with the reference's (1.0 and 2.2 ms) removes 12.1-14.9 ms of a
26.8 ms step: 3.10 tokens/step at 11.9-14.7 ms is **210-260 tok/s on four
cards, 52-65 per card**, against the reference's measured 55.9 (and 66.6 in
the earlier run). That is 1.8-2.1x, and it is the stated assumption — that the
reference's per-call rates are attainable at our shapes and accept — which the
two buckets' measured 5-8x per-call ratio makes plausible rather than proven.

## What this does not say

- It does not say the dense bucket can reach 2.2 ms on our stack. It says the
  precision is not the reason it does not.
- It does not separate w13 from down in the trace; that is the open join.
- It does not touch capacity. The same policy inverts on `bluewhale`, where
  residency is 18.4% and the host tier costs ~45 ms/token — that is the
  capacity page's result, and it is why decode is the part of this work that
  transfers.
