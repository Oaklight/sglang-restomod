# 2026-11-16 — The decode GEMM gap: where it is, and what removing it is worth

Status: **measured, with one partial capture.** The per-card decode deficit is
a kernel gap, not a capacity gap: dense and expert GEMM hold **63% of the
recorded verify-window kernel time** and run 5-8x slower per call than the
reference's, and the expert half localises to the projection our fast kernel
cannot read. One arm was served and one decode window was traced for this page;
the window records about one step in three, and every share below says so. The
capacity model is a separate page
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

## The decode-window census, and the correction it needed

One `profile_decode.py` run (`/start_profile` with `profile_stages=["decode"]`,
warm 4096-token prompt with a full cache hit) on the same arm; one TP-0 trace.

**Window totals, so every share below converts to milliseconds:** 34,107 kernel
instances totalling **1389.5 ms**, holding 40 `step[VERIFY bs=1]` spans of
**522.5 ms** wall. Of that kernel time only **238.3 ms / 23,897 instances** fall
inside the verify spans.

### Whole-capture shares — prefill-weighted, and not quotable as decode

| family | ms | % |
|---|---:|---:|
| collective | 343.4 | 24.7 |
| dense GEMM | 322.8 | 23.2 |
| expert GEMM (Marlin) | 299.6 | 21.6 |
| other | 279.6 | 20.1 |
| expert GEMM (other) | 104.3 | 7.5 |
| linear-attn (GDN) | 28.9 | 2.1 |
| attention | 10.8 | 0.8 |

**This table is not the decode window, and it was first reported as one.** The
capture also holds two `step[EXTEND bs=1 toks=4096]` spans (1536 ms) and two
`step[DRAFT toks=4096]` spans, despite the decode-only staging, and the large-M
prefill kernels dominate several families inside it: the Marlin calls average
0.48 ms across the capture but 43.6 us inside verify, and the 1-shot all-reduce
accounts for 285.1 ms across the capture against 8.0 ms inside verify.

### Verify-spans only, with per-step rates

| family | ms | % of 238.3 | per step (40) |
|---|---:|---:|---:|
| dense GEMM | 95.8 | 40.2 | 2.40 ms |
| other | 59.9 | 25.1 | 1.50 ms |
| expert GEMM (other: ours, Triton, reduce) | 34.1 | 14.3 | 0.85 ms |
| expert GEMM (Marlin) | 20.9 | 8.8 | 0.52 ms |
| linear-attn (GDN) | 11.9 | 5.0 | 0.30 ms |
| collective | 8.7 | 3.7 | 0.22 ms |
| attention | 6.9 | 2.9 | 0.17 ms |

Per name inside the verify spans: our `decode_moe_gemv_kernel` 480 calls at
**61.4 us**, `sm70_marlin_moe_gemm_kernel` 480 calls at **43.6 us**,
`all_reduce_1shot` 983 calls at **8.1 us**, NCCL ring 12 calls at 63.0 us.

**The capture is partial and the shares are upper bounds, not a census.** Twelve
GEMV calls and twelve Marlin calls per step against roughly 36 expert layers
(the module's own note), and 24.6 collectives per step against the ~96 a 48-layer
TP4 step issues, say one step in three is recorded. So the family table can rank
what is large, not what a fix would save — those savings need the ms totals
above plus a complete window.

**Two readings this replaces.** The collectives are **not** the largest decode
family: at bs=1 they are 8.1 us per call and 0.22 ms per captured step, i.e.
latency per hand-off rather than bandwidth, and the whole-capture 24.7% is the
4096-token prefills. And the expert time is **not** Marlin-dominated at decode:
inside verify the two paths are 20.9 ms and 29.5 ms. Both paths run, at
1.4:1 in Marlin's favour, which is a different statement from the 85% the
whole-capture table implies.

### Which shape each path serves — inference, and the instrument that would settle it

Per rank after TP4 the routed shapes are **w13 n=320, k=2560** and **down n=160,
k=2560**. Our kernel's admission (`decode_gemv.supports`) requires
`n % 64 == 0` with an odd `n/64`, so w13 passes (5) and down fails
(`160 % 64 = 32`); and it requires `size_m <= MAX_M = 8`, which at MTP4 admits
batch sizes up to 2. The served log confirms the capture side
(`decode gemv skipped: unsupported m=32 n=320 k=2560 topk=10 block=8`, the w13
call refused by the M cut at bs=8), and at runtime bs=1, m=4, so w13 is
admitted.

That the two 480-call series are one shape each is **inference from eligibility
plus counts**, not from the trace. The cheap instrument is a counter, not
another profile window: an `int[4]` incremented by `n` inside `maybe_call` and
inside the Marlin dispatch, printed at teardown, answers which shape each path
serves and what each costs per call.

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
all: `160 % 64 != 0`. The module records 49.3 us for our GEMV against Marlin's
48.3 us on that shape, which was read as "a wash, not worth pursuing" — that
reading is retired: Marlin is the slow path it has to beat, and in the verify
window above Marlin spends 43.6 us per down call where our w13 call, twice the
work, spends 61.4 us. So the wash says our kernel barely beat the slow path on
that shape, not that the shape is hopeless. The constraint is `160 % 64 != 0`,
so the fix is a build — a 192-wide padded tile or a different kernel — not a
threshold change.

## What closing the buckets is worth

Per verify step, from the **earlier traced arm** (lambda1 4-7, profiler on
both engines) and the clean-box dense measurement: expert 5.8-6.8 ms (17-20%)
and dense 10.5-12.3 ms (31-37%); on the clean box the dense cluster is
9.5-11.3 ms of a 26.8-28.6 ms step (35-42%). Today's window is consistent in
order after its one-in-three scaling (1.26 ms/step recorded for the two expert
paths, ~3.8 ms scaled) but is not precise enough to replace those figures. Replacing
both rates with the reference's (1.0 and 2.2 ms) removes 12.1-14.9 ms of a
26.8 ms step: 3.10 tokens/step at 11.9-14.7 ms is **210-260 tok/s on four
cards, 52-65 per card**, against the reference's measured 55.9 (and 66.6 in
the earlier run). That is 1.8-2.1x, and it is the stated assumption — that the
reference's per-call rates are attainable at our shapes and accept — which the
two buckets' measured 5-8x per-call ratio makes plausible rather than proven.

## What this does not say

- It does not say the dense bucket can reach 2.2 ms on our stack. It says the
  precision is not the reason it does not.
- It does not separate w13 from down in the trace; a wrapper counter settles
  that faster than a join, and it is the next instrument.
- It does not rank the decode families reliably: the capture holds roughly a
  third of a step's kernels, so the shares bound the order, not the savings.
- It does not touch capacity. The same policy inverts on `bluewhale`, where
  residency is 18.4% and the host tier costs ~45 ms/token — that is the
  capacity page's result, and it is why decode is the part of this work that
  transfers.
