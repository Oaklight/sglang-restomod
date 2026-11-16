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

### Which shape each path serves — what was inference, and what the counter settled

Per rank after TP4 the routed shapes are **w13 n=320, k=2560** and **down
n=2560, k=160**. (The `160` is down's K, not its N — `FF/tp = 640/4`; an earlier
version of this page wrote it as down's N and read every downstream number off
that error.) Our kernel's admission (`decode_gemv.supports`) requires
`n % 64 == 0` with an odd `n/64`, so w13 passes (`320/64 = 5`) and down fails
(`2560/64 = 40`, even); and it requires `size_m <= MAX_M = 8`, which at MTP4
admits batch sizes up to 2. The served log confirms the capture side
(`decode gemv skipped: unsupported m=32 n=320 k=2560 topk=10 block=8`, the w13
call refused by the M cut at bs=8), and at runtime bs=1, m=4, so w13 is
admitted.

That the two 480-call series are one shape each was **inference from eligibility
plus counts**, not from the trace. The counter that settles it was built and run
(PR #313, `SM70_MOE_CENSUS=1`): one dict keyed `(path, n, m)`, incremented at the
GEMV entry and unconditionally at the Marlin call site, printed as one INFO line
per 256 dispatches.

### What the counter found (lambda6 0-3, TP4, two served arms, 3328 dispatches each)

**w13 (`n=320`), both arms, identical counts:**

| path | m | calls |
| --- | --- | --- |
| `gemv` | 4 | 144 |
| `gemv` | 5 | 176 |
| `gemv` | 8 | 144 |
| `marlin` | 12, 16, 20, 24, 28, 32 | 144 each |
| `marlin` | 25 | 96 |
| `marlin` | 122 | 48 |
| `marlin` | 4096 | 192 |

**down (`n=2560`), both arms, identical counts:** `m=40` (144), `m=50` (176),
`m=80` (144), `m=120`, `m=160`, `m=200`, `m=240`, `m=250` (96), `m=280`,
`m=320` (144 each), `m=1220` (48), `m=40960` (192) — every one on Marlin.

Three things this settles:

- **The two series are one shape each, and the pairing proves it.** Every w13 row
  has a down row at exactly ten times its `m` (`4→40`, `5→50`, `8→80`, `12→120`,
  … `4096→40960`), which is `top_k = 10`: down's `m` counts route rows, as the
  eligibility note says. The counts match one for one, so the census sees the
  same MoE applications through both projections rather than two populations
  that happen to be the same size.
- **The M cut is real and it is at 8.** GEMV serves `m` 4/5/8 and nothing above;
  the first Marlin w13 row is `m=12`. The eligibility arithmetic predicted this
  and the served rows confirm it, so `MAX_M` is a genuine threshold on this path
  rather than a guess from the source.
- **Down never runs our kernel, at any `m`, exactly as the N rule says.** It is
  the `n=2560` series without exception. Its per-call cost is therefore a Marlin
  cost at a shape our kernel refuses, and the lever there is an admission
  decision (or a kernel that reads an even `n/64`), not `MAX_M`.

**The GEMV path earns its place.** The two arms differ in exactly three rows —
`m=4/5/8` on `n=320` — and nothing else: the down series is byte-identical in
both. With `SM70_MOE_DECODE_GEMV=0` the step went from **25.29 ms to 27.36 ms
(+2.07 ms, +8.2%)**, above the 3.8% drift floor between two serves. Deleting the
path would cost about 8%, not save it. The accept rate moved with the arm (3.12
against 2.74) because two implementations of the same GEMM round differently and
the draft chain amplifies it; that is a numerics effect, so the step is the
kernel result and the completion rate (123.4 against 100.3 tok/s) is not.

The counter's own cost: the arm that carried it ran 25.29 ms per step against
24.74 ms without it, accept equal at 3.12. That is ~2%, inside the 3.8% drift
floor, so the instrument is not what moved the arm.

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
is an illegal memory access) is the cheap half: the census shows the threshold is
where the source says it is, and the rows above it are Marlin's 64-row tile at
`m=12` and up.

### Correction: which width is down's, and what that changes

Two earlier versions of this section named the wrong quantity, and the census is
what exposed it.

- **The first version said down cannot reach our kernel because `160 % 64 = 32`.**
  That used down's **K** (`FF/tp = 640/4 = 160`) as its **N**. Down's N is 2560,
  and it fails the admission on the *other* half of the N rule: `2560/64 = 40`,
  which is even, and `decode_gemv.supports` requires an odd `n/64`. The census
  shows the consequence directly — every down call, at every `m` from 40 to
  40960, is on Marlin, and there is no GEMV row at `n=2560` at all.
- **The second version, written on the strength of "no `n=160` row anywhere",
  read that absence as "down never reaches this call site" and withdrew the
  attribution entirely.** There is no `n=160` row because no tensor in this
  model has that width. The down rows were in the census all along, at
  `n=2560`, paired one-for-one with the w13 rows at ten times their `m`.

**What the retraction took away is therefore restored, with a better basis:** the
down call is a Marlin call at `n=2560` at every served `m`, and its per-call
cost is the 64-row tile's cost at that width, observed rather than inferred. The
earlier 1.87x reading of it is a ratio between our w13 call and Marlin's down
call, which is two different shapes and must not be quoted as a per-shape
kernel gap.

**What does not change:** the lever on down is not `MAX_M`. It is either an
admission change (accept an even `n/64` at `n=2560`, which needs the kernel to
be verified at that width — the recorded 49.3 us against Marlin's 48.3 us there
says the win is not free) or a kernel that reads the shape well. Neither is a
threshold edit.

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
- Its per-call figures do not compose into a step: the 61.4 us GEMV call against
  the 43.6 us Marlin call points the wrong way, while turning the GEMV path off
  costs 8.2% per step. Any further expert-kernel proposal has to explain that
  gap before it can use a per-call number as its case.
- It does not rank the decode families reliably: the capture holds roughly a
  third of a step's kernels, so the shares bound the order, not the savings.
- It does not touch capacity. The same policy inverts on `bluewhale`, where
  residency is 18.4% and the host tier costs ~45 ms/token — that is the
  capacity page's result, and it is why decode is the part of this work that
  transfers.

---

## Addendum 2026-11-16: the reference's expert rate, and the 5-8x this page opened with

Status: **the 5-8x per-call ratio is retracted.** Our side of it stands; the
reference's side was measured through a window that holds 22% of a step's
kernels. Corrected, the per-call gap is 1.59x and 1.87x, per step 1.70x, and the
closing arithmetic in "What closing the buckets is worth" moves from 210-260
tok/s on four cards to 196-221.

### The measurement error, and the method that fixes it

`decode_slice.py`, and everything derived from it, attributes kernels only inside
a vLLM user annotation. The only decode annotation vLLM 1.5.1 emits on this
stack is `execute_context_0(0)_generation_1(5)`, one piecewise-graph piece: it
holds **8.4 ms of the 37.9 ms of kernels** that make up a step, 22%. This page
already warns that our own capture holds about one step in three; the
reference's number carried the same defect and nothing flagged it.

Method to reuse: pick an annotation that repeats exactly once per step, sort by
start, measure `[first.start, last.start]` and divide by `N-1`. Both ends are the
same phase of a step, so the CPU-span/GPU-lag clipping cancels. Then convert to a
per-step total with the structural launch count from the source and the mean
per-call duration, which is the quantity that survives the misalignment
(`sm70_fp4_moe/method.py`: "one decode step is 48 layers x 2 calls"). Tool:
`kernelcmp/scripts/phaseattr.py`.

### Per call, both sides

Both are modelopt NVFP4 on V100-SXM2, TP4, bs=1, top-10 of 512 experts, in a
CUDA-graphed decode.

| projection | ours | reference | ratio |
|---|---:|---:|---:|
| w13 / gate_up | 62.5 us (`decode_moe_gemv_kernel`, 48 calls) | 39.3 us (`nvfp4_qpn_m1_sm70_kernel<4>`, 48 calls) | 1.59x |
| w2 / down | 44.3 us (`sm70_marlin_moe_gemm_kernel`, 48 calls) | 23.7 us (`nvfp4_qpn_m1_sm70_kernel<1>`, 48 calls) | 1.87x |
| per step, target | 5.13 ms | 3.02 ms | 1.70x |
| per step, incl. the MTP layer | 6.61 ms | 3.88 ms | 1.70x |

**Bucket share and ceiling, both required beside the ratio:** the expert bucket is
**19.1% of a decode step**, so matching the reference's per-call times buys
**1.13x** and deleting the bucket outright buys 1.33x. "1.70x" is not "70% to win".

Our per-call figures are confirmed on an independent arm: this page's own
verify-window census measured 61.4 us and 43.6 us on lambda4, within 2% of the
62.5 and 44.3 above (lambda6). The reference's are means over 48 calls each, in
its own trace.

Two corrections to the attribution text above:

- The reference's per-layer routed MoE is **not `fused_moe_kernel`**. That Triton
  kernel appears 8.0 times per step in their trace and 5.3 in ours, which is the
  MTP layer's two projections (in our stack `un.py` keeps `mtp.*` unquantized and
  therefore on Triton, and the reference's count, 4 draft forwards x 2, fits the
  same owner). Their per-layer path is the TurboMind qpn pair in the table.
- `1 / 0.16 = 6.25` was the basis of the retired 5-8x. The M tile is real and it
  is the whole of the w2 gap (1.87x); it cannot carry a 6x, and it does not apply
  to w13, which already runs a no-tile kernel.

### What it does to the closing arithmetic

The expert term in "What closing the buckets is worth" was the reference's ~1.0
ms/step. Measured, it is 3.88 ms. Replacing only that term, keeping the page's
dense figures:

- expert: 6.61 -> 3.88 ms, -2.73 ms
- dense: 10.5-12.3 -> 2.2 ms, -8.3 to -10.1 ms
- step: 26.8 -> 14.0-15.8 ms
- at 3.10 tokens/step: 196-221 tok/s on four cards, 49-55 per card

**That projection is withdrawn the same day, because its dense term is the same
artifact as the expert term.** The reference's 2.2 ms for dense came from the 22%
window. De-clipped, same trace, same divisor:

| family, per step | ours (lambda6, QPN8 arm) | reference (lambda1) |
|---|---:|---:|
| **dense linears** | **7.90 ms** (709 launches, 11.2 us) | **12.41 ms** (843, 14.7 us) |
| routed experts | 5.13 ms (96) | 3.02 ms (96) |
| expert, third kernel | 1.48 ms | 0.86 ms |
| collectives | 0.61 ms | 10.96 ms (profiler-inflated) |
| memcpy / elementwise | 1.18 ms | 4.05 ms |

**We spend 1.6x less GPU time on dense linears, not 5x more.** So the projection
has no dense term: the end state is the expert-only 11.3%, about **1.11x**, and the
honest summary is that there is **no kernel work available on our side that changes
the step materially**.

The divisor is pinned rather than assumed: the reference's
`_sm70_qwen38_router_topk_kernel` fires **52.0 times per de-clipped step** — 48
layers plus 4 MTP forwards — so the per-step normalisation is right to +-2%.

**What is not established, and must not be written down yet: that the remaining
difference is "overlap".** The tempting sentence is that the reference's 37.9 ms of
kernels fit inside a 19.4-25.8 ms wall, hiding ~15 ms. **That wall is derived, not
measured** — it is `7.71 ms/output-token x accept`, and the reference's accept is
not separately measured, because their bench prints ms/output-token only. At the
marginal acceptance our own arm reaches (3.84), the same product gives ~29.6 ms and
the hidden time falls to ~8 ms. So overlap is **bounded between roughly 1.3x and
2x by an unknown accept**: a live candidate, not a result. What settles it is their
measured wall for a known output length, with no derivation in the chain.

### Which side each number is on

Three of the four arms below are ours, and an earlier draft of this addendum
compared our step against our own step from a different arm. Pinned, four
numbers per line:

| arm | host | ms/output-token | step | accept | tok/s |
|---|---|---:|---:|---:|---:|
| ours, kernelcmp arm | lambda1 4-7 | 10.49 | 28.0 ms (marginal, corrected) | 2.67 (marginal) | 95.3 |
| reference, wheel defaults | lambda1 4-7 | 7.71 (slope) | 19.7-25.8 (derived) | 2.55-3.34 (their metrics log) | 129.6 |
| ours, fp8 dense on | lambda4 4-7 | 7.93 | 24.74 ms (measured) | 3.12 (measured) | 126.2 |
| ours, the traced arm | lambda6 0-3 | 8.02 | 25.05 ms (measured); trace period 26.83 | 3.12 (measured) | 124.6 |

The lambda1 row carries a correction of its own. That bench file prints
`DECODE step 33.59 ms  accept 3.20 throughput 95.3 tok/s`, but it divided the
timed runs' step time by a `spec_verify_ct` taken from a separate calibration
request (40 steps). The timed runs had 48/36/48 steps, accept 2.67/3.56/2.67 and
a marginal 28.0 ms; the rate is unaffected because it is 1/slope across the timed
runs. So 33.59 / 3.20 describes no single request and is not used above.

The reference's step and accept are not separately measured: their bench prints
ms/output-token and their log prints a mean acceptance length per interval, so
only the product is pinned, and 19.7-25.8 spans the intervals observed. Their
accept range overlaps ours, so **no accept difference is established**. The one
quantity measured on both sides of the same host is ms/output-token: 10.49
against 7.71, **1.36x**. On our later arms it is 7.93-8.02, within 3-4% of the
reference, cross-host and therefore indicative only.

### Why the gap is 1.7x and not 6x: occupancy, not bandwidth

Per launch there are ~10 route tiles of work. w13 is N=320 per rank with
CTA_N=256, so the grid is ~2 CTAs wide by 10 routes: ~20 CTAs on 80 SMs.

| | w13 | w2 |
|---|---:|---:|
| bytes read per launch per rank, at most (10 distinct experts) | 4.61 MB | 2.30 MB |
| ours | 74 GB/s, 8.2% of 900 | 52 GB/s, 5.8% |
| reference | 117 GB/s, 13.0% | 97 GB/s, 10.8% |

Neither is bandwidth-bound, which closes the bandwidth hypothesis in one line and
rules out a bandwidth-shaped fix. What the reference does better is delete work:
its w2 has no M tile, and it fuses swiglu into w13 and the top-k reduce into w2,
where we launch `moe_aux` separately (0.64 ms/step) plus a separate
`act_and_mul`. Split-K is not the lever either: `_select_split_k` already picks
per projection from measurement, and one value for the model costs 8.42-8.60 ms
against 6.98-7.21.

The same Marlin kernel runs at 34.2% of the prefill window and 53.8% of fp16 peak
at M=4096, so the kernel is not the defect; small M is.

### Collectives, cross-stack

The reference's step spends **10.96 ms in two cross-device reduce kernels plus
4.05 ms of memcpy**; ours spends **0.61 ms in the 1-shot allreduce**. Not a ratio
and not comparable in isolation: spin-wait kernels are inflated by the profiler
and by whatever the other rank is doing, so their absolute duration is an upper
bound. What survives is the direction: our collective path is not the deficit,
and theirs pays for its own sharding far more than we pay for ours.

### What this addendum does not say

- It does not say 1.70x is fixable. w13's gap is reachable today; w2's is not,
  because `N=2560` gives `N/64 = 40`, an even multiple that the no-tile kernel's
  admission refuses, and the alternative is a kernel that reads that width —
  Marlin there is a prebuilt `.so` and stays the baseline.
- It does not establish the MTP owner from the trace. That is inference from our
  own loader and from the reference's call count.
- **It supersedes two figures in the page body above.** `bluewhale` residency is
  **9.8%**, not 18.4%, and the host tier costs **~92 ms/token**, not ~45, per the
  capacity page's per-tensor rebuild of the expert term. The original sentences
  stand, quoted, rather than edited — the same rule this addendum applies to the
  5-8x.
- It does not rank the step from kernel sums. The reference's 37.9 ms of kernels
  per step cannot fit its own 19.4-25.8 ms wall, so that step has multi-stream
  overlap and only per-call durations and source-side launch counts may be
  compared across the two traces.
