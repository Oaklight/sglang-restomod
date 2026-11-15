# 2026-11-15 — Competitor numbers by card count

Status: **partial**. 1Cat is measured at 4 cards only. Strata has measured
points at 1 and 2 cards from another lane, none from this one. Every cell not
measured is left empty; nothing here is interpolated.

Reading the cells:

- **measured** — run by this lane, on the box and protocol stated.
- **reported** — measured by another lane in this lab; I do not hold the trace.
- **claimed** — the project's own material, protocol quoted as published.
- **empty** — not measured.

Reading a rate: **every tok/s here carries its inputs — `completion_tokens`,
`spec_verify_ct`, `elapsed` — so a reader can re-derive it in one line instead
of taking it on trust.** Two derived figures have already circulated in this lab
as if measured, and both were caught by someone re-deriving them.

## Headline

At 4 cards, SPEC=1: **decode is not distinguishable from parity with 1Cat.** Our
median is **115.5 tok/s** on lambda6 against their 129.6 — a 12% gap — while our
own three-prompt spread is **104-145 tok/s**, about four times that gap. Any
claim finer than "roughly level" sits inside our own variance, and 1Cat's side is
a single measurement.

The decomposition survives the variance, so it is the useful statement:

| arm | ms per verify step | tokens per step | tok/s |
|---|---:|---:|---:|
| 1Cat TP4, MTP4 | 19.4 | 2.51 | 129.6 |
| ours TP4, SPEC=1, lambda6 median | 26.8 | 3.10 marginal | 115.5 |
| ratio | **1.38x slower per step** | **1.23x better accept** | 1.12x behind |

Read the decomposition carefully: the two step figures are **not measured the same
way**. Ours — 26.8 ms and 3.10 tokens per step — are both marginal and both from
our own two runs, so they are internally consistent. 1Cat's 19.4 ms is
`slope x average accept` at its published average 2.51, which assumes that
average held across the 32-to-128 window; we do not hold their short-run verify
count and will not invent one. The throughput ratio **1.12x is solid**, because
both sides are `1/slope` with no accept in the arithmetic. The step/accept split
is the shape of the difference, not a like-for-like measurement.

Read conditionally, it says: if 1Cat's average accept held over that window, our
per-step kernel time is 38% worse and our acceptance 23% better, and the two
nearly cancel. On that reading the kernel deficit is real and larger than the old
"27%" framing suggested, it is currently paid for by draft quality, and improving
accept further buys less than fixing the step.

Two arms, two boxes, not averaged: **lambda1 95.3** (one arm, old bench) and
**lambda6 115.5** median (range 104-145). Same tree and NEXTN on both; accept is
what moves (2.67 on lambda1, 2.84-3.56 on lambda6).

At SPEC=0 we measure **58.5-58.8 tok/s** (accept 1 by construction) on two boxes
and have nothing to compare it to: 1Cat publishes only its MTP arm.

**Prefill**, our first pair with MTP on both sides: **2971 tok/s**, SPEC=1 on
lambda6, cold (`cached=0`), against 1Cat's 4825 MTP4 — **1.62x theirs, we trail
38%**. Like for like in arm, unlike in box: the same arm on the reported clean
island runs at 4048-4092 tok/s, so about **13-14% of that gap is the lambda6
host** and the rest is not attributed. The contended-island 3450-3565 stays
withheld from the verdict.

**Cross-quant caveat that applies to the whole note.** Strata runs IQ2_XS
(35.5 GB of experts) against our fp8/fp4 (125.9 GiB). That is not a tok/s
comparison: the work per token differs by construction, and quality was never
measured. Strata's published claim is also a V100-**PCIE** card; these
measurements are SXM2 over NVLink 2.0.

## 1Cat (1cat-vllm 1.5.1, Qwen3.8-Flash-Next-NVFP4, TP4)

| cards | prefill tok/s | decode (accept) |
|---|---|---|
| 1 | — | — |
| 2 | — | — |
| 4 | **4825 measured** (4121 prompt tokens, 849 ms median) | **129.6 measured**; inputs: 128 completion tokens, 51 verify steps, 1317 ms long run (short 32 tokens in 576 ms) → slope 7.72 ms/output-token, accept 2.51 → 19.4 ms/step |
| 8 | — | — |

4-card protocol (measured 2026-11-15, lambda1 GPUs 4-7, one NVLink island,
load 5-7 on 80 cores; performance-tuner held 0-3):

- recipe = the wheel's own `serve_flash_next_nvfp4_v100.sh`: TP4, MTP4
  (`num_speculative_tokens: 4`), `max_model_len 131072`,
  `max_num_batched_tokens 8192`, `max_num_seqs 1`, `gpu_memory_utilization
  0.90`, prefix caching, `kv-cache-dtype auto`, `--no-async-scheduling`.
- checkpoint: NVIDIA's original NVFP4, whose MTP sits in the FP8 path.
- prefill: 4121 prompt tokens, `max_new_tokens=1`, unique nonce per run, 3 runs
  = 1232 / 849 / 828 ms → 3346 / 4854 / 4975 tok/s; median 4825. The first run
  includes JIT.
- decode: one fixed 4101-token prompt, timed at 32 and 128 output tokens; slope
  (1317 - 576) / 96 = 7.72 ms/output-token = 129.6 tok/s. vLLM metrics over a
  128-token request: 51 drafts, 79 accepted draft tokens, 128 completion → 2.51
  tokens per verify step → 19.4 ms/step.
- kernel time inside the step: 8.43 ms on rank0 (torch profiler). Collective
  kernels are inflated by the profiler; they are not used in the verdict.

Claimed, not measured here (lab note citing `benchmarks/flash-next-lambda4-tp4.md`):
TP4 Flash-Next at 8K context, prefill 5512 tok/s, decode 94.1 tok/s. Batch and
concurrency are not stated in that note, so it is not directly comparable.

1/2/8 cards: empty. The shipped launcher requires four peer-connected
V100-SXM2-32GB; no 1/2/8 recipe exists in its material.

## Strata (0.1.59.1 @ `13eedf55`, GGUF quant, `--spec 2`, single stream)

| cards | prefill tok/s | decode tok/s |
|---|---|---|
| 1 | **1287.4 reported** | **66.6 reported** (15.0 ms/token) |
| 2 | **1695.8 reported** (layer split K=24) | **80.1 reported** (12.5 ms/token; 40.1 per card = 60%) |
| 4 | — | — |
| 8 | — | — |

The reported rows are the multi-node lane's, on lambda11 (V100-SXM2), IQ2_XS,
6,521-token varied prompt, 64 new tokens, `temperature=0`, one stream
(`work/strata-run/RESULT-strata-two-card.md`). I do not hold the trace.

Two lines travel with the two-card row: two cards buy **1.20x decode and 1.32x
prefill** — pipeline parallelism on one host, which is capacity, not scaling —
and draft acceptance was 12/12 and 16/17, so **decode is the optimistic end of
these numbers**.

- Measured by this lane: **none completed**. Build at `13eedf55` (sm70, CUDA
  12.8) succeeded and the Q2_0 GGUF (66.4 GB) was downloaded; the single-card
  point was not run before the lane was reassigned.
- The earlier single-card figure of **36.8 tok/s** (`smoke2.log:58`: 8 tokens in
  217.3 ms) is a **cold 8-token smoke decode** — two or three draft rounds with
  the first-window overhead dominating. It is not steady state and not a peer
  point to the 66.6 above; the same file records 48.1 tok/s for an 8-token warm
  decode on a 57-token prompt, the same shape. **66.6 supersedes it as the
  single-card point** (6,521-token prompt, 64 generated, 961 ms, drafts 12/12,
  expert cache 98.5%). Both are now labelled by shape rather than set side by
  side.
- Claimed by the project (`strata/docs/NVIDIA_V100.md`, `OLDER_GPUS.md`): one
  V100-PCIE-32GB, **UD-IQ4_XS**, int8 KV, MTP `--spec 2`: prefill 1164 / 1251 /
  1123 tok/s at 7,194 / 28,650 / 114,338 prompt tokens; decode 38-51 tok/s,
  flat across that range. No V100 claim at 2/4/8 cards.
- 4 and 8 cards: empty; the project documents 2-4 cards and calls 8 untested.
- The reported rows and the claim use **different quants** (IQ2_XS vs
  UD-IQ4_XS). The reported 66.6 and the claimed 38-51 are not comparable to each
  other.

## Our stack, same table

| cards | prefill tok/s | decode (accept) |
|---|---|---|
| 1 | not runnable | not runnable |
| 2 | not runnable | not runnable |
| 4 (single node) | **3450-3565 reported, contended island, components include `gdn_tilelang_sm70` — withheld from the verdict**; **4199 reported, SPEC=0, host unconfirmed**; **1549 measured, pair off, SPEC=1** (4121 prompt tokens, 2644 ms median); 1663 in the SPEC=0 arm | **95.3 measured, SPEC=1** (slope 10.50 ms/output-token; 128 completion tokens, 48 verify steps, 1627 ms long run, short 32 tokens in 619 ms) → accept 2.67 → 28.0 ms/step. Accept varied by request: 2.67 / 3.56 / 2.67 at 48 / 36 / 48 steps, so the marginal step is 28.0-37.4 ms. **58.8 measured, SPEC=0** (slope 17.00 ms/token, accept 1); 1Cat ships no SPEC=0 arm, so it has no peer |
| 4 (dual node) | comm-bound at 66% (reported), no number held here | — |
| 4 (single node, lambda6) | **3593 measured, SPEC=0** (1147 ms, runs 1-2, run 0 JIT-excluded, spread 5.4%); **2971 measured, SPEC=1** (4121 prompt tokens, 1387 ms median, cold `cached=0`, spread 5.9%) | **115.5 measured median, SPEC=1** (104-145 across three prompts; per-run triples in the comparability table); **59.0 measured, SPEC=0** (16.95 ms/step, accept 1) |
| 8 | untested | untested |

Why 1 and 2 cards are called *not runnable* rather than empty: Flash-Next
carries 63.3 GiB of routed experts against 32 GB per card, so four cards are
the minimum. Eight is untested, not impossible.

4-card protocol: lambda1 GPUs 4-7, flashnext site (sglang v0.5.23 + patches,
restomod), model-sm70, TP4, NEXTN steps 3 / topk 1 / draft 4, ctx 65536,
chunk 4096, memfrac 0.82, attention qsa_v100, linear-attn prefill `triton`,
PLE offload, prefix cached (`cached_tokens=4096`).

## Prefill attribution (corrected)

An earlier draft of this note attributed the 2.1x prefill difference to
`--linear-attn-prefill-backend`. That was wrong: both arms ran `LATB=triton`.

The measured differentiator is the **component list**. Selecting
`gdn_tilelang_sm70` (the GDN TileLang pair from #275 + #287) on the calibration
box, same tree, same prompt, same `LATB=triton`:

- pair **on**: 3450-3565 tok/s
- pair **off** (`SM70_GDN_TILELANG=0`): 1476 tok/s

My pair-off numbers on lambda1 — 1549 at SPEC=1, 1663 at SPEC=0, a different
box — are that same configuration, not a conflicting result.

Neither arm of that A/B has run on a clean island, so its magnitude is open.

**What the prefill pairs now decompose into.** The MTP factor is measured on one
box; the host factor is only reported:

- measured, lambda6: SPEC=0 3593 → SPEC=1 2971 tok/s, factor **1.21**
- reported, clean 4-card island pair-on SPEC=0 at 1007-1018 ms, i.e. 4048-4092
  tok/s, against lambda6's 1147 ms: factor **1.13-1.14**

Product 1.37-1.38, which is exactly the ratio between the reported 4048-4092 and
2971 (1.36-1.38). So on the only lambda4 evidence that exists, the gap is **fully
accounted for by host x MTP with no residual**. The 4199 implies 976 ms, about 3%
faster than the reported 1007-1018 ms; using it leaves 2-3% unexplained, which is
one more reason not to build on it.

### The MoE-path lead is closed, and it closed the other way

`Using default MoE kernel config` is entered **only under SPEC=1**: eight times on
lambda6 SPEC=1, zero on lambda6 SPEC=0 on the same box (`fused_moe_kernel took`
also 0; `GDN prefill tracking guard installed` 4 in both). So the Triton fused-MoE
runner is the **draft model's** experts, not the target's and not a host property.
lambda4's log showing zero was never evidence of a better path — that server never
served a request, so it never reached the first forward where the lookup happens.
Dropped from the candidate list; the remaining cause is the box.

### The 4199 is reported, host unconfirmed

No lambda4 log in the shared log directory contains a served request; every
`launch-11480*.log` carrying `Prefill batch` lines is lambda6 or lambda1. The 4199
quoted earlier as "lambda4 0-3, clean island" therefore has **no log I can
produce**. It is corroborated only by the reported 1007-1018 ms and cannot be
attributed to a host from the record. It stays in the table as **reported, host
unconfirmed**; no row's verdict depends on it, and every decomposition that uses
it says so.

Rule: **never cite a prefill number without the component list.** The
difference is invisible at `register()` time and does not appear as a log
line, so the component list is the only record of which arm ran.

## Decode comparability

A per-step time means nothing without its accept length, and neither can be
re-derived without the three inputs:

| arm | completion tokens | verify steps | long run | short run | slope | tokens/step | step | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1Cat TP4, MTP4 | 128 | 51 | 1317 ms | 576 ms / 32 tok | 7.72 ms/token | 2.51 (request average) | 19.4 ms | 129.6 |
| ours SPEC=1, lambda1 | 128 | 48 | 1627 ms | 619 ms / 12 steps | 10.50 ms/token | 2.67 | 28.0 ms | 95.3 |
| ours SPEC=1, lambda6 run 0 | 128 | 45 | 1647 ms | 723 ms / 12 steps | 9.63 ms/token | 2.91 marginal | 28.0 ms | 103.9 |
| ours SPEC=1, lambda6 run 1 | 128 | 36 | 1384 ms | 723 ms / 12 steps | 6.89 ms/token | 4.00 marginal | 27.5 ms | 145.2 |
| ours SPEC=1, lambda6 run 2 | 128 | 43 | 1554 ms | 723 ms / 12 steps | 8.66 ms/token | 3.10 marginal | 26.8 ms | 115.5 |
| ours SPEC=0, lambda1 | 128 | none | 2377 ms | 745 ms / 32 tok | 17.00 ms/token | 1 | 17.0 ms | 58.8 |

How the columns follow: `slope = (long - short) / 96`, which is milliseconds per
output token, so `tok/s = 1000 / slope`; `step = slope x tokens per step`; and
`tokens per step` is the **marginal** `96 / (steps_long - steps_short)`. The
marginal and the request average coincide where accept is uniform (lambda1: 2.67
at both 12 steps and 48 steps). The short run's fixed cost cancels in the slope.
`elapsed / verify steps` is **not** the step time, because that fixed cost is
still in it: 1627 / 48 = 33.9 ms against a 28.0 ms marginal step.

1Cat's 2.51 is its **request average** (128 / 51), not a marginal figure, and its
19.4 ms step is `slope x average accept`. That is exact only if its average accept
also held over the 32-to-128 window, which we cannot show; it is the one cell in
this table whose two factors are not measured the same way. Our rows use marginal
counts where the arms differ.

The SPEC=0 row is faster per step only because it emits one token per step. The
comparable pair is the SPEC=1 rows: against lambda6 the median gap is 12%, against
lambda1 36%.

An independent SPEC=0 arm on a clean 4-card island measured **17.09 ms/step =
58.5 tok/s**. It carried no speculative flags, so there is no `spec_verify_ct`
to divide by: one step, one token, accept exactly 1. The 17.09 is a measured
step time and 58.5 is its arithmetic consequence, labelled that way; it agrees
with the 58.8 above across two boxes.

### Correction to the first version of this table

It published 33.6 ms/step at accept 3.20. Both numbers came from `bench.py`,
which reads `spec_verify_ct` from one extra long request taken *after* the timed
runs: that request took 40 steps (accept 3.20), while the timed long runs
recorded 48, 36 and 48 steps (accept 2.67, 3.56, 2.67). No single request
produced "33.6 ms at 3.20". **28.0 ms is re-derived under the corrected pairing
from the recorded requests — it is not a number the fixed tool printed**, since
the tool did not exist when that run was made. The fixed tool can confirm or
refute the pairing on new data; it cannot re-aggregate an old run. From those
three inputs, our accept is 2.67 on the median run and the marginal step is
28.0 ms. **`tok/s` is unaffected**: it is `1 / slope` and never touches
`accept`. The defect is in the old bench's cross-check, not in the timings.

### Second defect: a differential step paired with a point accept

Not yet fixed; the one-line fix waits on the code freeze. `bench.py` computes
`step` as a differential, `(t_long - t_short) / (steps_long - steps_short)`, and
`accept` as a point, `completion_long / steps_long`. On lambda6 those differ: the
marginal pair adds 31 steps for 96 tokens, **3.097 per step**, while the long
run's own average is 2.977. The printed rate is therefore **111.0** tok/s where
`Δcompletion / Δt` gives **115.5**. The printed identity `accept/step == 1/(ms per
output token)` cannot catch it, because both of its sides are built from the same
accept. Every rate in this note is derived from `Δcompletion / Δt`, not read off
the tool.

### Accept definition

In the pin, `eagle_utils.py:845` sets `accept_lens = num_correct_drafts + 1`,
`metrics_reporter.py:551` accumulates it and `:973` divides by
`spec_num_forward_ct`, and `tokenizer_manager.py:3167` divides
`completion_tokens` by `spec_verify_ct`. So **our accept already includes the
per-step bonus token**; adding another `+1` counts it twice and inflates the
rate by (accept+1)/accept, 1.37x at 2.67. Verified in the tree, not inferred.

For 1Cat we have not proven this from its code. What we have is that its
129.6 tok/s, 19.4 ms/step and accept 2.51 are **consistent with the same
definition**. Not "the same definition": if its docs define accept otherwise,
the ratio moves and we would be the ones who overclaimed.

## SPEC=0 and SPEC=1, per competitor

The tables above keep the competitors apart from our arms, which leaves the
reader to do the pairing. Pairing is where a comparison goes wrong, so it is done
here explicitly, with the arm each project actually publishes.

### Our own two arms, one box

lambda6 GPUs 0-3, one tree, one prompt set, one variable:

| arm | prefill | decode |
|---|---:|---:|
| SPEC=0 | **3593 tok/s** (1147 ms, runs 1-2; run 0 excluded) | 59.0 tok/s (16.95 ms/step, accept 1) |
| SPEC=1 | **2971 tok/s** (1387 ms) | 115.5 tok/s (26.8 ms/step, accept ~3.0 marginal) |
| MTP cost / benefit | **-17.3%** | **x1.96** |

That is the answer to "what does MTP cost and buy here". It replaces an earlier
lambda1-derived pair (prefill 1663 → 1549, -6.9%; decode 58.8 → 95.3, x1.62),
which is kept as a second observation from a different box and never presented as
a range spanning both. The two MTP costs, 6.9% and 17.3%, disagree and the
difference is itself unexplained.

SPEC=0 prefill runs were 1487 / 1178 / 1116 ms. Run 0 is the JIT run and is
**excluded**; runs 1-2 give 1147 ms median and a 5.4% spread. Printed with run 0
the median is 1178 ms and the spread 31.4%, which is a warm/cold artifact and not
the arm.

### Against 1Cat — SPEC=1 to SPEC=1

1Cat publishes only its MTP4 arm and has no no-spec number, so a SPEC=0 pairing
does not exist.

| | our arm | 1Cat arm | our tok/s | 1Cat tok/s | ratio |
|---|---|---|---:|---:|---|
| decode | SPEC=1 | MTP4 | 115.5 (104-145) | 129.6 | 12% behind, inside our own spread |
| prefill | **SPEC=1** | **MTP4** | **2971** | **4825** | 1.62x theirs; we trail 38% |
| prefill, other box | SPEC=0 | MTP4 | 4199 | 4825 | arm mismatch, not quoted |

The decode row is like for like in arm. The prefill row now is too, for the first
time: 2971 is our SPEC=1 arm on lambda6 and 4825 is 1Cat's MTP4. That is like for
like in arm and unlike in box: the same arm on the reported clean island runs at
4048-4092 tok/s, so roughly 13-14% of the 38% is the lambda6 host rather than
anything we do differently. The rest is not attributed. 4199
stays in the table only because it is the one clean-island prefill we hold; it is
a SPEC=0 arm against their MTP4, so it is recorded and not quoted as a ratio.

### Against Strata — SPEC=1 to `--spec 2`

Strata's runtime uses `--spec 2`, its MTP analog, so the decode pairing is
legitimate. Comparing card counts without dividing is not, so the per-card column
is carried as an **arithmetic consequence of the two cells, not a measured
per-card rate**:

| engine | arm | cards | decode tok/s | tok/s per card |
|---|---|---:|---:|---:|
| Strata | `--spec 2` | 1 | 66.6 reported | 66.6 |
| Strata | `--spec 2` | 2 | 80.1 reported | 40.1 |
| ours | SPEC=1, lambda1 | 4 | 95.3 measured | 23.8 |
| ours | SPEC=1, lambda6 | 4 | 115.5 measured | 28.9 |
| ours | SPEC=0 | 4 | 58.8 measured | no Strata peer |
| Strata | no-spec | — | — | no such arm |
| 1Cat | MTP4 | 1, 2, 8 | — | no such recipe |

**This is the most important number in the note and it is not flattering.** We
need four cards to beat Strata's one card by 1.43x. Per card we are **2.80x**
behind its single-card point on lambda1 and **2.31x** on lambda6; against its
two-card per-card point, **1.68x** and **1.39x**. Even the better box is more than
twice behind per card. The honest reading is that our four-card win is
**capacity, not efficiency**, and the per-card deficit is the kernel gap —
consistent with the traces, which put 62% of prefill in GDN chunk kernels, and a
better argument for the TileLang work than any single figure we have published.
This table is decode only.

**Cross-quant caveat, attached to this row.** Strata runs IQ2_XS (35.5 GB of
experts) against our fp8/fp4 (125.9 GiB), so it does less work per token. Part of
the 2.8x is quant and part is kernel; quality was never measured, so **we cannot
decompose the two** and do not.

Empty cells, with the reason in the row: Strata has no no-spec arm, so our SPEC=0
has no Strata peer; 1Cat ships no 1/2/8-card recipe.

## Protocol caveats that apply to any reuse

- **Engine trees differ by arm, and they are not the same tree.** The lambda1 and
  lambda6 arms ran `flashnext/repo/.build/sglang/python`, a source checkout at git
  tag `v0.5.23` (commit `297e54f`, checked with `git describe`). The 4199 arm is
  reported to have run the canonical pin site `sites/pin-v0.5.23/`
  (`sglang-0.5.23.dist-info`); I did not verify that from a launch record, since
  the only attestation file I can read describes a `.build` launch. Same tag,
  different trees, and the patch sets are not established as equal. The version
  banner in the `.build` tree reads `0.0.0.dev1+g73abfe1dd`, which is stale
  metadata — read the git tag, not the banner.
- Checkpoints differ by construction: 1Cat served NVIDIA's original NVFP4
  (MTP in FP8); we served model-sm70 (`mtp.*` unquantized).
- 1Cat's attention block size is 816 (mamba page alignment); ours is page 64.
  Attention tiles are not like-for-like.
- The SPEC=0 decode long run 2 returned **121 completion tokens, not 128** — a
  truncated generation. It is excluded from per-run accept arithmetic; the
  16.95 ms step is unaffected because the three runs agree to 4%.
- Prefill throughput moves with cache hits. Any number quoted without
  unique-prompt / cold-cache and prompt length is not comparable.
- Rate numerator: `bench.py` prints `requested_tokens / elapsed` (4096 / 2644 ms
  = 1549) while the server reports 4121 prompt tokens (1559 tok/s). Both engines
  use the same convention here, so the prefill ratios are unaffected and the
  absolute prefill rates read 0.6% low.
