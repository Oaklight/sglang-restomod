# 2026-11-16 — Competitor study: measurement protocol (pre-registered)

Status: **pre-registered**. This document fixes the method before any arm of the
study runs. A number that does not meet it is not a finding, whatever it shows.

Study questions, set by the user:

1. Strata's RAM/NVMe offload — what it keeps where, and how much of its 2.3-2.8x
   per card is capacity rather than kernels.
2. What 1Cat's vLLM has that our stack does not, ranked by effect on **prefill**,
   the only live gap.

Lanes: performance-tuner (lead, arbitration, the pages), kernel-crafter (source
mechanisms, filtered to "could this move prefill by ~15%"), multi-node (Strata's
realised residency and the matched-resident-set experiment), this lane (protocol,
the live 1Cat artifact, and the run records).

Both of these are excluded this round and are named as excluded rather than
omitted: **TileLang GEMV (#301)** and **QPN8 (#306)**, closed with numbers, and
**decode-only differences**, because decode is level within 2.7%.

## Why a protocol, in one paragraph

Every rule below comes from a measurement that was wrong tonight, not from
theory: a 13% optimistic decode rate from subtracting a separately measured
prefill; a 7.2x inflated wall from a profiled run's own clock; a bf16-dense arm
recorded as our served configuration; a per-step time and an accept length taken
from two different requests; a 3.8% control drift between two windows with
nothing changed but the hour; and a "captured but not replayed" conclusion drawn
from a launch census whose time ranges covered launches they did not own.

## Comparison rules

1. **One arm per window.** No exceptions. An arm whose window contained another
   arm is void.
2. **Same host within a comparison.** A cross-host absolute is never a finding;
   across hosts, quote a *ratio* with its own control in its own window.
3. **No cross-window comparisons.** If the two arms were not in the same window,
   they are not compared. A base control is re-run in each window and reported
   beside the arm.
4. **Ratios carry their own control.** Every ratio names the two arms, the
   window, and the control that brackets them.

## Prefill rows

A prefill row is valid only with all of:

- unique prompt nonce and `cached_tokens=0` recorded in the row;
- prompt token count and elapsed, median of 3 plus the spread;
- **the first request discarded and named** — it pays the Triton compile
  (~1.03 s on this stack);
- the component list **as it appears in `/proc/<pid>/environ`**, not as the
  launch script intended;
- the admitted/active count beside the component list — "it ran" and "0 layers
  were admitted" are identical in a log;
- the marlin `.so` md5 (`49458692777d78ed6dd44a10153cc09f` on 2026-11-15);
- absence of `Setup Custom allreduce failed` in the startup log.

## Decode rows

Valid only with `cached_tokens == prompt_tokens`, recorded. Otherwise report
`step = elapsed / spec_verify_ct` and **no rate** — `completion / elapsed`
includes prefill unless the prompt was fully cached, which is how the old
`bench.py` produced a 13% optimistic figure. Never subtract a separately
measured prefill; that is the same error in the other direction.

## Capacity claims

Throughput is a consequence, not evidence. A capacity claim needs the resident
set: per-card HBM by class, host RSS for what is offloaded, the NVMe path with
its hit rate, and the configuration that decides placement. Per-card tok/s alone
cannot separate capacity from kernels.

## Windows and slots

- The study owns **lambda4 4-7**.
- **Never lambda4 0-3.**
- lambda6 0-3 only with a per-window co-tenancy check; lambda12's two cards
  cannot hold TP4.
- Tests and serving cannot share four cards: test with no server, then launch,
  then bench.
- Nothing lands under `sglang_restomod/` without a GPU receipt.

## What each page must state

Each page carries: the question, the arms and their windows, the valid rows, the
numbers that are *not* findings and why (cross-window, cross-host, invalid row),
and the next experiment with its expected effect and its bound — where the bound
is derived from a measurement rather than asserted.

## Pre-registration

Written 2026-11-16 before any study arm ran. Changes to this document are
amendments, dated, and must precede the run they govern.

## Amendment 2026-11-16 (post-hoc: recorded, not applied retroactively)

The frame moved after this document was written. In order:

1. **Capacity, not the prefill gap.** The objective is highest inference
   performance on V100; capacity is a lever toward it. The phrase "the only live
   gap" in study question 2 is wrong as of this amendment, and the prefill
   ranking stopped being the organising principle.
2. **The placement question is answered** in
   `docs/runs/2026-11-16-capacity-model.md`.
3. **Quality at a given bit width is out of scope** (the quantization team's
   problem). Only kernel availability is recorded, because a format with no SM70
   kernel costs the whole model.
4. **The target model is called `bluewhale`** in everything we write.

**The rules above are unchanged and still bind every coefficient.** One arm per
window, same host within a comparison, the re-run control, the discarded first
request, the component list from `/proc/<pid>/environ`, and the admitted/active
count were all used as written.

Two rules were added downstream, in the pages rather than here: an all-reduce
count is quoted **with its per-call size, TP degree and span filter**; and an
expert byte total **names its MTP convention**.
