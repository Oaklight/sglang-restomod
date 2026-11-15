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

At 4 cards: **1Cat is 1.36x our decode throughput and 1.35-1.40x our prefill
throughput** — 129.6 vs 95.3 output tok/s, 4825 vs 3450-3565 prompt tok/s.
Equivalently, we trail by 27% on decode and 26-29% on prefill. The prefill pair
is quoted component-on; see the component rule below.

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
- An earlier peer figure of **36.8 tok/s** on a single card, also IQ2_XS but
  with the expert cache on auto and the opt-in switches off, does not agree with
  the 66.6 above. Different run, different switches; I cannot reconcile them
  from the material I hold, so both are recorded and neither is used in the
  headline.
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
| 4 (single node) | **3450-3565 reported, components include `gdn_tilelang_sm70`**; **1549 measured, pair off, SPEC=1** (4121 prompt tokens, 2644 ms median); 1663 in the SPEC=0 arm | **95.3 measured, SPEC=1** (slope 10.50 ms/output-token; 128 completion tokens, 48 verify steps, 1627 ms long run, short 32 tokens in 619 ms) → accept 2.67 → 28.0 ms/step. Accept varied by request: 2.67 / 3.56 / 2.67 at 48 / 36 / 48 steps, so the marginal step is 28.0-37.4 ms. **58.8 measured, SPEC=0** (slope 17.00 ms/token, accept 1) — not comparable to 1Cat |
| 4 (dual node) | comm-bound at 66% (reported), no number held here | — |
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

Rule: **never cite a prefill number without the component list.** The
difference is invisible at `register()` time and does not appear as a log
line, so the component list is the only record of which arm ran.

## Decode comparability

A per-step time means nothing without its accept length, and neither can be
re-derived without the three inputs:

| arm | completion tokens | verify steps | long run | short run | slope | accept | step | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1Cat TP4, MTP4 | 128 | 51 | 1317 ms | 576 ms / 32 tok | 7.72 ms/token | 2.51 | 19.4 ms | 129.6 |
| ours TP4, SPEC=1 | 128 | 48 | 1627 ms | 619 ms / 32 tok | 10.50 ms/token | 2.67 | 28.0 ms | 95.3 |
| ours TP4, SPEC=0 | 128 | none | 2377 ms | 745 ms / 32 tok | 17.00 ms/token | 1 | 17.0 ms | 58.8 |

How the columns follow: `slope = (long - short) / 96`; `accept = completion /
verify steps`; `step = slope x accept`; `tok/s = 1 / slope`. The short run's
fixed cost cancels in the slope. `elapsed / verify steps` is **not** the step
time, because that fixed cost is still in it: ours is 1627 / 48 = 33.9 ms
against a 28.0 ms marginal step.

The SPEC=0 row is faster per step only because it emits one token per step. The
comparable pair is the SPEC=1 row; there 1Cat is 1.36x ahead, i.e. we trail by
27%.

### Correction to the first version of this table

It published 33.6 ms/step at accept 3.20. Both numbers came from `bench.py`,
which reads `spec_verify_ct` from one extra long request taken *after* the timed
runs: that request took 40 steps (accept 3.20), while the timed long runs
recorded 48, 36 and 48 steps (accept 2.67, 3.56, 2.67). No single request
produced "33.6 ms at 3.20". Re-derived from its own three inputs, our accept is
2.67 on the median run and the marginal step is 28.0 ms. **`tok/s` is
unaffected**: it is `1 / slope` and never touches `accept`. The defect is in the
bench's cross-check, not in the timings.

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

## Protocol caveats that apply to any reuse

- Checkpoints differ by construction: 1Cat served NVIDIA's original NVFP4
  (MTP in FP8); we served model-sm70 (`mtp.*` unquantized).
- 1Cat's attention block size is 816 (mamba page alignment); ours is page 64.
  Attention tiles are not like-for-like.
- Prefill throughput moves with cache hits. Any number quoted without
  unique-prompt / cold-cache and prompt length is not comparable.
- Rate numerator: `bench.py` prints `requested_tokens / elapsed` (4096 / 2644 ms
  = 1549) while the server reports 4121 prompt tokens (1559 tok/s). Both engines
  use the same convention here, so the prefill ratios are unaffected and the
  absolute prefill rates read 0.6% low.
