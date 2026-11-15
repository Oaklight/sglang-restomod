# 2026-11-15 — Competitor numbers by card count

Status: **partial**. 1Cat is measured at 4 cards only. Strata has no completed
measurement from this lane. Every cell not measured is left empty; nothing here
is interpolated.

Reading the cells:

- **measured** — run by this lane, on the box and protocol stated.
- **reported** — measured by another lane in this lab; I do not hold the trace.
- **claimed** — the project's own material, protocol quoted as published.
- **empty** — not measured.

## Headline

At 4 cards, like for like: **1Cat is about 36% ahead on decode and about 28%
ahead on prefill** (4825 vs 3450-3565 prompt tok/s; 129.6 vs ~95 tok/s).

Both prefill numbers below carry their component list. That line is not
optional: 2.4x rides on it.

## 1Cat (1cat-vllm 1.5.1, Qwen3.8-Flash-Next-NVFP4, TP4)

| cards | prefill tok/s | decode (accept) |
|---|---|---|
| 1 | — | — |
| 2 | — | — |
| 4 | **4825 measured** (4121 prompt tokens, 849 ms median) | **19.4 ms/verify-step measured** (accept 2.51) = 129.6 tok/s |
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
- decode: one fixed 4101-token prompt served from prefix cache, timed at 32 and
  128 output tokens; slope 7.71 ms/output-token = 129.6 tok/s. vLLM metrics
  over a 128-token request: 51 drafts, 79 accepted draft tokens, 128 completion
  → 2.51 tokens per verify step → 19.4 ms/step.
- kernel time inside the step: 8.43 ms on rank0 (torch profiler). Collective
  kernels are inflated by the profiler; they are not used in the verdict.

Claimed, not measured here (lab note citing `benchmarks/flash-next-lambda4-tp4.md`):
TP4 Flash-Next at 8K context, prefill 5512 tok/s, decode 94.1 tok/s. Batch and
concurrency are not stated in that note, so it is not directly comparable.

1/2/8 cards: empty. The shipped launcher requires four peer-connected
V100-SXM2-32GB; no 1/2/8 recipe exists in its material.

## Strata

| cards | prefill tok/s | decode (accept) |
|---|---|---|
| 1 | — (published 1164 / 1251 / 1123 at 7.2k / 28.7k / 114k, UD-IQ4_XS) | — (published 38-51 tok/s; peer measured 36.8 on IQ2_XS) |
| 2 | — (peer measured the 7.2k point; value not held here) | — |
| 4 | — | — |
| 8 | — | — |

- Measured by this lane: **none completed**. Build at `13eedf55` (sm70, CUDA
  12.8) succeeded and the Q2_0 GGUF (66.4 GB) was downloaded; the single-card
  point was not run before the lane was reassigned.
- Reported by the multi-node peer, not my measurement: single-card decode
  **36.8 tok/s**, **IQ2_XS**, spec 2, expert cache auto, opt-in switches off;
  the two-card layer split on lambda11/12 measured the 7.2k prefill point.
- Claimed by the project: one V100-PCIE-32GB, **UD-IQ4_XS**, int8 KV,
  MTP `--spec 2`: prefill 1164 / 1251 / 1123 tok/s at 7,194 / 28,650 / 114,338
  prompt tokens; decode 38-51 tok/s, flat across that range.
- 4 and 8 cards: empty; the project documents 2-4 cards and calls 8 untested.
- The published range and the peer's 36.8 use **different quants**
  (UD-IQ4_XS vs IQ2_XS). They are not comparable to each other.

## Our stack, same table

| cards | prefill tok/s | decode (accept) |
|---|---|---|
| 1 | not runnable | not runnable |
| 2 | not runnable | not runnable |
| 4 (single node) | **3450-3565 reported, components include `gdn_tilelang_sm70`**; **1663 measured, pair off** | **33.6 ms/step median measured, SPEC=1 (accept 3.20)** = 95.3 tok/s; **17.0 ms/step measured, SPEC=0 (accept 1)** — not comparable to 1Cat |
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

My 1663 tok/s on lambda1 is a pair-off measurement on a different box. It is
not a conflicting result; it is the same configuration.

Rule: **never cite a prefill number without the component list.** The
difference is invisible at `register()` time and does not appear as a log
line, so the component list is the only record of which arm ran.

## Decode comparability

A per-step time means nothing without its accept length:

| arm | ms/step | accept | tok/s |
|---|---:|---:|---:|
| 1Cat TP4, MTP4 | 19.4 | 2.51 | 129.6 |
| ours TP4, SPEC=1 | 33.6 | 3.20 | 95.3 |
| ours TP4, SPEC=0 | 17.0 | 1 | 58.8 |

The SPEC=0 row is faster per step only because it emits one token per step. The
comparable pair is the SPEC=1 row; there 1Cat is about 36% ahead.

## Protocol caveats that apply to any reuse

- Checkpoints differ by construction: 1Cat served NVIDIA's original NVFP4
  (MTP in FP8); we served model-sm70 (`mtp.*` unquantized).
- 1Cat's attention block size is 816 (mamba page alignment); ours is page 64.
  Attention tiles are not like-for-like.
- Prefill throughput moves with cache hits. Any number quoted without
  unique-prompt / cold-cache and prompt length is not comparable.
