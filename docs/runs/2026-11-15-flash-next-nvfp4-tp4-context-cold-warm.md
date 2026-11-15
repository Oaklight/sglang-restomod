# 2026-11-15 — Qwen3.8-Flash-Next-NVFP4 on 4x V100: context length, cold vs warm

Status: **measured; prefill is the ceiling.**

Prefill throughput is flat at roughly 900 tok/s from 8k to 64k prompt tokens. The
per-token cost does not grow with context, so attention is not the bottleneck on
this model: a constant per-token term is, and the routed-expert GEMM is the only
candidate large enough to explain the magnitude.

## Setup

| item | value |
| --- | --- |
| model | Qwen3.8-Flash-Next-NVFP4 (125.9 GiB, 48 layers, 512 experts, topk 10) |
| hardware | 4x Tesla V100-SXM2-32GB, one host, TP4, ep 1 |
| attention | `qsa_v100`, page size 64 |
| speculative | NEXTN, 3 steps, 1 topk, 4 draft tokens, rejection sampling |
| context | `--context-length 65536`, `--mem-fraction-static 0.82` |
| cache | `--enable-hierarchical-cache --hicache-size 16` |
| KV pool | 80 064 tokens per rank after the hierarchy reservation |
| MoE | `--moe-runner-backend marlin`, `--quantization modelopt_fp4` |

Two log gates were both zero for the whole run: `Setup Custom allreduce failed`
and `Using default MoE kernel config`. A nonzero count makes the numbers below
non-comparable.

A second tenant held about 2.2 GB per card on two of the four GPUs during the
run. This is a shared cluster and the run is reported with that tenant present.

## Method

Requests went to `/generate` directly, not through `sglang.benchmark.serving`.
That harness's TPOT disagreed with three independent direct measurements by
roughly 3x on this build and the disagreement is still unreconciled, so its
latency columns were not used. Step time is `wall_ms / spec_verify_ct`.

Cache state is forced rather than assumed:

- **miss**: the prompt carries a unique nonce, so the prefix cannot be resident;
- **hit**: the identical prompt is sent again immediately after the miss.

Three repetitions per length, averaged. Prompt length is the server's own
`meta_info.prompt_tokens`, not the intended target.

## Results, mean of 3

| prompt tokens | cold TTFT | cold prefill | warm TTFT | warm prefill | decode step | decode rate | accept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 176 | 8.873 s | 921 tok/s | 0.454 s | 17 996 tok/s | 49.17 ms | 44.1 tok/s | 2.17 |
| 16 337 | 17.734 s | 921 tok/s | 0.471 s | 34 661 tok/s | 52.42 ms | 58.6 tok/s | 3.08 |
| 64 017 | 73.750 s | 868 tok/s | 0.607 s | 105 407 tok/s | 54.71 ms | 46.2 tok/s | 2.53 |

Per-repetition step times were tight (8k: 49.02 / 51.69 / 46.79 ms; 64k: 53.25 /
53.35 / 57.52 ms), so the decode column is stable. The decode rate column is not
a clean comparison between rows: it is driven by the accepted-token count, which
varied from 2.0 to 3.76 per step across prompts, not by the step time.

## What the shape of the curve says

**Cold prefill is linear in length.** 921 / 921 / 868 tok/s across a 7.8x span of
context. Per token the cost is about 1.1 ms and it does not grow. A quadratic
attention term would show up as a falling rate at 64k; it does not.

At 921 tok/s and roughly 2.36 GFLOP per token of active expert work, the four
GPUs are producing on the order of 2 TFLOPS against a peak in the hundreds. That
is a utilisation problem, not a context-length problem.

The consistent explanation is the shape of the expert GEMM. With a
4096-token chunk and topk 10, about 40 960 routings spread over 512 experts give
roughly 80 tokens per expert, so prefill issues hundreds of small expert GEMMs
per layer rather than one large one. Upstream Marlin is a stub below SM80, so
NVFP4 on Volta has no alternative backend here and the component in
`sglang_restomod/quant/sm70_fp4_moe` is the only path. This paragraph is
inference from the flat curve and the launch configuration, not a profile.

**The warm path is not free.** Warm TTFT is 0.45-0.61 s and roughly constant in
length, so the hit path pays a fixed cost of about half a second rather than the
milliseconds a pure radix-cache hit would imply. With the host hierarchy enabled
the plausible component is a host-to-device KV transfer, which for 64k tokens is
on the order of 700 MB. Unconfirmed; it needs a transfer-time measurement.

**Decode is flat in context.** Step time moves only from 49 to 55 ms between 8k
and 64k while the KV set grows eightfold, so decode is dominated by weight
bandwidth rather than by attention over the KV.

## Consequences

- Prefill is the axis to attack. Making the expert path efficient moves every
  cold turn in an agentic session; making the decode step faster moves a number
  that is already flat.
- Long context is affordable here. The model has 48 layers of which 12 are full
  attention and 36 are linear attention with a constant state, so the cache grows
  with 12 layers only; that is why 64k fits in a 1 GB pool.
- Do not extrapolate past 64k with this configuration. `--context-length` is a
  per-request cap and the pool is 80 064 tokens, so a single session cannot
  exceed that regardless of the model's 262 144 position limit.
