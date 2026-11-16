# Documentation map

The front page is [`../README.md`](../README.md). The architecture and the
rules that keep this repository a layer rather than a fork are in
[`../RESTMOD.md`](../RESTMOD.md). This page says which document answers which
question, so a reader does not have to guess.

## Start here

| Question | Document |
|---|---|
| What does the layer contain, and which components ship? | [`../README.md`](../README.md) |
| How is it built, and how does a component get added? | [`../RESTMOD.md`](../RESTMOD.md) |
| Which hardware and stack is validated, and with what receipt? | [`../COMPATIBILITY.md`](../COMPATIBILITY.md) |
| Which launch profile produced a number? | [`../recipes/README.md`](../recipes/README.md) |
| How does an upstream edit enter, and when is it retired? | [`../patches/README.md`](../patches/README.md) |
| What are the branch and tag rules? | [`../BRANCHES.md`](../BRANCHES.md) |
| How do I run the checks and the tests? | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |

## Component design

| Document | Covers |
|---|---|
| [`quantization-dispatch.md`](quantization-dispatch.md) | The contract between a checkpoint format and the kernel that runs it: FP8, NVFP4, INT4, the FP8 environment variables, and NVFP4 decode GEMV eligibility |
| [`qpn2-layout.md`](qpn2-layout.md) | The QPN2 NVFP4 weight layout: buffer, lane order, K order, scale encoding, and what a reader must not do |
| [`qpn2-cross-run-runbook.md`](qpn2-cross-run-runbook.md) | Running a QPN2 A/B arm against Marlin and reading the result |
| [`lsa-sm70.md`](lsa-sm70.md) | Latent Sparse Attention design, what is ported, its limits and stop conditions |
| [`spec-schemes.md`](spec-schemes.md) | Which speculative decoding schemes are obtainable on SM70, and what each costs |
| [`kernel-sync.md`](kernel-sync.md) | Where external kernels come from, how a fix reaches a launch, and what is currently stranded |

## Measurement and verification

A run record is one dated measurement with its setup, method, result and
caveats. A receipt is a signed evidence bundle; [`gpu-receipts.md`](gpu-receipts.md)
is the operator runbook for producing one, and
[`measurement-prerequisites.md`](measurement-prerequisites.md) is what a host
must satisfy before a number counts.

| Document | Covers |
|---|---|
| [`measurement-prerequisites.md`](measurement-prerequisites.md) | Device class, kernel libraries, build space, site check, log gates, contexts, and what must not run alongside |
| [`gpu-receipts.md`](gpu-receipts.md) | The out-of-band receipt mechanism, roles, and the signing runbook |
| [`../scripts/gpu_suite.sh`](../scripts/gpu_suite.sh) | The suite that produces the evidence a receipt signs |
| [`runs/2026-11-16-competitor-study-protocol.md`](runs/2026-11-16-competitor-study-protocol.md) | The competitor study's pre-registered protocol: one arm per window, control drift, the discarded first request, the component list from `/proc`, admitted/active counts, and the page contract |

## Run records

Newest first. Each entry states what the run established, not what it hoped to.

| Date | Record | Established |
|---|---|---|
| 2026-11-16 | [Capacity model for V100 serving](runs/2026-11-16-capacity-model.md) | Capacity, not throughput, is the difference: the competitor holds 80.2% of its experts on one card and the whole model on two where we need four, with per-card throughput level once shard count is matched (31.6 against 31.5-35.4) and 2.26x in our favour per model; quantization and placement multiply; fp8 dense is free (9.3% of a decode step, prefill-neutral); for bluewhale the same policy inverts (9.8% residency, ~92 ms/token host cost) and a 204 GB Engram table is an I/O tier rather than a capacity term |
| 2026-11-16 | [The decode GEMM gap](runs/2026-11-16-decode-gemm-gap.md) | The per-card decode deficit is a kernel gap: dense and expert GEMM are 63% of the recorded verify window and 5-8x per call; the expert half sits in Marlin's 64-row M tile (1/0.16 = 6.25 against 5-8x), the down projection being unreachable for our no-tile kernel at n=160/rank; fp8 dense is worth 9% of a decode step and nothing at prefill; the capture holds about one step in three, and the whole-capture shares are prefill-weighted |
| 2026-11-15 | [Competitor numbers by card count](runs/2026-11-15-competitor-numbers.md) | 1Cat at 4 cards: prefill 4825 tok/s, decode 129.6. Decode is level with ours on lambda6 (115.5 median, 104-145 across prompts) and on the served arm (126.2 reported); on the served arm prefill is 3912 tok/s quiet (3641 with other work on the box, 7% apart and not reconciled), 18.9% behind, with allreduce at 38% of peak and marlin at 54%, so neither bounds it; per-card deficit against the capacity reference is 1.77x (2.11x against that reference's earlier run) while per model we lead 2.26x; the 2.3-2.8x band this row used to quote is retired, see [the decode GEMM gap](runs/2026-11-16-decode-gemm-gap.md) |
| 2026-11-15 | [Attribution and TP4 placement](runs/2026-11-15-attribution-and-tp4-placement.md) | A/B/C separates the effects: the GDN pair is prefill 2.4x and decode 0, while the open #286 discovery is decode 2.4x and prefill -46%, an open defect; TP4 belongs on one host (65.9% cross-host all-reduce); the #287 tail gate was costing a decode step; receipt `3880f8e058` is signed and `fail` |
| 2026-11-15 | [GDN pair in serving, and token equality](runs/2026-11-15-gdn-pair-serving-and-repeatability.md) | The pair is reached in a served prefill (TileLang grids, zero Triton chunks) and prefill is 2.48x on a pair-only arm; token equality cannot gate it, because the pure-Triton control diverges cold-to-cold |
| 2026-11-15 | [Flash-Next NVFP4 TP4, 8k/16k/64k, cold vs warm](runs/2026-11-15-flash-next-nvfp4-tp4-context-cold-warm.md) | Cold prefill is flat at 921 / 921 / 868 tok/s from 8k to 64k; the run reads that as an expert-GEMM shape problem and marks the reading as inference; warm TTFT pays a fixed ~0.5 s; decode step is flat at 49-55 ms |
| 2026-11-15 | [The dense-GEMM dive](runs/2026-11-15-dense-gemm-dive.md) | The decode dense cluster is 39.5% of the step at ~17% of its byte floor on bf16, and on fp8 the 15% cut is bytes not kernel (bandwidth falls 152 to ~120 GB/s); the remaining 6.42 ms is launch-bound; a Triton GEMV is 3-24x slower than `torch.mv` and a TileLang one fails the same gate even with split-K (split-K: 1.21x/2.22x at large N, 0.58x at `(320, 10240)`), so the GEMV lane is closed; the exact-dense prefill dispatch is null on the 36 of ~68 coverable layers per rank that a name gate admits, and 2.1% once every listed shape reaches it (100 per rank, one launch each) |
| 2026-11-15 | [GDN prefill kernels: Triton vs TileLang](runs/2026-11-15-gdn-tilelang-prefill-ab.md) | 108x and 33x on the two GDN prefill kernels; 1.96x measured on a contended box with unequal arms, 1.84x patch-attributable; the durable number is the 1390 ms kernel saving, not the tok/s multiple |
| 2026-11-08 | [Flash-Next accept length vs context](runs/2026-11-08-flash-next-accept-vs-context.md) | Accept length and verify step time are flat from 3.8k to 30.8k; neither draft nor verify degrades in that range |
| 2026-10-26 | [Flash-Next decode under CUDA graph](runs/2026-10-26-flash-next-cuda-graph.md) | 25.9-36.9 tok/s with a graph against 5.5-6.3 without, on 8x V100 TP8/EP8 |
| 2026-10-26 | [Flash-Next decode without per-layer host reads](runs/2026-10-26-flash-next-decode-syncs.md) | Decode 2.9 to 5.5-6.3 tok/s; the remainder is launch overhead and expert-parallel all-reduce |
| 2026-10-26 | [27B: KV dtype and MTP by context](runs/2026-10-26-qwen3.8-27b-kv-dtype-mtp-crossover.md) | MTP pays below about 12k and costs 33% at 32k; the KV dtype question is decided by the decode partial |
| 2026-10-26 | [MTP on 27B](runs/2026-10-26-qwen3.8-27b-mtp-node-b.md) | The checkpoint's own MTP head works: decode 2.0-2.3x over the graphed baseline |
| 2026-10-25 | [CUDA graph on 27B](runs/2026-10-25-qwen3.8-27b-fp8-cuda-graph-node-b.md) | The graph metadata hook fixes capture; decode improves 3.0-3.4x |
| 2026-10-25 | [E5M2 KV on 27B](runs/2026-10-25-qwen3.8-27b-fp8-e5m2-kv-node-b.md) | E5M2 KV is unusable with the default partial decode kernel, and without that kernel gives no throughput gain |
| 2026-10-25 | [27B FP8 at the pin](runs/2026-10-25-qwen3.8-27b-fp8-node-b.md) | Correctness validated; performance is limited by the dense decode path, not by FP8 weights |
| 2026-10-25 | [Flash-Next FP8 expert GEMM](runs/2026-10-25-qwen3.8-flash-next-fp8-sm70-moe-node-a.md) | The block-FP8 expert method is numerically validated; the model then stopped at the QSA decode path, not at the MoE |

## Operations and CI

| Document | Covers |
|---|---|
| [`self-hosted-runner.md`](self-hosted-runner.md) | The protocol for a GPU host in CI, and what must never be recorded in this repository |
| [`../scripts/check_site.py`](../scripts/check_site.py) | Verifying that a site carries the patch queue and the tuned MoE configs before a run trusts it |

## Historical

These describe work that is complete or superseded. They are kept because the
reasoning is still referenced.

| Document | Status |
|---|---|
| [`engine-diff.md`](engine-diff.md) | One-time comparison of `dg1kjd/sglang-sxm2` against `v0.5.22`: which fork pieces the two target models need, and which can live here |
| [`engineering-plan.md`](engineering-plan.md) | The early work plan. Its layout tree predates most components; read `RESTMOD.md` and `sglang_restomod/plugins.py` for the current structure |
