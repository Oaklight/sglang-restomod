# 2026-11-15 — GDN prefill kernels: Triton vs TileLang on V100

Status: **measured on a contended box; the ratio is a reference, not a target.
A clean re-derivation on an isolated host is in flight.**

The two Triton GDN prefill kernels were replaced with TileLang implementations
and the prefill wall fell from 3045 ms to ~1555 ms at 4096 prompt tokens. The
kernel-level saving is the durable result; the tok/s multiple is not, because
the box was shared and the two arms did not run the same memory fraction.

## Setup

A share of kernel time is not transferable across models or launch profiles, so
the profile comes before the number.

| item | value |
|---|---|
| host | 8 x V100-SXM2-32GB (NVLink, two islands of four), **two servers resident**: baseline on GPUs 0-3, patched on 4-7 |
| model | Qwen3.8-Flash-Next, NVFP4, our re-export with `mtp.*` excluded from quantization |
| precision | fp16 activations, NVFP4 weights, `--moe-runner-backend marlin` |
| TP / EP | TP4 / EP1 per server, 32 GB per rank |
| stack | SGLang v0.5.23 (`297e54fd105e`) + this repository's 3 patches; `sglang-restomod` at the #266 merge |
| context | `--context-length 65536`, `--chunked-prefill-size 4096`, page size 64, HiCache 16 GB |
| spec decode | NEXTN, 3 steps / 4 draft tokens, rejection sampling (SPEC=1) except where a row says SPEC=0 |
| mem-fraction | **0.82 baseline arm, 0.75 patched arm** (see the confounds below) |
| prompt | 4096 tokens requested; 4123 actual on the baseline arm, 4106 on the patched arm |

## What was replaced

| kernel | Triton | TileLang | ratio |
|---|---|---|---|
| `chunk_fwd_kernel_o` | 25.12 ms | 0.233 ms | 108x |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 15.4 ms | 0.47 ms | 33x |

Isolated numbers, fixed-seed tensors on a free V100. The kernel pair accounted
for **1436.9 ms of the SPEC=1 prefill wall and 63.6% of prefill kernel time**
in a no-draft trace, which is why this pair was the first target.

In the SPEC=1 wall rather than in kernel time, the same pair is **46.9%**:
1425 ms of a 3041 ms wall on the pre-fix trace. The two shares differ because
they have different denominators — 63.6% is of prefill *kernel* time in a
no-draft trace, 46.9% is of the *production* wall with the draft layer running.
Quote whichever matches the question, and never the two together.

After the fix the ceiling moves rather than disappears. In the post-fix 4096
window (1256.0 ms of kernel time) the largest remaining item is the MTP/draft
MoE at 42.4%, then routed Marlin at 21.8%, then dense GEMM at 13.5%; the GDN
pair is down to 1.7%. That ordering is why the next work is the draft layer's
MoE and the decode dense kernels, not another attention kernel.

> **Withdrawn 2026-11-15: the 42.4% and the priority sentence above.** Both
> came from a window collected on a shared box, and neither survives a quiet
> re-measurement. See the correction below before quoting either.

### Correction: the draft's MoE is 4.0% of prefill kernel time

Measured on an isolated lambda6 (4x V100-SXM2, TP4, co-tenants holding
~1.7 GiB per card, no concurrent load), one cold 4096-token prefill under
`/start_profile`, rank 0: **7139 kernels, 994.1 ms of kernel time**. The
draft's `fused_moe_kernel` is **40.2 ms, 4.0%**, in four calls — 25.9 ms of
gate_up over the full 4096 tokens and 13.7 ms of down, inside a
`step[DRAFT bs=1 toks=4096]` window of 27.5 ms.

The 1256.0 ms window this note previously quoted is 26% larger than the quiet
measurement on the same model and protocol; treat it as stale.

The draft's **total** prefill cost is real but it is not its MoE. On the same
box, SPEC=1 prefill is 3439 tok/s (median of three, 4121 prompt tokens,
`cached_tokens=0`) against 3593-3639 at SPEC=0, so the whole draft pass —
attention, dense projections and the extend over the prompt included — is
about **15% of the wall**. Its MoE is 4 points of *kernel* time inside that.

Two independent checks say the withdrawn 304 ms gate_up / 226 ms down pair was
not one request:

- gate_up to down is 2:1 in FLOPs and measures 1.89 here (25.9 / 13.7), while
  304 / 226 = 1.35;
- ~268 GFLOP in 25.9 ms is **10.4 TFLOPS**, which is what an fp16 kernel
  without tensor cores reaches on V100 (FMA peak ~15.7); 304 ms would be
  0.9 TFLOPS, which is a broken measurement rather than a slow kernel.

Consequence: quantising the draft's experts to NVFP4 caps out at **~3% of
prefill**, not a third. The component (`mtp_fp4_moe`) was built and is left
off by default for that reason; its PROVENANCE records the same ceiling.

Root cause, verified in the emitted code rather than inferred from timings:
Triton 3.5.1 has no sm70 MMA lowering, so `chunk_fwd_kernel_o.ptx` contains
**0 `mma.sync` and 10,272 `fma.rn.f32`**, and the kernel reports
**n_spills = 5258** registers per thread. Every dot product is scalar FMA
paying spill traffic. TileLang emits `mma.sync` for the same shapes.

## End-to-end, as measured

| | baseline | patched |
|---|---|---|
| prompt tokens | 4096 (4123 actual) | 4096 (4106 actual) |
| wall | 3045 ms | ~1555 ms |
| prompt tok/s | 1347 | 2634 |
| ms per prompt token | 0.7424 | 0.3797 |

Unprofiled wall, median of 3 warm runs. **1.96x measured.**

## Attribution, and why it is 1.84x

Total kernel time fell 1511 ms. The GDN pair accounts for 1390 ms of that. The
remaining 121 ms came from non-GDN kernels that also got 9% faster
(1330.3 -> 1209.2 ms) with **no identified mechanism** — the patch cannot make
unrelated kernels faster, so that 121 ms is measured but not attributable.

Patch-creditable wall is therefore 3045 - 1390 = **~1655 ms = 2475 tok/s =
1.84x**. The ~100 ms between 1.84x and 1.96x is recorded as unexplained, most
plausibly less contention between the two runs.

**Ship expectation: 1.84x. 1.96x is what that box produced.**

## Two confounds in this run, both mine

1. **The box was shared.** Both arms ran with two servers resident on the same
   eight GPUs. The clean number on an isolated host for the same config is 1633 tok/s,
   **21% above this baseline**, which is what an isolated box buys.
2. **The arms were not configured alike.** The baseline half ran
   `--mem-fraction-static 0.82` and the patched half ran 0.75, because 0.82
   leaves 2.3 GB free and 0.70 crashes the hybrid mamba state cache
   (`total_rest_memory=-3.68 GB`). Prefill throughput for a fixed prompt should
   not depend on it, but "should not" is not a measurement.

Consequence: **the absolute numbers here are superseded** and the ratio must be
re-derived on two isolated hosts with one config and an alternating arm.

## Correctness

Token-for-token gate at **SPEC=0**, four server starts, 3 fixed prompts x 128
greedy tokens x 3 repeats per side:

- unpatched, twice: one distinct output per prompt per pass, zero flips;
- patched, twice: one distinct output per prompt per pass, zero flips;
- cross: all three prompts identical token for token, 128/128.

The negative control is what makes this a gate rather than a spot check: if
identical configurations still flipped, "identical across configurations" would
mean nothing. Both sides deterministic, so the comparison discriminates.

**Scope:** 3 prompts, 128 tokens, greedy, the served path. Not a proof of
equivalence on arbitrary inputs.

At **SPEC=1 the gate is void**: `--speculative-use-rejection-sampling` makes
even the unpatched baseline non-repeatable (2 distinct outputs in 5 runs); both
sides flip at token 63 and share the first 63. The flip is in the accept step,
not the arithmetic. A prefill-only control (max_new_tokens=1, 4096 tokens) is
repeatable and matches across servers.

Numerics: `chunk_o` is 0.74 ULP against an fp32 reference (Triton 0.64; the
difference between them is the sum of roundings). `chunk_h`, `v_new` and the
final state are within 1.13 ULP.

## A domain limit that belongs in the kernel, not the bug tracker

The recurrence is **not contractive for arbitrary unnormalized inputs**. Triton
and TileLang produce identical failure counts on the same random tensors
(nan = 11,795,324, inf = 1,160, absmax = 65,504). This is a property of the
recurrence, shared by both implementations, not a TileLang defect — and it is
documented in both kernels so the next reader does not file it as one.

## Retracted numbers

Do not quote any of these; each was published and then contradicted by a later
measurement or a corrected trace:

| retracted | why |
|---|---|
| 74.0%, 3.85x kernel-only cap | a trace filtered to `dur > 1 ms` dropped ~3000 short kernels |
| 68% of the post-fix wall outside kernels | same filter; the corrected figure is ~49% |
| "1619 dense-GEMM launches per step is the decode defect" | decode is CUDA-graph captured: 2 `cudaGraphLaunch` per step, median inter-kernel gap 0.93 us |
| "NVFP4 is 17% slower on prefill than MIXED_PRECISION" | the MIXED run was SPEC=0, so its MTP never ran; normalised they tie. Speculative decoding adds 17.9% of prefill wall |

## Status of the change itself

The kernels are merged in `Oaklight/sm70-kernels` (`gdn_chunk_o`, `gdn_chunk_h`).
The integration used here was a **site-local monkeypatch on `fla/chunk.py`** and
is not in this repository, so this is a **lab result**: the speedup is real and
the mechanism is understood, but nothing in `sglang-restomod` currently produces
it. A dispatchable component — gate, Triton fallback, `SM70_GDN_TILELANG=0`
toggle, PROVENANCE — is in progress; when it lands the number becomes a property
of the repo and the caveat goes away.

## Reproducing

The honest reproduction is the alternating arm on one isolated node:

```
on/off/off/on with SM70_GDN_TILELANG, 4096 prompt tokens, median of 5,
SPEC=1, one mem-fraction for both arms, box otherwise idle
```

and then the same on a second node, comparing the on/off **ratios** rather than
the absolute tok/s — two nodes differ in ways a single comparison cannot
separate from the thing under test.
