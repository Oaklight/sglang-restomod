# 2026-11-15 — GDN prefill kernels: Triton vs TileLang on V100

Status: **measured on a contended box; the ratio is a reference, not a target.
A clean re-derivation on lambda4 is in flight.**

The two Triton GDN prefill kernels were replaced with TileLang implementations
and the prefill wall fell from 3045 ms to ~1555 ms at 4096 prompt tokens. The
kernel-level saving is the durable result; the tok/s multiple is not, because
the box was shared and the two arms did not run the same memory fraction.

## What was replaced

| kernel | Triton | TileLang | ratio |
|---|---|---|---|
| `chunk_fwd_kernel_o` | 25.12 ms | 0.233 ms | 108x |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 15.4 ms | 0.47 ms | 33x |

Isolated numbers, fixed-seed tensors on a free V100. The kernel pair accounted
for **1436.9 ms of the SPEC=1 prefill wall and 63.6% of prefill kernel time**
in a no-draft trace, which is why this pair was the first target.

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
   eight GPUs. The clean number on lambda4 for the same config is 1633 tok/s,
   **21% above this baseline**, which is what an isolated box buys.
2. **The arms were not configured alike.** The baseline half ran
   `--mem-fraction-static 0.82` and the patched half ran 0.75, because 0.82
   leaves 2.3 GB free and 0.70 crashes the hybrid mamba state cache
   (`total_rest_memory=-3.68 GB`). Prefill throughput for a fixed prompt should
   not depend on it, but "should not" is not a measurement.

Consequence: **the absolute numbers here are superseded** and the ratio must be
re-derived on lambda4/lambda1 with one config and an alternating arm.

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
