# DSV4 paged MQA logits: kernel design (not implemented)

Design only. The acceptance gate below is written **before** any run, because
a faster kernel that moves the top-k set is not a win.

## What it replaces

`fp8_paged_mqa_logits_torch` (`srt/layers/attention/dsv4/indexer.py:82`), the
only in-tree non-flash_mla implementation of this op. Measured at 0.2872 ms
(2048 pages) and 0.2898 ms (8192 pages) on V100 — 4x the data for 0.7% more
time, so the cost is per-call, not per-byte
(`docs/runs/2026-11-17-fallback-microbench.md`).

## The math, read from that implementation

Inputs: `q_fp8 [B, 1, H, 128]` e4m3, `kvcache_fp8 [pages, 64, 1, 132]` where
the first `64*128` bytes of a page are e4m3 values and the last `64*4` are one
fp32 scale per token, `weight [B, H]` fp32, `seq_lens [B]`, `page_table
[B, max_pages]`, `max_seq_len`, `clean_logits` (the fallback asserts False).

Per batch `b` and token `t` gathered through `page_table[b]`:

```
logit[b,t] = kv_scale[b,t] * Σ_h  weight[b,h] * relu( Σ_d kv[b,t,d,h] * q[b,0,h,d] )
logit[b,t] = 0   where t >= seq_lens[b]
```

then padded (or truncated) to `max_seq_len`. Two details to preserve exactly:
the `relu` is applied **before** the head weighting, and `page_table` entries
below zero are clamped to page 0 before the gather.

`ks`/`ke` are not parameters of this function. If the contract's §3 defines
them for a different entry point, the two must be reconciled before
implementation — that is an open dependency, not something to guess at here.

## Shape of the kernel

One launch per call. Grid over (batch, token blocks); each block pulls the
whole query for its batch (`H=32`, `128` wide: 32x128 bf16 = 8 KB) into shared
memory once, then streams the gathered pages:

- dequantize e4m3 -> bf16 in registers/shared on load, which is the same
  precision the torch path uses (`kv_values_fp8.to(torch.bfloat16)`), so the
  contract allows 16-bit MMA on sm70 without changing semantics;
- accumulate each 128-wide dot in fp32, apply `relu`, fold `weight[h]` and the
  per-token fp32 scale, then mask by `seq_lens`;
- write `[B, max_seq_len]` fp32.

The point is the launch: the torch path is eight kernels (clamp, gather,
slice, view, cast, bmm, relu+weight+sum, scale+mask+pad) over a 0.29 ms budget
that does not move with data size.

## Acceptance gate, stated in advance (superseded -- see below)

**Superseded 2026-11-17, not withdrawn.** The gate below was aimed at a shape
whose cost halves under graph capture, while the served shape barely moves.
`docs/runs/2026-11-17-indexer-graph-capture.md` measures 0.2883 -> 0.1403 ms at
B=1 but 1.4781 -> 1.3425 at B=32, so the "at 2048 pages" floor is a floor on
the small half of the problem. It is left here because a gate stated in advance
is judged on its own terms, and the honest way to change it is to add the shape
that matters beside it. The restated gate follows it.

## Acceptance gate, stated in advance

Arm: one V100-SXM2-32GB, torch 2.9.1+cu128, CUDA 12.8, no CUDA graphs, 10
warm-up iterations discarded, 50 repeats, median of per-iteration events —
identical to the microbenchmark that produced the baseline.

1. **Speed.** At **2048 pages** the kernel must be **>= 3x faster than torch**
   (<= 0.096 ms) and at **8192 pages** likewise (<= 0.097 ms). Both must pass;
   a win at one size is not a result.
2. **Agreement.** On a non-zero cache (random e4m3 values, scale byte 127 in
   every slot, random `weight`, `seq_lens` shorter than the table):
   `max |logit_kernel - logit_torch| <= 2e-2` absolute, **and** the **top-64
   index set is identical on every row**.
3. **Failure rule.** If speed passes and agreement fails, the kernel is
   rejected, not retuned against the top-k. If agreement passes and speed is
   under 3x, the result is reported as "not worth it" rather than as a partial
   win.

Why 2e-2: bf16 carries an 8-bit significand (~4e-3 relative per element) and
the dot is 128 terms with fp32 accumulation, so 2e-2 absolute on logits of
order 10 is a few times the expected accumulation error — tight enough that a
broken reduction fails it, loose enough that a correct one passes without
tuning.

## What this design does not claim

- No bluewhale performance claim: the baseline is the torch fallback on
  synthetic tensors, not a model run.
- The 8 index layers of V4.1-Flash (`index_source_layer_ids`) mean this op is
  ~2.3 ms/step in torch, not 11.6; the design targets that fallback cost, not
  the CUDA path's.
- Correctness of the fallback itself is unverified on this hardware — no
  forward has run. This kernel must be compared against torch, and torch
  against the model's own reference when one exists.

## Re-priced in the captured regime, 2026-11-17

`docs/runs/2026-11-17-indexer-graph-capture.md` measures this call inside a
CUDA graph, because decode is graph-captured and the 0.287 ms baseline was not.
Unlike the prenorm, it does not collapse: at B=1 capture takes it 0.2883 ->
0.1403 ms (about half), and at B>=8 only ~9% (0.4772 -> 0.4332 at B=8,
1.4781 -> 1.3425 at B=32). So the per-step cost in the served regime is
**1.12 ms/step at B=1, 3.47 at B=8, 10.74 at B=32** across the 8 index layers —
the batch-scaled end is the larger target, and it is real work rather than
launch overhead. The gate above is left as stated; it is judged against the
fallback, and this only re-prices that fallback.

## Acceptance gate, restated at the served shape (B=32)

Supersedes the 2048-page gate above; same three properties, aimed at the shape
whose cost survives graph capture. Arm as in
`docs/runs/2026-11-17-indexer-graph-capture.md`: one V100-SXM2-32GB, torch
2.9.1+cu128, CUDA 12.8, **inside a captured CUDA graph, replay-only**, 10 warm-up
replays discarded, 50 timed, median — the regime the baseline is priced in.

**Baseline: 1.3425 ms/call at B=32, 128 pages/sequence, seq_len 8192**, i.e.
10.74 ms/step across the eight index layers. That multiplication is the prize
and the reason this kernel exists; at B=1 the same call is 0.1403 ms and the
case is much weaker.

1. **Speed.** At least **3x** the captured baseline: **<= 0.447 ms/call** at
   B=32, 128 pages, seq_len 8192. Measured in the same captured regime, because
   an eager figure and a captured one are not comparable.
2. **Agreement.** `max |logit_kernel - logit_torch| <= 2e-2` absolute on a
   non-zero cache (random e4m3 values, real per-token fp32 scales, random
   `weight`, `seq_lens` shorter than the table), **and** the **top-64 index set
   identical on every row** — set equality, not overlap. Why 2e-2: the dot is
   128 terms accumulated in fp32 with an 8-bit-significand operand (bf16), so a
   correct kernel sits a few times 1e-3 below this; it is loose enough that a
   correct reduction passes without tuning and tight enough that a broken one
   fails. The top-64 equality is the check that a tolerance cannot hide behind.
3. **Failure rule, unchanged.** Speed passes and agreement fails: rejected, not
   retuned against the top-k. Agreement passes and speed is under 3x: reported
   as "not worth it", not as a partial win.

### What the win has to come from, named

The baseline is not launch-bound (capture removes 9% at B=32, against 49% at
B=1), so the 3x has to come from removing intermediate tensors. At B=32, 128
pages, 8192 tokens, in the reference:

| intermediate | size | what the reference does with it |
| --- | --- | --- |
| gathered pages | 34.6 MB | `kvcache_flat[pages_clamped]`, unavoidable input traffic |
| `kv_values` cast | 67.1 MB bf16 | materialized by `.to(torch.bfloat16)`, read once by the bmm |
| `scores` before the head sum | 33.6 MB fp32 | written by `bmm`, then read/written by `relu`, the `weight` multiply, and `sum(dim=2)` |

A fused kernel keeps the gathered pages as the only large read, casts in
registers, and never materializes `scores`: the head reduction happens in
accumulators. If the implementation keeps the 67 MB cast or the 33.6 MB score
tensor, most of the 10.74 ms/step survives and the gate will say so honestly.

### Cast semantics: what actually needs pinning

The reference casts e4m3 to bf16 **before** the product. That cast is exact —
e4m3 carries 3 mantissa bits against bf16's 7, and its exponent range sits
inside bf16's — so a kernel that widens e4m3 in registers cannot differ from it
on that step, and no tolerance is being spent to hide a cast difference.

What must be pinned is **where the per-token scale is applied**: the reference
multiplies by `kv_scales` *after* the relu-weighted head sum, per token, not per
dimension. A kernel that folds the scale into the dot products is a different
function, and the top-64 set check is what would catch it.

### Caveat on the prize

10.74 ms/step exists only at a real batch. At B=1 the op is 1.12 ms/step and a
3x kernel buys 0.7 ms; at B=32 it buys ~7 ms. Both are stated so the number is
not read as a per-request cost.
