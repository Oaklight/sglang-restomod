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
