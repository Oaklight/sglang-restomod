# DSV4 HC prenorm: kernel design (not implemented)

Design only, with the acceptance gate fixed before any run. This is the first
kernel to write on this path — see "Why first" below, which reverses the order
I argued yesterday on launch arithmetic rather than preference.

## What it replaces

`hc_pre_torch_impl` (`srt/models/deepseek_v4.py:3358`), the torch side of
`hc_pre` when DeepGEMM and TileLang are off — which on SM70 is always, since
the TileLang split-K kernel cannot compile here (`docs/sm70-tilelang-dtypes.md`).
Measured at 0.117 ms per call flat from M=1 to M=32 and 2.62 ms at M=4096
(`docs/runs/2026-11-17-fallback-microbench.md`).

## The math, read from that implementation

```
x_flat = x.flatten(1).float()                       # [T, hc_mult*hidden] = [T, 20480]
rsqrt  = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)   # [T, 1]
mixes  = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)                  # [T, 1, 24]
```

`hc_fn` is `[mix_hc, hc_hidden]` = `[24, 20480]` fp32. `mixes` feeds
`hc_split_sinkhorn` immediately afterwards. Nothing else in the call has
state, so the whole thing is one reduction plus one skinny GEMM over the same
matrix.

## Why first

Launch counts per decode step, from the two measurements and the call sites:

| | torch cost/call | calls/step | torch ms/step | launches/call |
| --- | --- | --- | --- | --- |
| prenorm | 0.117 ms | 40 (all layers) | **4.7** (pre only) | ~6 (cast, square, mean, add, rsqrt, linear, mul) |
| indexer | 0.287 ms | 8 (`index_source_layer_ids`) | **2.3** | ~10 (clamp, gather, 2 slices, 2 copies, 2 casts, bmm, relu, mul, sum, mul, mask, pad) |

The prenorm has fewer launches per call but five times the calls, so fusing it
removes more step time: ~5 launches x 40 = 200 launches against ~10 x 8 = 80.
Its floor is 0.027 ms (one cuBLAS matmul), so a fused kernel targets ~0.03-0.04
ms/call, i.e. **~1.3 ms/step against 4.7** — and the post step, unmeasured,
adds to that. That is why it goes first.

## Shape of the kernel

One launch. `hc_fn` is 24 x 20480 x 4 B = 1.97 MB, too large for shared
memory, so the grid is over (token block, K split) and each block streams its
slice of `hc_fn` and of `x_flat`:

- accumulate the 24 dot products per token in registers, and the sum of
  squares for `rsqrt`, then combine the K splits through fp32 atomics into
  `[T, 24]` and `[T]` partials;
- the normalization `* rsqrt(sqrsum/T + eps)` is folded into the consumer
  (`hc_split_sinkhorn`), which already reads `mixes` and would take the
  sqrsum alongside it. That keeps it one launch; the alternative is a second
  tiny epilogue over `[T, 24]`, still 3-4x fewer launches than torch. The
  consumer change is the risk, and it is why the gate below is unchanged by
  either choice.

Input stays bf16: dequantize nothing, cast to fp32 in registers. The torch
path itself computes in fp32 (`flatten(1).float()`), so the contract is
unchanged — this is not a reduced-precision substitution.

## Acceptance gate, stated in advance

Arm: one V100-SXM2-32GB, torch 2.9.1+cu128, CUDA 12.8, no CUDA graphs, 10
warm-up iterations discarded, 50 repeats, median of per-iteration events,
identical to the baseline.

1. **Speed.** Per call, at `M in {1, 4, 32}`: **<= 0.040 ms** (>= 3x torch's
   0.117). At `M = 256`: <= 0.106 ms and `M = 1024`: <= 0.369 ms (>= 2x, the
   shapes are bandwidth-bound so the fused ceiling is lower). At `M = 4096`:
   <= 1.0 ms (>= 2.6x; the memory floor is ~0.37 ms and a 24-output GEMM does
   not reach it).
2. **Agreement.** With a fixed seed, random bf16 `x` and fp32 `hc_fn`:
   `max |mixes_kernel - mixes_torch| <= 1e-2 * max |mixes_torch|`, no NaN and
   no Inf in either.
3. **Failure rule.** Speed without agreement is rejected, not retuned against
   the tolerance. Agreement without the stated speed is reported as "not worth
   it" rather than as a partial win.

Why a relative tolerance here and an absolute one for the indexer: `mixes`
feeds a normalization whose output scale is set by the row, so an absolute
number would be arbitrary; the top-k structure the indexer feeds is what made
"identical top-64" the right absolute test there.

## What this design does not claim

- No bluewhale performance claim: the baseline is the torch fallback on
  synthetic tensors, not a model run.
- The post step (`mhc_post`) is unmeasured; the 4.7 ms/step above is the pre
  step only, and the real gain may be larger.
- Correctness of the torch fallback itself is unverified on this hardware — no
  forward has run, because the DSV4 attention backend stops at its terminal
  flash_mla call.

## First gate run: missed

`docs/runs/2026-11-17-prenorm-kernel-gate.md` records the first implementation
against this gate: agreement passes (rel err ~1e-6, two-run delta 0.0), speed
misses at every shape — 0.22 ms against a 0.040 ms gate below M=32, 1.28x torch
at M=4096. The kernel is not landed. The diagnosis (grid of 8 programs at
small M, `MIX=24` padded to 32, fp32 having no tensor cores on sm70) and the
three next things to try are in that page.
