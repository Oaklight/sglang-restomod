# What TileLang can emit on SM70

Read this before proposing a TileLang kernel for Volta. It is one measured
limit and the reason it is easy to get wrong.

## The limit: no fp32 load layout in the MMA macro generator

An fp32 TileLang GEMM does not compile on sm70. It fails in layout inference,
not at launch and not with a shape error:

```
File ".../tilelang/tileop/gemm/__init__.py", line 118, in infer_layout
File ".../tilelang/tileop/gemm/gemm_mma_sm70.py", line 44, in infer_layout
File ".../tilelang/intrinsics/mma_sm70_macro_generator.py", line 359, in make_mma_load_layout
ValueError: Unsupported dtype float32
```

Measured 2026-11-16 on a V100-SXM2-32GB, pin `297e54f` (v0.5.23), launching
`mhc_pre`'s split-K prenorm kernel through `_prewarm_mhc_kernels`. The GEMM
carries its `fn` operand as float32; nothing about tile size, block size or
split-K changes the outcome, because the failure is at load-layout selection
for the operand dtype.

## Why it is easy to get wrong

"TileLang gives us real MMA on sm70 while Triton emits FMA only" is true for
the 16-bit dtypes our kernels use, and it is commonly repeated without the
qualifier. The qualifier is what matters here: the MMA path exists on sm70,
and it accepts a limited dtype set. An fp32 kernel gets no MMA path at all —
so "use TileLang instead of Triton" is not a route out of an fp32 problem.

## What to do

- Check the operand dtypes of a candidate kernel first. float16/bfloat16
  operands are the supported case; float32 is not, and package-level
  precision parameters (a `float32` accumulator input, a fused stats/sumsq
  operand like `mhc_pre`'s `fn`) are operands, not options.
- If a kernel is fp32 by design, the routes are: change the operand dtype and
  show the numerics still hold, or take a torch/Triton implementation and
  accept its speed.
- Do not generalize a size table hoping to make an fp32 kernel compile. That
  is exactly what `patches/0006-mhc-prenorm-block-selection.patch` looked
  like it would do: it replaces `mhc_pre`'s two-entry `hc_hidden_size` table
  with the rule that produced it (largest of {256,128} dividing
  `split_size`), and it is correct for platforms whose TileLang has the
  dtype. On SM70 it is **inert — a dtype limit, not a size limit**. The SM70
  route for that path is the torch implementation, selected by
  `SGLANG_OPT_USE_TILELANG_MHC_PRE=0` (see `patches/0005`).
