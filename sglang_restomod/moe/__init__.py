"""SM70 MoE plumbing that is not the expert GEMM itself.

The expert GEMM lives in ``sglang_restomod.quant.sm70_fp4_moe``; this package
holds the surrounding routing kernels, which every MoE backend shares.
"""
