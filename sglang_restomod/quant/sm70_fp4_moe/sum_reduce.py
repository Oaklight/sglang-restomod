"""Triton replacement for the missing ``sgl_kernel.moe_sum_reduce``.

The installed ``sgl_kernel`` build exports no ``moe_sum_reduce`` op, and the
Marlin MoE runner calls it to sum the top-k expert outputs:

    AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute
    'moe_sum_reduce'

The kernel here is the one SGLang keeps as the Triton reference for that op
(``kernels/aot/benchmark/bench_sum_scale.py``), so the arithmetic is the same:
a float32 sum over the top-k dimension, multiplied by the routed scaling
factor, stored in the input dtype.

Ported from SGLang (Apache-2.0).
"""

from __future__ import annotations

import torch


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def moe_sum_reduce_kernel(
        input_ptr,
        input_stride_0,
        input_stride_1,
        input_stride_2,
        output_ptr,
        output_stride_0,
        output_stride_1,
        token_num: int,
        topk_num: int,
        hidden_dim: int,
        routed_scaling_factor: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        NUM_STAGE: tl.constexpr,
    ):
        input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
        input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
        output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

        token_block_id = tl.program_id(0)
        dim_block_id = tl.program_id(1)

        offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

        mask_token = offs_token < token_num
        mask_dim = offs_dim < hidden_dim

        base_ptrs = (
            input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tile = tl.load(
                base_ptrs + i * input_stride_1,
                mask=mask_token[:, None] & mask_dim[None, :],
                other=0.0,
            )
            accumulator += tile.to(tl.float32)
        accumulator *= routed_scaling_factor

        store_ptrs = (
            output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
        )
        tl.store(
            store_ptrs,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=mask_token[:, None] & mask_dim[None, :],
        )

    return moe_sum_reduce_kernel


_KERNEL = None


def moe_sum_reduce(input_tensor, output_tensor, routed_scaling_factor=0):
    """Sum ``input`` over its top-k dimension into ``output``."""
    global _KERNEL
    import triton

    if _KERNEL is None:
        _KERNEL = _kernel()
    if not (input_tensor.is_contiguous() and output_tensor.is_contiguous()):
        raise ValueError("moe_sum_reduce expects contiguous tensors")
    token_num, topk_num, hidden_dim = input_tensor.shape
    block = min(triton.next_power_of_2(hidden_dim), 2048)
    grid = (triton.cdiv(token_num, 1), triton.cdiv(hidden_dim, block))
    _KERNEL[grid](
        input_tensor,
        *input_tensor.stride(),
        output_tensor,
        *output_tensor.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=float(routed_scaling_factor),
        BLOCK_M=1,
        BLOCK_DIM=block,
        NUM_STAGE=1,
        num_warps=16,
    )
    return output_tensor


def available() -> bool:
    """Whether the replacement is needed, i.e. the op is absent."""
    try:
        getattr(torch.ops.sgl_kernel, "moe_sum_reduce")
        return False
    except AttributeError:
        return True
