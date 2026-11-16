"""MXFP4 (E8M0 per 32) to NVFP4 (E4M3 per 16) scale expansion for the SM70 GEMM.

The DeepSeek-V4 FP8 checkpoints carry FP4 expert weights with one E8M0 scale
per 32 values. Our Marlin kernel takes one E4M3 scale per 16 values plus a
per-expert global scale, so the weights need no work at all: the nibble grid is
E2M1 on both sides. Only the scale grid changes, and it is exact when every
resulting ratio is representable in E4M3.

Measured on `models/bluewhale/base-fp8` (2026-11-16): exponent spread is 3-4
bits and no block clamps across 8,847,360 sampled scale slots, so a per-expert
global equal to the largest block scale is both sufficient and exact. The
helper still counts clamping rather than assuming it, because a checkpoint with
a wider spread would lose the smallest blocks silently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..format_contract import E8M0_BIAS, MIN_EXACT_E4M3_EXPONENT

if TYPE_CHECKING:
    import torch


def _as_exponents(scale) -> torch.Tensor:
    """Return the raw E8M0 exponent bytes of a scale tensor as int16."""
    import torch

    if scale.dtype == torch.uint8 or scale.dtype == torch.int8:
        return scale.view(torch.uint8).to(torch.int16)
    if scale.dtype == torch.float8_e8m0fnu:
        return scale.view(torch.uint8).to(torch.int16)
    # float containers hold the numerical 2**e; recover the exponent.
    return torch.log2(scale.to(torch.float32)).round().to(torch.int16) + E8M0_BIAS


def expand_mxfp4_scales(scale, dtype=None):
    """[.., K/32] E8M0 ``scale`` -> ([.., K/16] E4M3 scales, global, clamped).

    ``global`` is one float32 per expert: the largest block scale, so every
    ``e4m3_scale = block / global`` is at most 1 and the product
    ``e4m3_scale * global`` is the value the per-32 block carried. Every block
    is repeated twice, because a 32-value block is two 16-value blocks that
    share it, and the values inside them do not change.

    ``clamped`` counts the 32-value blocks whose ratio falls below 2**-9 and so
    cannot be represented in E4M3 — those two halves lose their scale. Zero is
    the expected result for the DeepSeek-V4 checkpoints measured so far; a
    non-zero count is a numeric loss the caller has to accept knowingly.
    """
    import torch

    out_dtype = dtype or torch.float8_e4m3fn
    exponents = _as_exponents(scale)
    if exponents.ndim < 1:
        raise ValueError("scale tensor must have at least one dimension")

    reduce_dims = tuple(range(1, exponents.ndim))
    global_exp = exponents.amax(dim=reduce_dims) if reduce_dims else exponents
    ratio_exp = exponents - global_exp.reshape(
        global_exp.shape + (1,) * len(reduce_dims)
    )
    clamped = int((ratio_exp < MIN_EXACT_E4M3_EXPONENT).sum())

    values = torch.exp2(ratio_exp.to(torch.float32))
    e4 = values.to(out_dtype)
    expanded = e4.repeat_interleave(2, dim=-1).contiguous()
    global_scale = torch.exp2(global_exp.to(torch.float32) - E8M0_BIAS)
    return expanded, global_scale, clamped
