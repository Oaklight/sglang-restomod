"""NVFP4 scale encoding for the SM70 Marlin expert GEMM.

The SM70 iterator reconstructs an FP16 scale from one byte with integer shifts
(``(byte << 8) >> 1``, i.e. ``byte << 7``). It therefore consumes a compact
non-negative S0E5M3 byte, not the checkpoint's S1E4M3 value. The byte order is
logical ``[experts, K/16, N]``: the iterator indexes the per-expert scale block
by logical ``n``, so ``marlin_permute_scales`` must not be applied here.

Ported from ``sglang-v100-plus`` (``v100_plus/sglang_v100_plus/quantization.py``).
"""

from __future__ import annotations


def encode_scales(scales, activation_dtype):
    """Encode logical ``[E, K/16, N]`` E4M3 scales for the SM70 iterator.

    Returns ``(encoded, factor)``. ``factor`` is a power of two applied to the
    scales for bfloat16; the caller divides the processed global scale by it.
    """
    import torch

    if activation_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"SM70 NVFP4 Marlin supports FP16/BF16 activations, got {activation_dtype}"
        )
    if scales.dtype != torch.float8_e4m3fn or scales.ndim != 3:
        raise ValueError(
            "SM70 NVFP4 Marlin expects [E, K/16, N] E4M3 scales, got "
            f"shape={tuple(scales.shape)}, dtype={scales.dtype}"
        )

    logical = scales.to(torch.float16)
    if bool((logical < 0).any()):
        raise ValueError("NVFP4 block scales must be non-negative")

    # FP16 has enough exponent range for ModelOpt's normalized E4M3 scales.
    # BF16 may need a shared rescale so every non-zero S0E5M3 byte keeps its
    # high bit set, which the integer dequantizer relies on.
    scale_factor = 1.0
    if activation_dtype == torch.bfloat16:
        nonzero = logical[logical > 0]
        if nonzero.numel() > 0:
            min_scaled = nonzero.float().min() * (2**7)
            if min_scaled < 2:
                scale_factor = float(torch.ceil(torch.log2(2 / min_scaled)).exp2())
                logical = (logical.float() * scale_factor).to(torch.float16)

    num_experts, num_groups, size_n = logical.shape
    if size_n % 4 != 0:
        raise ValueError(f"SM70 NVFP4 Marlin requires N divisible by 4, got N={size_n}")

    flat = logical.reshape(-1, size_n)
    # The iterator reverses each pair when it expands metadata, so the pairs
    # are reordered here to cancel that reversal.
    flat = flat.view(-1, 4)[:, [0, 2, 1, 3]].reshape(-1, size_n)
    encoded = (flat * (2**7)).view(torch.int16) << 1
    encoded = encoded.view(torch.float8_e4m3fn).reshape(-1, size_n * 2)
    encoded = encoded[:, 1::2].contiguous()
    return encoded.reshape(num_experts, num_groups, size_n), scale_factor


def encode_global_scale(global_scale, activation_dtype):
    """Compensate an NVFP4 global scale for SM70's integer dequantizer."""
    import torch

    if activation_dtype == torch.float16:
        target_exponent = 5
    elif activation_dtype == torch.bfloat16:
        target_exponent = 8
    else:
        raise ValueError(
            f"SM70 NVFP4 Marlin supports FP16/BF16 activations, got {activation_dtype}"
        )
    fp4_exponent = 2
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return (
        global_scale.to(torch.float32) * (2.0 ** (exponent_bias - 7))
    ).contiguous()
