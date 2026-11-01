"""NVFP4 scale encoding for the SM70 Marlin expert GEMM.

The SM70 iterator rebuilds an FP16 scale from one byte with integer shifts
(``(byte << 8) >> 1``, i.e. ``byte << 7``). What this encoder emits is the high
byte of ``fp16(scale * 2**7)`` with the low pad bit dropped -- a Q0.7-style
code with the exponent in the high bits, not a float field. For a scale of
``1.0`` the byte is ``176``. The result is bit-identical to upstream's
``nvfp4_marlin_process_scales``, which the SM80 path validates.

The byte order is logical ``[experts, K/16, N]``: the iterator indexes the
per-expert scale block by logical ``n``, so ``marlin_permute_scales`` must not
be applied here.

Ported from ``sglang-v100-plus`` (``v100_plus/sglang_v100_plus/quantization.py``).
"""

from __future__ import annotations


def encode_scales(scales, activation_dtype):
    """Encode logical ``[E, K/16, N]`` E4M3 scales for the SM70 iterator."""
    import torch

    if activation_dtype != torch.float16:
        raise ValueError(
            "SM70 NVFP4 Marlin supports FP16 activations only, got "
            f"{activation_dtype}"
        )
    if scales.dtype != torch.float8_e4m3fn or scales.ndim != 3:
        raise ValueError(
            "SM70 NVFP4 Marlin expects [E, K/16, N] E4M3 scales, got "
            f"shape={tuple(scales.shape)}, dtype={scales.dtype}"
        )

    logical = scales.to(torch.float16)
    if bool((logical < 0).any()):
        raise ValueError("NVFP4 block scales must be non-negative")

    # FP16's exponent range covers ModelOpt's normalized E4M3 scales: the
    # largest (448) times 2**7 is 57,344, still inside fp16. BF16 is rejected
    # above rather than rescaled, because the compensation would also have to
    # land on the global scale and that pairing is not exercised anywhere.
    scale_factor = 1.0

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

    if activation_dtype != torch.float16:
        raise ValueError(
            "SM70 NVFP4 Marlin supports FP16 activations only, got "
            f"{activation_dtype}"
        )
    # E5 (fp16) to E2 (fp4): 2**4 - 2**1 = 14. The -7 stands for the 128 that
    # ``encode_scales`` multiplied into every block scale.
    exponent_bias = 2**4 - 2**1
    return (
        global_scale.to(torch.float32) * (2.0 ** (exponent_bias - 7))
    ).contiguous()
