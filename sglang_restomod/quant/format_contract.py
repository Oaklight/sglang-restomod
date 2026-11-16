"""What each 4-bit weight format on SM70 expects, in one place.

Three encodings of the same knowledge had grown up separately: the expert MoE
scale encoder, the dense NVFP4 wrapper, and the MXFP4-scale expansion. They
agreed, but nothing made them agree, and on 2026-11-16 a missing global scale
that all three knew about reached the kernel and died in a ``TORCH_CHECK``
three layers down with no format named.

So this module holds the parts that are format knowledge and nothing else:

* a descriptor per format -- the value grid, the scale grid, whether a global
  scale is part of the encoding, and how two values are ordered inside a byte;
* assertions a repack site calls on the tensors it is about to hand over, which
  raise :class:`FormatError` naming the format and the tensor;
* the Q0.7 per-block scale encoding the SM70 iterator reads.

It is deliberately not "marlin": QPN2 and QPN8 are our own layouts, and their
knowledge stays with them. Nothing here is loaded at import -- torch is imported
inside each function, as ``CLAUDE.md`` requires for a plugin that loads at
server startup.
"""

from __future__ import annotations

from dataclasses import dataclass

# One E8M0 byte is an exponent with this bias; 127 encodes 2**0.
E8M0_BIAS = 127

# E4M3 (e4m3fn) represents every power of two from 2**-9 (its subnormal step)
# up to 2**8 exactly. A ratio outside that band is a numeric loss, not a
# rounding: the block scale cannot be carried at all.
MIN_EXACT_E4M3_EXPONENT = -9

# The SM70 iterator rebuilds a per-block scale from one byte with
# ``(byte << 8) >> 1``, so the byte carries ``scale * 2**Q07_SHIFT`` and the
# global scale is compensated by the same factor. See :func:`encode_q07`.
Q07_SHIFT = 7


class FormatError(ValueError):
    """A tensor does not match the format the kernel will read it as."""


@dataclass(frozen=True)
class Format:
    """One 4-bit weight format as the SM70 kernels see it."""

    name: str
    values: str
    value_bits: int
    scale_dtype: str
    scale_group: int
    global_scale: str  # "required" or "none"
    packing: str

    @property
    def values_per_byte(self) -> int:
        return 8 // self.value_bits

    def logical_k(self, packed_k: int) -> int:
        return packed_k * self.values_per_byte


MXFP4 = Format(
    name="MXFP4",
    values="e2m1",
    value_bits=4,
    scale_dtype="e8m0",
    scale_group=32,
    # E8M0 is an exponent, so the grid is self-contained: there is nothing for
    # a global scale to compensate.
    global_scale="none",
    packing="two values per byte, low nibble = even k",
)

NVFP4 = Format(
    name="NVFP4",
    values="e2m1",
    value_bits=4,
    scale_dtype="e4m3",
    scale_group=16,
    # The block scale is quantized, so its own magnitude is a separate number.
    # Dropping it silently rescales every weight in the tensor.
    global_scale="required",
    packing="two values per byte, low nibble = even k",
)

FORMATS = {fmt.name: fmt for fmt in (MXFP4, NVFP4)}


def check_packed_weight(weight, fmt: Format, where: str) -> int:
    """Check the packed container and return the logical K it holds."""
    import torch

    if weight.dtype not in (torch.int8, torch.uint8):
        raise FormatError(
            f"{fmt.name}: {where} is {weight.dtype}, not the packed 8-bit container "
            f"this format is read from ({fmt.packing})"
        )
    if weight.ndim < 2:
        raise FormatError(
            f"{fmt.name}: {where} has shape {tuple(weight.shape)}; a packed weight "
            "needs at least [N, K/packed] with K in the last dimension"
        )
    return fmt.logical_k(int(weight.shape[-1]))


def check_scale_grid(scale, weight, fmt: Format, where: str, k: int | None = None) -> int:
    """Check that ``scale`` holds one entry per ``fmt.scale_group`` values of K.

    ``k`` is the logical K when the caller already knows it; otherwise it is
    read off ``weight``. Returns the logical K, so a caller can check the scale
    and the weight in one call.
    """
    if k is None:
        k = check_packed_weight(weight, fmt, where + "_weight")
    want = tuple(weight.shape[:-1]) + (k // fmt.scale_group,)
    if tuple(scale.shape) != want:
        raise FormatError(
            f"{fmt.name}: {where} holds {tuple(scale.shape)} scales, but "
            f"{tuple(weight.shape)} packed implies one {fmt.scale_dtype} scale per "
            f"{fmt.scale_group} of K={k}, i.e. {want}"
        )
    return k


def check_global_scale(global_scale, fmt: Format, where: str) -> None:
    """Check that a format whose encoding needs a global scale has one."""
    if fmt.global_scale != "required" or global_scale is not None:
        return
    raise FormatError(
        f"{fmt.name}: {where} has no per-expert global scale. A {fmt.scale_dtype} "
        f"scale per {fmt.scale_group} is half of the encoding, not a self-contained "
        "grid, and the kernel refuses the pair with no global (ops.cu: 'the "
        "global_scale parameter must be passed for nvfp4 format')"
    )


def encode_q07(scales):
    """Logical E4M3 block scales -> the Q0.7 byte the SM70 iterator reads.

    The iterator rebuilds a scale as ``byte << 7``, so what it needs is the high
    byte of ``fp16(scale * 2**7)`` with the pad bit dropped. For a scale of 1.0
    the byte is 176. This is bit-identical to the pin's
    ``nvfp4_marlin_process_scales``; ``tests/test_sm70_format_contract.py``
    holds that equivalence, along with the expert component's own copy.

    The 4-wide pair reversal cancels the reversal the iterator applies when it
    expands its metadata. The last dimension is N and must be a multiple of 4.
    """
    import torch

    if scales.dtype != torch.float8_e4m3fn:
        raise FormatError(
            f"Q0.7 encoding expects E4M3 block scales, got {scales.dtype}"
        )
    if scales.ndim < 1 or int(scales.shape[-1]) % 4 != 0:
        raise FormatError(
            f"Q0.7 encoding needs N divisible by 4 in the last dimension, got "
            f"shape {tuple(scales.shape)}"
        )
    logical = scales.to(torch.float16)
    if bool((logical < 0).any()):
        raise FormatError("NVFP4 block scales must be non-negative")

    shape = tuple(scales.shape)
    size_n = shape[-1]
    flat = logical.reshape(-1, size_n)
    flat = flat.view(-1, 4)[:, [0, 2, 1, 3]].reshape(-1, size_n)
    encoded = (flat * (2**Q07_SHIFT)).view(torch.int16) << 1
    encoded = encoded.view(torch.float8_e4m3fn).reshape(-1, size_n * 2)
    return encoded[:, 1::2].contiguous().reshape(shape)
