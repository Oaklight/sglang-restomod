"""Golden-vector tests for the SM70 NVFP4 scale encoder (CPU).

These pin the byte the encoder emits, independent of the GEMM. The GPU check in
``sm70_fp4_moe/check.py`` dequantizes with the same conventions it quantizes
with, so a wrong scale layout cancels out there; this file does not.

The mechanism under test: the SM70 iterator rebuilds an fp16 scale from one
byte with ``byte << 7``, so the encoder emits the high byte of
``fp16(scale * 2**7)`` -- equivalently ``fp16_bits(scale * 128) >> 7``.

The constants below were derived from the implementation, so this file locks
the current output against future edits rather than proving the layout. The
``[0,2,1,3]`` pair order had no coverage at all before it; an error already
inside the encoder would still pass. The real-checkpoint comparison against an
independent dequantization is still open, and the PR test plan leaves that box
unchecked.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _byte(scale: float) -> int:
    """High byte of ``fp16(scale * 2**7)``, written the long way."""
    bits = int((torch.tensor([scale], dtype=torch.float16) * 128).view(torch.int16)[0])
    return (bits >> 7) & 0xFF


#: (scale, emitted byte). Every value here is exact in both e4m3 and fp16.
GOLDEN = (
    (1.0, 176),  # fp16(128.0) = 0x5800
    (0.5, 168),  # fp16(64.0)  = 0x5400
    (2.0, 184),  # fp16(256.0) = 0x5C00
    (3.0, 188),  # fp16(384.0) = 0x5E00
    (0.25, 160),  # fp16(32.0)  = 0x5000
    (448.0, 246),  # fp16(57344.0) = 0x7B00, the e4m3 maximum
    (0.0, 0),
    (6.0, 196),  # fp16(768.0) = 0x6200
)


def test_golden_bytes_and_pair_order():
    from sglang_restomod.quant.sm70_fp4_moe.scales import encode_scales

    values = [scale for scale, _ in GOLDEN]
    scales = (
        torch.tensor(values, dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .reshape(1, 1, len(values))
    )
    encoded, factor = encode_scales(scales, torch.float16)

    assert factor == 1.0
    got = encoded.reshape(-1).view(torch.uint8).tolist()
    # The encoder reverses each group of four to cancel the iterator's own
    # reversal, so the emitted order inside a group is 0, 2, 1, 3.
    want = [GOLDEN[i][1] for i in (0, 2, 1, 3, 4, 6, 5, 7)]
    assert got == want


def test_bytes_match_the_scaled_fp16_high_byte():
    from sglang_restomod.quant.sm70_fp4_moe.scales import encode_scales

    torch.manual_seed(0)
    size_n = 64
    scales = (torch.rand(2, 3, size_n) * 400).to(torch.float8_e4m3fn)
    encoded, _ = encode_scales(scales, torch.float16)

    swapped = scales.float().reshape(-1, size_n)
    swapped = swapped.view(-1, 4)[:, [0, 2, 1, 3]].reshape(-1, size_n)
    want = ((swapped.to(torch.float16) * 128).view(torch.int16) >> 7).to(torch.uint8)
    assert encoded.reshape(-1).view(torch.uint8).tolist() == want.reshape(-1).tolist()


def test_experts_are_independent():
    from sglang_restomod.quant.sm70_fp4_moe.scales import encode_scales

    scales = (
        torch.tensor([1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 0.125, 0.0625])
        .to(torch.float8_e4m3fn)
        .reshape(2, 1, 4)
    )
    encoded, _ = encode_scales(scales, torch.float16)
    got = encoded.reshape(-1).view(torch.uint8).tolist()
    want = [_byte(scale.item()) for scale in scales.reshape(-1)]
    # swap 0,2,1,3 within each expert's single group of four
    assert got == [want[i] for i in (0, 2, 1, 3, 4, 6, 5, 7)]


def test_bfloat16_is_rejected():
    from sglang_restomod.quant.sm70_fp4_moe.scales import (
        encode_global_scale,
        encode_scales,
    )

    scales = torch.ones(1, 1, 4).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="FP16"):
        encode_scales(scales, torch.bfloat16)
    with pytest.raises(ValueError, match="FP16"):
        encode_global_scale(torch.ones(1), torch.bfloat16)


def test_global_scale_is_multiplied_by_128():
    from sglang_restomod.quant.sm70_fp4_moe.scales import encode_global_scale

    global_scale = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32)
    got = encode_global_scale(global_scale, torch.float16)
    assert got.tolist() == [128.0, 64.0, 256.0]
