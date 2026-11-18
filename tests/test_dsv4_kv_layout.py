"""The DSV4 KV page layouts: offsets, E8M0 decoding, and a round trip.

Every expected number here is copied from ``sgl_kernel/deepseek_v4/kv_layout.cuh``
at the pin, not derived from the implementation, so a change in either is a
failure rather than a silent agreement.
"""

from __future__ import annotations

import pytest

from sglang_restomod.attention.dsv4_sm70 import kv_layout

torch = pytest.importorskip("torch")


def test_traits_match_the_header():
    v4 = kv_layout.get_layout("V4")
    assert (v4.data_bytes, v4.scale_bytes, v4.tile) == (576, 8, 64)
    assert v4.bytes_per_token() == 584
    assert v4.rope_bf16 is True

    v41 = kv_layout.get_layout("V41")
    assert (v41.data_bytes, v41.scale_bytes, v41.tile) == (512, 16, 32)
    assert v41.bytes_per_token() == 528
    assert v41.interleaved is False

    new = kv_layout.get_layout("V41_NEW")
    assert new.bytes_per_token() == 528
    assert new.interleaved is True

    fp4 = kv_layout.get_layout("V41_FP4")
    assert (fp4.data_bytes, fp4.scale_bytes, fp4.tile) == (256, 32, 16)


def test_page_bytes_pads_to_the_row_stride():
    assert kv_layout.page_bytes(kv_layout.get_layout("V4"), 256) == 149760  # 260 * 576
    assert kv_layout.page_bytes(kv_layout.get_layout("V41"), 256) == 135168  # 264 * 512
    assert kv_layout.page_bytes(kv_layout.get_layout("V41_NEW"), 256) == 135168


def test_blocked_offsets_put_all_scales_after_all_data():
    layout = kv_layout.get_layout("V41")
    data_0, scale_0 = kv_layout.row_offsets(layout, 256, 0)
    data_5, scale_5 = kv_layout.row_offsets(layout, 256, 5)
    assert (data_0, scale_0) == (0, 256 * 512)
    assert (data_5, scale_5) == (5 * 512, 256 * 512 + 5 * 16)


def test_interleaved_offsets_keep_scales_beside_their_token():
    layout = kv_layout.get_layout("V41_NEW")
    assert kv_layout.row_offsets(layout, 256, 3) == (3 * 528, 3 * 528 + 512)


def test_slot_outside_the_page_is_refused():
    with pytest.raises(ValueError, match="outside a page"):
        kv_layout.row_offsets(kv_layout.get_layout("V41"), 256, 256)


def test_e8m0_decodes_to_powers_of_two():
    raw = torch.tensor([127, 126, 128, 0, 119, 255], dtype=torch.uint8)
    got = kv_layout.decode_e8m0(raw)
    assert got[0] == 1.0
    assert got[1] == 0.5
    assert got[2] == 2.0
    assert got[3] == pytest.approx(2.0**-127)
    assert got[4] == pytest.approx(2.0**-8)
    assert torch.isnan(got[5])


def test_v41_round_trip():
    """Quantize a row the way the store kernel does, then read it back."""
    layout = kv_layout.get_layout("V41")
    dims = 512
    value = torch.randn(dims, dtype=torch.float32)
    fp8_max = 448.0
    # Per-32 groups: scale = amax/448 as an E8M0 power of two, values divided by it.
    grouped = value.view(dims // 32, 32)
    amax = grouped.abs().amax(dim=1).clamp_min(1e-4)
    exponent = torch.log2(amax / fp8_max).ceil().clamp(-127, 127)
    mult = torch.exp2(exponent)
    quantized = (grouped / mult[:, None]).to(torch.float8_e4m3fn)

    data = quantized.view(torch.uint8).reshape(-1)
    scales = (exponent.to(torch.int32) + kv_layout.E8M0_BIAS).to(torch.uint8)
    back = kv_layout.dequantize_token(layout, data, scales)

    err = (back - value).abs()
    # E4M3 carries a 4-bit significand, so round-to-nearest is within 2**-4 of
    # the element's own magnitude -- not of the group scale, which is why the
    # bound is relative to the largest value, not to `mult`. A wrong group or a
    # misread scale misses this by orders of magnitude, not by a factor of two.
    assert float(err.max()) <= float(value.abs().max()) / 16 + 1e-6
