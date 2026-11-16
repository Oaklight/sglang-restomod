"""E8M0 per-32 -> E4M3 per-16 expansion: exact where it can be, counted where not."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sglang_restomod.quant.sm70_fp4_moe.mxfp4_scales import (  # noqa: E402
    expand_mxfp4_scales,
)


def _e8m0(exponents):
    """Build an E8M0 tensor from integer exponents (127 == 2**0)."""
    return (torch.tensor(exponents, dtype=torch.int16) + 127).to(torch.uint8).view(
        torch.float8_e8m0fnu
    )


def test_every_block_is_reproduced_exactly_within_range():
    exponents = [[-8, -7, -6, -5], [-7, -7, -7, -7]]
    scales = _e8m0(exponents)

    e4, global_scale, clamped = expand_mxfp4_scales(scales)

    assert clamped == 0
    assert e4.shape == (2, 8)
    assert e4.dtype == torch.float8_e4m3fn
    want = torch.exp2(torch.tensor(exponents, dtype=torch.float32))
    # Compare per block: the two halves of a block share one scale.
    got = (e4[:, 0::2].to(torch.float32) * global_scale.unsqueeze(-1))
    assert torch.equal(got, want)
    assert torch.equal(global_scale, torch.exp2(torch.tensor([-5.0, -7.0])))


def test_a_block_below_the_representable_band_is_counted():
    exponents = [[0, 0, -12], [-3, -3, -3]]
    scales = _e8m0(exponents)

    _, _, clamped = expand_mxfp4_scales(scales)

    # 2**-12 relative to 2**0 is below E4M3's 2**-9 floor: one 32-value block.
    assert clamped == 1


def test_uint8_and_e8m0_inputs_agree():
    exponents = [[-9, -8, -8, -7, -7, -6]]
    as_e8m0 = _e8m0(exponents)
    as_bytes = (torch.tensor(exponents, dtype=torch.int16) + 127).to(torch.uint8)

    a, ga, ca = expand_mxfp4_scales(as_e8m0)
    b, gb, cb = expand_mxfp4_scales(as_bytes)

    assert ca == cb == 0
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert torch.equal(ga, gb)


def test_shape_follows_the_grid_change():
    scales = torch.full((4, 2304, 160), 120, dtype=torch.uint8)

    e4, global_scale, clamped = expand_mxfp4_scales(scales)

    assert e4.shape == (4, 2304, 320)
    assert global_scale.shape == (4,)
    assert clamped == 0
