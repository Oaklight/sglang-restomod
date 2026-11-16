"""The format contract, and the encoders already in the tree, must agree.

Nothing here migrates the two existing implementations: they serve Flash-Next
and they work. The point is that the knowledge now exists once and is checked,
so the next component imports it instead of re-deriving it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sglang_restomod.quant import format_contract as fc  # noqa: E402
from sglang_restomod.quant.sm70_fp4_moe import mxfp4_scales as mx  # noqa: E402
from sglang_restomod.quant.sm70_fp4_moe import scales as moe_scales  # noqa: E402


def _representative():
    """[E, K/16, N] E4M3 scales spanning what ModelOpt emits."""
    columns = torch.tensor([2.0**-9, 2.0**-4, 0.5, 1.0, 1.5, 8.0, 64.0, 448.0])
    return columns.to(torch.float8_e4m3fn).repeat(2, 3).reshape(2, 3, 8).contiguous()


def test_the_expert_encoder_agrees_with_the_contract():
    scales = _representative()

    want = fc.encode_q07(scales)
    got, factor = moe_scales.encode_scales(scales, torch.float16)

    assert factor == 1.0
    assert tuple(got.shape) == tuple(want.shape)
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def test_the_pins_encoder_agrees_with_the_contract():
    utils = pytest.importorskip("sglang.srt.layers.quantization.marlin_utils_fp4")
    if not torch.cuda.is_available():
        # The pin's first line is ``scales >= 0``; fp8 comparison is
        # implemented on CUDA only, so on CPU this arm cannot run at all.
        pytest.skip("the pin compares E4M3 with >=, which CPU does not implement")

    scales = _representative().cuda()
    want = fc.encode_q07(scales)
    got = utils.nvfp4_marlin_process_scales(scales.reshape(-1, scales.shape[-1]))

    assert torch.equal(
        got.view(torch.uint8).cpu(),
        want.reshape(-1, want.shape[-1]).view(torch.uint8).cpu(),
    )


def test_a_scale_grid_that_does_not_match_the_weights_is_refused():
    weight = torch.zeros((4, 64, 32), dtype=torch.int8)  # K = 64, one per 16 -> 4
    scale = torch.zeros((4, 64, 2), dtype=torch.uint8)

    with pytest.raises(fc.FormatError, match="NVFP4"):
        fc.check_scale_grid(scale, weight, fc.NVFP4, "w13_weight_scale")


def test_a_packed_weight_that_is_not_8_bit_is_refused():
    with pytest.raises(fc.FormatError, match="MXFP4"):
        fc.check_packed_weight(torch.zeros((4, 64, 32), dtype=torch.int32), fc.MXFP4, "w13")


def test_a_missing_global_scale_names_the_format():
    with pytest.raises(fc.FormatError, match="NVFP4.*global scale"):
        fc.check_global_scale(None, fc.NVFP4, "w13")

    # MXFP4 is self-contained, so no global is the correct state there.
    fc.check_global_scale(None, fc.MXFP4, "w13")


def test_the_expansion_takes_its_grid_constants_from_the_contract():
    assert mx.E8M0_BIAS == fc.E8M0_BIAS
    assert mx.MIN_EXACT_E4M3_EXPONENT == fc.MIN_EXACT_E4M3_EXPONENT
    # 2**-12 against a 2**0 block is below the floor, so exactly one block clamps.
    exponents = torch.tensor([[0, 0, -12]], dtype=torch.int16) + fc.E8M0_BIAS
    _, _, clamped = mx.expand_mxfp4_scales(exponents.to(torch.uint8))
    assert clamped == 1
