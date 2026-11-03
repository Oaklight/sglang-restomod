"""CPU test for the block-FP8 LM head quantization.

The kernel contract is exercised on the GPU host; this covers the layout the
component hands to it, which is where a shape or scale mistake would show up
before any GPU is involved.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _quantize():
    from sglang_restomod.quant.sm70_fp8_lm_head import _quantize as fn

    return fn


def test_shapes_and_dtypes():
    q_fn = _quantize()
    w = torch.randn(256, 128, dtype=torch.float16)

    q, scale = q_fn(w)

    assert q.shape == w.shape
    assert q.dtype == torch.float8_e4m3fn
    assert q.is_contiguous()
    assert scale.shape == (2, 1)
    assert scale.dtype == torch.float32
    assert (scale > 0).all()


def test_round_trip_error():
    q_fn = _quantize()
    torch.manual_seed(0)
    w = torch.randn(512, 256, dtype=torch.float16) * 0.02

    q, scale = q_fn(w)
    blocks = q.float().view(512 // 128, 128, 256 // 128, 128)
    deq = (blocks * scale[:, None, :, None]).view(512, 256)
    rel = (deq - w.float()).norm() / w.float().norm()

    # e4m3 keeps 3 mantissa bits, so a 128x128 block scale cannot do better
    # than a few percent on Gaussian data.
    assert rel < 4e-2, f"relative error {rel:.4f} exceeds the 4e-2 budget"


def test_unaligned_shape_is_refused():
    q_fn = _quantize()
    with pytest.raises(ValueError):
        q_fn(torch.randn(300, 128, dtype=torch.float16))
