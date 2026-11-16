"""The MXFP4 hook supplies what the SM70 NVFP4 layout needs from the layer.

Both halves are CPU work: ``_expand`` reads shapes and rewrites a scale grid,
and ``_build_quant_info`` reads the layout off the repacked weights. The GEMM
itself is not reached here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sglang_restomod.quant.sm70_fp4_moe import mxfp4_hook  # noqa: E402


def _info():
    """A stand-in for ``MarlinMoeQuantInfo``: only the two globals matter."""
    return SimpleNamespace(w13_global_scale=None, w2_global_scale=None)


def test_globals_are_supplied_for_the_marlin_layout(monkeypatch):
    layer = SimpleNamespace(
        w13_weight=torch.zeros((1, 8, 4), dtype=torch.int32),
        w2_weight=torch.zeros((1, 4, 8), dtype=torch.int32),
        w13_weight_scale_2=torch.tensor([1.0]),
        w2_weight_scale_2=torch.tensor([2.0]),
    )
    monkeypatch.setattr(mxfp4_hook, "_ORIGINAL_BUILD", lambda _layer: _info())

    info = mxfp4_hook._build_quant_info(layer)

    # ``ops.cu`` refuses E2M1 weights with E4M3 scales when the global is None.
    assert info.w13_global_scale is layer.w13_weight_scale_2
    assert info.w2_global_scale is layer.w2_weight_scale_2


def test_a_prefix_outside_the_marlin_layout_is_left_alone(monkeypatch):
    layer = SimpleNamespace(
        w13_weight=torch.zeros((1, 8, 4), dtype=torch.uint8),  # QPN2
        w2_weight=torch.zeros((1, 4, 8), dtype=torch.int32),  # Marlin
        w13_weight_scale_2=torch.tensor([3.0]),
        w2_weight_scale_2=torch.tensor([4.0]),
    )
    monkeypatch.setattr(mxfp4_hook, "_ORIGINAL_BUILD", lambda _layer: _info())

    info = mxfp4_hook._build_quant_info(layer)

    assert info.w13_global_scale is None
    assert info.w2_global_scale is layer.w2_weight_scale_2


def test_a_scale_grid_that_is_not_one_per_32_is_refused():
    layer = SimpleNamespace(
        w13_weight=torch.zeros((2, 64, 32), dtype=torch.int8),  # K = 64
        w13_weight_scale_inv=torch.full((2, 64, 4), 127, dtype=torch.uint8),
    )

    with pytest.raises(RuntimeError, match="one E8M0 scale per 32"):
        mxfp4_hook._expand(layer, "w13", torch.device("cpu"))


def test_the_checkpoint_grid_is_expanded_in_place():
    layer = SimpleNamespace(
        w13_weight=torch.zeros((2, 64, 32), dtype=torch.int8),  # K = 64
        w13_weight_scale_inv=torch.full((2, 64, 2), 120, dtype=torch.uint8),
    )

    clamped = mxfp4_hook._expand(layer, "w13", torch.device("cpu"))

    assert clamped == 0
    assert tuple(layer.w13_weight_scale.shape) == (2, 64, 4)
    assert layer.w13_weight_scale.dtype == torch.float8_e4m3fn
    assert tuple(layer.w13_weight_scale_2.shape) == (2,)
