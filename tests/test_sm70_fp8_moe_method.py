"""Numeric test for the SM70 block-FP8 MoE method (GPU)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)


def test_apply_matches_reference_dense():
    from sglang_restomod.quant.sm70_fp8_moe.check import run

    assert run(False) < 5e-2


def test_apply_matches_reference_expert_parallel():
    from sglang_restomod.quant.sm70_fp8_moe.check import run

    assert run(True) < 5e-2


def test_apply_handles_empty_routing():
    from sglang_restomod.quant.sm70_fp8_moe.check import run_empty

    assert run_empty() == 0
