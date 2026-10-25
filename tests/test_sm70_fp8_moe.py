"""GPU test for the SM70 FP8 MoE binding.

Deselected on CPU CI (`-m "not gpu"`); run on a V100 with
`scripts/test_gpu.sh` or `python -m pytest -m gpu`.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu


def test_binding_matches_dequantized_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")

    from sglang_restomod.quant.sm70_fp8_moe import selftest

    try:
        rel = selftest.run()
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        pytest.skip(f"SM70 environment unavailable: {exc}")

    assert rel < 5e-3, f"relative error {rel:.5f} exceeds the 5e-3 budget"
