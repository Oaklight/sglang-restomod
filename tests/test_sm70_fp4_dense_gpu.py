"""GPU test for the marlin_v100 dense NVFP4 linear GEMM.

`sglang_restomod.quant.sm70_fp4_dense.check` builds a synthetic ModelOpt NVFP4
linear in the checkpoint layout, runs the engine's preparation and
`apply_fp4_marlin_linear` through the patched `marlin_utils_fp4`, and compares
against a dequantized FP16 reference on three shapes: aligned, N-padded and
K-padded. This turns that check into a test so the suite records it; the check
itself stays runnable by hand.

Skips when `SM70_MARLIN_DIR` is unset: the extension is built outside this
repository (see `quant/sm70_fp4_dense/PROVENANCE.md`) and is not shipped with
it. See `tests/test_sm70_fp4_moe_gpu.py` for the symlink layout the loader
expects.
"""

import os

import pytest

pytestmark = pytest.mark.gpu


def _require_sm70():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cap = torch.cuda.get_device_capability()
    if cap != (7, 0):
        pytest.skip(f"needs SM70, found {cap[0]}.{cap[1]}")
    if not os.environ.get("SM70_MARLIN_DIR"):
        pytest.skip("SM70_MARLIN_DIR is unset; the marlin_v100 extension is external")
    return torch


def test_dense_nvfp4_gemm_matches_dequantized_reference():
    _require_sm70()

    from sglang_restomod.quant.sm70_fp4_dense.check import run

    error = run()
    assert error < 5e-2, f"max relative error {error:.3e} against the dequantized reference"


def test_the_dense_path_declines_a_width_and_a_group_it_cannot_pack():
    _require_sm70()

    from sglang_restomod.quant.sm70_fp4_dense.check import declines

    cases = declines()
    assert set(cases) == {"pack-width", "group-size"}
    assert "4-bit" in cases["pack-width"]
    assert "group_size=16" in cases["group-size"]
