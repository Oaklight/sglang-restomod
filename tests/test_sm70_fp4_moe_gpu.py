"""GPU test for the marlin_v100 NVFP4 expert GEMM.

`sglang_restomod.quant.sm70_fp4_moe.check` builds a small NVFP4 expert bank,
runs the SM70 conversion and `moe_wna16_marlin_gemm`, and compares against a
dequantized FP16 reference for both `w13` and `w2`. This turns that check into
a test so the suite records it; the check itself stays runnable by hand.

Skips when `SM70_MARLIN_DIR` is unset: the extension is built outside this
repository (see `quant/sm70_fp4_moe/PROVENANCE.md`) and is not shipped with it.

The loader looks for `_C*.so` and `_moe_C*.so` by name, while SGLang's prebuilt
copies are `_sm70_marlin_v100_dense.abi3.so` and `_sm70_marlin_v100_moe.abi3.so`,
so point the variable at a directory of symlinks:

    mkdir -p ~/marlin-ops && cd ~/marlin-ops
    ln -sf .../prebuilt/_sm70_marlin_v100_dense.abi3.so _C.so
    ln -sf .../prebuilt/_sm70_marlin_v100_moe.abi3.so _moe_C.so
"""

import os

import pytest

pytestmark = pytest.mark.gpu


def test_nvfp4_expert_gemm_matches_dequantized_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cap = torch.cuda.get_device_capability()
    if cap != (7, 0):
        pytest.skip(f"needs SM70, found {cap[0]}.{cap[1]}")
    if not os.environ.get("SM70_MARLIN_DIR"):
        pytest.skip("SM70_MARLIN_DIR is unset; the marlin_v100 extension is external")

    from sglang_restomod.quant.sm70_fp4_moe.check import run

    error = run()
    assert error < 5e-2, f"max relative error {error:.3e} against the dequantized reference"
