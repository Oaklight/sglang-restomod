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


def _skip_when_the_layout_is_qpn2():
    """Refuse to run a Marlin-reader check against a QPN2 bank.

    ``prepare_moe_nvfp4`` reads ``SGLANG_SM70_FP4_MOE_LAYOUT`` at call time, so
    an ambient ``qpn2`` converts this check's checkpoint-layout bank before the
    Marlin reader it exercises sees it, and the reader rejects the bank width
    (``b_q_weight.size(1) = 36864, size_k = 256``). The QPN2 path is covered by
    its own tests: ``test_sm70_fp4_moe_qpn2.py`` (CPU) and
    ``test_sm70_qpn2_gemv_gpu.py`` (decode kernel).
    """
    from sglang_restomod.quant.sm70_fp4_moe import qpn2

    if qpn2.layout() == "qpn2":
        prefixes = ",".join(sorted(qpn2.selected_prefixes())) or "none"
        pytest.skip(
            f"{qpn2.ENV_LAYOUT}=qpn2 (tensors={prefixes}) converts this check's "
            "bank to the QPN2 expert layout before the Marlin reader runs; the "
            "QPN2 path is covered by tests/test_sm70_fp4_moe_qpn2.py and "
            "tests/test_sm70_qpn2_gemv_gpu.py"
        )


def test_nvfp4_expert_gemm_matches_dequantized_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cap = torch.cuda.get_device_capability()
    if cap != (7, 0):
        pytest.skip(f"needs SM70, found {cap[0]}.{cap[1]}")
    if not os.environ.get("SM70_MARLIN_DIR"):
        pytest.skip("SM70_MARLIN_DIR is unset; the marlin_v100 extension is external")
    _skip_when_the_layout_is_qpn2()

    from sglang_restomod.quant.sm70_fp4_moe.check import run

    error = run()
    assert error < 5e-2, f"max relative error {error:.3e} against the dequantized reference"
