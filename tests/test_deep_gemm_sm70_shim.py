"""The SM70 deep_gemm stand-in: when it installs, and how it fails.

Two properties matter more than any kernel: it must never shadow a real
package, and every entry it cannot answer must raise a named error rather than
return something plausible. The tests that need torch use a CPU tensor and skip
when torch is absent; the ones that are about selection and failure modes run
everywhere.
"""

from __future__ import annotations

import importlib
import sys

import pytest

shim = importlib.import_module("sglang_restomod.kernels.deep_gemm_sm70")


class _Fake:
    """Just enough of a tensor for shape/dtype reporting, without torch."""

    def __init__(self, shape, dtype="float8_e4m3fn"):
        self.shape = shape
        self.dtype = dtype


@pytest.fixture
def clean_modules(monkeypatch):
    for k in [m for m in sys.modules if m == "deep_gemm" or m.startswith("deep_gemm.")]:
        monkeypatch.delitem(sys.modules, k, raising=False)
    monkeypatch.setattr(shim, "_real_deep_gemm_installed", lambda: False)
    monkeypatch.setattr(shim, "_device_sm", lambda: 70)
    yield
    for k in [m for m in sys.modules if m == "deep_gemm" or m.startswith("deep_gemm.")]:
        del sys.modules[k]


@pytest.fixture
def installed(clean_modules):
    shim.register()
    return sys.modules["deep_gemm"]


def test_never_shadows_a_real_package(monkeypatch):
    monkeypatch.setattr(shim, "_real_deep_gemm_installed", lambda: True)
    monkeypatch.setattr(shim, "_device_sm", lambda: 70)
    shim.register()
    assert "deep_gemm" not in sys.modules


def test_only_on_sm70(monkeypatch):
    monkeypatch.setattr(shim, "_real_deep_gemm_installed", lambda: False)
    monkeypatch.setattr(shim, "_device_sm", lambda: 80)
    shim.register()
    assert "deep_gemm" not in sys.modules


def test_installed_module_is_marked(installed):
    assert getattr(installed, shim.SUBSTITUTE_FLAG) is True
    assert sys.modules["deep_gemm.utils"] is installed.utils


@pytest.mark.parametrize(
    "name", ["fp8_fp4_mqa_logits", "fp8_mega_moe", "mega_moe_pre_dispatch_sm90"]
)
def test_capability_probes_stay_false(installed, name):
    """These gate path selection upstream; a stub would switch the path on."""
    assert not hasattr(installed, name)


def test_unknown_attribute_raises_attribute_error(installed):
    with pytest.raises(AttributeError):
        installed.not_a_real_entry  # noqa: B018


def test_stub_names_the_entry_and_the_shapes(installed):
    with pytest.raises(NotImplementedError) as err:
        installed.fp8_gemm_nt((_Fake((4, 512)), _Fake((4, 4))), (_Fake((8, 512)), None), _Fake((4, 8)))
    message = str(err.value)
    assert "fp8_gemm_nt" in message
    assert "(4, 512)" in message and "(8, 512)" in message


def test_utils_helper_raises_when_called(installed):
    with pytest.raises(NotImplementedError) as err:
        installed.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(None)
    assert "deep_gemm.utils.layout" in str(err.value)


def test_probing_a_submodule_does_not_raise(installed):
    """The import system probes for dunders; those must be AttributeError.

    A NotImplementedError here escapes ``except ImportError`` guards and breaks
    an unrelated import (torch's chain is one), which is how this was found.
    """
    assert installed.utils.__name__ == "deep_gemm.utils"
    assert not hasattr(installed.utils, "__file__")
    assert not hasattr(installed.utils.math, "no_such_helper")


def test_config_surface(installed):
    installed.set_num_sms(80)
    assert installed.get_num_sms() == 80
    installed.set_compile_mode(1)
    assert installed.get_compile_mode() == 1
    installed.set_pdl(True)  # no-op, must not raise on a CPU-only box
    assert installed.get_mk_alignment_for_contiguous_layout() == 128


def test_bf16_gemm_nt_matches_torch(installed):
    torch = pytest.importorskip("torch")
    lhs = torch.randn(3, 4)
    rhs = torch.randn(5, 4)
    out = torch.empty(3, 5)
    installed.bf16_gemm_nt(lhs, rhs, out)
    assert torch.allclose(out, lhs @ rhs.t(), atol=1e-5)


def test_bf16_gemm_nt_rejects_ragged_k(installed):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="NT product"):
        installed.bf16_gemm_nt(torch.zeros(2, 4), torch.zeros(3, 5), torch.zeros(2, 3))


def test_tf32_hc_prenorm_matches_the_torch_fallback(installed):
    """Same numbers as deepseek_v4.hc_pre_torch_impl, which is what it replaces."""
    torch = pytest.importorskip("torch")
    x = torch.randn(6, 8, dtype=torch.bfloat16)
    fn = torch.randn(3, 8, dtype=torch.float32)
    out = torch.empty(6, 3, dtype=torch.float32)
    sqrsum = torch.empty(6, dtype=torch.float32)
    installed.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_splits=None)
    xf = x.float()
    assert torch.allclose(out, xf @ fn.t(), atol=1e-4)
    assert torch.allclose(sqrsum, xf.square().sum(-1), atol=1e-4)


def test_empty_prenorm_is_a_noop(installed):
    torch = pytest.importorskip("torch")
    x = torch.zeros(0, 8, dtype=torch.bfloat16)
    fn = torch.zeros(3, 8)
    out = torch.full((0, 3), 7.0)
    sqrsum = torch.full((0,), 7.0)
    installed.tf32_hc_prenorm_gemm(x, fn, out, sqrsum)
    assert out.numel() == 0 and sqrsum.numel() == 0
