"""The fused HC prenorm component: registration, guards, and split choice.

Nothing here runs the kernel -- no Triton and no CUDA. The point is that the
component is wired into the shim's ``fused`` arm, that an input it cannot
represent raises instead of computing a different reduction, and that the
split count always divides K.
"""

from __future__ import annotations

import sys
import types

import pytest

from sglang_restomod.kernels import hc_prenorm_sm70

torch = pytest.importorskip("torch")


def test_registers_as_a_component():
    from sglang_restomod import plugins

    component = plugins.COMPONENTS["hc_prenorm_sm70"]
    assert component.module == "sglang_restomod.kernels.hc_prenorm_sm70"


def test_register_installs_the_op(monkeypatch):
    shim = types.ModuleType(hc_prenorm_sm70.SHIM_MODULE)
    installed = []
    shim.install_fused_hc_prenorm = installed.append
    monkeypatch.setitem(sys.modules, hc_prenorm_sm70.SHIM_MODULE, shim)

    hc_prenorm_sm70.register()

    assert installed == [hc_prenorm_sm70.hc_prenorm]


def test_register_without_the_installer_is_inactive(monkeypatch):
    from sglang_restomod import attest

    attest.reset()
    shim = types.ModuleType(hc_prenorm_sm70.SHIM_MODULE)
    monkeypatch.setitem(sys.modules, hc_prenorm_sm70.SHIM_MODULE, shim)

    hc_prenorm_sm70.register()

    snapshot = attest.snapshot()
    assert "hc_prenorm_sm70" in snapshot["inactive"]
    detail = snapshot["components"]["hc_prenorm_sm70"]["detail"]
    assert "install_fused_hc_prenorm" in detail


def test_split_always_divides_k():
    # 20480 = 2**12 * 5: 8 divides it, and so does every power of two below it.
    assert hc_prenorm_sm70._split_count(20480, 8) == 8
    assert hc_prenorm_sm70._split_count(20480, 6) == 5
    assert hc_prenorm_sm70._split_count(192, 8) == 8
    assert hc_prenorm_sm70._split_count(129, 8) == 3
    assert hc_prenorm_sm70._split_count(7, 8) == 7


def test_split_rejects_zero():
    with pytest.raises(ValueError, match="num_splits"):
        hc_prenorm_sm70._split_count(20480, 0)


def _args(t=4, mix=24, hc=256, out_shape=None, ss_shape=None):
    x = torch.empty((t, hc), dtype=torch.bfloat16)
    fn = torch.empty((mix, hc), dtype=torch.float32)
    out = torch.empty(out_shape or (t, mix), dtype=torch.float32)
    ss = torch.empty(ss_shape or (t,), dtype=torch.float32)
    return x, fn, out, ss


def test_rejects_missing_outputs():
    x, fn, _, ss = _args()
    with pytest.raises(ValueError, match="preallocated"):
        hc_prenorm_sm70.hc_prenorm(x, fn, None, ss)


def test_rejects_3d_input():
    x, fn, out, ss = _args()
    with pytest.raises(ValueError, match="2-D"):
        hc_prenorm_sm70.hc_prenorm(x.view(2, 2, 256), fn, out, ss)


def test_rejects_mismatched_width():
    x, _, out, ss = _args()
    fn = torch.empty((24, 128), dtype=torch.float32)
    with pytest.raises(ValueError, match="fn must be"):
        hc_prenorm_sm70.hc_prenorm(x, fn, out, ss)


def test_rejects_wrong_out_shape():
    x, fn, out, ss = _args(out_shape=(4, 32))
    with pytest.raises(ValueError, match="out must be"):
        hc_prenorm_sm70.hc_prenorm(x, fn, out, ss)


def test_rejects_wrong_sqrsum_shape():
    x, fn, out, ss = _args(ss_shape=(4, 1))
    with pytest.raises(ValueError, match="sqrsum must be"):
        hc_prenorm_sm70.hc_prenorm(x, fn, out, ss)


def test_rejects_non_fp32_output():
    x, fn, _, ss = _args()
    out = torch.empty((4, 24), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="float32"):
        hc_prenorm_sm70.hc_prenorm(x, fn, out, ss)


def test_rejects_unaligned_k():
    x, fn, out, ss = _args(hc=200)
    with pytest.raises(ValueError, match="multiple of 128"):
        hc_prenorm_sm70.hc_prenorm(x, fn, out, ss)


def test_rejects_cpu_tensors():
    x, fn, out, ss = _args()
    with pytest.raises(ValueError, match="CUDA"):
        hc_prenorm_sm70.hc_prenorm(x, fn, out, ss)
