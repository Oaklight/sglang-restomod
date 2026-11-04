"""CPU test for the QPN8 source guard.

The QPN8 decoder is JIT-built from a `.cu` this distribution does not ship, and
`_QPN8_SRC_PATH` resolves through `sglang.kernels.sm70_paths`, which the pinned
install has no copy of. Before the guard, the first layer whose shape matched
raised `AttributeError: 'NoneType' object has no attribute 'is_file'` inside the
scheduler at startup. The prep must now log and return instead.
"""

from __future__ import annotations

import pytest


def _prep_module():
    pytest.importorskip("torch")
    return pytest.importorskip("sglang_restomod.quant.sm70_fp8_dense._method")


class _Layer:
    def __init__(self, shape, prefix):
        torch = pytest.importorskip("torch")
        self.tp_size = 4
        self.weight = torch.empty(shape)
        self.prefix = prefix


def test_missing_source_is_a_no_op(monkeypatch):
    module = _prep_module()
    monkeypatch.setattr(module, "_QPN8_SRC_PATH", None)
    monkeypatch.setattr(module, "_QPN8_MISSING_LOGGED", False)
    layer = _Layer((8704, 5120), "m.layers.0.mlp.gate_up_proj")
    module._maybe_prepare_sm70_fp8_qpn8(layer)
    assert not getattr(layer, "sm70_fp8_qpn8_ready", False)


def test_nonexistent_source_path_is_a_no_op(monkeypatch):
    module = _prep_module()
    monkeypatch.setattr(module, "_QPN8_SRC_PATH", "/nonexistent/qpn8.cu")
    monkeypatch.setattr(module, "_QPN8_MISSING_LOGGED", False)
    layer = _Layer((8704, 5120), "m.layers.0.mlp.gate_up_proj")
    module._maybe_prepare_sm70_fp8_qpn8(layer)
    assert not getattr(layer, "sm70_fp8_qpn8_ready", False)
