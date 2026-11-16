"""The QPN8 decode gate, and the Flash-Next shape set behind it.

QPN8 never ran anywhere, and it had two independent reasons, both silent:

* the gate opened with ``getattr(layer, "tp_size", 1) != 4``, and no linear
  module in the pinned engine carries that attribute, so every layer was
  refused before a shape was compared;
* its table is the 27B's, and Flash-Next's projections are not in it.

The table convention differs from the prefill table in this module on purpose.
This gate is read before ``fp8_prepare`` replaces the tensor, so the weight is
still the checkpoint layout, ``(N, K)``; the prefill gate is read after the
repack, where the tensor is logical KxN. A test pins that, because a mistaken
transpose here produces a kernel that runs and returns wrong numbers rather
than one that refuses.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

_ABSENT = object()


def _gate():
    pytest.importorskip("sglang.srt.layers.quantization.fp8")
    from sglang_restomod.quant.sm70_fp8_dense import _method

    return _method


class _Layer:
    def __init__(self, prefix: str | None, shape: tuple[int, int], tp_size=_ABSENT):
        if prefix is not None:
            self.prefix = prefix
        if tp_size is not _ABSENT:
            self.tp_size = tp_size
        self.weight = torch.empty(shape, device="meta")


class _Parallel:
    def __init__(self, tp_size: int):
        self.tp_size = tp_size


def _mock_parallel(monkeypatch, tp_size: int | None):
    rc = pytest.importorskip("sglang.srt.runtime_context")
    if tp_size is None:

        def _raise():
            raise RuntimeError("parallel state unavailable")

        monkeypatch.setattr(rc, "get_parallel", _raise)
    else:
        monkeypatch.setattr(rc, "get_parallel", lambda: _Parallel(tp_size))


def test_qpn8_is_off_by_default(monkeypatch):
    """The old default was "1", and it never enabled anything because the gate
    was dead. Fixing the gate makes the default load-bearing, so it is off."""
    method = _gate()
    monkeypatch.delenv("SGLANG_SM70_FP8_DECODE_QPN8", raising=False)
    assert method._get_sm70_fp8_decode_qpn8_enabled() is False

    monkeypatch.setenv("SGLANG_SM70_FP8_DECODE_QPN8", "1")
    assert method._get_sm70_fp8_decode_qpn8_enabled() is True


def test_the_27b_layer_is_admitted_without_a_tp_size_attribute(monkeypatch):
    """The regression: this failed for every layer on every site."""
    method = _gate()
    monkeypatch.delenv(method.ENV_QPN8_SHAPES, raising=False)
    _mock_parallel(monkeypatch, 4)

    layer = _Layer("model.layers.0.mlp.down_proj", (5120, 4352))
    assert not hasattr(layer, "tp_size")
    assert method._is_sm70_fp8_qpn8_exact_dense_layer(layer)


def test_tp_size_must_be_four(monkeypatch):
    method = _gate()
    _mock_parallel(monkeypatch, 2)

    assert not method._is_sm70_fp8_qpn8_exact_dense_layer(
        _Layer("model.layers.0.mlp.down_proj", (5120, 4352))
    )


def test_flashnext_shapes_are_off_until_asked(monkeypatch):
    method = _gate()
    _mock_parallel(monkeypatch, 4)
    layer = _Layer("model.layers.0.linear_attn.in_proj_qkvz", (4096, 2560))

    monkeypatch.delenv(method.ENV_QPN8_SHAPES, raising=False)
    assert not method._is_sm70_fp8_qpn8_exact_dense_layer(layer)

    monkeypatch.setenv(method.ENV_QPN8_SHAPES, "flashnext")
    assert method._is_sm70_fp8_qpn8_exact_dense_layer(layer)


def test_the_table_is_n_by_k_not_k_by_n(monkeypatch):
    """The checkpoint layout at this point is (N, K); the transpose is refused."""
    method = _gate()
    monkeypatch.setenv(method.ENV_QPN8_SHAPES, "flashnext")
    _mock_parallel(monkeypatch, 4)

    assert method._is_sm70_fp8_qpn8_exact_dense_layer(
        _Layer("model.layers.0.linear_attn.in_proj_qkvz", (4096, 2560))
    )
    assert not method._is_sm70_fp8_qpn8_exact_dense_layer(
        _Layer("model.layers.0.linear_attn.in_proj_qkvz", (2560, 4096))
    )


def test_a_nameless_module_is_admitted_by_shape_under_the_env(monkeypatch):
    """68 of the 104 Flash-Next candidates per rank carry no prefix."""
    method = _gate()
    _mock_parallel(monkeypatch, 4)
    layer = _Layer(None, (2560, 1536))

    monkeypatch.delenv(method.ENV_QPN8_SHAPES, raising=False)
    assert not method._is_sm70_fp8_qpn8_exact_dense_layer(layer)

    monkeypatch.setenv(method.ENV_QPN8_SHAPES, "flashnext")
    assert method._is_sm70_fp8_qpn8_exact_dense_layer(layer)


def test_the_shape_fallback_does_not_widen_the_default_table(monkeypatch):
    method = _gate()
    monkeypatch.delenv(method.ENV_QPN8_SHAPES, raising=False)
    _mock_parallel(monkeypatch, 4)

    # A 27B shape with no name stays refused by the default table.
    assert not method._is_sm70_fp8_qpn8_exact_dense_layer(_Layer(None, (5120, 4352)))
    # And a Flash-Next shape stays refused with the env unset, above.


def test_shapes_outside_both_tables_are_refused(monkeypatch):
    method = _gate()
    monkeypatch.setenv(method.ENV_QPN8_SHAPES, "flashnext")
    _mock_parallel(monkeypatch, 4)

    # The hyperconnection shapes never enter QPN8: they are not 128-aligned.
    assert not method._is_sm70_fp8_qpn8_exact_dense_layer(_Layer(None, (10240, 320)))
    assert not method._is_sm70_fp8_qpn8_exact_dense_layer(_Layer(None, (320, 10240)))


def test_an_unknown_shape_set_raises(monkeypatch):
    method = _gate()
    _mock_parallel(monkeypatch, 4)

    monkeypatch.setenv(method.ENV_QPN8_SHAPES, "flash-next")
    with pytest.raises(ValueError, match="empty or 'flashnext'"):
        method._is_sm70_fp8_qpn8_exact_dense_layer(_Layer(None, (2560, 1536)))

    monkeypatch.setenv(method.ENV_QPN8_SHAPES, "FLASHNEXT")
    assert method._is_sm70_fp8_qpn8_exact_dense_layer(_Layer(None, (2560, 1536)))
