"""CPU test for the SM70 WNA16 capability gate and its extension probe.

Issue #39: ``CompressedTensorsWNA16.get_min_capability`` reports 80, which
refuses the config on sm70 before any weight is read. The gate answers 70 only
when the marlin_v100 extension can serve a dense WNA16 GEMM, and the upstream
answer otherwise.
"""

from __future__ import annotations

import pytest


def _wna16_class():
    schemes = pytest.importorskip(
        "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16"
    )
    return schemes.CompressedTensorsWNA16


def test_gate_returns_70_only_with_the_extension(monkeypatch):
    from sglang_restomod import overrides

    config_cls = _wna16_class()
    original = config_cls.get_min_capability
    try:
        assert overrides._install_wna16_min_capability() is True

        monkeypatch.setattr(overrides, "_sm70_marlin_v100_available", lambda: True)
        assert config_cls.get_min_capability() == 70

        monkeypatch.setattr(overrides, "_sm70_marlin_v100_available", lambda: False)
        assert config_cls.get_min_capability() == 80
    finally:
        config_cls.get_min_capability = original


def test_install_is_idempotent(monkeypatch):
    from sglang_restomod import overrides

    config_cls = _wna16_class()
    original = config_cls.get_min_capability
    try:
        assert overrides._install_wna16_min_capability() is True
        assert overrides._install_wna16_min_capability() is False
    finally:
        config_cls.get_min_capability = original


def test_probe_needs_the_directory(monkeypatch):
    from sglang_restomod.quant import sm70_marlin_v100

    monkeypatch.delenv(sm70_marlin_v100.ENV_DIR, raising=False)
    monkeypatch.setattr(sm70_marlin_v100, "_OPS", sm70_marlin_v100._UNRESOLVED)

    with pytest.raises(RuntimeError, match=sm70_marlin_v100.ENV_DIR):
        sm70_marlin_v100.directory()
    assert sm70_marlin_v100.available() is False


def test_probe_needs_the_library(monkeypatch, tmp_path):
    from sglang_restomod.quant import sm70_marlin_v100

    monkeypatch.setenv(sm70_marlin_v100.ENV_DIR, str(tmp_path))
    monkeypatch.setattr(sm70_marlin_v100, "_OPS", sm70_marlin_v100._UNRESOLVED)

    assert sm70_marlin_v100.available() is False
    with pytest.raises(RuntimeError, match=sm70_marlin_v100.ENV_DIR):
        sm70_marlin_v100.ops()
