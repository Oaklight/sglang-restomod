"""CPU tests for the SM70 WNA16 capability gate."""

from __future__ import annotations

from sglang_restomod.quant.sm70_fp4_moe import wna16_capability


class _Scheme:
    @classmethod
    def get_min_capability(cls) -> int:
        return 80


def _installed():
    cls = type("S", (_Scheme,), {})
    assert wna16_capability.install_on(cls) is True
    return cls


def test_sm70_with_marlin_lowers_the_gate(monkeypatch):
    cls = _installed()
    monkeypatch.setattr(wna16_capability, "_sm70_with_marlin", lambda: True)
    assert cls.get_min_capability() == 70


def test_without_marlin_the_upstream_answer_stands(monkeypatch):
    cls = _installed()
    monkeypatch.setattr(wna16_capability, "_sm70_with_marlin", lambda: False)
    assert cls.get_min_capability() == 80


def test_install_is_idempotent():
    cls = _installed()
    assert wna16_capability.install_on(cls) is False


def test_subclasses_resolve_through_the_wrapper(monkeypatch):
    cls = _installed()
    monkeypatch.setattr(wna16_capability, "_sm70_with_marlin", lambda: False)

    class Sub(cls):
        pass

    assert Sub.get_min_capability() == 80


def test_non_sm70_devices_fall_through():
    """The helper reads the device; anything that is not sm70 stays closed."""
    import pytest

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        assert wna16_capability._sm70_with_marlin() is False
        return
    if torch.cuda.get_device_capability()[0] != 7:
        assert wna16_capability._sm70_with_marlin() is False
