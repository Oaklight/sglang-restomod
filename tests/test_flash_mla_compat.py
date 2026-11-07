"""CPU tests for the guarded ``flash_mla`` stub.

The stub exists because v0.5.23 imports the external flash_mla wheel on the
decode-graph path even when the launch runs a backend that never calls it. What
matters is its surface, not its dispatch: the two names the pin binds at import
must be importable, every other name and every call must fail with a message
that says what to install, and a working wheel must never be shadowed.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

pytest.importorskip("torch")


@pytest.fixture
def compat():
    """A clean compat module: no stub, no flash_mla, nothing left over."""
    from sglang_restomod.attention.v100 import flash_mla_compat

    saved = {
        name: sys.modules.pop(name)
        for name in ("flash_mla", "flash_mla.flash_mla_interface")
        if name in sys.modules
    }
    flash_mla_compat._INSTALLED = False
    try:
        yield flash_mla_compat
    finally:
        for name in ("flash_mla", "flash_mla.flash_mla_interface"):
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        flash_mla_compat._INSTALLED = False


def test_stub_serves_the_names_the_pin_binds_at_import(compat):
    assert compat.install_stub()

    # Exactly what sglang/kernels/ops/attention/flash_mla_interface.py does.
    import flash_mla
    from flash_mla import fused_norm_rope_attn_rope_cast
    from flash_mla.flash_mla_interface import FlashMLASchedMeta

    assert callable(fused_norm_rope_attn_rope_cast)
    assert isinstance(FlashMLASchedMeta, type)
    assert flash_mla.flash_mla_interface.FlashMLASchedMeta is FlashMLASchedMeta
    assert compat.is_stub_installed()
    assert compat.flash_mla_status() == "stub"


def test_every_other_name_raises_with_the_guidance(compat):
    compat.install_stub()
    import flash_mla
    from flash_mla import flash_mla_interface

    for module, name in (
        (flash_mla, "flash_mla_with_kvcache"),
        (flash_mla, "get_mla_metadata"),
        (flash_mla_interface, "flash_mla_sparse_fwd"),
    ):
        with pytest.raises(ImportError, match=r"not implemented by the restomod stub"):
            getattr(module, name)

    with pytest.raises(ImportError, match=r"sgl-flash-mla"):
        flash_mla.fused_norm_rope_attn_rope_cast()
    with pytest.raises(ImportError, match=r"sgl-flash-mla"):
        flash_mla.flash_mla_interface.FlashMLASchedMeta()


def test_a_working_wheel_is_not_shadowed(compat):
    wheel = types.ModuleType("flash_mla")
    sys.modules["flash_mla"] = wheel

    assert compat.ensure_flash_mla_stub() is False
    assert not compat.is_stub_installed()
    assert sys.modules["flash_mla"] is wheel


def test_import_failure_installs_the_stub_and_logs_once(compat, monkeypatch, caplog):
    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == "flash_mla":
                raise ImportError("simulated: no flash_mla on this host")
            return None

    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])
    caplog.set_level(logging.INFO, logger="sglang_restomod.overrides")

    assert compat.ensure_flash_mla_stub() is True
    assert compat.is_stub_installed()
    lines = [r for r in caplog.records if "guarded stub" in r.getMessage()]
    assert len(lines) == 1
    assert "simulated: no flash_mla" in lines[0].getMessage()

    # Second call is a no-op: no second line, still installed.
    assert compat.ensure_flash_mla_stub() is True
    assert len([r for r in caplog.records if "guarded stub" in r.getMessage()]) == 1


def test_plugin_registration_installs_it(compat, monkeypatch):
    """register() must run the stub before anything imports an attention backend."""
    from sglang_restomod.attention import v100

    called = []
    monkeypatch.setattr(compat, "ensure_flash_mla_stub", lambda: called.append("stub") or True)
    for name in (
        "backend_registry",
        "draft_backend",
        "graph_metadata",
    ):
        module = types.ModuleType(f"sglang_restomod.attention.v100.{name}")
        for attr in ("install", "install_draft_choices", "install_qsa_swap"):
            setattr(
                module,
                attr,
                lambda *a, _attr=attr, **k: called.append(_attr),
            )
        monkeypatch.setitem(sys.modules, module.__name__, module)

    v100.register()
    assert called[0] == "stub"
