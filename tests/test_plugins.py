from __future__ import annotations

import sys
import types

import pytest

from sglang_restomod import plugins


def test_empty_selection():
    assert plugins.selection({}) == []
    assert plugins.selection({plugins.ENV_COMPONENTS: ""}) == []
    assert plugins.selection({plugins.ENV_COMPONENTS: " , "}) == []


def test_parsing_and_dedup():
    env = {plugins.ENV_COMPONENTS: " fp8_moe, attention_v100 , fp8_moe "}
    assert plugins.selection(env) == ["fp8_moe", "attention_v100"]


def test_unknown_component_lists_known_names():
    with pytest.raises(ValueError, match="unknown component"):
        plugins.selection({plugins.ENV_COMPONENTS: "nope"})


def test_load_enabled_registers_once(monkeypatch):
    calls: list[str] = []
    module = types.ModuleType("restomod_test_component")
    module.register = lambda: calls.append("ok")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "restomod_test_component", module)
    monkeypatch.setitem(
        plugins.COMPONENTS,
        "test",
        plugins.Component("test", "restomod_test_component", "test component"),
    )

    env = {plugins.ENV_COMPONENTS: "test"}
    assert plugins.is_enabled("test", env)
    assert plugins.load_enabled(env) == ["test"]
    assert calls == ["ok"]


def test_component_without_register_is_recorded_not_raised(monkeypatch):
    """A broken component definition must not take the whole plugin down.

    This used to raise ``AttributeError``, which reads as fail-fast and is not:
    ``sglang.srt.plugins`` wraps the plugin call in ``except Exception``, so the
    raise dropped *every* component and the server started normally on the
    pinned kernels. The failure is now recorded with the same message and the
    neighbouring components still load.
    """
    from sglang_restomod import attest

    attest.reset()
    module = types.ModuleType("restomod_broken_component")
    monkeypatch.setitem(sys.modules, "restomod_broken_component", module)
    monkeypatch.setitem(
        plugins.COMPONENTS,
        "broken",
        plugins.Component("broken", "restomod_broken_component", "no register"),
    )
    assert plugins.load_enabled({plugins.ENV_COMPONENTS: "broken"}) == []
    detail = attest.snapshot()["components"]["broken"]["detail"]
    assert "has no register" in detail
    attest.reset()
