"""The SM70 MoE align override: what it rebinds and what it leaves alone."""

from __future__ import annotations

import sys
import types

import pytest

from sglang_restomod.moe import align_sm70


def _fake_module(limit=64):
    module = types.ModuleType(align_sm70._MODULE)
    if limit is not None:
        module.SMALL_NUMEL_LIMIT = limit
    return module


def test_registers_as_a_component():
    from sglang_restomod import plugins

    component = plugins.COMPONENTS["moe_align_sm70"]
    assert component.module == "sglang_restomod.moe.align_sm70"


def test_sets_the_admission_limit_to_zero(monkeypatch):
    module = _fake_module(limit=64)
    monkeypatch.setitem(sys.modules, align_sm70._MODULE, module)

    align_sm70.register()

    # The selection branch is ``numel <= SMALL_NUMEL_LIMIT``, so zero makes the
    # single-CTA Triton kernel unreachable for every non-empty input.
    assert module.SMALL_NUMEL_LIMIT == 0


def test_second_call_is_a_no_op(monkeypatch):
    module = _fake_module(limit=64)
    monkeypatch.setitem(sys.modules, align_sm70._MODULE, module)

    align_sm70.register()
    align_sm70.register()

    assert module.SMALL_NUMEL_LIMIT == 0


def test_missing_limit_is_a_no_op(monkeypatch):
    """A non-CUDA build never defines the name; the override must not raise."""
    module = _fake_module(limit=None)
    monkeypatch.setitem(sys.modules, align_sm70._MODULE, module)

    align_sm70.register()

    assert not hasattr(module, "SMALL_NUMEL_LIMIT")


def test_a_restored_limit_is_overridden_again(monkeypatch):
    """The decision reads the value, so a reload cannot leave the path enabled."""
    module = _fake_module(limit=64)
    monkeypatch.setitem(sys.modules, align_sm70._MODULE, module)

    align_sm70.register()
    module.SMALL_NUMEL_LIMIT = 64  # something restored the upstream value
    align_sm70.register()

    assert module.SMALL_NUMEL_LIMIT == 0
