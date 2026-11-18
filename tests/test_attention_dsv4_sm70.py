"""The DSV4 SM70 attention component: what it rebinds, and its refusals.

No sglang import and no GPU: a fake backend module stands in for the pin, so the
test checks the contract this component depends on -- that it rebinds exactly
two names, that it forces the sparse prefill branch off, and that it refuses
rather than silently doing nothing when patch 0007 is missing.
"""

from __future__ import annotations

import sys
import types

import pytest

from sglang_restomod.attention import dsv4_sm70


@pytest.fixture(autouse=True)
def _reset_hook():
    dsv4_sm70._HOOKED = False
    yield
    dsv4_sm70._HOOKED = False


def test_registers_as_a_component():
    from sglang_restomod import plugins

    component = plugins.COMPONENTS["attention_dsv4_sm70"]
    assert component.module == "sglang_restomod.attention.dsv4_sm70"


def test_forces_dense_prefill(monkeypatch):
    monkeypatch.setenv(dsv4_sm70.SPARSE_PREFILL_ENV, "1")
    dsv4_sm70._force_dense_prefill()
    assert dsv4_sm70.SPARSE_PREFILL_ENV in __import__("os").environ
    assert __import__("os").environ[dsv4_sm70.SPARSE_PREFILL_ENV] == "0"


def test_leaves_dense_prefill_alone_when_already_off(monkeypatch):
    monkeypatch.setenv(dsv4_sm70.SPARSE_PREFILL_ENV, "0")
    dsv4_sm70._force_dense_prefill()
    assert __import__("os").environ[dsv4_sm70.SPARSE_PREFILL_ENV] == "0"


def _fake_backend(with_resolver=True):
    module = types.ModuleType(dsv4_sm70.BACKEND_MODULE)
    calls = []

    if with_resolver:

        def resolve_flash_mla(swa_interleaved):
            calls.append(("resolve", swa_interleaved))
            return ("flash_mla", 0)

        module.resolve_flash_mla = resolve_flash_mla

    class DeepseekV4AttnBackend:
        def _forward_prefill_sparse(self, *a, **k):  # pragma: no cover - replaced
            raise AssertionError("the unpatchd sparse prefill must not run")

    module.DeepseekV4AttnBackend = DeepseekV4AttnBackend
    return module, calls


def test_without_patch_0007_it_is_inactive(monkeypatch):
    from sglang_restomod import attest

    attest.reset()
    module, _ = _fake_backend(with_resolver=False)
    monkeypatch.setitem(sys.modules, dsv4_sm70.BACKEND_MODULE, module)
    monkeypatch.setattr(dsv4_sm70, "_gpu_is_sm70", lambda: True, raising=False)

    dsv4_sm70.register()

    snapshot = attest.snapshot()
    assert "attention_dsv4_sm70" in snapshot["inactive"]
    assert "0007" in snapshot["components"]["attention_dsv4_sm70"]["detail"]
