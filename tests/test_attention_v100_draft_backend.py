"""CPU tests for the draft backend hook."""

from __future__ import annotations

from sglang_restomod.attention.v100 import draft_backend
from sglang_restomod.attention.v100.backend_registry import FLASH_ATTN_V100


class _Backend:
    def __init__(self, runner):
        self.runner = runner
        self.prefill_attention_backend_str = None
        self.decode_attention_backend_str = None


class _Factory:
    """Stand-in for ``DraftBackendFactory``, with the engine's two entry points."""

    def __init__(self, name):
        self.draft_attn_backend = name
        self.qsa_profile = None
        self.draft_model_runner = object()
        self.calls = []

    def create_draft_extend_backend(self):
        self.calls.append(("extend", self.draft_attn_backend))
        return "engine-extend"

    def create_decode_backend(self):
        self.calls.append(("decode", self.draft_attn_backend))
        return "engine-decode"


def _wrap(runner, backend):
    backend.wrapped = True
    return backend


def _installed(name):
    factory = type("F", (_Factory,), {})
    assert draft_backend.install_on(factory, _Backend, _wrap) is True
    return factory(name)


def test_other_names_pass_through():
    f = _installed("triton")
    assert f.create_draft_extend_backend() == "engine-extend"
    assert f.create_decode_backend() == "engine-decode"
    assert f.calls == [("extend", "triton"), ("decode", "triton")]


def test_v100_extend_builds_the_component_backend():
    f = _installed(FLASH_ATTN_V100)
    backend = f.create_draft_extend_backend()
    assert isinstance(backend, _Backend)
    assert backend.runner is f.draft_model_runner
    assert backend.wrapped is True
    assert backend.prefill_attention_backend_str == FLASH_ATTN_V100
    assert backend.decode_attention_backend_str == FLASH_ATTN_V100


def test_v100_decode_falls_back_to_triton():
    f = _installed(FLASH_ATTN_V100)
    assert f.create_decode_backend() == "engine-decode"
    assert f.calls == [("decode", "triton")]
    assert f.draft_attn_backend == FLASH_ATTN_V100


def test_v100_extend_keeps_the_qsa_profile_path():
    f = _installed(FLASH_ATTN_V100)
    f.qsa_profile = object()
    assert f.create_draft_extend_backend() == "engine-extend"


def test_install_is_idempotent():
    factory = type("F", (_Factory,), {})
    assert draft_backend.install_on(factory, _Backend, _wrap) is True
    assert draft_backend.install_on(factory, _Backend, _wrap) is False
