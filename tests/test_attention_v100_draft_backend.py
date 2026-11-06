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

    def __init__(self, name, topk=1, speculative_num_steps=3):
        self.draft_attn_backend = name
        self.qsa_profile = None
        self.draft_model_runner = object()
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
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


def _installed(name, native=None, calls=None, **factory_args):
    factory = type("F", (_Factory,), {})
    native_calls = calls if calls is not None else []

    def build_native(runner, topk, steps, stamp):
        native_calls.append((runner, topk, steps, stamp))
        return native

    assert (
        draft_backend.install_on(factory, _Backend, _wrap, build_native=build_native)
        is True
    )
    return factory(name, **factory_args), native_calls


def test_other_names_pass_through():
    f, native_calls = _installed("triton")
    assert f.create_draft_extend_backend() == "engine-extend"
    assert f.create_decode_backend() == "engine-decode"
    assert f.calls == [("extend", "triton"), ("decode", "triton")]
    assert native_calls == []


def test_v100_extend_builds_the_component_backend():
    f, _ = _installed(FLASH_ATTN_V100)
    backend = f.create_draft_extend_backend()
    assert isinstance(backend, _Backend)
    assert backend.runner is f.draft_model_runner
    assert backend.wrapped is True
    assert backend.prefill_attention_backend_str == FLASH_ATTN_V100
    assert backend.decode_attention_backend_str == FLASH_ATTN_V100


def test_v100_decode_uses_the_native_container():
    native = object()
    f, native_calls = _installed(FLASH_ATTN_V100, native=native)
    assert f.create_decode_backend() is native
    assert f.calls == []
    assert native_calls == [(f.draft_model_runner, 1, 3, FLASH_ATTN_V100)]
    assert f.draft_attn_backend == FLASH_ATTN_V100


def test_v100_decode_falls_back_to_triton():
    f, native_calls = _installed(FLASH_ATTN_V100, native=None)
    assert f.create_decode_backend() == "engine-decode"
    assert f.calls == [("decode", "triton")]
    assert len(native_calls) == 1
    assert f.draft_attn_backend == FLASH_ATTN_V100


def test_v100_decode_keeps_the_qsa_path():
    native = object()
    f, native_calls = _installed(FLASH_ATTN_V100, native=native)
    f.qsa_profile = object()
    assert f.create_decode_backend() == "engine-decode"
    assert native_calls == []


def test_single_step_decode_has_no_container():
    native = object()
    f, native_calls = _installed(
        FLASH_ATTN_V100, native=native, speculative_num_steps=1
    )
    assert f.create_decode_backend() == "engine-decode"
    assert native_calls == []


def test_install_is_idempotent():
    factory = type("F", (_Factory,), {})
    assert draft_backend.install_on(factory, _Backend, _wrap) is True
    assert draft_backend.install_on(factory, _Backend, _wrap) is False
