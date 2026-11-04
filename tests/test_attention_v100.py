"""CPU tests for the attention_v100 component."""

from __future__ import annotations

from sglang_restomod.attention.v100 import graph_metadata


class _Backend:
    """Stand-in for FlashAttnV100Backend without importing sglang."""


class _BackendWithMethod:
    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        return "engine"


def test_install_on_adds_the_method():
    cls = type("Backend", (_Backend,), {})
    assert graph_metadata.install_on(cls) is True
    assert graph_metadata.METHOD in vars(cls)
    assert cls.init_forward_metadata_out_graph is graph_metadata._init_forward_metadata_out_graph


def test_install_on_is_idempotent():
    cls = type("Backend", (_Backend,), {})
    assert graph_metadata.install_on(cls) is True
    assert graph_metadata.install_on(cls) is False


def test_install_on_keeps_an_engine_implementation():
    assert graph_metadata.install_on(_BackendWithMethod) is False
    assert _BackendWithMethod().init_forward_metadata_out_graph(None) == "engine"


def test_delegates_decode_metadata_to_triton():
    class Triton:
        def __init__(self):
            self.called = None

        def init_forward_metadata_out_graph(self, forward_batch, in_capture):
            self.called = in_capture

    class Mode:
        def is_decode_or_idle(self):
            return True

        def is_draft_extend_v2(self):
            return False

        def is_target_verify(self):
            return False

    backend = type("Backend", (_Backend,), {})()
    backend._triton = Triton()
    backend.forward_metadata = object()
    backend._uses_sm70_fp8_kv = False
    backend._uses_native_linear_verify = lambda mode: False
    backend._uses_fp8_prefill_scratch = lambda mode: False

    graph_metadata._init_forward_metadata_out_graph(
        backend, type("Batch", (), {"forward_mode": Mode()})(), True
    )
    assert backend._triton.called is True
    assert backend.forward_metadata is None


def test_encoder_only_layer_keeps_its_window():
    from sglang_restomod.attention.v100.masks import resolve_layer_mask

    causal, window = resolve_layer_mask(
        default_causal=True,
        is_cross_attention=False,
        is_encoder_only=True,
        sliding_window_size=2047,
    )
    assert causal is False
    assert window == 2047


def test_causal_layer_keeps_its_window():
    from sglang_restomod.attention.v100.masks import resolve_layer_mask

    assert resolve_layer_mask(
        default_causal=True,
        is_cross_attention=False,
        is_encoder_only=False,
        sliding_window_size=2047,
    ) == (True, 2047)


def test_layer_without_a_window_gets_minus_one():
    from sglang_restomod.attention.v100.masks import resolve_layer_mask

    assert resolve_layer_mask(
        default_causal=True,
        is_cross_attention=False,
        is_encoder_only=True,
        sliding_window_size=None,
    ) == (False, -1)
