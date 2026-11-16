"""The exact-dense prefill gate, and the Flash-Next shape set behind it.

The 27B table is admitted by default and validated against real weights. The
Flash-Next shapes are opt-in because they were derived from the checkpoint and
the launch, not from an output comparison, so a launch that does not ask for
them must behave exactly as before.

Two admission defects are pinned here because both were silent. The gate used
``getattr(layer, "tp_size", 1) != 4``, and no linear module in the pinned
engine carries that attribute, so it refused every layer on every site while
every other gate reported success. And modules converted by
``fp8_dense_quant`` keep no ``prefix``, so the suffix is empty and a
name-matched gate can never admit them: on Flash-Next that refused 272 of 416
candidates whose shapes the table listed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

#: Marks a layer built the way the engine builds one: no ``tp_size`` attribute.
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
    """Stand in for ``get_parallel()``; ``None`` makes the accessor raise."""
    rc = pytest.importorskip("sglang.srt.runtime_context")
    if tp_size is None:

        def _raise():
            raise RuntimeError("parallel state unavailable")

        monkeypatch.setattr(rc, "get_parallel", _raise)
    else:
        monkeypatch.setattr(rc, "get_parallel", lambda: _Parallel(tp_size))


def test_the_27b_table_is_admitted_by_default(monkeypatch):
    method = _gate()
    monkeypatch.delenv(method.ENV_EXACT_DENSE_SHAPES, raising=False)
    _mock_parallel(monkeypatch, 4)

    # Table entries and layer shapes are both (K, N) at this point.
    assert method._is_sm70_fp8_prefill_exact_dense_layer(_Layer("model.x.o_proj", (1536, 5120)))


def test_a_layer_without_a_tp_size_attribute_can_still_be_admitted(monkeypatch):
    """The regression that made both tables dead on every site.

    ``LinearBase`` has no ``tp_size``; before the fix this assertion failed for
    every layer in every model, and no log line said so.
    """
    method = _gate()
    monkeypatch.delenv(method.ENV_EXACT_DENSE_SHAPES, raising=False)
    _mock_parallel(monkeypatch, 4)

    layer = _Layer("model.x.o_proj", (1536, 5120))
    assert not hasattr(layer, "tp_size")
    assert method._is_sm70_fp8_prefill_exact_dense_layer(layer)


def test_the_engine_accessor_supplies_the_tp_size(monkeypatch):
    method = _gate()
    monkeypatch.delenv(method.ENV_EXACT_DENSE_SHAPES, raising=False)
    _mock_parallel(monkeypatch, 4)

    tp_size, source = method._sm70_fp8_layer_tp_size(_Layer("model.x.o_proj", (1536, 5120)))
    assert (tp_size, source) == (4, "get_parallel().tp_size")


def test_a_present_attribute_wins_over_the_accessor(monkeypatch):
    method = _gate()
    _mock_parallel(monkeypatch, 4)

    tp_size, source = method._sm70_fp8_layer_tp_size(
        _Layer("model.x.o_proj", (1536, 5120), tp_size=2)
    )
    assert (tp_size, source) == (2, "layer.tp_size")


def test_tp_size_must_be_four(monkeypatch):
    method = _gate()
    monkeypatch.setenv(method.ENV_EXACT_DENSE_SHAPES, "flashnext")
    _mock_parallel(monkeypatch, 2)

    assert not method._is_sm70_fp8_prefill_exact_dense_layer(
        _Layer("model.layers.0.self_attn.qkv_proj", (2560, 3584))
    )


def test_an_unavailable_accessor_falls_back_to_the_shape_tables(monkeypatch):
    method = _gate()
    monkeypatch.delenv(method.ENV_EXACT_DENSE_SHAPES, raising=False)
    _mock_parallel(monkeypatch, None)

    tp_size, source = method._sm70_fp8_layer_tp_size(_Layer("model.x.o_proj", (1536, 5120)))
    assert (tp_size, source) == (None, "unavailable")
    assert method._is_sm70_fp8_prefill_exact_dense_layer(_Layer("model.x.o_proj", (1536, 5120)))


def test_flashnext_shapes_are_off_until_asked(monkeypatch):
    method = _gate()
    _mock_parallel(monkeypatch, 4)
    layer = _Layer("model.layers.0.self_attn.qkv_proj", (2560, 3584))

    monkeypatch.delenv(method.ENV_EXACT_DENSE_SHAPES, raising=False)
    assert not method._is_sm70_fp8_prefill_exact_dense_layer(layer)

    monkeypatch.setenv(method.ENV_EXACT_DENSE_SHAPES, "flashnext")
    assert method._is_sm70_fp8_prefill_exact_dense_layer(layer)


@pytest.mark.parametrize(
    "prefix, shape",
    [
        ("model.layers.0.self_attn.o_proj", (1536, 2560)),
        ("model.layers.0.linear_attn.in_proj_qkvz", (2560, 4096)),
        ("model.layers.0.linear_attn.out_proj", (1536, 2560)),
    ],
)
def test_each_flashnext_shape_is_admitted(prefix, shape, monkeypatch):
    method = _gate()
    monkeypatch.setenv(method.ENV_EXACT_DENSE_SHAPES, "flashnext")
    _mock_parallel(monkeypatch, 4)

    assert method._is_sm70_fp8_prefill_exact_dense_layer(_Layer(prefix, shape))


def test_a_nameless_module_is_admitted_by_shape_under_the_flashnext_table(monkeypatch):
    """``fp8_dense_quant`` leaves 272 of 416 Flash-Next candidates unnamed."""
    method = _gate()
    _mock_parallel(monkeypatch, 4)
    layer = _Layer(None, (1536, 2560))

    monkeypatch.delenv(method.ENV_EXACT_DENSE_SHAPES, raising=False)
    assert not method._is_sm70_fp8_prefill_exact_dense_layer(layer)

    monkeypatch.setenv(method.ENV_EXACT_DENSE_SHAPES, "flashnext")
    assert method._is_sm70_fp8_prefill_exact_dense_layer(layer)


def test_the_shape_fallback_does_not_widen_the_default_table(monkeypatch):
    """A nameless layer with a 27B shape stays refused unless asked for."""
    method = _gate()
    monkeypatch.delenv(method.ENV_EXACT_DENSE_SHAPES, raising=False)
    _mock_parallel(monkeypatch, 4)

    assert not method._is_sm70_fp8_prefill_exact_dense_layer(_Layer(None, (1536, 5120)))


def test_a_wrong_shape_is_refused(monkeypatch):
    method = _gate()
    monkeypatch.setenv(method.ENV_EXACT_DENSE_SHAPES, "flashnext")
    _mock_parallel(monkeypatch, 4)

    # Right suffix, wrong N: the gate is the shape, not the name.
    assert not method._is_sm70_fp8_prefill_exact_dense_layer(
        _Layer("model.layers.0.self_attn.qkv_proj", (2560, 3072))
    )
    # 3328 was the first derivation; the checkpoint's fused tensor is 3584.
    assert not method._is_sm70_fp8_prefill_exact_dense_layer(
        _Layer("model.layers.0.self_attn.qkv_proj", (2560, 3328))
    )
    # A shape outside both tables stays on the TurboMind GEMM.
    assert not method._is_sm70_fp8_prefill_exact_dense_layer(_Layer(None, (4608, 1152)))


def test_an_unknown_shape_set_raises_instead_of_meaning_off(monkeypatch):
    """A misspelt flag must not report the baseline as if the table worked."""
    method = _gate()
    _mock_parallel(monkeypatch, 4)
    layer = _Layer("model.layers.0.self_attn.qkv_proj", (2560, 3584))

    monkeypatch.setenv(method.ENV_EXACT_DENSE_SHAPES, "flash-next")
    with pytest.raises(ValueError, match="empty or 'flashnext'"):
        method._is_sm70_fp8_prefill_exact_dense_layer(layer)

    # Case folding still works; only a genuinely unknown value is refused.
    monkeypatch.setenv(method.ENV_EXACT_DENSE_SHAPES, "FLASHNEXT")
    assert method._is_sm70_fp8_prefill_exact_dense_layer(layer)


def test_the_workspace_is_sized_over_every_table():
    method = _gate()

    needed = max(
        k * n
        for table in (
            method._SM70_FP8_PREFILL_DENSE_SHAPES,
            method._SM70_FP8_PREFILL_DENSE_SHAPES_FLASHNEXT,
        )
        for k, n in table.values()
    )
    assert method._SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS >= needed
