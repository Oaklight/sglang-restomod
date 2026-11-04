"""Admission rules for the post-load block-FP8 dense quantization.

The conversion itself is the same ``_quantize`` the LM head component uses and
is exercised on the GPU host; what is testable without a card is which layers
are allowed through, because a wrong admission either quantizes an expert
(the NVFP4 MoE path owns those) or a layer whose forward never reads
``quant_method``.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from sglang_restomod.quant import sm70_fp8_dense_quant as comp  # noqa: E402


class _UnquantizedLinearMethod:
    pass


class _Fp8LinearMethod:
    pass


def _layer(shape=(256, 256), dtype=torch.bfloat16, method=None):
    layer = types.SimpleNamespace()
    layer.weight = torch.nn.Parameter(
        torch.zeros(*shape, dtype=dtype), requires_grad=False
    )
    layer.quant_method = method or _UnquantizedLinearMethod()
    layer.output_size_per_partition = shape[0]
    return layer


def test_dense_unquantized_layer_is_admitted():
    assert comp._admitted("model.layers.0.self_attn.q_proj", _layer())


def test_expert_layers_are_refused():
    layer = _layer((128, 128))
    assert not comp._admitted("model.layers.0.mlp.experts.3.up_proj", layer)


def test_lm_head_is_refused():
    assert not comp._admitted("lm_head", _layer((128, 128)))


def test_unaligned_widths_are_refused():
    # 320 is the Flash-Next hyper-connection low-rank width: not a 128 multiple.
    assert not comp._admitted("model.layers.0.hc.down", _layer((320, 2560)))


def test_already_quantized_layer_is_refused():
    layer = _layer()
    layer.weight_block_size = [comp.BLOCK, comp.BLOCK]
    assert not comp._admitted("model.layers.0.self_attn.q_proj", layer)


def test_a_quantized_method_is_refused():
    layer = _layer(method=_Fp8LinearMethod())
    assert not comp._admitted("model.layers.0.self_attn.q_proj", layer)


def test_a_plain_linear_without_quant_method_is_refused():
    layer = types.SimpleNamespace(weight=torch.zeros(256, 256, dtype=torch.bfloat16))
    assert not comp._admitted("model.layers.0.mlp.dense", layer)


def test_a_real_nn_linear_is_admitted():
    # The model's non-LinearBase projections are plain nn.Linear; convert()
    # wraps their forward so quant_method.apply is reached.
    assert comp._admitted("model.layers.0.mlp.dense", torch.nn.Linear(256, 256).to(torch.bfloat16))


def test_a_plain_linear_is_not_admitted_when_unaligned():
    assert not comp._admitted("model.layers.0.mlp.dense", torch.nn.Linear(320, 256).to(torch.bfloat16))


def test_a_non_2d_weight_is_refused():
    layer = _layer((256,))
    assert not comp._admitted("model.layers.0.norm", layer)


def test_fp32_weight_is_refused():
    layer = _layer(dtype=torch.float32)
    assert not comp._admitted("model.layers.0.self_attn.q_proj", layer)
