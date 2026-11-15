"""CPU tests for the draft NVFP4 conversion (no GPU, no Marlin build).

The quantizer's convention is not asserted against itself: the round-trip test
dequantizes with the engine's own ``dequantize_nvfp4``, which is the reader the
checkpoint layout is defined by. A layout error therefore fails here instead of
producing a draft whose experts are subtly wrong.

The layer tests never build a real ``FusedMoE``: they exercise the gate, the
selection predicate, and rollback, which is where a refusal can leave a layer
half-converted.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _quantizer():
    from sglang_restomod.quant.sm70_mtp_fp4_moe import quantize

    return quantize


def _engine_dequantizer():
    module = pytest.importorskip("sglang.srt.layers.quantization.dequantization")
    return module.dequantize_nvfp4


def test_layout_matches_the_checkpoint_shapes():
    quantize = _quantizer()
    bank = torch.randn(3, 8, 64, dtype=torch.float16)

    packed = quantize.quantize_experts(bank)

    assert packed.weight.shape == (3, 8, 32)
    assert packed.weight.dtype == torch.uint8
    assert packed.scales.shape == (3, 8, 4)
    assert packed.scales.dtype == torch.float8_e4m3fn
    assert packed.global_scale.shape == (3,)
    assert packed.global_scale.dtype == torch.float32


def test_round_trip_through_the_engine_dequantizer():
    quantize = _quantizer()
    dequantize_nvfp4 = _engine_dequantizer()
    torch.manual_seed(0)
    bank = torch.randn(4, 16, 128, dtype=torch.float32)

    packed = quantize.quantize_experts(bank)
    # The engine's reader broadcasts w_s2 over the scale tensor, so a
    # per-expert global scale has to arrive as [E, 1, 1].
    restored = dequantize_nvfp4(
        packed.weight,
        packed.scales,
        packed.global_scale.view(-1, 1, 1),
        out_dtype=torch.float32,
    )

    # E2M1's coarsest step is 2 units out of a range of 6, so the nearest-level
    # error is at most one unit of the block scale, amax/6, plus the e4m3
    # rounding of that scale.
    block_amax = bank.view(4, 16, 8, 16).abs().amax(dim=-1)
    limit = (block_amax / 6.0 * 1.15).repeat_interleave(16, dim=-1)
    assert bool(((restored - bank).abs() <= limit + 1e-6).all())


def test_grid_values_round_trip_exactly():
    quantize = _quantizer()
    dequantize_nvfp4 = _engine_dequantizer()
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    signs = torch.tensor([1.0, -1.0])
    # One expert, one row, 16 columns; amax is exactly 6 so the block scale is
    # exactly 1.0 and every value lands on a grid point.
    row = (levels.repeat_interleave(2) * signs.repeat(8)).view(1, 1, 16)

    packed = quantize.quantize_experts(row)
    # The engine's reader broadcasts w_s2 over the scale tensor, so a
    # per-expert global scale has to arrive as [E, 1, 1].
    restored = dequantize_nvfp4(
        packed.weight,
        packed.scales,
        packed.global_scale.view(-1, 1, 1),
        out_dtype=torch.float32,
    )

    assert torch.equal(restored.view(-1), row.view(-1))


def test_gate_and_up_halves_share_one_global_scale():
    quantize = _quantizer()
    # w13 is [E, 2*I, K]: the two halves are rows of one expert bank, so one
    # global scale per expert covers both by construction.
    bank = torch.randn(2, 8, 32, dtype=torch.float16)

    packed = quantize.quantize_experts(bank)

    assert packed.global_scale.shape == (2,)


def test_zero_expert_stays_zero_without_poisoning_its_neighbours():
    quantize = _quantizer()
    dequantize_nvfp4 = _engine_dequantizer()
    torch.manual_seed(1)
    bank = torch.randn(2, 4, 32, dtype=torch.float32)
    bank[0] = 0.0

    packed = quantize.quantize_experts(bank)
    # The engine's reader broadcasts w_s2 over the scale tensor, so a
    # per-expert global scale has to arrive as [E, 1, 1].
    restored = dequantize_nvfp4(
        packed.weight,
        packed.scales,
        packed.global_scale.view(-1, 1, 1),
        out_dtype=torch.float32,
    )

    assert bool(torch.isfinite(restored).all())
    assert torch.equal(restored[0], torch.zeros_like(restored[0]))
    assert bool((restored[1] - bank[1]).abs().max() > 0)


@pytest.mark.parametrize(
    "weight, expected",
    [
        (torch.zeros(4, 32), "expected [experts, N, K]"),
        (torch.zeros(2, 4, 24), "not a multiple of the 16-element"),
        (torch.zeros(2, 4, 32, dtype=torch.int32), "already packed"),
        (torch.full((2, 4, 32), float("nan")), "inf or nan"),
    ],
)
def test_refusals_name_the_reason(weight, expected):
    quantize = _quantizer()

    reason = quantize.refusal(weight)

    assert reason is not None and expected in reason
    with pytest.raises(ValueError):
        quantize.quantize_experts(weight)


def test_packed_weights_are_refused():
    """Re-quantizing an NVFP4 bank would otherwise read codes as magnitudes."""
    quantize = _quantizer()

    reason = quantize.refusal(torch.zeros(2, 4, 16, dtype=torch.uint8))

    assert reason is not None and "already packed" in reason


def test_selection_takes_unquantized_experts_only():
    from sglang_restomod.quant.sm70_mtp_fp4_moe import _admits

    class FusedMoE:
        pass

    class UnquantizedFusedMoEMethod:
        pass

    class ModelOptNvFp4FusedMoEMethod:
        pass

    layer = FusedMoE()
    layer.quant_method = UnquantizedFusedMoEMethod()
    assert _admits(layer, FusedMoE, UnquantizedFusedMoEMethod)

    # The target's experts: same layer type, already NVFP4.
    layer.quant_method = ModelOptNvFp4FusedMoEMethod()
    assert not _admits(layer, FusedMoE, UnquantizedFusedMoEMethod)

    # A dense line that happens to carry an unquantized method.
    linear = type("Linear", (), {})()
    linear.quant_method = UnquantizedFusedMoEMethod()
    assert not _admits(linear, FusedMoE, UnquantizedFusedMoEMethod)


def test_draft_is_identified_by_model_class():
    from sglang_restomod.quant.sm70_mtp_fp4_moe import _is_draft_model

    class Qwen3NextForCausalLMMTP:
        pass

    class Qwen3NextForCausalLM:
        pass

    assert _is_draft_model(Qwen3NextForCausalLMMTP())
    assert not _is_draft_model(Qwen3NextForCausalLM())


class _NvFp4Config:
    group_size = 16


class _QuantConfig:
    nvfp4_config = _NvFp4Config()


class _RunnerConfig:
    activation = "silu"


class _Layer:
    """Just enough of a ``FusedMoE`` for the conversion gates."""

    def __init__(self):
        self.w13_weight = torch.randn(2, 8, 32, dtype=torch.float16)
        self.w2_weight = torch.randn(2, 8, 32, dtype=torch.float16)
        self.quant_config = _QuantConfig()
        self.moe_runner_config = _RunnerConfig()
        self.quant_method = object()
        self.runner = object()


def _patch_backend(monkeypatch, backend):
    from sglang.srt.layers.moe import utils as moe_utils
    from sglang.srt.layers.quantization import modelopt_quant

    # Both modules imported the name by value; patching one leaves the other
    # reading the launch's real backend. The value is the real enum so every
    # predicate the constructor calls still resolves.
    monkeypatch.setattr(moe_utils, "get_moe_runner_backend", lambda: backend)
    monkeypatch.setattr(modelopt_quant, "get_moe_runner_backend", lambda: backend)


def test_conversion_refuses_a_non_marlin_backend(monkeypatch):
    pytest.importorskip("sglang.srt.layers.quantization.modelopt_quant")
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    from sglang_restomod.quant.sm70_mtp_fp4_moe import convert_layer

    _patch_backend(monkeypatch, MoeRunnerBackend.TRITON)
    layer = _Layer()

    with pytest.raises(RuntimeError, match="moe-runner-backend marlin"):
        convert_layer(layer)

    assert not hasattr(layer, "w13_weight_scale")


def test_failed_swap_restores_the_original_layer(monkeypatch):
    pytest.importorskip("sglang.srt.layers.quantization.modelopt_quant")
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    from sglang_restomod.quant.sm70_mtp_fp4_moe import convert_layer

    _patch_backend(monkeypatch, MoeRunnerBackend.MARLIN)

    def boom(self, layer, config):
        raise RuntimeError("swap failed")

    monkeypatch.setattr(ModelOptNvFp4FusedMoEMethod, "create_moe_runner", boom)
    layer = _Layer()
    before = {
        "w13_weight": layer.w13_weight,
        "w2_weight": layer.w2_weight,
        "quant_method": layer.quant_method,
        "runner": layer.runner,
    }

    with pytest.raises(RuntimeError, match="swap failed"):
        convert_layer(layer)

    assert layer.w13_weight is before["w13_weight"]
    assert layer.w2_weight is before["w2_weight"]
    assert layer.quant_method is before["quant_method"]
    assert layer.runner is before["runner"]
    assert not hasattr(layer, "w13_weight_scale")
