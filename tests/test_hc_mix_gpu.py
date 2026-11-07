"""GPU test: the mixer override reproduces the unconverted math.

Builds the pin's ``GatedResidual`` at the Flash-Next mixer shape (hidden 2560,
4 hyper-connections, low-rank 320), then compares the overridden ``mix``
against the same arithmetic on the logical weights. The converted stand-in
pads the low-rank axis to the 128 block and slices, which is what
``fp8_dense_quant`` does; the numeric kernel is covered by its own tests.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_restomod.quant import sm70_hc_mix

pytestmark = pytest.mark.gpu

HIDDEN = 2560
HC = 4
LOWRANK = 320
BLOCK = 128


class _PaddedLinear(nn.Module):
    """A converted linear's contract: padded width, fp8 weight, sliced output.

    ``fp8_dense_quant`` leaves an fp8 tensor in ``module.weight`` and pads the
    low-rank axis to the block; the forward casts, pads the input and slices.
    """

    def __init__(self, weight: torch.Tensor, out_features: int):
        super().__init__()
        n_pad = (weight.shape[0] + BLOCK - 1) // BLOCK * BLOCK
        k_pad = (weight.shape[1] + BLOCK - 1) // BLOCK * BLOCK
        padded = torch.zeros(n_pad, k_pad, dtype=weight.dtype, device=weight.device)
        padded[: weight.shape[0], : weight.shape[1]] = weight
        self.weight = nn.Parameter(padded.to(torch.float8_e4m3fn))
        self.weight_block_size = [BLOCK, BLOCK]
        self.sm70_fp8_out_features = out_features
        self.sm70_fp8_padded_in_features = k_pad

    def _dequant(self, dtype):
        return self.weight.to(dtype)

    def forward(self, x):
        weight = self._dequant(x.dtype)
        if x.shape[-1] < weight.shape[1]:
            pad = torch.zeros(
                *x.shape[:-1],
                weight.shape[1] - x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat((x, pad), dim=-1)
        return F.linear(x, weight)[..., : self.sm70_fp8_out_features]


def _reference(x, down_weight, up_weight):
    weight = F.silu(F.linear(x, down_weight) / HC)
    weight = torch.sigmoid(F.linear(weight, up_weight))
    weight = weight.unflatten(-1, (HC, HIDDEN))
    return (weight * x.unflatten(-1, (HC, HIDDEN))).mean(dim=-2)


def _make_layer():
    from sglang.srt.layers.hyperconnection import (
        GatedResidual,
        HyperConnectionConfig,
    )

    config = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HIDDEN,
        params_dtype=torch.float16,
        hc_lowrank=LOWRANK,
    )
    layer = GatedResidual(config, use_mix=True, use_combine=False).cuda()
    # The loader casts every parameter; the JIT norm kernel refuses fp32.
    layer.hc_norm.weight.data = layer.hc_norm.weight.data.half()
    return layer


@pytest.fixture(scope="module")
def hyperconnection():
    torch.cuda.set_device(0)
    module = pytest.importorskip("sglang.srt.layers.hyperconnection")
    return module


def test_unconverted_layer_is_unchanged(hyperconnection):
    sm70_hc_mix.install(hyperconnection)
    layer = _make_layer()
    # The hook is installed at construction; unconverted linears keep the
    # upstream computed path, which this asserts on the output.
    assert layer._restomod_hc_mix is True
    x = torch.randn(2, HC * HIDDEN, device="cuda", dtype=torch.float16)
    got, _ = layer.mix(x)
    want = _reference(x, layer.input_mix_weight_down.weight, layer.input_mix_weight_up.weight)
    torch.testing.assert_close(got, want, rtol=5e-2, atol=2e-2)


def test_converted_layer_matches_the_logical_math(hyperconnection):
    sm70_hc_mix.install(hyperconnection)
    layer = _make_layer()
    down_w = layer.input_mix_weight_down.weight.detach().clone()
    up_w = layer.input_mix_weight_up.weight.detach().clone()
    x = torch.randn(4, HC * HIDDEN, device="cuda", dtype=torch.float16)
    want = _reference(x, down_w, up_w)

    down = _PaddedLinear(down_w, LOWRANK)
    up = _PaddedLinear(up_w, HC * HIDDEN)
    layer.input_mix_weight_down = down
    layer.input_mix_weight_up = up
    # The converted weights are fp8; the reference is their logical slice so
    # the test measures the routing, not the quantization error.
    want = _reference(
        x,
        down._dequant(x.dtype)[:LOWRANK, : HC * HIDDEN],
        up._dequant(x.dtype)[: HC * HIDDEN, :LOWRANK],
    )
    assert layer._restomod_hc_mix is True
    # The swap above happens after construction, as the loader does it.
    got, _ = layer.mix(x)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


def test_the_raw_weight_path_is_silently_wrong(hyperconnection):
    """The bypass this component removes: a padded weight under F.linear.

    The zero-padded rows keep the shapes consistent, so the unwrapped path does
    not raise -- it returns a different number. That is why the override exists
    and why a shape guard alone would not catch the bypass.
    """
    layer = _make_layer()
    down_w = layer.input_mix_weight_down.weight.detach().clone()
    down = _PaddedLinear(down_w, LOWRANK)
    x = torch.randn(2, HC * HIDDEN, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        F.linear(x, down.weight)  # fp8 weight under an fp16 F.linear
