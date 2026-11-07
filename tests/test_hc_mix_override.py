"""CPU tests for the hyper-connection mixer override (``hc_mix_fp8``).

The pin's ``GatedResidual`` needs torch and kernel imports, so these tests use
a stand-in class with the same structure: two ``nn.Linear`` mixers and a
``_mix_compute`` that calls ``F.linear`` on their raw weights. ``install()``
takes the module object, which is how the stand-in is injected.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_restomod import plugins
from sglang_restomod.quant import sm70_hc_mix


def _mix_math(x, down_weight, up_weight, hc, hs):
    weight = F.silu(F.linear(x, down_weight) / hc)
    weight = torch.sigmoid(F.linear(weight, up_weight))
    weight = weight.unflatten(-1, (hc, hs))
    return (weight * x.unflatten(-1, (hc, hs))).mean(dim=-2)


class _PaddedLinear(nn.Module):
    """Stand-in for a converted, zero-padded linear.

    Real modules slice to ``n`` and pad the input to ``k_pad``; this one does
    the same arithmetic through ``F.linear`` so the test needs no kernel.
    """

    def __init__(self, real_n: int, k_pad: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(real_n, k_pad) * 0.1)
        self.weight_block_size = [128, 128]
        self.sm70_fp8_out_features = real_n
        self.sm70_fp8_padded_in_features = k_pad

    def forward(self, x):
        if x.shape[-1] < self.weight.shape[1]:
            pad = torch.zeros(
                *x.shape[:-1],
                self.weight.shape[1] - x.shape[-1],
                dtype=x.dtype,
            )
            x = torch.cat((x, pad), dim=-1)
        return F.linear(x, self.weight)[..., : self.sm70_fp8_out_features]


class FakeGatedResidual:
    def __init__(self, hidden=8, hc=2, lowrank=4, converted=False):
        self.config = SimpleNamespace(hidden_size=hidden, hc_lowrank=lowrank)
        self.hc_count = hc
        self.input_mix_weight_down = nn.Linear(hidden * hc, lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(lowrank, hidden * hc, bias=False)

        def _mix_compute(
            hyper_input_normed,
            input_mix_weight_down,
            input_mix_weight_up,
            hc,
            hs,
        ):
            input_mix_weight = F.silu(F.linear(hyper_input_normed, input_mix_weight_down) / hc)
            input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
            input_mix_weight = torch.sigmoid(input_mix_weight)
            input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
            return (input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))).mean(dim=-2)

        self._mix_compute = _mix_compute
        if converted:
            self.input_mix_weight_down = _PaddedLinear(lowrank, hidden * hc)
            self.input_mix_weight_up = _PaddedLinear(hidden * hc, 128)

    def mix(self, hyper_input):
        mixed = self._mix_compute(
            hyper_input,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
            self.hc_count,
            self.config.hidden_size,
        )
        return mixed, (hyper_input, hyper_input, None)


class NoMixerClass:
    def __init__(self):
        self.config = SimpleNamespace(hidden_size=8, hc_lowrank=4)
        self.hc_count = 2


class ChangedLayoutClass:
    """A pin bump that removed the F.linear mixer this override hooks."""

    def mix(self):  # keeps the class shape check happy
        return None

    def __init__(self):
        self.config = SimpleNamespace(hidden_size=8, hc_lowrank=4)
        self.hc_count = 2
        self.input_mix_weight_down = nn.Linear(16, 4, bias=False)
        self.input_mix_weight_up = nn.Linear(4, 16, bias=False)

        def _mix_compute(x, down, up, hc, hs):
            return torch.zeros_like(x)

        self._mix_compute = _mix_compute


@pytest.fixture(scope="module")
def fake_module():
    module = types.ModuleType("fake_hyperconnection")
    module.GatedResidual = FakeGatedResidual
    assert sm70_hc_mix.install(module) == 1
    return module


def _reference(instance, x):
    """The mixer's arithmetic on the logical (unpadded) weights."""
    down = instance.input_mix_weight_down
    up = instance.input_mix_weight_up
    hs = instance.config.hidden_size
    hc = instance.hc_count
    lowrank = instance.config.hc_lowrank
    return _mix_math(
        x,
        down.weight[:lowrank, : hc * hs],
        up.weight[: hc * hs, :lowrank],
        hc,
        hs,
    )


def test_component_is_registered():
    assert "hc_mix_fp8" in plugins.COMPONENTS
    component = plugins.COMPONENTS["hc_mix_fp8"]
    assert component.module == "sglang_restomod.quant.sm70_hc_mix"


def test_unconverted_mixer_keeps_the_upstream_path(fake_module):
    instance = FakeGatedResidual()
    assert instance._restomod_hc_mix is True  # hook installed, inert until conversion
    x = torch.randn(3, 16)
    got, _residuals = instance.mix(x)
    assert torch.allclose(got, _reference(instance, x))


def test_conversion_after_construction_is_picked_up(fake_module):
    """``fp8_dense_quant`` converts after load, i.e. after ``__init__`` ran."""
    instance = FakeGatedResidual()
    x = torch.randn(2, 16)
    instance.mix(x)  # the pre-conversion call, through the upstream path
    instance.input_mix_weight_down = _PaddedLinear(4, 16)
    instance.input_mix_weight_up = _PaddedLinear(16, 128)
    got, _residuals = instance.mix(x)
    assert torch.allclose(got, _reference(instance, x))


def test_converted_mixer_routes_through_the_modules(fake_module):
    torch.manual_seed(0)
    instance = FakeGatedResidual(converted=True)
    assert instance._restomod_hc_mix is True
    x = torch.randn(2, 16)
    got, _residuals = instance.mix(x)
    assert torch.allclose(got, _reference(instance, x))


def test_second_install_is_a_noop(fake_module):
    assert sm70_hc_mix.install(fake_module) == 0


def test_install_raises_without_the_class():
    module = types.ModuleType("empty")
    with pytest.raises(RuntimeError, match="no GatedResidual"):
        sm70_hc_mix.install(module)


def test_install_raises_when_the_class_lost_mix():
    module = types.ModuleType("noclass")
    module.GatedResidual = NoMixerClass
    with pytest.raises(RuntimeError, match="no mix"):
        sm70_hc_mix.install(module)


def test_install_raises_when_the_mixer_layout_changed():
    module = types.ModuleType("changed")
    module.GatedResidual = ChangedLayoutClass
    with pytest.raises(RuntimeError, match="pin bump changed"):
        sm70_hc_mix.install(module)


def test_instance_guard_fires_on_a_wrong_shape(fake_module):
    instance = FakeGatedResidual(converted=False)
    instance.input_mix_weight_down = _PaddedLinear(6, 16)  # wrong output width
    with pytest.raises(RuntimeError, match="input_mix_weight_down"):
        sm70_hc_mix._install_instance(instance)


def test_layout_guard_survives_an_earlier_init_wrapper(fake_module):
    """``overrides.py`` wraps ``__init__`` before components load.

    ``_install_hyperconnection_fp16`` replaces ``GatedResidual.__init__`` with a
    closure whose own source has none of the mixer's lines, so a guard that
    reads ``cls.__init__`` rejects a correct pin. The guard reads the class
    body instead, and the install must chain the earlier wrapper.
    """
    original = FakeGatedResidual.__init__

    def earlier_override(self, *args, **kwargs):
        original(self, *args, **kwargs)

    FakeGatedResidual.__init__ = earlier_override
    try:
        sm70_hc_mix._check_class(FakeGatedResidual)  # must not raise
        assert sm70_hc_mix.install(fake_module) == 1
        instance = FakeGatedResidual()
        assert instance.config.hidden_size == 8  # the earlier wrapper ran
    finally:
        FakeGatedResidual.__init__ = original


def test_instance_guard_fires_on_a_non_linear(fake_module):
    instance = FakeGatedResidual(converted=False)
    instance.input_mix_weight_up = nn.Identity()
    with pytest.raises(RuntimeError, match="not a linear"):
        sm70_hc_mix._install_instance(instance)
