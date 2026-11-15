"""CPU tests for the TileLang GDN pair: gates, fallbacks, rebinding, state pool.

The kernels need a V100 and TileLang; everything here runs on CPU against the
dispatch logic around them. The gate must never answer a call the kernel
cannot compute exactly, and a refused call must reach the Triton function with
its arguments unchanged.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest
import torch

from sglang_restomod.linear_attn_sm70 import gdn_pair

B, T, HG, H, K, V = 1, 128, 1, 2, 16, 64
LOGGER = "sglang_restomod.linear_attn_sm70.gdn_pair"


@pytest.fixture
def pair(monkeypatch):
    monkeypatch.setattr(gdn_pair, "_ORIGINALS", {})
    monkeypatch.setattr(gdn_pair, "_NOTES", set())
    monkeypatch.setattr(gdn_pair, "_INSTALLED", False)
    monkeypatch.setattr(gdn_pair, "_KERNELS", None)
    monkeypatch.setattr(gdn_pair, "_KERNEL_IMPORT_FAILED", False)
    return gdn_pair


def _chunk_o_inputs(t=T):
    q = torch.zeros(B, t, HG, K, dtype=torch.float16)
    k = torch.zeros(B, t, HG, K, dtype=torch.float16)
    v = torch.zeros(B, t, H, V, dtype=torch.float16)
    h = torch.zeros(B, t // 64, H, V, K, dtype=torch.float16)
    g = torch.zeros(B, t, H, dtype=torch.float32)
    return q, k, v, h, g


def _chunk_h_inputs(t=T):
    k = torch.zeros(B, t, HG, K, dtype=torch.float16)
    w = torch.zeros(B, t, H, K, dtype=torch.float16)
    u = torch.zeros(B, t, H, V, dtype=torch.float16)
    g = torch.zeros(B, t, H, dtype=torch.float32)
    state = torch.zeros(B, H, V, K, dtype=torch.float16)
    return k, w, u, g, state


class _Recorder:
    """Stands in for a wrapped function; records the exact call."""

    def __init__(self, result="triton"):
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


# -- chunk_o gate ----------------------------------------------------------


def test_chunk_o_gate_accepts_the_supported_shape(pair):
    q, k, v, h, g = _chunk_o_inputs()
    assert pair._gate_chunk_o(q, k, v, h, g, None, None, 64) == ""
    assert pair._gate_chunk_o(q, k, v, h, g, K**-0.5, None, 64) == ""


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda a: (None, a[1], a[2], a[3], a[4]), id="no-q"),
        pytest.param(lambda a: (a[0], a[1], a[2], a[3], None), id="no-gate"),
        pytest.param(lambda a: tuple(_chunk_o_inputs(96)), id="odd-tokens"),
        pytest.param(lambda a: (a[0], a[1], a[2], a[3][:, :1], a[4]), id="short-h"),
        pytest.param(
            lambda a: (a[0], a[1], a[2].float(), a[3], a[4]),
            id="fp32-values",
        ),
        pytest.param(
            lambda a: (a[0], a[1], a[2], a[3], a[4].half()),
            id="fp16-gate",
        ),
    ],
)
def test_chunk_o_gate_refuses_unsupported_calls(pair, mutate):
    assert pair._gate_chunk_o(*mutate(_chunk_o_inputs()), None, None, 64) != ""


def test_chunk_o_gate_refuses_varlen_and_other_scale(pair):
    q, k, v, h, g = _chunk_o_inputs()
    assert "varlen" in pair._gate_chunk_o(q, k, v, h, g, None, torch.zeros(2), 64)
    assert "scale" in pair._gate_chunk_o(q, k, v, h, g, 0.5, None, 64)
    assert "chunk_size" in pair._gate_chunk_o(q, k, v, h, g, None, None, 32)


def test_device_gate_refuses_a_cpu_tensor(pair):
    q, _, _, _, _ = _chunk_o_inputs()
    assert pair._device_reason(q, "chunk_o") != ""


# -- chunk_o dispatch ------------------------------------------------------


def test_chunk_o_serves_when_the_gate_passes(pair, monkeypatch):
    sentinel = object()
    original = _Recorder()
    pair._ORIGINALS["chunk_fwd_o"] = original
    monkeypatch.setattr(pair, "_device_reason", lambda *a: "")
    monkeypatch.setattr(pair, "_run_chunk_o", lambda *a: sentinel)

    q, k, v, h, g = _chunk_o_inputs()
    assert pair.chunk_fwd_o(q, k, v, h, g=g, scale=K**-0.5) is sentinel
    assert original.calls == []


def test_chunk_o_falls_back_with_the_arguments_intact(pair, monkeypatch):
    original = _Recorder()
    pair._ORIGINALS["chunk_fwd_o"] = original
    monkeypatch.setattr(pair, "_device_reason", lambda *a: "")
    monkeypatch.setattr(pair, "_run_chunk_o", lambda *a: pytest.fail("ran on refusal"))

    q, k, v, h, g = _chunk_o_inputs()
    cu = torch.zeros(2, dtype=torch.int64)
    assert pair.chunk_fwd_o(q, k, v, h, g=g, scale=None, cu_seqlens=cu) == "triton"
    expected = ((q, k, v, h), {"g": g, "scale": None, "cu_seqlens": cu, "chunk_size": 64})
    assert original.calls == [expected]


def test_chunk_o_falls_back_when_the_kernel_raises(pair, monkeypatch):
    original = _Recorder()
    pair._ORIGINALS["chunk_fwd_o"] = original
    monkeypatch.setattr(pair, "_device_reason", lambda *a: "")

    def boom(*a):
        raise RuntimeError("launch failed")

    monkeypatch.setattr(pair, "_run_chunk_o", boom)
    q, k, v, h, g = _chunk_o_inputs()
    assert pair.chunk_fwd_o(q, k, v, h, g=g) == "triton"
    assert len(original.calls) == 1
    assert any("fault" in note for note in pair._NOTES)


def test_a_repeated_refusal_logs_once(pair, monkeypatch, caplog):
    pair._ORIGINALS["chunk_fwd_o"] = _Recorder()
    monkeypatch.setattr(pair, "_device_reason", lambda *a: "")
    q, k, v, h, g = _chunk_o_inputs()
    cu = torch.zeros(2, dtype=torch.int64)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        pair.chunk_fwd_o(q, k, v, h, g=g, cu_seqlens=cu)
        pair.chunk_fwd_o(q, k, v, h, g=g, cu_seqlens=cu)
    assert len([r for r in caplog.records if r.name == LOGGER]) == 1


# -- chunk_h gate ----------------------------------------------------------


def _good_h_args(k, w, u, g, state):
    """Positional argument tuple for ``_gate_fwd_h`` that it accepts as-is."""
    idx = torch.zeros(B, dtype=torch.int64)
    return (k, w, u, g, None, state, idx, True, None, None, False, True, None, None)


def test_chunk_h_gate_accepts_the_supported_shape(pair):
    k, w, u, g, state = _chunk_h_inputs()
    assert pair._gate_fwd_h(*_good_h_args(k, w, u, g, state)) == ""


@pytest.mark.parametrize(
    "overrides",
    [
        {"gk": torch.zeros(B, T, HG, K)},
        {"use_exp2": True},
        {"cu_seqlens": torch.zeros(2)},
        {"chunk_indices": torch.zeros(1, 2)},
        {"track_state": torch.zeros(1)},
        {"track_chunk_idx": torch.zeros(1, dtype=torch.int64)},
        {"save_new_value": False},
        {"g": None},
        {"initial_state_indices": None},
    ],
)
def test_chunk_h_gate_refuses_unsupported_calls(pair, overrides):
    k, w, u, g, state = _chunk_h_inputs()
    kwargs = {
        "gk": None,
        "initial_state": state,
        "initial_state_indices": torch.zeros(B, dtype=torch.int64),
        "save_new_value": True,
        "cu_seqlens": None,
        "chunk_indices": None,
        "use_exp2": False,
        "inplace_update": True,
        "track_state": None,
        "track_chunk_idx": None,
    }
    kwargs.update(overrides)
    assert pair._gate_fwd_h(k, w, u, kwargs.pop("g", g), **kwargs) != ""


def test_chunk_h_gate_refuses_odd_tokens(pair):
    k, w, u, g, state = _chunk_h_inputs(96)
    reason = pair._gate_fwd_h(*_good_h_args(k, w, u, g, state))
    assert "multiple" in reason


# -- chunk_h state pool ----------------------------------------------------


class _StateKernels:
    """A fake kernel pair whose state pass adds one to every element of s0."""

    def __init__(self):
        self.states = []

    def chunk_h(self, b, t, h, hg, k, v):
        def kernel(kt, wt, ut, gt, s0):
            self.states.append(s0.clone())
            s0.add_(1.0)
            return (
                torch.zeros(b, t // 64, h, v, k, dtype=torch.float16),
                torch.zeros_like(ut),
            )

        return kernel

    def chunk_o(self, b, t, h, hg, k, v):
        def kernel(qt, kt, vt, ht, gt):
            return torch.zeros(b, t, h, v, dtype=torch.float16)

        return kernel


def test_chunk_h_writes_the_state_back_to_its_own_slot(pair, monkeypatch):
    kernels = _StateKernels()
    monkeypatch.setattr(pair, "_KERNELS", kernels)
    pool = torch.zeros(4, H, V, K, dtype=torch.float16)
    pool[2] = 5.0
    idx = torch.tensor([2], dtype=torch.int64)

    k, w, u, g, _ = _chunk_h_inputs()
    h, v_new = pair._run_fwd_h(k, w, u, g, pool, idx, True)

    assert tuple(h.shape) == (B, T // 64, H, V, K)
    assert tuple(v_new.shape) == (B, T, H, V)
    assert torch.equal(kernels.states[0], pool.new_full((B, H, V, K), 5.0))
    assert torch.equal(pool[2], pool.new_full((H, V, K), 6.0))
    for slot in (0, 1, 3):
        assert torch.equal(pool[slot], pool.new_zeros(H, V, K))


def test_chunk_h_leaves_a_negative_slot_untouched_and_starts_at_zero(pair, monkeypatch):
    kernels = _StateKernels()
    monkeypatch.setattr(pair, "_KERNELS", kernels)
    pool = torch.zeros(4, H, V, K, dtype=torch.float16)
    pool[0] = 9.0
    idx = torch.tensor([-1], dtype=torch.int64)

    k, w, u, g, _ = _chunk_h_inputs()
    pair._run_fwd_h(k, w, u, g, pool, idx, True)

    assert torch.equal(kernels.states[0], torch.zeros(B, H, V, K, dtype=torch.float16))
    assert torch.equal(pool[0], pool.new_full((H, V, K), 9.0))


def test_chunk_h_honours_inplace_update_false(pair, monkeypatch):
    kernels = _StateKernels()
    monkeypatch.setattr(pair, "_KERNELS", kernels)
    pool = torch.zeros(4, H, V, K, dtype=torch.float16)
    pool[1] = 3.0
    idx = torch.tensor([1], dtype=torch.int64)

    k, w, u, g, _ = _chunk_h_inputs()
    pair._run_fwd_h(k, w, u, g, pool, idx, False)
    assert torch.equal(pool[1], pool.new_full((H, V, K), 3.0))


# -- installation ----------------------------------------------------------


def _original_pair():
    def chunk_fwd_o(*args, **kwargs):
        return "triton_o"

    def chunk_gated_delta_rule_fwd_h(*args, **kwargs):
        return "triton_h"

    return chunk_fwd_o, chunk_gated_delta_rule_fwd_h


def _fake_modules(monkeypatch):
    chunk_fwd_o, fwd_h = _original_pair()
    source_o = types.ModuleType("fake.fla.chunk_o")
    source_o.chunk_fwd_o = chunk_fwd_o
    source_h = types.ModuleType("fake.fla.chunk_delta_h")
    source_h.chunk_gated_delta_rule_fwd_h = fwd_h
    caller = types.ModuleType("fake.fla.chunk")
    caller.chunk_fwd_o = chunk_fwd_o
    caller.chunk_gated_delta_rule_fwd_h = fwd_h
    other = types.ModuleType("fake.fla.kda")
    other.chunk_gated_delta_rule_fwd_h = fwd_h
    for module in (source_o, source_h, caller, other):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return source_o, source_h, caller, other


def test_bind_rebinds_every_holder(pair, monkeypatch):
    source_o, source_h, caller, other = _fake_modules(monkeypatch)
    assert pair._bind(source_o, source_h, caller) is True

    assert caller.chunk_fwd_o is pair.chunk_fwd_o
    assert caller.chunk_gated_delta_rule_fwd_h is pair.chunk_gated_delta_rule_fwd_h
    assert source_o.chunk_fwd_o is pair.chunk_fwd_o
    assert other.chunk_gated_delta_rule_fwd_h is pair.chunk_gated_delta_rule_fwd_h


def test_bind_rolls_back_when_the_engine_caller_never_sees_it(pair, monkeypatch):
    source_o, source_h, caller, other = _fake_modules(monkeypatch)
    foreign = _Recorder()
    caller.chunk_fwd_o = foreign

    assert pair._bind(source_o, source_h, caller) is False
    assert source_o.chunk_fwd_o is not pair.chunk_fwd_o
    assert source_h.chunk_gated_delta_rule_fwd_h is not pair.chunk_gated_delta_rule_fwd_h
    assert other.chunk_gated_delta_rule_fwd_h is not pair.chunk_gated_delta_rule_fwd_h
    assert caller.chunk_fwd_o is foreign


def test_control_arm_installs_nothing(pair, monkeypatch):
    monkeypatch.setenv("SM70_GDN_TILELANG", "0")
    assert pair.install() is False
    assert pair._INSTALLED is False
    assert pair._ORIGINALS == {}
    assert pair._KERNELS is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", True), ("1", True), ("on", True), ("0", False), ("off", False), ("No", False)],
)
def test_switch_values(pair, monkeypatch, value, expected):
    monkeypatch.setenv("SM70_GDN_TILELANG", value)
    assert pair.enabled() is expected


def test_switch_rejects_an_unknown_value(pair, monkeypatch):
    monkeypatch.setenv("SM70_GDN_TILELANG", "maybe")
    with pytest.raises(ValueError):
        pair.enabled()
