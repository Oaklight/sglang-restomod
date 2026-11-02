"""CPU tests for the SM70 dense W4A16 wiring."""

from __future__ import annotations

import sys
import types

import pytest

from sglang_restomod.quant.sm70_fp4_moe import marlin_dense


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(marlin_dense, "_STATE", {"gemm": None, "repack": None, "probed": True})
    monkeypatch.setattr(marlin_dense, "_ORIGINALS", {}, raising=False)


def _record(name):
    calls = []

    def fn(*args, **kwargs):
        calls.append((args, kwargs))
        return name

    return fn, calls


def test_gemm_falls_back_when_the_extension_is_absent():
    original, calls = _record("pinned")
    marlin_dense._ORIGINALS["gemm"] = original

    out = marlin_dense.gptq_marlin_gemm(
        "a", None, "w", "s", None, None, None, None, "ws", types.SimpleNamespace(id=3),
        1, 2, 3,
    )
    assert out == "pinned"
    assert calls[0][0][9].id == 3


def test_gemm_maps_the_extension_schema(monkeypatch):
    captured = {}

    def fake_gemm(*args):
        captured["args"] = args
        return "sm70"

    monkeypatch.setattr(marlin_dense, "_STATE", {"gemm": fake_gemm, "repack": None, "probed": True})
    out = marlin_dense.gptq_marlin_gemm(
        "a", None, "w", "s", "gs", "z", "g", "p", "ws",
        types.SimpleNamespace(id=3), 1, 2, 3, True, False, True, False,
    )
    assert out == "sm70"
    args = captured["args"]
    # a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros,
    # g_idx, perm, workspace, b_type_id, size_m, size_n, size_k, ...
    assert args[3] is None and args[5] is None
    assert args[11] == 3
    assert args[15:] == (True, False, True, False)


def test_repack_appends_the_extension_flag(monkeypatch):
    captured = {}

    def fake_repack(*args):
        captured["args"] = args
        return "packed"

    monkeypatch.setattr(marlin_dense, "_STATE", {"gemm": None, "repack": fake_repack, "probed": True})
    assert marlin_dense.gptq_marlin_repack("w", "p", 128, 64, 4) == "packed"
    assert captured["args"] == ("w", "p", 128, 64, 4, False)


def test_repack_falls_back_to_five_arguments():
    original, calls = _record("pinned")
    marlin_dense._ORIGINALS["repack"] = original
    assert marlin_dense.gptq_marlin_repack("w", "p", 128, 64, 4) == "pinned"
    assert calls[0][0] == ("w", "p", 128, 64, 4)


def test_rebind_only_touches_matching_attributes():
    def original(*args):
        return "o"

    def replacement(*args):
        return "r"

    original.__name__ = "gptq_marlin_gemm"
    holder = types.ModuleType("holder_gemm")
    holder.gptq_marlin_gemm = original
    other = types.ModuleType("holder_other")
    other.gptq_marlin_gemm = lambda *a: "x"
    sys.modules["holder_gemm"] = holder
    sys.modules["holder_other"] = other
    try:
        assert marlin_dense._rebind(original, replacement) == 1
        assert holder.gptq_marlin_gemm is replacement
        assert other.gptq_marlin_gemm() == "x"
    finally:
        del sys.modules["holder_gemm"], sys.modules["holder_other"]
