"""Availability contract of the SM70 dense TurboMind FP8 loader (CPU).

``_load_sm70_turbomind_fp8_ops()`` returns a bool and every caller treats it
as one: ``can_use_sm70_turbomind_fp8`` falls back to the upstream kernel and
the GPU suite skips. On 2026-11-15 a receipt run found it raising
``TypeError`` instead, from ``Path(None)``, when neither
``SGLANG_SM70_TURBOMIND_OPS_PATH`` nor the fork's ``sm70_prebuilt`` helper
named the library -- the exact configuration of the pinned upstream.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from sglang_restomod.quant.sm70_fp8_dense import _method as kernels  # noqa: E402


def _reset(monkeypatch, default_path):
    """Clear the once-only load state so each case probes from scratch."""
    monkeypatch.setattr(kernels, "_OPS_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(kernels, "_OPS_AVAILABLE", False)
    monkeypatch.setattr(kernels, "_DEFAULT_OPS_PATH", default_path)


def test_no_library_path_reports_unavailable(monkeypatch):
    _reset(monkeypatch, None)
    monkeypatch.delenv("SGLANG_SM70_TURBOMIND_OPS_PATH", raising=False)

    assert kernels._load_sm70_turbomind_fp8_ops() is False


def test_a_library_path_that_does_not_exist_reports_unavailable(monkeypatch, tmp_path):
    _reset(monkeypatch, None)
    monkeypatch.setenv("SGLANG_SM70_TURBOMIND_OPS_PATH", str(tmp_path / "absent.so"))

    assert kernels._load_sm70_turbomind_fp8_ops() is False


def test_the_default_path_is_used_when_the_environment_is_unset(monkeypatch, tmp_path):
    # The environment wins; with it unset the helper's path is tried. A file
    # that is not a library still fails at load, which is the point: the
    # function reports the failure instead of raising.
    missing = tmp_path / "from_helper.so"
    _reset(monkeypatch, str(missing))
    monkeypatch.delenv("SGLANG_SM70_TURBOMIND_OPS_PATH", raising=False)

    assert kernels._load_sm70_turbomind_fp8_ops() is False


# --- which variable may name the library -----------------------------------
#
# The dense path predates the MoE one and used SGLANG_SM70_TURBOMIND_OPS_PATH,
# while the site launchers export SM70_TURBOMIND_SO / SGLANG_SM70_TURBOMIND_SO
# for the experts. A host that serves MoE and not the dense variable silently
# ran the dense linears on the upstream kernel; a receipt run found it by
# skipping the whole dense GPU test file.


def _clear_all_names(monkeypatch):
    for name in (
        "SGLANG_SM70_TURBOMIND_OPS_PATH",
        "SM70_TURBOMIND_SO",
        "SGLANG_SM70_TURBOMIND_SO",
    ):
        monkeypatch.delenv(name, raising=False)


def test_the_moe_variable_names_the_library(monkeypatch, tmp_path):
    _clear_all_names(monkeypatch)
    _reset(monkeypatch, None)
    monkeypatch.setenv("SM70_TURBOMIND_SO", str(tmp_path / "moe.so"))

    assert kernels._resolve_ops_path() == str(tmp_path / "moe.so")


def test_the_sglang_moe_alias_names_the_library(monkeypatch, tmp_path):
    _clear_all_names(monkeypatch)
    _reset(monkeypatch, None)
    monkeypatch.setenv("SGLANG_SM70_TURBOMIND_SO", str(tmp_path / "alias.so"))

    assert kernels._resolve_ops_path() == str(tmp_path / "alias.so")


def test_the_dense_variable_wins_over_the_moe_alias(monkeypatch, tmp_path):
    _clear_all_names(monkeypatch)
    _reset(monkeypatch, None)
    monkeypatch.setenv("SGLANG_SM70_TURBOMIND_OPS_PATH", str(tmp_path / "dense.so"))
    monkeypatch.setenv("SM70_TURBOMIND_SO", str(tmp_path / "moe.so"))

    assert kernels._resolve_ops_path() == str(tmp_path / "dense.so")


def test_the_fork_resolver_wins_over_the_moe_alias(monkeypatch, tmp_path):
    # The engine's own resolver describes the launch that is running; the
    # generic environment names are a fallback, not an override.
    _clear_all_names(monkeypatch)
    _reset(monkeypatch, str(tmp_path / "from_helper.so"))
    monkeypatch.setenv("SM70_TURBOMIND_SO", str(tmp_path / "moe.so"))

    assert kernels._resolve_ops_path() == str(tmp_path / "from_helper.so")


def test_the_moe_alias_reaches_the_loader(monkeypatch, tmp_path):
    seen = []

    def fake_load_library(path):
        seen.append(path)
        raise OSError("not a library")

    _clear_all_names(monkeypatch)
    _reset(monkeypatch, None)
    monkeypatch.setenv("SGLANG_SM70_TURBOMIND_SO", str(tmp_path / "alias.so"))
    monkeypatch.setattr(kernels.torch.ops, "load_library", fake_load_library)

    assert kernels._load_sm70_turbomind_fp8_ops() is False
    assert seen == [str(tmp_path / "alias.so")]

