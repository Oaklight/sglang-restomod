"""The decode-path census is an instrument; it must count and never raise.

Written for one question a profile window could not answer: which expert
projection each decode path serves, at which token count. ``n`` separates the
two routed projections (w13 n=320 per rank, down n=160) and ``m`` is what the
M cut acts on, so the counter keys on all three.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _fresh(monkeypatch):
    from sglang_restomod.quant.sm70_fp4_moe import decode_gemv

    monkeypatch.setattr(decode_gemv, "_CENSUS", {})
    monkeypatch.setattr(decode_gemv, "_CENSUS_CALLS", 0)
    return decode_gemv


def test_census_is_off_unless_enabled(monkeypatch):
    gemv = _fresh(monkeypatch)
    monkeypatch.delenv(gemv.ENV_CENSUS, raising=False)

    gemv._census("gemv", 320, 4)
    gemv.note_marlin(160, 4)

    assert gemv._CENSUS == {}
    assert gemv._CENSUS_CALLS == 0


def test_census_counts_per_path_shape_and_token_count(monkeypatch):
    gemv = _fresh(monkeypatch)
    monkeypatch.setenv(gemv.ENV_CENSUS, "1")

    gemv._census("gemv", 320, 4)
    gemv._census("gemv", 320, 4)
    gemv.note_marlin(160, 4)
    gemv._census("gemv", "320", 8)  # a call site may hand the size as a string

    assert gemv._CENSUS == {
        ("gemv", 320, 4): 2,
        ("marlin", 160, 4): 1,
        ("gemv", 320, 8): 1,
    }
    line = gemv.report()
    assert "gemv:n=320:m=4=2" in line
    assert "marlin:n=160:m=4=1" in line
    assert "4 dispatches" in line


def test_report_names_an_empty_census(monkeypatch):
    gemv = _fresh(monkeypatch)
    assert "<none>" in gemv.report()
