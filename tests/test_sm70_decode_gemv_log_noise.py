"""Log-volume tests for the SM70 NVFP4 decode GEMV dispatch (CPU).

The dispatch runs once per expert layer per call. A skip reason logged at INFO
on every call therefore writes one line per layer per rank per step: a decode
step on the 48-layer Flash-Next profile produced 144 lines, and prefill chunks
produced thousands. The rule these tests pin is that a repeated reason is
reported once and then only at DEBUG, without losing the first report.
"""

from __future__ import annotations

import logging

import pytest

# No torch gate: the module loads it lazily and every path exercised here stops
# before the first tensor operation, so this test runs on a CPU-only runner.
from sglang_restomod.quant.sm70_fp4_moe import decode_gemv


@pytest.fixture(autouse=True)
def _fresh_once_set(monkeypatch):
    monkeypatch.setattr(decode_gemv, "_LOG_ONCE", set())
    monkeypatch.setattr(decode_gemv, "_LOG_ONCE_MAX", 32)


def _info(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.INFO]


def test_repeated_reason_is_logged_once(caplog):
    caplog.set_level(logging.DEBUG, logger=decode_gemv.__name__)
    for _ in range(5):
        decode_gemv._log_once("k", "decode gemv skipped: reason %s", "x")
    assert len(_info(caplog)) == 1
    assert "reason x" in caplog.text


def test_distinct_reasons_are_each_logged(caplog):
    caplog.set_level(logging.DEBUG, logger=decode_gemv.__name__)
    decode_gemv._log_once("a", "first")
    decode_gemv._log_once("b", "second")
    decode_gemv._log_once("a", "first")
    assert [r.getMessage() for r in _info(caplog)] == ["first", "second"]


def test_beyond_the_cap_falls_back_to_debug(caplog, monkeypatch):
    monkeypatch.setattr(decode_gemv, "_LOG_ONCE_MAX", 2)
    caplog.set_level(logging.DEBUG, logger=decode_gemv.__name__)
    for i in range(4):
        decode_gemv._log_once(f"k{i}", "reason %d", i)
    # The cap bounds memory; a key past it is still reported, just not at INFO.
    assert len(_info(caplog)) == 2
    assert "reason 3" in caplog.text


def test_skip_path_in_maybe_call_logs_once_per_projection(caplog, monkeypatch):
    """The call site, not just the helper: a varying chunk length is one key."""
    caplog.set_level(logging.DEBUG, logger=decode_gemv.__name__)
    monkeypatch.setattr(decode_gemv, "_is_cuda", lambda _t: True)

    sentinel = object()
    for size_m in (4096, 27, 512):
        decode_gemv.maybe_call(
            sentinel, sentinel, sentinel, sentinel, sentinel, sentinel,
            sentinel, sentinel, sentinel,
            moe_block_size=decode_gemv.BLOCK_M,
            top_k=10,
            mul_topk_weights=True,
            size_m=size_m,
            size_n=320,
            size_k=2560,
        )

    records = _info(caplog)
    assert len(records) == 1, [r.getMessage() for r in records]
    # The per-call variable is still in the message, the key just ignores it.
    assert "m=4096" in records[0].getMessage()
