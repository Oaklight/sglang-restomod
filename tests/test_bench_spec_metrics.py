"""bench.py records the speculative accept metrics from the final meta_info.

The GPU suite's bench row is the only place a spec launch's accept rate is
visible next to its rate: the engine fills ``spec_accept_length`` only when it
verified drafts, so a missing field has to read as "not speculative", not as
zero acceptance.
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "bench", ROOT / "scripts" / "bench.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load()


def test_absent_meta_yields_no_spec_fields():
    assert bench.spec_fields(None) == {}
    assert bench.spec_fields({}) == {}


def test_non_spec_meta_yields_no_spec_fields():
    meta = {"completion_tokens": 64, "prompt_tokens": 32}
    assert bench.spec_fields(meta) == {}


def test_accept_metrics_are_rounded():
    # Values captured from a 32k MTP launch on the pin, 64 output tokens.
    meta = {
        "completion_tokens": 64,
        "spec_accept_rate": 0.5529411764705883,
        "spec_accept_length": 3.764705882352941,
        "spec_verify_ct": 17,
    }
    assert bench.spec_fields(meta) == {
        "accept_length": 3.765,
        "accept_rate": 0.553,
        "verify_ct": 17,
    }


def test_summary_carries_the_accept_fields():
    r = {
        "t0": 0.0,
        "first_t": 1.0,
        "end_t": 2.0,
        "n_tok": 32,
        "chunk_t": [],
        "usage": {"prompt_tokens": 4096, "completion_tokens": 64},
        "meta": {"spec_accept_length": 3.5, "spec_verify_ct": 18},
    }
    s = bench.summarize(r, "ctx4096 rep1")
    assert s["accept_length"] == 3.5
    assert s["verify_ct"] == 18
    assert "accept_rate" not in s


def test_a_record_without_meta_still_summarizes():
    r = {
        "t0": 0.0,
        "first_t": 1.0,
        "end_t": 2.0,
        "n_tok": 32,
        "chunk_t": [],
        "usage": {"prompt_tokens": 4096, "completion_tokens": 64},
    }
    s = bench.summarize(r, "ctx4096 rep1")
    assert s["pure_decode_tps"] == 63.0
    assert "accept_length" not in s
