"""Split-K selection for the SM70 NVFP4 expert GEMM (CPU).

``_select_split_k`` only writes an environment variable, so the rule is
testable without a GPU or the extension. What matters: the long-K call gets 2,
the short-K call and an unaligned K get 1, ``off`` leaves the environment
alone, and a typo fails loudly.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

from sglang_restomod.quant.sm70_fp4_moe import method  # noqa: E402


@pytest.fixture()
def split_k(monkeypatch):
    def select(mode: str | None, size_k: int) -> str | None:
        if mode is None:
            monkeypatch.delenv(method.ENV_SPLIT_K, raising=False)
        else:
            monkeypatch.setenv(method.ENV_SPLIT_K, mode)
        monkeypatch.delenv(method.MARLIN_SPLIT_K, raising=False)
        method._select_split_k(size_k)
        return os.environ.get(method.MARLIN_SPLIT_K)

    return select


def test_default_splits_only_the_long_call(split_k):
    assert split_k(None, 2560) == "2"  # w13 per rank at TP4
    assert split_k(None, 160) == "1"  # w2 per rank at TP4


def test_default_leaves_an_unaligned_k_unsplit(split_k):
    # Above the threshold but not 64-aligned, so CTA_K need not divide it.
    assert split_k(None, 2592) == "1"


def test_off_leaves_the_environment_alone(split_k):
    assert split_k("off", 2560) is None
    assert split_k("0", 160) is None


def test_pinned_value_applies_to_every_call(split_k):
    assert split_k("4", 2560) == "4"
    assert split_k("4", 160) == "4"


def test_per_shape_wins_over_an_outside_value(monkeypatch):
    monkeypatch.setenv(method.MARLIN_SPLIT_K, "8")
    monkeypatch.delenv(method.ENV_SPLIT_K, raising=False)
    method._select_split_k(160)
    assert os.environ[method.MARLIN_SPLIT_K] == "1"


def test_unknown_value_is_rejected(split_k):
    with pytest.raises(ValueError, match=method.ENV_SPLIT_K):
        split_k("three", 2560)
