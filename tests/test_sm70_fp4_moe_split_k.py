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
    def select(mode: str | None, size_m: int, size_k: int) -> str | None:
        if mode is None:
            monkeypatch.delenv(method.ENV_SPLIT_K, raising=False)
        else:
            monkeypatch.setenv(method.ENV_SPLIT_K, mode)
        monkeypatch.delenv(method.MARLIN_SPLIT_K, raising=False)
        method._select_split_k(size_m, size_k)
        return os.environ.get(method.MARLIN_SPLIT_K)

    return select


def test_default_splits_only_the_long_call(split_k):
    assert split_k(None, 4, 2560) == "2"  # w13 per rank at TP4
    assert split_k(None, 40, 160) == "1"  # w2 per rank at TP4


def test_default_leaves_prefill_alone(split_k):
    # A prefill chunk reaches the same K=2560 gate_up call with a large M; it
    # has no partial wave to fill, and it was never measured.
    assert split_k(None, 2048, 2560) == "1"


def test_default_leaves_an_unaligned_k_unsplit(split_k):
    # Above the threshold but not 64-aligned, so CTA_K need not divide it.
    assert split_k(None, 4, 2592) == "1"


def test_default_splits_k_that_the_chunk_does_not_divide(split_k):
    # 2624 is 41 CTA_K tiles at CTA_K=64, so each split gets whole tiles
    # (21 and 20); divisibility by K/2 is not required.
    assert split_k(None, 4, 2624) == "2"


def test_off_leaves_the_environment_alone(split_k):
    assert split_k("off", 4, 2560) is None
    assert split_k("0", 40, 160) is None


def test_pinned_value_applies_to_every_call(split_k):
    assert split_k("4", 4, 2560) == "4"
    assert split_k("4", 40, 160) == "4"


def test_per_shape_wins_over_an_outside_value(monkeypatch):
    monkeypatch.setenv(method.MARLIN_SPLIT_K, "8")
    monkeypatch.delenv(method.ENV_SPLIT_K, raising=False)
    method._select_split_k(40, 160)
    assert os.environ[method.MARLIN_SPLIT_K] == "1"


def test_unknown_value_is_rejected(split_k):
    with pytest.raises(ValueError, match=method.ENV_SPLIT_K):
        split_k("three", 4, 2560)


def test_auto_is_not_an_alias_for_off(split_k):
    # The extension has its own auto selector; this knob calls that `off`.
    with pytest.raises(ValueError, match=method.ENV_SPLIT_K):
        split_k("auto", 4, 2560)


def test_the_gemm_wrapper_writes_the_value(monkeypatch):
    """Deleting the call inside the wrapper has to fail a test, not just a trace."""
    from types import SimpleNamespace

    seen = {}

    class _Ops:
        def gemm(self, *args, **kwargs):
            seen["split_k"] = os.environ.get(method.MARLIN_SPLIT_K)
            return None

    monkeypatch.delenv(method.ENV_SPLIT_K, raising=False)
    monkeypatch.delenv(method.MARLIN_SPLIT_K, raising=False)
    monkeypatch.setattr(method, "ensure_loaded", lambda: _Ops())
    monkeypatch.setattr(method, "_record_moe_stats", lambda *a, **k: None)

    def call(size_m: int, size_n: int, size_k: int) -> str | None:
        method.moe_wna16_marlin_gemm(
            a=None,
            c_or_none=None,
            b_q_weight=None,
            b_bias_or_none=None,
            b_scales=None,
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=None,
            sorted_token_ids=None,
            expert_ids=None,
            num_tokens_post_padded=None,
            topk_weights=None,
            moe_block_size=8,
            top_k=1,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type=SimpleNamespace(id=0),
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
        )
        return seen["split_k"]

    assert call(4, 320, 2560) == "2"
    assert call(40, 2560, 160) == "1"
