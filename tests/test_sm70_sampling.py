"""Contract tests for the SM70 EAGLE renorm replacements (CPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _reference_top_k(probs, k):
    out = torch.zeros_like(probs)
    for row in range(probs.shape[0]):
        index = torch.topk(probs[row], int(k)).indices
        out[row, index] = probs[row, index]
    return out / out.sum(dim=-1, keepdim=True)


def _reference_top_p(probs, top_p):
    out = torch.zeros_like(probs)
    for row in range(probs.shape[0]):
        values, index = torch.sort(probs[row], descending=True)
        cumulative = values.cumsum(dim=-1)
        crossing = int((cumulative >= float(top_p)).nonzero()[0])
        keep = index[: crossing + 1]
        out[row, keep] = probs[row, keep]
    return out / out.sum(dim=-1, keepdim=True)


def test_top_k_matches_reference():
    from sglang_restomod.spec.sm70_sampling import top_k_renorm_probs

    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(3, 64), dim=-1)
    k = torch.tensor([1, 5, 20])
    got = top_k_renorm_probs(probs, k)
    want = torch.stack(
        [_reference_top_k(probs[i : i + 1], k[i])[0] for i in range(probs.shape[0])]
    )
    assert torch.allclose(got, want, atol=1e-6)
    assert torch.allclose(got.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_top_k_scalar():
    from sglang_restomod.spec.sm70_sampling import top_k_renorm_probs

    torch.manual_seed(1)
    probs = torch.softmax(torch.randn(2, 32), dim=-1)
    got = top_k_renorm_probs(probs, 4)
    want = _reference_top_k(probs, 4)
    assert torch.allclose(got, want, atol=1e-6)


def test_top_p_matches_reference():
    from sglang_restomod.spec.sm70_sampling import top_p_renorm_probs

    torch.manual_seed(2)
    probs = torch.softmax(torch.randn(4, 128), dim=-1)
    top_p = torch.tensor([0.1, 0.5, 0.9, 1.0])
    got = top_p_renorm_probs(probs, top_p)
    want = torch.stack(
        [_reference_top_p(probs[i : i + 1], top_p[i])[0] for i in range(4)]
    )
    assert torch.allclose(got, want, atol=1e-6)
    assert torch.allclose(got.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_dtype_is_preserved():
    from sglang_restomod.spec.sm70_sampling import top_k_renorm_probs

    probs = torch.softmax(torch.randn(2, 16), dim=-1).to(torch.float16)
    got = top_k_renorm_probs(probs, 3)
    assert got.dtype == torch.float16


def test_top_p_returns_float32():
    # FlashInfer's top_p_renorm_probs casts to float() before allocating its
    # output, while top_k_renorm_probs uses empty_like. The replacement mirrors
    # both, so a chained call keeps the dtype upstream would produce.
    from sglang_restomod.spec.sm70_sampling import (
        top_k_renorm_probs,
        top_p_renorm_probs,
    )

    probs = torch.softmax(torch.randn(2, 16), dim=-1).to(torch.float16)
    assert top_k_renorm_probs(probs, 3).dtype == torch.float16
    assert top_p_renorm_probs(probs, 0.9).dtype == torch.float32
