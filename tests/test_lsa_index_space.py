"""CPU tests for the LSA index space port."""

from __future__ import annotations

import torch

from sglang_restomod.attention.lsa_sm70.index_space import (
    LearnedProjection,
    RotaryLayout,
    position_free,
    project,
    regroup,
    regroup_partial,
)


def _qk(batch=1, heads=24, kv_heads=4, q_len=8, k_len=16, dim=256, dtype=torch.float32):
    torch.manual_seed(0)
    q = torch.randn(batch, heads, q_len, dim, dtype=dtype)
    k = torch.randn(batch, kv_heads, k_len, dim, dtype=dtype)
    return q, k


def test_regroup_matches_the_head_sum_identity() -> None:
    q, k = _qk()
    index_q, index_k = regroup_partial(q, k)
    assert index_q.shape == (1, 8, 4 * 256)
    assert index_k.shape == (1, 16, 4 * 256)
    group = q.shape[1] // k.shape[1]
    for head in range(k.shape[1]):
        q_sum = q[0, head * group : (head + 1) * group].sum(0)  # [Tq, dh]
        # One query position and one key position, the head's block of the index dim.
        block = slice(head * 256, (head + 1) * 256)
        assert torch.allclose(index_q[0, 3, block], q_sum[3].float(), atol=1e-5, rtol=1e-5)
        assert torch.allclose(index_k[0, 5, block], k[0, head, 5].float(), atol=1e-5, rtol=1e-5)


def test_regroup_returns_unit_rows() -> None:
    q, k = _qk()
    index_q, index_k = regroup(q, k)
    assert index_q.dtype == torch.float32
    assert torch.allclose(index_q.norm(dim=-1), torch.ones(1, 8), atol=1e-5)
    assert torch.allclose(index_k.norm(dim=-1), torch.ones(1, 16), atol=1e-5)


def test_zero_rows_stay_zero() -> None:
    q, k = _qk()
    k[0, 0, 0] = 0
    _, index_k = regroup(q, k)
    assert torch.equal(index_k[0, 0, :256], torch.zeros(256))


def test_position_free_prefix_suffix_and_none() -> None:
    q = torch.arange(2 * 8, dtype=torch.float32).reshape(1, 1, 2, 8)
    k = q.clone()
    qp, kp = position_free(q, k, RotaryLayout("prefix", 2))
    assert torch.equal(qp, q[..., 2:])
    assert torch.equal(kp, k[..., 2:])
    qs, ks = position_free(q, k, RotaryLayout("suffix", 2))
    assert torch.equal(qs, q[..., :6])
    assert torch.equal(ks, k[..., :6])
    qn, kn = position_free(q, k, RotaryLayout("none", 0))
    assert torch.equal(qn, q)
    assert torch.equal(kn, k)


def test_position_free_refuses_a_full_rotation() -> None:
    q = torch.zeros(1, 1, 1, 8)
    try:
        position_free(q, q, RotaryLayout("prefix", 8))
    except ValueError as exc:
        assert "nothing is position-free" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a full rotation must be refused")


def test_learned_projection_shape_and_unit_rows() -> None:
    torch.manual_seed(0)
    q, k = _qk()
    index_q, index_k = regroup(q, k)
    query_map = torch.randn(4 * 256, 64)
    key_map = torch.randn(4 * 256, 64)
    LearnedProjection(query_map, key_map)
    projected_q = project(index_q, query_map)
    projected_k = project(index_k, key_map)
    assert projected_q.shape == (1, 8, 64)
    assert projected_k.shape == (1, 16, 64)
    assert torch.allclose(projected_q.norm(dim=-1), torch.ones(1, 8), atol=1e-5)
    assert torch.allclose(projected_k.norm(dim=-1), torch.ones(1, 16), atol=1e-5)
