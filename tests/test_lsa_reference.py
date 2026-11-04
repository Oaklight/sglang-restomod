"""CPU tests for the flat LSA selection port."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang_restomod.attention.lsa_sm70.budget import EarlyBudget
from sglang_restomod.attention.lsa_sm70.reference import build_buckets, read

DIM = 64


def _unit(count: int, dim: int = DIM, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return F.normalize(torch.randn(count, dim), dim=-1)


def _buckets(keys: torch.Tensor):
    return build_buckets(
        keys,
        pool_size=4,
        bucket_size=8,
        bucket_cap=16,
        cluster_block_size=512,
    )


def test_build_buckets_partitions_every_closed_key() -> None:
    keys = _unit(1024)
    buckets = _buckets(keys)
    members = buckets.bucket_members.flatten()
    live = members[members >= 0]
    assert live.numel() == 1024
    assert torch.equal(torch.sort(live).values, torch.arange(1024))
    assert int(buckets.bucket_sizes.sum()) == 1024
    assert torch.allclose(
        buckets.bucket_centroids.norm(dim=-1),
        torch.ones(buckets.bucket_centroids.shape[0]),
        atol=1e-5,
    )


def test_read_with_a_full_budget_covers_the_causal_prefix() -> None:
    keys = _unit(1024)
    buckets = _buckets(keys)
    queries = _unit(2, seed=3)
    positions = torch.tensor([511, 1023], dtype=torch.int64)
    selected, lengths = read(
        queries,
        positions,
        buckets,
        top_p=1024,
        top_k=4096,
        sink_size=4,
        window_size=512,
    )
    assert lengths.tolist() == [512, 1024]
    for row, position in enumerate(positions.tolist()):
        live = selected[row][selected[row] >= 0].tolist()
        assert sorted(live) == list(range(position + 1))


def test_read_orders_expansion_then_window_then_sinks() -> None:
    keys = _unit(1024)
    buckets = _buckets(keys)
    queries = _unit(1, seed=4)
    positions = torch.tensor([1023], dtype=torch.int64)
    selected, lengths = read(
        queries,
        positions,
        buckets,
        top_p=1024,
        top_k=4096,
        sink_size=4,
        window_size=64,
    )
    row = selected[0, : lengths[0]].tolist()
    assert row[-4:] == [0, 1, 2, 3]
    assert row[-68:-4] == list(range(960, 1024))
    assert all(4 <= key < 960 for key in row[:-68])


def test_read_sees_only_closed_cluster_blocks() -> None:
    keys = _unit(1024)
    buckets = _buckets(keys)
    queries = _unit(1, seed=5)
    positions = torch.tensor([100], dtype=torch.int64)
    selected, lengths = read(
        queries,
        positions,
        buckets,
        top_p=1024,
        top_k=4096,
        sink_size=4,
        window_size=32,
    )
    live = selected[0, : lengths[0]].tolist()
    assert live == list(range(69, 101)) + [0, 1, 2, 3]


def test_early_budget_reads_fewer_long_range_keys() -> None:
    keys = _unit(1024)
    buckets = _buckets(keys)
    queries = _unit(1, seed=6)
    positions = torch.tensor([600], dtype=torch.int64)
    late, late_lengths = read(
        queries, positions, buckets, top_p=1024, top_k=4096, sink_size=4, window_size=32
    )
    early, early_lengths = read(
        queries,
        positions,
        buckets,
        top_p=1024,
        top_k=4096,
        sink_size=4,
        window_size=32,
        early_budget=EarlyBudget(position=700, top_k=32, top_p=4),
    )
    assert int(early_lengths[0]) < int(late_lengths[0])
    assert int(early_lengths[0]) <= 32 + 32 + 4


def test_read_is_deterministic() -> None:
    keys = _unit(1024)
    buckets = _buckets(keys)
    queries = _unit(4, seed=7)
    positions = torch.tensor([200, 400, 600, 800], dtype=torch.int64)
    first = read(queries, positions, buckets, top_p=16, top_k=128, sink_size=4, window_size=64)
    second = read(queries, positions, buckets, top_p=16, top_k=128, sink_size=4, window_size=64)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
