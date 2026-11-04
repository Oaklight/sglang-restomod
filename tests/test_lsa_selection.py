"""CPU tests: the vectorized selection matches the reference oracle."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang_restomod.attention.lsa_sm70.budget import EarlyBudget
from sglang_restomod.attention.lsa_sm70.reference import Buckets, build_buckets, read
from sglang_restomod.attention.lsa_sm70.selection import select_lists


def _buckets(seed: int = 0):
    torch.manual_seed(seed)
    keys = F.normalize(torch.randn(2048, 64), dim=-1)
    return build_buckets(keys, pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=512)


def _queries(rows: int, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return F.normalize(torch.randn(rows, 64), dim=-1)


def test_select_lists_matches_the_reference() -> None:
    buckets = _buckets()
    queries = _queries(64)
    positions = torch.arange(0, 2048, 32, dtype=torch.int64)
    got, got_lengths = select_lists(
        queries, positions, buckets, top_p=16, top_k=64, sink_size=4, window_size=32
    )
    want, want_lengths = read(
        queries, positions, buckets, top_p=16, top_k=64, sink_size=4, window_size=32
    )
    assert torch.equal(got_lengths, want_lengths)
    for row in range(queries.shape[0]):
        n = int(want_lengths[row])
        assert got[row, :n].tolist() == want[row, :n].tolist()
        assert torch.all(got[row, n:] == -1)


def test_select_lists_matches_the_reference_with_an_early_budget() -> None:
    buckets = _buckets(seed=2)
    queries = _queries(32, seed=3)
    positions = torch.arange(0, 2048, 64, dtype=torch.int64)
    early = EarlyBudget(position=1024, top_k=32, top_p=8)
    got, got_lengths = select_lists(
        queries,
        positions,
        buckets,
        top_p=16,
        top_k=64,
        sink_size=4,
        window_size=32,
        early_budget=early,
    )
    want, want_lengths = read(
        queries,
        positions,
        buckets,
        top_p=16,
        top_k=64,
        sink_size=4,
        window_size=32,
        early_budget=early,
    )
    assert torch.equal(got_lengths, want_lengths)
    for row in range(queries.shape[0]):
        n = int(want_lengths[row])
        assert got[row, :n].tolist() == want[row, :n].tolist()


def test_select_lists_covers_the_causal_prefix_with_a_full_budget() -> None:
    buckets = _buckets(seed=4)
    queries = _queries(4, seed=5)
    positions = torch.tensor([511, 1023, 1535, 2047], dtype=torch.int64)
    selected, lengths = select_lists(
        queries,
        positions,
        buckets,
        top_p=1024,
        top_k=4096,
        sink_size=4,
        window_size=512,
    )
    for row, position in enumerate(positions.tolist()):
        live = selected[row][selected[row] >= 0].tolist()
        assert sorted(live) == list(range(position + 1))
        assert int(lengths[row]) == position + 1


def test_select_lists_without_closed_blocks_reads_the_window() -> None:
    empty = Buckets(
        torch.empty(0, 64),
        torch.empty(0, 16, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        64,
        512,
    )
    queries = _queries(4, seed=6)
    positions = torch.arange(4, dtype=torch.int64)

    got, got_lengths = select_lists(
        queries, positions, empty, top_p=16, top_k=64, sink_size=2, window_size=32
    )
    want, want_lengths = read(
        queries, positions, empty, top_p=16, top_k=64, sink_size=2, window_size=32
    )

    assert torch.equal(got_lengths, want_lengths)
    for row in range(queries.shape[0]):
        n = int(want_lengths[row])
        assert got[row, :n].tolist() == want[row, :n].tolist()
    # Position zero has no closed block and no sink outside the window.
    assert got[0, : int(got_lengths[0])].tolist() == [0]


def test_select_lists_at_position_zero_matches_the_reference() -> None:
    buckets = _buckets(seed=7)
    queries = _queries(2, seed=8)
    positions = torch.tensor([0, 1], dtype=torch.int64)

    got, got_lengths = select_lists(
        queries, positions, buckets, top_p=16, top_k=64, sink_size=4, window_size=32
    )
    want, want_lengths = read(
        queries, positions, buckets, top_p=16, top_k=64, sink_size=4, window_size=32
    )

    assert torch.equal(got_lengths, want_lengths)
    for row in range(queries.shape[0]):
        n = int(want_lengths[row])
        assert got[row, :n].tolist() == want[row, :n].tolist()
