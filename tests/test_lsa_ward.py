"""CPU tests for the LSA Ward clustering port."""

from __future__ import annotations

import torch

from sglang_restomod.attention.lsa_sm70.ward import (
    members_from_labels,
    pool_keys,
    ward_merge,
    ward_rounds,
)


def test_pool_keys_sums_consecutive_runs() -> None:
    keys = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4)
    pools = pool_keys(keys, 4)
    assert pools.shape == (2, 2, 4)
    assert torch.equal(pools[0, 0], keys[0, :4].sum(0))
    assert torch.equal(pools[1, 1], keys[1, 4:].sum(0))


def test_ward_merge_partitions_pools_and_respects_the_cap() -> None:
    torch.manual_seed(0)
    pools = torch.randn(2, 128, 64)
    sizes = torch.full((2, 128), 4.0)
    labels = ward_merge(pools, sizes, bucket_count=64, bucket_cap=16)
    assert labels.shape == (2, 128)
    for block in range(2):
        unique = torch.unique(labels[block])
        assert unique.numel() == 64
        assert int(unique.min()) == 0 and int(unique.max()) == 63
        for bucket in unique.tolist():
            held = sizes[block][labels[block] == bucket].sum()
            assert held <= 16


def test_ward_merge_is_deterministic() -> None:
    torch.manual_seed(1)
    pools = torch.randn(1, 64, 32)
    sizes = torch.full((1, 64), 4.0)
    first = ward_merge(pools, sizes, bucket_count=32, bucket_cap=16)
    second = ward_merge(pools, sizes, bucket_count=32, bucket_cap=16)
    assert torch.equal(first, second)


def test_ward_rounds_reports_a_short_block() -> None:
    # One pool of 8 keys cannot merge with another under a cap of 8.
    pools = torch.randn(1, 4, 16)
    sizes = torch.full((1, 4), 8.0)
    _, reached = ward_rounds(pools, sizes, bucket_count=2, bucket_cap=8)
    assert not bool(reached.item())


def test_members_from_labels_keeps_position_order_and_pads() -> None:
    labels = torch.tensor([1, 0, 1, 2, 0], dtype=torch.int64)
    members = members_from_labels(labels, bucket_count=3, bucket_cap=3)
    assert members.tolist() == [[1, 4, -1], [0, 2, -1], [3, -1, -1]]
