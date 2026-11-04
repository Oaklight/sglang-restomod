"""The request index and the vectorized selection against the torch oracle.

Both run on a CPU, so this covers the index maintenance and the selection
without sglang or a GPU.
"""

from __future__ import annotations

import torch

from sglang_restomod.attention.lsa_sm70.budget import EarlyBudget
from sglang_restomod.attention.lsa_sm70.design import LsaDesign, RequestIndex, row_segments
from sglang_restomod.attention.lsa_sm70.reference import build_buckets, read
from sglang_restomod.attention.lsa_sm70.selection import select_lists

DESIGN = LsaDesign(
    top_k=32,
    top_p=16,
    window_size=64,
    sink_size=2,
    pool_size=4,
    bucket_size=8,
    bucket_cap=16,
    index_dim=8,
)


def _unit_keys(rows: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(rows, dim, generator=generator)
    return keys / keys.norm(dim=1, keepdim=True)


def _filled_index(design: LsaDesign, keys: torch.Tensor) -> RequestIndex:
    """Build an index whose open block is deliberately partial."""
    index = RequestIndex(design, torch.device("cpu"))
    for start, stop in ((0, 50), (50, 64), (64, 128), (128, 192)):
        index.append(keys[start:stop])
    index.append(keys[192:222])
    return index


def _buckets_of(design: LsaDesign, keys: torch.Tensor):
    return build_buckets(
        keys[:192],
        pool_size=design.pool_size,
        bucket_size=design.bucket_size,
        bucket_cap=design.bucket_cap,
        cluster_block_size=design.cluster_block_size,
    )


def test_index_length_counts_closed_and_open_keys():
    keys = _unit_keys(222, DESIGN.index_dim, 0)
    index = _filled_index(DESIGN, keys)

    assert index.closed == 192
    assert index.open_count == 30
    assert index.length == 222


def test_index_buckets_match_the_oracle():
    keys = _unit_keys(222, DESIGN.index_dim, 1)
    index = _filled_index(DESIGN, keys)
    expected = _buckets_of(DESIGN, keys)

    buckets = index.buckets()

    assert torch.equal(buckets.bucket_members, expected.bucket_members)
    assert torch.allclose(buckets.bucket_centroids, expected.bucket_centroids)
    assert torch.equal(buckets.bucket_sizes, expected.bucket_sizes)
    assert buckets.block_bucket_count == expected.block_bucket_count
    assert buckets.cluster_block_size == expected.cluster_block_size


def test_empty_index_has_no_buckets():
    index = RequestIndex(DESIGN, torch.device("cpu"))

    buckets = index.buckets()

    assert buckets.bucket_centroids.shape == (0, DESIGN.index_dim)
    assert buckets.bucket_members.shape == (0, DESIGN.bucket_cap)


def test_reset_drops_the_table():
    keys = _unit_keys(222, DESIGN.index_dim, 2)
    index = _filled_index(DESIGN, keys)

    index.reset()

    assert index.length == 0
    assert index.buckets().bucket_centroids.shape[0] == 0


def test_selection_matches_the_oracle_over_every_position():
    keys = _unit_keys(222, DESIGN.index_dim, 3)
    queries = _unit_keys(222, DESIGN.index_dim, 4)
    positions = torch.arange(222, dtype=torch.int32)
    index = _filled_index(DESIGN, keys)
    buckets = _buckets_of(DESIGN, keys)

    selected, lengths = select_lists(
        queries,
        positions,
        index.buckets(),
        top_p=DESIGN.top_p,
        top_k=DESIGN.top_k,
        sink_size=DESIGN.sink_size,
        window_size=DESIGN.window_size,
    )
    expected, expected_lengths = read(
        queries,
        positions,
        buckets,
        top_p=DESIGN.top_p,
        top_k=DESIGN.top_k,
        sink_size=DESIGN.sink_size,
        window_size=DESIGN.window_size,
    )

    assert torch.equal(lengths, expected_lengths)
    for row, length in enumerate(lengths.tolist()):
        assert torch.equal(selected[row, :length], expected[row, :length])
        assert (selected[row, length:] == -1).all()


def test_selection_matches_the_oracle_under_an_early_budget():
    keys = _unit_keys(222, DESIGN.index_dim, 5)
    queries = _unit_keys(222, DESIGN.index_dim, 6)
    positions = torch.arange(222, dtype=torch.int32)
    index = _filled_index(DESIGN, keys)
    buckets = _buckets_of(DESIGN, keys)
    early = EarlyBudget(position=128, top_k=16, top_p=8)

    selected, lengths = select_lists(
        queries,
        positions,
        index.buckets(),
        top_p=DESIGN.top_p,
        top_k=DESIGN.top_k,
        sink_size=DESIGN.sink_size,
        window_size=DESIGN.window_size,
        early_budget=early,
    )
    expected, expected_lengths = read(
        queries,
        positions,
        buckets,
        top_p=DESIGN.top_p,
        top_k=DESIGN.top_k,
        sink_size=DESIGN.sink_size,
        window_size=DESIGN.window_size,
        early_budget=early,
    )

    assert torch.equal(lengths, expected_lengths)
    for row, length in enumerate(lengths.tolist()):
        assert torch.equal(selected[row, :length], expected[row, :length])


def test_design_reads_knobs_from_the_environment():
    design = LsaDesign.from_env(
        {
            "SGLANG_RESTOMOD_LSA_TOP_K": "1024",
            "SGLANG_RESTOMOD_LSA_TOP_P": "256",
            "SGLANG_RESTOMOD_LSA_WINDOW": "512",
            "SGLANG_RESTOMOD_LSA_SINK": "2",
        }
    )

    assert (design.top_k, design.top_p, design.window_size, design.sink_size) == (1024, 256, 512, 2)


def test_design_refuses_inconsistent_sizes():
    try:
        LsaDesign.from_env({"SGLANG_RESTOMOD_LSA_WINDOW": "500"})
    except ValueError as error:
        assert "must divide" in str(error)
    else:
        raise AssertionError("a window the bucket size does not divide was accepted")


def test_row_segments_split_a_prefill_batch_by_request():
    assert row_segments([7, 9], [3, 2]) == [(7, 0, 3), (9, 3, 5)]


def test_row_segments_give_a_decode_batch_one_row_a_request():
    assert row_segments([7, 9], None) == [(7, 0, 1), (9, 1, 2)]


def test_row_segments_of_an_empty_batch_are_empty():
    assert row_segments([], []) == []
    assert row_segments([], None) == []
