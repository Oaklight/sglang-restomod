"""CPU oracles for the QSA selection outputs the kernels consume.

The selection math is the pin's (``qsa.kernel``), but the backend decides
whether an index tensor means compressed blocks, builds the expansion
arguments, and maps logical positions to physical slots. These tests pin those
boundaries without a GPU:

- ``_uses_block_indices``: block ids vs key positions, by width;
- ``_expand_block_indices``: the expansion rule the kernels receive, compared
  with an explicit oracle (block coverage, sequence masking, the current-block
  tail, valid-first compaction);
- ``_logical_to_physical``: logical position -> slot, with ``-1``,
  out-of-range and duplicate handling across sequences.

The module needs the pin on ``PYTHONPATH``; restomod-ci installs the layer
without its engine dependency, so these tests skip there and run wherever
SGLang is present.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang")

from sglang.srt.layers.attention.qsa.kernel import (  # noqa: E402
    expand_qsa_block_indices,
)

from sglang_restomod.attention.v100.qwen_sparse_attn_backend import (  # noqa: E402
    QwenSparseAttnBackend,
)


class _BlockBackend:
    """The two backend methods, bound without a model runner."""

    _uses_block_indices = QwenSparseAttnBackend._uses_block_indices
    _expand_block_indices = QwenSparseAttnBackend._expand_block_indices

    def __init__(self, block_topk: int, budget: int, compress_ratio: int) -> None:
        self.qsa_profile = SimpleNamespace(block_topk=block_topk, budget=budget)
        self.compress_ratio = compress_ratio


def _oracle_expand(blocks, positions, seq_lens, ratio: int, budget: int):
    """The rule the kernels assume, written straight from the semantics.

    Each block covers ``ratio`` consecutive positions. The expanded list is cut
    at ``budget`` entries, entries at or past the sequence end are dropped, the
    current block's visible tail is appended after them, and valid entries are
    packed to the front in that order.
    """
    width = budget + ratio - 1
    out = torch.full((blocks.shape[0], width), -1, dtype=torch.int64)
    for row in range(blocks.shape[0]):
        length = int(seq_lens[row])
        expanded = []
        for block in blocks[row].tolist():
            if block < 0:
                continue
            expanded.extend(range(block * ratio, (block + 1) * ratio))
        expanded = [position for position in expanded[:budget] if position < length]
        tail_start = ((int(positions[row]) + 1) // ratio) * ratio
        tail = [
            position
            for position in range(tail_start, int(positions[row]) + 1)
            if position < length
        ]
        values = (expanded + tail)[:width]
        out[row, : len(values)] = torch.tensor(values, dtype=torch.int64)
    return out


def test_block_indices_are_recognized_by_width():
    backend = _BlockBackend(block_topk=512, budget=2048, compress_ratio=4)
    assert backend._uses_block_indices(torch.zeros((2, 512), dtype=torch.int32))
    assert not backend._uses_block_indices(torch.zeros((2, 2048), dtype=torch.int32))

    backend.qsa_profile = None
    assert not backend._uses_block_indices(torch.zeros((2, 512), dtype=torch.int32))


def test_block_expansion_matches_the_oracle():
    ratio, budget = 4, 16
    block_topk = (budget + ratio - 1) // ratio
    backend = _BlockBackend(block_topk, budget, ratio)
    blocks = torch.tensor(
        [
            [0, 1, 2, -1],
            [3, 0, 1, 2],
            [10, 4, -1, 5],  # block 10 is past the third row's sequence end
        ],
        dtype=torch.int32,
    )
    positions = torch.tensor([10, 15, 40])
    seq_lens = torch.tensor([12, 20, 41])
    metadata = SimpleNamespace(
        indexer_metadata=SimpleNamespace(
            decode_logical_positions=positions,
            get_seqlens_int32=lambda: seq_lens,
        )
    )

    expanded = backend._expand_block_indices(blocks, metadata)
    expected = _oracle_expand(blocks, positions, seq_lens, ratio, budget)

    assert expanded.shape == (blocks.shape[0], budget + ratio - 1)
    assert torch.equal(expanded.long(), expected), (
        f"expansion mismatch\n got: {expanded}\nwant: {expected}"
    )


def test_block_expansion_rejects_a_wrong_width():
    with pytest.raises(ValueError):
        expand_qsa_block_indices(
            torch.zeros((2, 5), dtype=torch.int32),
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
            4,
            16,
        )


def test_logical_to_physical_maps_and_masks():
    metadata = SimpleNamespace(
        token_to_batch_idx=torch.tensor([0, 0, 1]),
        sequence_lengths=torch.tensor([5, 3]),
        token_slot_table=torch.tensor(
            [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 0, 0]]
        ),
    )
    # Rows 0 and 1 belong to sequence 0 (length 5); row 2 to sequence 1 (length 3).
    logical = torch.tensor(
        [
            [-1, 0, 4, 5, 99],  # -1 padding, in range, two past the end
            [1, 4, 3, 2, -1],
            [2, 3, 0, 0, 1],
        ]
    )

    slots = QwenSparseAttnBackend._logical_to_physical(logical, metadata)

    expected = torch.tensor(
        [
            [-1, 10, 14, -1, -1],
            [11, 14, 13, 12, -1],
            [22, -1, 20, 20, 21],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(slots, expected), f"mapping mismatch\n got: {slots}\nwant: {expected}"


def test_logical_to_physical_rejects_a_row_mismatch():
    metadata = SimpleNamespace(
        token_to_batch_idx=torch.tensor([0, 0]),
        sequence_lengths=torch.tensor([5]),
        token_slot_table=torch.tensor([[10, 11, 12, 13, 14]]),
    )
    with pytest.raises(ValueError):
        QwenSparseAttnBackend._logical_to_physical(
            torch.zeros((3, 2), dtype=torch.long), metadata
        )
