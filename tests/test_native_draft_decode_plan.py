"""CPU tests for the native draft-decode planning helpers."""

from __future__ import annotations

import pytest
import torch

from sglang_restomod.attention.v100 import draft_decode_plan as plan


def test_native_is_enabled_by_default():
    assert plan.native_enabled({}) is True
    assert plan.native_enabled({plan.ENV_FLAG: "1"}) is True
    assert plan.native_enabled({plan.ENV_FLAG: "off"}) is False


def test_native_env_rejects_a_non_boolean():
    with pytest.raises(ValueError, match=plan.ENV_FLAG):
        plan.native_enabled({plan.ENV_FLAG: "maybe"})


def test_page_table_reads_the_page_start_of_each_row():
    page_size = 4
    # Row 0 has 9 tokens over physical pages 0, 3, 2. Row 1 has 5 over 5, 1.
    row0 = [0, 1, 2, 3, 12, 13, 14, 15, 8]
    row1 = [20, 21, 22, 23, 4]
    kv_indices = torch.tensor(row0 + row1, dtype=torch.int64)
    kv_indptr = torch.tensor([0, len(row0), len(row0) + len(row1)], dtype=torch.int32)
    page_table = torch.zeros((2, 3), dtype=torch.int32)
    seq_lens = torch.zeros((2,), dtype=torch.int32)

    rows = plan.page_table_from_indices(
        kv_indptr, kv_indices, page_table, seq_lens, page_size
    )

    assert rows == 2
    assert seq_lens.tolist() == [9, 5]
    assert page_table.tolist() == [[0, 3, 2], [5, 1, 0]]


def test_page_table_marks_unused_pages_zero():
    kv_indices = torch.arange(32, dtype=torch.int64)
    kv_indptr = torch.tensor([0, 4, 4], dtype=torch.int32)
    page_table = torch.full((2, 2), 7, dtype=torch.int32)
    seq_lens = torch.zeros((2,), dtype=torch.int32)

    plan.page_table_from_indices(kv_indptr, kv_indices, page_table, seq_lens, 4)

    assert seq_lens.tolist() == [4, 0]
    assert page_table.tolist() == [[0, 0], [0, 0]]


def test_page_table_handles_an_empty_index_list():
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
    kv_indices = torch.empty((0,), dtype=torch.int64)
    page_table = torch.full((1, 2), 5, dtype=torch.int32)
    seq_lens = torch.ones((1,), dtype=torch.int32)

    rows = plan.page_table_from_indices(kv_indptr, kv_indices, page_table, seq_lens, 4)

    assert rows == 1
    assert seq_lens.tolist() == [0]
    assert page_table.tolist() == [[0, 0]]


def test_page_table_refuses_a_short_buffer():
    kv_indptr = torch.tensor([0, 4, 8], dtype=torch.int32)
    kv_indices = torch.arange(8, dtype=torch.int64)
    page_table = torch.zeros((1, 1), dtype=torch.int32)
    seq_lens = torch.zeros((1,), dtype=torch.int32)

    assert plan.page_table_from_indices(kv_indptr, kv_indices, page_table, seq_lens) == 0


class _Layer:
    def __init__(self, **overrides):
        self.tp_q_head_num = 12
        self.tp_k_head_num = 2
        self.qk_head_dim = 256
        self.v_head_dim = 256
        self.is_cross_attention = False
        self.sliding_window_size = None
        self.logit_cap = None
        self.xai_temperature_len = 0
        for key, value in overrides.items():
            setattr(self, key, value)


def test_supported_layer_has_no_reason():
    assert plan.layer_shape_reason(_Layer()) is None


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"tp_k_head_num": 5}, "gqa_ratio"),
        ({"v_head_dim": 128}, "qk_dim"),
        ({"qk_head_dim": 250, "v_head_dim": 250}, "not_a_multiple"),
        ({"is_cross_attention": True}, "cross_attention"),
        ({"sliding_window_size": 4096}, "sliding_window"),
        ({"logit_cap": 30.0}, "logit_cap"),
        ({"xai_temperature_len": 8}, "xai_temperature"),
    ],
)
def test_unsupported_layer_reports_the_reason(overrides, reason):
    assert reason in plan.layer_shape_reason(_Layer(**overrides))
