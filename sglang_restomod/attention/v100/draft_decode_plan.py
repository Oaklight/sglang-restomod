"""Planning helpers for the native draft-decode path.

The engine's multi-step draft container precomputes one ``kv_indptr`` /
``kv_indices`` pair per draft step. The native split-KV decode kernel takes a
page table and one visible length per row instead. This module converts the
first form into the second and decides which layer shapes the kernel accepts.

It imports torch only, so the CPU test suite can exercise it without the
engine. The container that calls it lives in ``draft_decode``.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# The native decode kernel and the engine's paged KV cache share this page size.
PAGE_SIZE = 16
ENV_FLAG = "SGLANG_V100_DRAFT_DECODE_NATIVE"



def _as_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in ("0", "false", "off", "no", "1", "true", "on", "yes"):
        raise ValueError(f"{name} must be a boolean value, got {value!r}.")
    return normalized in ("1", "true", "on", "yes")


def native_enabled(env=None) -> bool:
    """Return whether the launch allows the native draft-decode path."""
    value = (env if env is not None else os.environ).get(ENV_FLAG)
    if value is None or not value.strip():
        return True
    return _as_bool(value, ENV_FLAG)


def page_table_from_indices(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> int:
    """Fill a per-step page table and per-row lengths from a flat index list.

    ``kv_indptr`` has ``rows + 1`` entries and ``kv_indices`` holds the rows
    concatenated in order. Row ``r`` starts at ``kv_indptr[r]`` and has
    ``kv_indptr[r + 1] - kv_indptr[r]`` visible tokens. Logical page ``p`` of
    the row starts at its token ``p * page_size``; with a page-aligned token
    pool that token's physical id divided by ``page_size`` is the page index.

    Returns the number of rows written. Never synchronizes with the host.
    """
    rows = kv_indptr.numel() - 1
    if rows <= 0 or page_table.shape[0] < rows or seq_lens.shape[0] < rows:
        return 0
    if kv_indices.numel() == 0:
        page_table[:rows].zero_()
        seq_lens[:rows].zero_()
        return rows

    seq = kv_indptr[1 : rows + 1].to(torch.int64) - kv_indptr[:rows].to(torch.int64)
    seq_lens[:rows].copy_(seq.to(seq_lens.dtype))

    max_pages = page_table.shape[1]
    offsets = torch.arange(max_pages, dtype=torch.int64, device=kv_indices.device) * page_size
    index = kv_indptr[:rows].to(torch.int64).unsqueeze(1) + offsets.unsqueeze(0)
    valid = offsets.unsqueeze(0) < seq.unsqueeze(1)
    index.clamp_(max=kv_indices.numel() - 1)

    pages = (kv_indices[index] // page_size).to(torch.int32)
    page_table[:rows].copy_(torch.where(valid, pages, torch.zeros_like(pages)))
    return rows


def layer_shape_reason(layer) -> str | None:
    """Return why the native decode kernel cannot serve ``layer``, or None."""
    q_heads = int(getattr(layer, "tp_q_head_num", 0))
    kv_heads = int(getattr(layer, "tp_k_head_num", 0))
    qk_dim = int(getattr(layer, "qk_head_dim", 0))
    v_dim = int(getattr(layer, "v_head_dim", 0))
    if q_heads <= 0 or kv_heads <= 0:
        return "no_head_count"
    if q_heads % kv_heads:
        return f"gqa_ratio(h={q_heads},hkv={kv_heads})"
    if qk_dim != v_dim:
        return f"qk_dim({qk_dim})!=v_dim({v_dim})"
    if qk_dim % PAGE_SIZE:
        return f"dim({qk_dim})_not_a_multiple_of_{PAGE_SIZE}"
    if getattr(layer, "is_cross_attention", False):
        return "cross_attention"
    window = getattr(layer, "sliding_window_size", None)
    if window is not None and window >= 0:
        return f"sliding_window({window})"
    if getattr(layer, "logit_cap", None) not in (None, 0, 0.0):
        return f"logit_cap({layer.logit_cap})"
    if getattr(layer, "xai_temperature_len", 0) > 0:
        return "xai_temperature"
    return None
