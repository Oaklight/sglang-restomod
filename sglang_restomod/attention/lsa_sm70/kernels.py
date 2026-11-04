"""SM70 Triton selection: a chunked bucket scan with a top-p pick, then expansion.

Two programs a row:

- ``_scan_top_p`` scores every bucket of the row's visible blocks in fp32,
  packs ``(score, bucket)`` into one int64 key and sorts it descending, so ties
  go to the lower bucket as the oracle's stable argsort does, and writes the
  best ``top_p`` ids;
- ``_expand`` walks those picks best first, drops the sinks and the window,
  writes the members up to ``top_k`` and then appends the window and the
  sinks.

The tensors are sized by constexpr powers of two; ``select_lists_triton``
computes them from the tensors and the design. Nothing here imports sglang.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .reference import Buckets


@triton.jit
def _scan_top_p(
    queries,
    centroids,
    positions,
    picked,
    bucket_count,
    index_dim: tl.constexpr,
    buckets_pow2: tl.constexpr,
    picks_pow2: tl.constexpr,
    top_p: tl.constexpr,
    cluster_size: tl.constexpr,
    block_bucket_count: tl.constexpr,
):
    row = tl.program_id(0)
    bucket = tl.arange(0, buckets_pow2)
    position = tl.load(positions + row).to(tl.int32)
    visible = tl.minimum((position // cluster_size) * block_bucket_count, bucket_count)
    scores = tl.zeros((buckets_pow2,), tl.float32)
    for dim in tl.static_range(index_dim):
        query = tl.load(queries + row * index_dim + dim).to(tl.float32)
        values = tl.load(
            centroids + bucket * index_dim + dim, mask=bucket < bucket_count, other=0.0
        ).to(tl.float32)
        scores += values * query
    scores = tl.where(bucket < visible, scores, float("-inf"))
    # Unsigned order that matches float order: positives flip the sign bit,
    # negatives invert. The sign bit of the packed key is flipped again so a
    # signed int64 sort orders by score, then by the lower id on a tie.
    bits = scores.to(tl.int32, bitcast=True)
    ordered = bits ^ tl.where(bits < 0, -1, -(1 << 31))
    high = ordered ^ -(1 << 31)
    key = high.to(tl.int64) * (1 << 32) + (((1 << 32) - 1) - bucket.to(tl.int64))
    key = tl.sort(key, descending=True)
    low = key & ((1 << 32) - 1)
    chosen = (((1 << 32) - 1) - low).to(tl.int32)
    columns = tl.arange(0, buckets_pow2)
    tl.store(picked + row * top_p + columns, chosen, mask=columns < top_p)


@triton.jit
def _expand(
    picked,
    members,
    positions,
    lists,
    lengths,
    bucket_count,
    bucket_cap: tl.constexpr,
    top_p: tl.constexpr,
    picks_pow2: tl.constexpr,
    top_k: tl.constexpr,
    window_size: tl.constexpr,
    window_pow2: tl.constexpr,
    sink_size: tl.constexpr,
    sink_pow2: tl.constexpr,
    width: tl.constexpr,
    width_pow2: tl.constexpr,
    cluster_size: tl.constexpr,
    block_bucket_count: tl.constexpr,
    pad: tl.constexpr,
):
    row = tl.program_id(0)
    position = tl.load(positions + row).to(tl.int32)
    visible = tl.minimum((position // cluster_size) * block_bucket_count, bucket_count)
    window_start = tl.maximum(position + 1 - window_size, 0)
    columns = tl.arange(0, width_pow2)
    tl.store(
        lists + row * width + columns,
        tl.full((width_pow2,), pad, tl.int32),
        mask=columns < width,
    )
    picks = tl.load(
        picked + row * top_p + tl.arange(0, picks_pow2)[:, None],
        mask=tl.arange(0, picks_pow2)[:, None] < top_p,
        other=-1,
    )
    live_pick = (tl.arange(0, picks_pow2)[:, None] < top_p) & (picks >= 0) & (picks < visible)
    slots = tl.arange(0, bucket_cap)[None, :]
    taken = tl.load(
        members + picks * bucket_cap + slots,
        mask=live_pick & (slots < bucket_cap),
        other=-1,
    )
    keep = live_pick & (taken >= sink_size) & (taken < window_start)
    keep_i = keep.to(tl.int32)
    within = tl.cumsum(keep_i, axis=1) - keep_i
    totals = tl.sum(keep_i, axis=1)
    base = tl.cumsum(totals, axis=0) - totals
    offsets = base[:, None] + within
    tl.store(
        lists + row * width + offsets,
        taken,
        mask=keep & (offsets < top_k),
    )
    selected = tl.minimum(tl.sum(totals, axis=0), top_k)
    window_columns = tl.arange(0, window_pow2)
    window_count = tl.minimum(position + 1, window_size)
    tl.store(
        lists + row * width + selected + window_columns,
        window_start + window_columns,
        mask=window_columns < window_count,
    )
    sink_columns = tl.arange(0, sink_pow2)
    sink_count = tl.minimum(sink_size, window_start)
    tl.store(
        lists + row * width + selected + window_count + sink_columns,
        sink_columns,
        mask=sink_columns < sink_count,
    )
    tl.store(lengths + row, selected + window_count + sink_count)


def _pow2(value: int) -> int:
    size = 1
    while size < max(1, value):
        size *= 2
    return size


def select_lists_triton(
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    buckets: Buckets,
    top_p: int,
    top_k: int,
    sink_size: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select each row's keys with the two Triton programs."""
    rows = index_queries.shape[0]
    index_dim = index_queries.shape[1]
    bucket_count = buckets.bucket_centroids.shape[0]
    width = top_k + window_size + sink_size
    picked = torch.full((rows, top_p), -1, dtype=torch.int32, device=index_queries.device)
    lists = torch.empty((rows, width), dtype=torch.int32, device=index_queries.device)
    lengths = torch.empty((rows,), dtype=torch.int32, device=index_queries.device)
    if rows == 0:
        return lists, lengths
    candidates = {
        "index_dim": index_dim,
        # tl.sort has no size-1 path; an empty table sorts two masked lanes.
        "buckets_pow2": max(2, _pow2(bucket_count)),
        "picks_pow2": _pow2(top_p),
        "top_p": top_p,
        "cluster_size": buckets.cluster_block_size,
        "block_bucket_count": buckets.block_bucket_count,
    }
    _scan_top_p[(rows,)](
        index_queries,
        buckets.bucket_centroids,
        positions,
        picked,
        bucket_count,
        **candidates,
    )
    _expand[(rows,)](
        picked,
        buckets.bucket_members,
        positions,
        lists,
        lengths,
        bucket_count,
        bucket_cap=buckets.bucket_members.shape[1],
        top_p=top_p,
        picks_pow2=_pow2(top_p),
        top_k=top_k,
        window_size=window_size,
        window_pow2=_pow2(window_size),
        sink_size=sink_size,
        sink_pow2=_pow2(max(1, sink_size)),
        width=width,
        width_pow2=_pow2(width),
        cluster_size=buckets.cluster_block_size,
        block_bucket_count=buckets.block_bucket_count,
        pad=-1,
    )
    return lists, lengths
