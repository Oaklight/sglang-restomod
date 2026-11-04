"""Vectorized flat LSA selection over a built bucket table.

``reference.read`` is the oracle: it loops over query rows and uses Python
``argsort``. A prefill chunk has tens of thousands of rows, so the serving path
selects a chunk at a time with the same rules:

- a query sees only the buckets of the cluster blocks closed before its own;
- the best ``top_p`` buckets are picked, a tie going to the lower bucket;
- their members are expanded best bucket first, dropping the sinks and the
  window, and cut at ``top_k``;
- the window and then the sinks follow, and the row is ``-1`` padded.

The early budget is per row: a row before ``early.position`` reads
``early.top_k`` and ``early.top_p``.
"""

from __future__ import annotations

import torch

from .budget import EarlyBudget
from .constants import PAD
from .reference import Buckets


def select_lists(
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    buckets: Buckets,
    top_p: int,
    top_k: int,
    sink_size: int,
    window_size: int,
    early_budget: EarlyBudget | None = None,
    chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the keys each query reads, a chunk of rows at a time.

    Args:
        index_queries: One index query a row [Tq, index_dim] fp32 or bf16, unit rows.
        positions: Each query's position [Tq] int32 or int64.
        buckets: The buckets of the cache, from ``build_buckets``.
        top_p: Buckets a query picks.
        top_k: Long-range keys a query reads at most.
        sink_size: First positions always read.
        window_size: Recent positions always read, the query's own included.
        early_budget: The budget of queries before its position, or None.
        chunk: Rows selected at once.

    Returns:
        Selected key ids [Tq, top_k + window_size + sink_size] int32 - the
        expansion, then the window, then the sinks, -1 padded.
        Keys each row holds [Tq] int32.
    """
    rows = index_queries.shape[0]
    width = top_k + window_size + sink_size
    selected = torch.full((rows, width), PAD, dtype=torch.int32, device=index_queries.device)
    lengths = torch.zeros(rows, dtype=torch.int32, device=index_queries.device)
    if rows == 0:
        return selected, lengths
    bucket_count = buckets.bucket_centroids.shape[0]
    bucket_ids = torch.arange(bucket_count, device=index_queries.device)
    pick_count = min(top_p, bucket_count)
    early_top_k = early_budget.top_k if early_budget is not None else top_k
    early_top_p = early_budget.top_p if early_budget is not None else top_p
    expand_count = max(top_k, early_top_k)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        query = index_queries[start:stop].float()
        position = positions[start:stop].long()
        scores = query @ buckets.bucket_centroids.T
        visible = (position // buckets.cluster_block_size * buckets.block_bucket_count).clamp(
            max=bucket_count
        )
        scores = scores.masked_fill(bucket_ids[None, :] >= visible[:, None], float("-inf"))
        if early_budget is not None:
            row_top_p = torch.where(
                position < early_budget.position,
                torch.full_like(position, early_top_p),
                torch.full_like(position, top_p),
            )
            row_top_k = torch.where(
                position < early_budget.position,
                torch.full_like(position, early_top_k),
                torch.full_like(position, top_k),
            )
        else:
            row_top_p = torch.full_like(position, top_p)
            row_top_k = torch.full_like(position, top_k)
        # Best score first, a tie to the lower bucket.
        picked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :pick_count]
        pick_columns = torch.arange(pick_count, device=index_queries.device)
        pick_valid = pick_columns[None, :] < row_top_p[:, None]
        pick_valid &= picked < visible[:, None]
        members = buckets.bucket_members[picked].long()
        members = torch.where(pick_valid[..., None], members, -1)
        flat = members.reshape(stop - start, -1)
        if flat.shape[1] < expand_count:
            # A short table - no visible bucket, or few - still needs a row to sort.
            pad = expand_count - flat.shape[1]
            flat = torch.cat(
                [
                    flat,
                    torch.full((stop - start, pad), PAD, dtype=flat.dtype, device=flat.device),
                ],
                dim=1,
            )
        window_start = (position + 1 - window_size).clamp(min=0)
        valid = (flat >= sink_size) & (flat < window_start[:, None])
        order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)
        order = order[:, :expand_count]
        expansion = flat.gather(1, order)
        expansion_valid = valid.gather(1, order)
        expansion_valid &= (
            torch.arange(expand_count, device=index_queries.device)[None, :] < row_top_k[:, None]
        )
        n_sel = expansion_valid.sum(1)
        block = selected[start:stop]
        # Expansion at the row's start, valid entries first.
        sel_columns = torch.arange(expand_count, device=index_queries.device)
        sel_mask = sel_columns[None, :] < n_sel[:, None]
        block[:, :expand_count] = torch.where(sel_mask, expansion, PAD).to(torch.int32)
        # Window after the expansion.
        window_columns = torch.arange(window_size, device=index_queries.device)
        n_window = (position + 1).clamp(max=window_size)
        window_mask = window_columns[None, :] < n_window[:, None]
        window_values = window_start[:, None] + window_columns[None, :]
        block.scatter_(
            1,
            n_sel[:, None] + window_columns[None, :],
            torch.where(window_mask, window_values, PAD).to(torch.int32),
        )
        # Sinks after the window.
        n_sink = torch.minimum(torch.full_like(window_start, sink_size), window_start)
        sink_columns = torch.arange(sink_size, device=index_queries.device)
        sink_mask = sink_columns[None, :] < n_sink[:, None]
        block.scatter_(
            1,
            n_sel[:, None] + n_window[:, None] + sink_columns[None, :],
            torch.where(sink_mask, sink_columns[None, :].expand_as(sink_mask), PAD).to(torch.int32),
        )
        lengths[start:stop] = n_sel + n_window + n_sink
    return selected, lengths
