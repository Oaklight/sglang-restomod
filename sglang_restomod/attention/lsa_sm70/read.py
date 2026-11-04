"""Read the KV slots an LSA selection names, through the pinned Triton decode kernel.

The pinned SGLang exposes ``decode_attention_fwd`` with ``kv_indptr`` and
``kv_indices`` at ``page_size=1``; it has no architecture gate and the V100
backend already delegates decode to it. Each query row is treated as its own
sequence whose keys are its list, which is exactly what LSA needs: one softmax
over the selected keys, the window and the sinks.

The reference does the same in ``lsa/sglang/models/qwen3_5/backend.py``
(``triton_decode``), from which the packing below is adapted.
"""

from __future__ import annotations

import torch

from .constants import PAD

# Splits of one query's list the Triton read reduces over. A prefill chunk has
# rows enough to fill the GPU; a decode batch does not.
TRITON_SPLITS_PREFILL: int = 1
TRITON_SPLITS_DECODE: int = 8

# More sequences than this is a prefill chunk's rows rather than a decode batch's.
TRITON_PREFILL_ROWS: int = 4096


def _slot_index_dtype(keys: torch.Tensor, values: torch.Tensor) -> torch.dtype:
    """Return the dtype the kernel's slot indices need to address the pool.

    The kernel offsets a slot by its row stride in the index's own dtype, so an
    int32 slot past 2**31 elements wraps. A 128k pool of 4 heads of 256 is well
    under that, but a 10M pool is not.
    """
    largest = max(keys.shape[0] * keys.stride(0), values.shape[0] * values.stride(0))
    return torch.int64 if largest >= 2**31 else torch.int32


def lsa_sparse_read(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    lists: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    logit_cap: float = 0.0,
    splits: int | None = None,
) -> torch.Tensor:
    """Attend each query row over the KV slots its list names.

    Args:
        q: One query row per sequence [S, H, dh], its heads grouped over the pool's key heads.
        k_buffer: The pool's keys [slots, Hkv, dh].
        v_buffer: The pool's values [slots, Hkv, dv].
        lists: Slots per sequence [S, width] int32, the first `lengths` of each row real,
            -1 padded.
        lengths: Real entries of each row [S] int32, at least 1.
        scale: The attention softmax scale.
        logit_cap: A softcap on the logits, or 0.
        splits: KV splits per row, or None to choose from the row count.

    Returns:
        The attention output [S, H, dv] in the query's dtype.
    """
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

    sequences, width = lists.shape
    heads = q.shape[1]
    device = lists.device
    if sequences == 0:
        return torch.empty((0, heads, v_buffer.shape[-1]), dtype=q.dtype, device=device)
    kv_indptr = torch.zeros(sequences + 1, dtype=torch.int32, device=device)
    torch.cumsum(lengths, 0, dtype=torch.int32, out=kv_indptr[1:])
    # Each real entry lands at its row's start plus its column; the padding
    # lands on one spare slot past the end.
    columns = torch.arange(width, dtype=torch.int32, device=device)
    spare = sequences * width
    destination = torch.where(
        columns[None, :] < lengths[:, None],
        kv_indptr[:-1, None] + columns,
        spare,
    )
    kv_indices = torch.zeros(spare + 1, dtype=_slot_index_dtype(k_buffer, v_buffer), device=device)
    kv_indices.scatter_(
        0,
        destination.flatten().long(),
        lists.clamp(min=PAD).flatten().to(kv_indices.dtype),
    )
    split_count = splits
    if split_count is None:
        split_count = (
            TRITON_SPLITS_PREFILL if sequences > TRITON_PREFILL_ROWS else TRITON_SPLITS_DECODE
        )
    num_kv_splits = torch.full((sequences,), split_count, dtype=torch.int32, device=device)
    value_dim = v_buffer.shape[-1]
    attn_logits = torch.empty(
        (sequences, heads, split_count, value_dim), dtype=torch.float32, device=device
    )
    attn_lse = torch.full(
        (sequences, heads, split_count), float("-inf"), dtype=torch.float32, device=device
    )
    out = torch.empty((sequences, heads, value_dim), dtype=q.dtype, device=device)
    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        out,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        split_count,
        scale,
        1.0,
        1.0,
        logit_cap=logit_cap,
        enable_lean=False,
    )
    return out
