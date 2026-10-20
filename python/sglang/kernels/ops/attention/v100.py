"""Thin SM70 adapters for the canonical flash-attention-v100 CUDA package.

No implementation fallback, global GPU scratch or eager CUDA-library import.
Sparse selection and cache/state lifetimes belong to the runtime caller.
"""

import torch


def _ops():
    from sglang_fa_sm70 import ensure_loaded

    ensure_loaded()
    return torch.ops.sglang_fa_sm70


def sparse_attention_v100(
    q,
    kv,
    indices,
    scale,
    lengths=None,
    workspace=None,
    kv_splits=None,
    *,
    return_lse=False,
):
    """Selected physical keys, FP16 latent512; prefill or bounded split decode.

    Negative/out-of-pool IDs are masked. Rows1..4 use split work; all other
    sizes use the same engine without splitting. Scratch is call-owned unless
    a backend supplies its own per-stream dictionary. Shapes use host metadata.
    Optional LSE is FP32/base-2 per query/head, with -inf for empty support;
    DCP combines the FP16 partials using that normalizer in FP32.
    """
    if q.ndim != 3 or q.shape[1] not in (8, 16, 32) or q.shape[2] != 512:
        raise ValueError("V100 sparse attention requires Q[T,H8/16/32,512]")
    if indices.shape[0] != q.shape[0] or indices.ndim not in (2, 3):
        raise ValueError("V100 sparse attention requires one ID row per query")
    if (indices.ndim == 3 and indices.shape[1] != 1) or (
        kv.ndim not in (2, 3) or (kv.ndim == 3 and kv.shape[1] != 1)
    ):
        raise ValueError("V100 sparse attention requires a single latent KV head")
    if kv.shape[-1] != 512 or indices.shape[-1] < 1:
        raise ValueError(
            "V100 sparse attention requires latent512 and a nonempty ID width"
        )
    ids = indices.reshape(q.shape[0], indices.shape[-1]).contiguous()
    kv = kv.reshape(-1, 512)
    if not kv.is_contiguous():
        raise ValueError("V100 sparse attention requires a contiguous FP16 KV cache")
    if lengths is not None:
        lengths = lengths.reshape(-1).contiguous()
    ops = _ops()
    rows, heads, dim = q.shape
    if q.data_ptr() % 16:
        q = q.clone(memory_format=torch.contiguous_format)
    if not 1 <= rows <= 4:
        q = q.contiguous()
        if return_lse:
            return ops.prefill_lse(q, kv, ids, lengths, float(scale))
        return ops.prefill(q, kv, ids, lengths, float(scale))
    # Match the baseline contiguous split grouping without reading GPU counts.
    tiles = (ids.shape[1] + 63) // 64
    desired = max(
        1, torch.cuda.get_device_properties(q.device).multi_processor_count // rows
    )
    if kv_splits is not None:
        if not isinstance(kv_splits, int) or not 1 <= kv_splits <= 80:
            raise ValueError("V100 sparse attention requires 1–80 split groups")
        desired = kv_splits
    per_split = (tiles + desired - 1) // desired
    splits = (tiles + per_split - 1) // per_split
    padded_heads = max(16, heads)
    shape = (rows, splits, padded_heads)
    if workspace is None:
        scratch = None
    else:
        scratch = workspace.get("sparse")
        if scratch is not None and scratch[0].numel() < rows * splits * padded_heads:
            scratch = None
    if scratch is None:
        scratch = (
            torch.empty(
                rows * splits * padded_heads, device=q.device, dtype=torch.float32
            ),
            torch.empty(
                rows * splits * padded_heads * dim, device=q.device, dtype=torch.float16
            ),
        )
        if workspace is not None:
            workspace["sparse"] = scratch
    lse = scratch[0][: rows * splits * padded_heads].view(shape)
    partial = scratch[1][: rows * splits * padded_heads * dim].view(*shape, dim)
    output = torch.empty(q.shape, device=q.device, dtype=torch.float16)
    ops.decode_out(q, kv, ids, lengths, lse, partial, float(scale))
    if return_lse:
        output_lse = torch.empty((rows, heads), device=q.device, dtype=torch.float32)
        ops.reduce_out(lse, partial, output, output_lse)
        return output, output_lse
    ops.reduce_out(lse, partial, output)
    return output


def dense_attention_v100(q, k, v, offsets, max_length, scale, out=None, lse=None):
    """Noncausal packed vision attention, D64, through Volta Tensor Cores."""
    if out is None:
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if lse is None:
        lse = torch.empty(
            (q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32
        )
    _ops().dense_out(q, k, v, offsets, out, lse, int(max_length), float(scale))
    return out


def recurrent_attention_v100(
    q,
    k,
    v,
    gate,
    beta,
    state,
    slots,
    offsets,
    *,
    A_log=None,
    dt_bias=None,
    snapshots=None,
    snapshot_slots=None,
    parents=None,
    scale=128**-0.5,
    lower_bound=-5.0,
    raw_beta=False,
    write_state=True,
    out=None,
    prepared=None,
    track_state=None,
    track_steps=None,
):
    """KDA FP32 recurrence; FP16 pool writes and per-step verification carry."""
    if out is None:
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _ops().kda_out(
        q,
        k,
        v,
        gate,
        beta,
        state,
        slots,
        offsets,
        out,
        A_log,
        dt_bias,
        snapshots,
        snapshot_slots,
        parents,
        float(scale),
        float(lower_bound),
        raw_beta,
        write_state,
        prepared,
        track_state,
        track_steps,
    )
    return out
