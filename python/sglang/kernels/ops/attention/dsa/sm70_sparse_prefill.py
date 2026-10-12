"""Selected-key FP16 sparse MLA through the standalone V100 FA AOT library.

The personal flash-attention-v100 fork owns CUDA/CUTLASS code. This adapter
preserves the selected support, FP16 KV/probabilities/output and FP32 reductions.
No weights or full selected KV bank are materialized.
"""

import torch

from sglang.kernels.ops.kvcache.cache_ops import q8kv8_topk_length_from_indices


def sparse_mla_prefill_sm70(q, kv, indices, sm_scale, topk_length=None):
    """Return [1, tokens, heads8/16, 512] with the AOT sparse SM70 FA kernel.

    Index rows address physical KV slots, with negative padding. Optional lengths
    bound the last valid column; otherwise a device scan derives that bound.
    Both gathered matrices remain FP16; the KV cache is never quantized/copied.
    """
    if not (
        q.is_cuda
        and torch.cuda.get_device_capability(q.device) == (7, 0)
        and q.ndim == 3
        and q.shape[1] in (8, 16)
        and q.shape[2] == 512
        and q.dtype == kv.dtype == torch.float16
        and kv.device == indices.device == q.device
        and kv.shape[-1] == 512
        and indices.dtype == torch.int32
        and indices.shape[0] == q.shape[0]
        and indices.shape[-1] > 0
        and (indices.ndim == 2 or (indices.ndim == 3 and indices.shape[1] == 1))
        and (kv.ndim == 2 or (kv.ndim == 3 and kv.shape[1] == 1))
    ):
        raise ValueError(
            "SM70 sparse prefill requires FP16 heads8/16, latent512 and CUDA int32 indices"
        )
    indices = indices.reshape(q.shape[0], indices.shape[-1]).contiguous()
    if topk_length is None:
        topk_length = q8kv8_topk_length_from_indices(indices)
    elif not (
        topk_length.device == q.device
        and topk_length.dtype == torch.int32
        and topk_length.numel() == q.shape[0]
    ):
        raise ValueError("SM70 sparse prefill lengths must be one CUDA int32 per query")
    from sglang_fa_sm70 import ensure_loaded

    ensure_loaded()
    q = q.contiguous()
    kv = kv.reshape(-1, 512).contiguous()
    # Contiguous offset views can still violate the vector-load alignment.
    if q.data_ptr() % 16:
        q = q.clone()
    if kv.data_ptr() % 16:
        kv = kv.clone()
    return torch.ops.sglang_fa_sm70.prefill(
        q, kv, indices, topk_length.reshape(-1).contiguous(), float(sm_scale)
    ).unsqueeze(0)
