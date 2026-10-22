"""Shared-history SM70 index scoring without global per-head intermediates.

Volta Tensor Cores reconstruct the existing FP16 operands and accumulate each
full 128-element dot in FP32. ReLU, head weighting/reduction and key scaling
remain FP32. History width is dynamic: chunked prefill reuses one compiled
kernel instead of compiling for each newly appended history bank.
"""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(
    compile_flags=["--fmad=false"],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def _index_score_prefill_kernel():
    rows = T.dynamic("rows")
    capacity = T.dynamic("capacity")
    heads, dim, block = 32, 128, 64

    @T.prim_func
    def mqa_logits_prefill_sm70_kernel(
        Q: T.Tensor([rows, heads, dim], T.float16),
        K: T.Tensor([capacity, dim], T.float16),
        S: T.Tensor([capacity], T.float32),
        W: T.Tensor([rows, heads], T.float32),
        L: T.Tensor([rows], T.int32),
        O: T.Tensor([rows, capacity], T.float32),
    ):
        with T.Kernel(T.ceildiv(capacity, block), T.ceildiv(rows, 2), threads=128) as (
            tile,
            pair,
        ):
            qs = T.alloc_shared([heads, dim], T.float16)
            ks = T.alloc_shared([block, dim], T.float16)
            dots = T.alloc_fragment([heads, block], T.float32)
            result = T.alloc_fragment([block], T.float32)
            first = pair * 2
            last = T.min(first + 1, rows - 1)
            longest = T.min(capacity, T.max(0, T.max(L[first], L[last])))
            start = tile * block
            if start < longest:
                for n, d in T.Parallel(block, dim):
                    ks[n, d] = T.if_then_else(start + n < longest, K[start + n, d], 0)
                # Retain one key tile; reuse query/accumulator storage so pairing
                # does not double the shared-memory or register footprint.
                for item in T.serial(2):
                    row = first + item
                    if row < rows:
                        length = T.min(capacity, T.max(0, L[row]))
                        T.copy(Q[row, :, :], qs)
                        T.clear(dots)
                        T.gemm(
                            qs,
                            ks,
                            dots,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for h, n in T.Parallel(heads, block):
                            dots[h, n] = (
                                T.if_then_else(dots[h, n] < 0, 0, dots[h, n])
                                * W[row, h]
                            )
                        T.reduce_sum(dots, result, dim=0)
                        for n in T.Parallel(block):
                            if start + n < capacity:
                                O[row, start + n] = T.if_then_else(
                                    start + n < length,
                                    result[n] * S[start + n],
                                    -T.infinity(T.float32),
                                )
            else:
                for item in T.serial(2):
                    row = first + item
                    if row < rows:
                        for n in T.Parallel(block):
                            if start + n < capacity:
                                O[row, start + n] = -T.infinity(T.float32)

    return mqa_logits_prefill_sm70_kernel


def mqa_logits_prefill_sm70(query, keys, key_scales, weights, lengths):
    """Score a shared FP16 index bank with 32 heads of width 128 on SM70.

    The caller converts the unchanged index bank once per prefill chunk. No
    latent KV or selected-key budget changes. Only reduced [query,history]
    scores reach global memory; causal future positions are always -inf.
    """
    if not (
        query.is_cuda
        and torch.cuda.get_device_capability(query.device) == (7, 0)
        and query.ndim == 3
        and query.shape[1:] == (32, 128)
        and query.dtype in (torch.float16, torch.float8_e4m3fn)
        and keys.ndim == 2
        and keys.shape[1] == 128
        and keys.dtype == torch.float16
        and keys.is_contiguous()
        and keys.data_ptr() % 16 == 0
        and key_scales.shape in ((keys.shape[0],), (keys.shape[0], 1))
        and weights.shape == query.shape[:2]
        and key_scales.dtype == weights.dtype == torch.float32
        and lengths.shape == (query.shape[0],)
        and lengths.dtype == torch.int32
        and all(t.device == query.device for t in (keys, key_scales, weights, lengths))
    ):
        raise ValueError(
            "SM70 prefill scoring requires [rows,32,128] FP16/FP8 queries, "
            "aligned shared FP16 keys, FP32 scales/weights and int32 lengths"
        )
    output = torch.empty(
        (query.shape[0], keys.shape[0]), device=query.device, dtype=torch.float32
    )
    if query.shape[0] and keys.shape[0]:
        query = query.half().contiguous()
        if query.data_ptr() % 16:
            raise ValueError("SM70 prefill scoring queries require 16-byte alignment")
        _index_score_prefill_kernel()(
            query,
            keys,
            key_scales.reshape(-1).contiguous(),
            weights.contiguous(),
            lengths.contiguous(),
            output,
        )
    return output
