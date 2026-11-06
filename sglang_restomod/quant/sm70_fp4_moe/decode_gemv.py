"""Decode-shaped NVFP4 MoE GEMV: dispatch and extension loader.

The pinned Marlin kernel runs a 64-row M tile whatever M is, so at decode one
expert's 64-row tile carries about one real row and the call reads its weights
at 16% of the card's bandwidth (``sglang-restomod#201``). This module picks a
kernel that has no M tile for the shapes it can serve and returns ``None`` for
everything else, leaving the Marlin path untouched.

Eligibility is deliberately narrow, because the kernel decodes the repacked
weight layout by hand:

* the tensor is NVFP4 with no bias, zeros, ``g_idx`` or permutation;
* ``N % 64 == 0`` and ``N / 64`` is **odd**. Both orderings are decoded now
  (``notes/nvfp4-marlin-layout-2026-11-05.md``), but the even case is the down
  projection, where ``k / CtaK = 5`` leaves too little work to amortise the
  top-k reduction: 49.3 us against Marlin's 48.3 us. So the restriction is
  deliberate, not a limitation;
* ``K % 16 == 0`` and the route tile is 8 rows, which is what the align kernel
  emits at decode;
* the output is routed by ``sorted_token_ids`` as the Marlin call expects.

Set ``SM70_MOE_DECODE_GEMV=0`` to force the Marlin path.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_ENABLE = "SM70_MOE_DECODE_GEMV"
BLOCK_M = 8
# Largest token count the decode path is admitted for; the same cut
# ``method._select_split_k`` uses.
MAX_M = 8
K_SPLIT = 4
# Workspace is sized for this many route tiles, above any decode capture batch.
_WORKSPACE_ROUTE_TILES = 1024

_LOCK = threading.Lock()
_MODULE = None
_FAILED = False
_WORKSPACE = None
_DEBUGGED: list[int] = []

_CSRC = Path(__file__).resolve().parent / "csrc" / "decode_gemv.cu"


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "1") != "0"


def _load():
    """Build and cache the extension, once per process."""
    global _MODULE, _FAILED
    with _LOCK:
        if _MODULE is not None or _FAILED:
            return _MODULE
        try:
            from torch.utils.cpp_extension import load

            _MODULE = load(
                name="sm70_decode_moe_gemv",
                sources=[str(_CSRC)],
                extra_cuda_cflags=[
                    "-O3", "--use_fast_math", "-std=c++17",
                    "--expt-relaxed-constexpr",
                ],
                verbose=False,
            )
            logger.info("SM70 NVFP4 MoE: decode GEMV extension ready")
        except Exception:  # pragma: no cover - build failure is a fallback
            _FAILED = True
            logger.warning(
                "SM70 NVFP4 MoE: decode GEMV unavailable, using Marlin",
                exc_info=True,
            )
        return _MODULE


def supports(size_m: int, size_n: int, size_k: int, top_k: int,
             moe_block_size: int) -> bool:
    """Whether the decode kernel covers this shape."""
    if not enabled():
        return False
    if moe_block_size != BLOCK_M:
        return False
    if size_n % 64 != 0 or (size_n // 64) % 2 != 1:
        return False
    if size_k % 16 != 0:
        return False
    # Split-K divides the K groups four ways.
    k_groups = size_k // 16
    if k_groups % K_SPLIT != 0:
        return False
    # Decode only. The kernel has no M tile and every number behind it is a
    # decode number; a prefill chunk routes many tokens per block, which is a
    # regime Marlin already covers. `size_m` is the token count for the gate_up
    # call and `tokens * top_k` for the down call, which this shape rejects on
    # N before the cut matters.
    if size_m > MAX_M or top_k <= 0:
        return False
    return True


def _workspace_for(rows: int, size_n: int, device):
    global _WORKSPACE
    # Size for the largest capture batch up front: a resize inside a
    # torch.cuda.graph capture is what turns into an illegal memory access
    # on the server, because each batch size is captured exactly once.
    need = K_SPLIT * max(rows, _WORKSPACE_ROUTE_TILES) * size_n
    if _WORKSPACE is None or _WORKSPACE.numel() < need or _WORKSPACE.device != device:
        import torch

        _WORKSPACE = torch.empty(need, dtype=torch.float32, device=device)
    return _WORKSPACE[:need]


def maybe_call(
    a,
    c,
    b_q_weight,
    b_scales,
    global_scale,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    topk_weights,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    size_m: int,
    size_n: int,
    size_k: int,
):
    """Run the decode kernel, or return ``None`` to leave the call to Marlin."""
    if c is None or global_scale is None or topk_weights is None:
        logger.info("decode gemv skipped: missing tensor")
        return None
    if not _is_cuda(a):
        logger.info("decode gemv skipped: not cuda")
        return None
    if not supports(size_m, size_n, size_k, top_k, moe_block_size):
        logger.info("decode gemv skipped: unsupported m=%d n=%d k=%d topk=%d block=%d",
                    size_m, size_n, size_k, top_k, moe_block_size)
        return None
    module = _load()
    if module is None:
        logger.info("decode gemv skipped: extension unavailable")
        return None

    rows = c.size(0)
    # ``rows`` is the route count, ``size_m * top_k``: the align kernel emits
    # route indices (``token * top_k + slot``) and the kernel recovers the
    # activation row as ``row / top_k``. The reference Marlin kernel reads the
    # same metadata, so the two only agree when every block holds one route --
    # which is what the decode shape produces. Refuse anything else rather
    # than compute garbage.
    # The kernel addresses a and c as dense row-major buffers. A sliced or
    # otherwise strided view would write outside the tensor.
    if not (a.is_contiguous() and c.is_contiguous()):
        logger.info("decode gemv skipped: non-contiguous a=%s c=%s",
                    a.is_contiguous(), c.is_contiguous())
        return None
    # The reduce kernel indexes topk_weights by route, so it needs one entry
    # per route. The server passes ``topk_weights`` shaped ``[size_m, top_k]``,
    # which flattens to exactly that.
    if topk_weights.numel() < rows:
        logger.info("decode gemv skipped: topk_weights %d < rows %d",
                    topk_weights.numel(), rows)
        return None
    # `expert_ids` is allocated for the worst case and the align kernel writes
    # only the blocks it fills; the rest of the tensor is uninitialised. The
    # padded token count is the authority on how many blocks exist, and the
    # kernel bounds itself on it before it reads `expert_ids`.
    if not (_is_cuda(num_tokens_post_padded) and num_tokens_post_padded.numel() >= 1):
        logger.info("decode gemv skipped: no padded-token count")
        return None
    if os.environ.get("SM70_MOE_GEMV_DEBUG") == "1" and not _DEBUGGED:
        _DEBUGGED.append(1)
        import torch as _t

        if not _t.cuda.is_current_stream_capturing():
            logger.info(
                "decode gemv inputs: a=%s%s c=%s%s bq=%s bs=%s%s gs=%s%s "
                "si=[%d,%d] ei=[%d,%d] tw=%s topk=%d rows=%d",
                tuple(a.shape), a.dtype, tuple(c.shape), c.dtype,
                tuple(b_q_weight.shape), tuple(b_scales.shape), b_scales.dtype,
                tuple(global_scale.shape), global_scale.dtype,
                int(sorted_token_ids.min()), int(sorted_token_ids.max()),
                int(expert_ids.min()), int(expert_ids.max()),
                tuple(topk_weights.shape), top_k, rows,
            )
    logger.info("decode gemv: rows=%d tiles=%d n=%d k=%d", rows, expert_ids.size(0), size_n, size_k)
    workspace = _workspace_for(rows, size_n, c.device)
    return module.decode_moe_gemv(
        a, c, b_q_weight, b_scales, global_scale, sorted_token_ids,
        expert_ids, topk_weights, rows, top_k, size_n, size_k,
        bool(mul_topk_weights), workspace, K_SPLIT, num_tokens_post_padded,
    )


def _is_cuda(tensor) -> bool:
    import torch

    return isinstance(tensor, torch.Tensor) and tensor.is_cuda
