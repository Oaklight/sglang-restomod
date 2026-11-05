"""Decode-shaped NVFP4 MoE GEMV: dispatch and extension loader.

The pinned Marlin kernel runs a 64-row M tile whatever M is, so at decode one
expert's 64-row tile carries about one real row and the call reads its weights
at 16% of the card's bandwidth (``sglang-restomod#201``). This module picks a
kernel that has no M tile for the shapes it can serve and returns ``None`` for
everything else, leaving the Marlin path untouched.

Eligibility is deliberately narrow, because the kernel decodes the repacked
weight layout by hand:

* the tensor is NVFP4 with no bias, zeros, ``g_idx`` or permutation;
* ``N % 64 == 0`` and ``N / 64`` is **odd** -- the repack tiles the output
  differently for an even number of 64-column tiles and only the odd ordering
  is decoded (``notes/nvfp4-marlin-layout-2026-11-05.md``);
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
K_SPLIT = 4

_LOCK = threading.Lock()
_MODULE = None
_FAILED = False
_WORKSPACE = None

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
    return size_m > 0 and top_k > 0


def _workspace_for(rows: int, size_n: int, device):
    global _WORKSPACE
    need = K_SPLIT * rows * size_n
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
        return None
    if not _is_cuda(a):
        return None
    if not supports(size_m, size_n, size_k, top_k, moe_block_size):
        return None
    module = _load()
    if module is None:
        return None

    rows = c.size(0)
    workspace = _workspace_for(rows, size_n, c.device)
    return module.decode_moe_gemv(
        a, c, b_q_weight, b_scales, global_scale, sorted_token_ids,
        expert_ids, topk_weights, rows, top_k, size_n, size_k,
        bool(mul_topk_weights), workspace, K_SPLIT,
    )


def _is_cuda(tensor) -> bool:
    import torch

    return isinstance(tensor, torch.Tensor) and tensor.is_cuda
