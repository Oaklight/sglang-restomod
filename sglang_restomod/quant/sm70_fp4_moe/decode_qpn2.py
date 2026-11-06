"""QPN2 decode GEMV: dispatch and extension loader.

Sibling of ``decode_gemv``: same routing, same output layout, different weight
buffer. QPN2 keeps the E4M3 group scales inside the weight tensor, so this path
takes no ``b_scales``; the global scale is the raw per-expert FP32 value.

Eligibility is the decode regime the kernel covers:

* a QPN2 weight buffer (uint8) with ``N % 32 == 0`` and ``K % 16 == 0``;
* ``size_m <= 8`` and a route block of 8, which is what the align kernel emits
  at decode. A prefill chunk is refused: the QPN2 prefill reader is not
  integrated yet, and the Marlin reader cannot read this layout either, so the
  caller raises rather than computing garbage;
* contiguous ``a`` and ``c``, and one route per block or more, both handled by
  the kernel.

Set ``SM70_MOE_DECODE_QPN2=0`` to disable the kernel (useful when diagnosing;
it does not make a QPN2 buffer readable by Marlin).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_ENABLE = "SM70_MOE_DECODE_QPN2"
BLOCK_M = 8
MAX_M = 8
# Split-K 4 for the gate_up shape (160 groups); the down shape has 10 groups
# and a wider reduce than weight stream, so it keeps one split.
SPLIT_K_SHORT = 1
SPLIT_K_LONG = 4
SPLIT_K_LONG_MIN_GROUPS = 64
_WORKSPACE_ROUTE_TILES = 1024

_LOCK = threading.Lock()
_MODULE = None
_FAILED = False
_WORKSPACE = None

_CSRC = Path(__file__).resolve().parent / "csrc" / "decode_gemv_qpn2.cu"


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "1") != "0"


def split_k_for(size_k: int) -> int:
    return SPLIT_K_LONG if size_k // 16 >= SPLIT_K_LONG_MIN_GROUPS else SPLIT_K_SHORT


def _load():
    """Build and cache the extension, once per process."""
    global _MODULE, _FAILED
    with _LOCK:
        if _MODULE is not None or _FAILED:
            return _MODULE
        try:
            from torch.utils.cpp_extension import load

            _MODULE = load(
                name="sm70_decode_moe_gemv_qpn2",
                sources=[str(_CSRC)],
                extra_cuda_cflags=[
                    "-O3", "--use_fast_math", "-std=c++17",
                    "--expt-relaxed-constexpr",
                ],
                verbose=False,
            )
            logger.info("SM70 NVFP4 MoE: QPN2 decode GEMV extension ready")
        except Exception:  # pragma: no cover - build failure is a fallback
            _FAILED = True
            logger.warning(
                "SM70 NVFP4 MoE: QPN2 decode GEMV unavailable", exc_info=True
            )
        return _MODULE


def supports(size_m: int, size_n: int, size_k: int, top_k: int,
             moe_block_size: int) -> bool:
    """Whether the QPN2 decode kernel covers this shape."""
    if not enabled():
        return False
    if moe_block_size != BLOCK_M:
        return False
    if size_n % 32 or size_k % 16:
        return False
    if size_m > MAX_M or top_k <= 0:
        return False
    return True


def _workspace_for(rows: int, size_n: int, size_k: int, device):
    global _WORKSPACE
    # Size for the largest capture batch up front: a resize inside a
    # torch.cuda.graph capture is what turns into an illegal memory access.
    planes = 4 * split_k_for(size_k)
    need = planes * max(rows, _WORKSPACE_ROUTE_TILES) * size_n
    if _WORKSPACE is None or _WORKSPACE.numel() < need or _WORKSPACE.device != device:
        import torch

        _WORKSPACE = torch.empty(need, dtype=torch.float32, device=device)
    return _WORKSPACE[:need]


def maybe_call(
    a,
    c,
    b_q_weight,
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
    """Run the QPN2 decode kernel, or return ``None`` when it cannot."""
    import torch

    if c is None or global_scale is None or topk_weights is None:
        logger.info("qpn2 gemv skipped: missing tensor")
        return None
    if not (isinstance(a, torch.Tensor) and a.is_cuda):
        logger.info("qpn2 gemv skipped: not cuda")
        return None
    if not supports(size_m, size_n, size_k, top_k, moe_block_size):
        logger.info(
            "qpn2 gemv skipped: unsupported m=%d n=%d k=%d topk=%d block=%d",
            size_m, size_n, size_k, top_k, moe_block_size,
        )
        return None
    module = _load()
    if module is None:
        return None
    rows = c.size(0)
    if not (a.is_contiguous() and c.is_contiguous()):
        logger.info("qpn2 gemv skipped: non-contiguous a or c")
        return None
    if topk_weights.numel() < rows:
        logger.info(
            "qpn2 gemv skipped: topk_weights %d < rows %d",
            topk_weights.numel(), rows,
        )
        return None
    if not (num_tokens_post_padded.is_cuda and num_tokens_post_padded.numel() >= 1):
        logger.info("qpn2 gemv skipped: no padded-token count")
        return None
    split_k = split_k_for(size_k)
    workspace = _workspace_for(rows, size_n, size_k, c.device)
    logger.info(
        "qpn2 gemv: rows=%d tiles=%d n=%d k=%d split_k=%d",
        rows, expert_ids.size(0), size_n, size_k, split_k,
    )
    return module.decode_moe_gemv_qpn2(
        a, c, b_q_weight, global_scale, sorted_token_ids, expert_ids,
        topk_weights, rows, top_k, size_n, size_k, bool(mul_topk_weights),
        workspace, split_k, num_tokens_post_padded,
    )
