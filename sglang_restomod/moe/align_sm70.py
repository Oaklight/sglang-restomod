"""Route the tiny-batch MoE alignment to the CUDA pair on SM70.

``moe_align_block_size`` builds the routing metadata every MoE backend consumes
(``sorted_token_ids``, ``expert_ids``, ``num_tokens_post_pad``). Upstream has
two implementations on CUDA and picks between them by input size:

* ``moe_align_small_numel``, a single-CTA Triton kernel that does O(numel**2)
  pairwise work and is admitted when ``numel <= SMALL_NUMEL_LIMIT`` (64),
  ``num_experts + 1 > 64`` and ``ignore_invalid_expert`` is false;
* the generic pair ``moe_align_block_size_kernel`` + ``count_and_sort_expert_tokens_kernel``.

The Triton kernel was added for one-CTA launch cost. On SM70 it loses badly at
decode shapes: for the Flash-Next TP4 NVFP4 MoE (numel 40, 512 experts,
block 8) it measures 29.3 us per call in-server, while the CUDA pair costs 5.0
+ 2.3 us for the same work. The Triton kernel is one CTA on an 80-SM card, so
its O(numel**2) loop runs at single-SM throughput while the whole GPU idles.

Setting ``SMALL_NUMEL_LIMIT`` to zero makes the Triton branch unreachable and
every caller falls through to the generic CUDA pair, which keeps its own
single-block fast path for ``num_experts + 1 <= 64``. The two paths produce the
same metadata; only the kernel changes.

Issue: sglang-restomod#95.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MODULE = "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size"


def register() -> None:
    """Disable the single-CTA Triton tiny-batch align path on this device."""
    import importlib

    module = importlib.import_module(_MODULE)
    limit = getattr(module, "SMALL_NUMEL_LIMIT", None)
    if limit is None:
        # Defined only on a CUDA build; nothing selects the Triton path without it.
        logger.warning(
            "sglang-restomod: %s has no SMALL_NUMEL_LIMIT; the tiny-batch "
            "align path is not present and nothing was changed",
            _MODULE,
        )
        return
    if limit == 0:
        return
    module.SMALL_NUMEL_LIMIT = 0
    logger.info(
        "sglang-restomod: tiny-batch MoE align now uses the CUDA pair "
        "(was SMALL_NUMEL_LIMIT=%s)", limit,
    )
