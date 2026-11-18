"""One-launch ``fp8_paged_mqa_logits`` for SM70's small decode batches.

The DSV4 indexer scores each compressed KV token against the query heads and
keeps a top-k; the pin's only non-flash_mla implementation is an eight-launch
torch chain that materializes a 67 MB bf16 cast and a 33.6 MB fp32 score tensor
which `relu`, the head weight and the head sum each rewrite. This kernel is one
launch, reads each KV byte once, keeps the dot in fp32 registers, and writes
only the ``[B, max_seq_len]`` result.

**Where it wins, measured** (``docs/runs/2026-11-17-indexer-kernel-first-run.md``,
captured graph, replay-only, one V100): 3.51x at B=1, 2.48x at B=2, 2.25x at
B=4, then ~1.95x from B=8 up. Agreement is exact to 1.2e-7 against a 2e-2
tolerance, and the top-64 index set is identical on every row at every batch.
The **B=32 gate (>= 3x) is not met and is not claimed**; reaching it needs a
real MMA, which Triton does not emit on sm70 (see PROVENANCE and the run page).

The dispatch edge is **noise-limited, not resolved**. B=8 measured 2.04x and
1.95x minutes apart on the same 2.0x bar, so the wired band is B <= 4, where
2.25x has margin. More repeats could move the edge to 8; a TileLang rewrite
that clears 3x at B=32 would remove the edge entirely.

**Worth, labelled by arm** — both are single-op inferences from a synthetic
call, not served numbers. At B<=4: 0.2647 -> 0.1178 ms per call, ~0.147 ms
across the eight index layers, **~1.2 ms/step**. At B=32, if the large-batch
arm were ever shipped: 10.74 -> 5.51 ms/step, ~5.2 ms/step — and that one needs
TileLang, not this kernel.

Import is cheap: no torch and no triton at module import. ``register()``
installs a thunk into the shim, and the Triton module is imported on first call.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SHIM_MODULE = "sglang_restomod.kernels.deep_gemm_sm70"

_entry = None


def _load_entry():
    """Import the Triton module once, on first call rather than at register()."""
    global _entry
    if _entry is None:
        from .kernel import fused_paged_mqa_logits

        _entry = fused_paged_mqa_logits
    return _entry


def _thunk(q_fp8, kvcache_fp8, weight, seq_lens, page_table, metadata,
           max_seq_len, clean_logits=False) -> Any:
    """The shim's signature, with the kernel imported only when it is called."""
    return _load_entry()(q_fp8, kvcache_fp8, weight, seq_lens, page_table,
                         metadata, max_seq_len, clean_logits)


def register() -> None:
    """Install the kernel as the shim's ``fused`` arm for paged logits."""
    import importlib
    import sys

    shim = sys.modules.get("deep_gemm")
    if shim is None or not getattr(shim, "__restomod_substitute__", False):
        logger.info(
            "sglang-restomod: no deep_gemm stand-in is installed, so the paged "
            "logits kernel has no entry to bind; nothing was changed"
        )
        return
    installer = getattr(importlib.import_module(SHIM_MODULE),
                        "install_fused_paged_mqa_logits", None)
    if installer is None:
        logger.warning(
            "sglang-restomod: %s has no install_fused_paged_mqa_logits; the pin "
            "moved and the paged logits kernel was not bound", SHIM_MODULE,
        )
        return
    installer(_thunk)
    logger.info(
        "sglang-restomod: paged logits kernel bound as the shim's fused arm "
        "(auto dispatches to it at B <= 4)"
    )
