"""DSV4 attention on SM70 without flash_mla.

The pin's DSV4 attention is "flash_mla plus the metadata layer": every
``init_forward_metadata*`` builds its metadata from the KV pool's layout, never
from the kernel, and the HIP backend rebinds fourteen of them unbound. So a
platform without flash_mla needs to replace exactly two terminal calls:

* ``flash_mla_with_kvcache`` -- decode (and any forward the sparse-prefill
  branch declines), resolved once in ``__init__`` by ``resolve_flash_mla``;
* ``_forward_prefill_sparse`` -- the sparse prefill body, whose
  ``flash_mla_sparse_fwd`` and ``dequantize_k_cache_paged_dsv4`` are also
  unavailable.

This component replaces both with torch and changes nothing else. It also
forces ``SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`` off: that branch is on by
default and is selected before the dense fallback, so a backend supplying no
sparse implementation would never reach its own code.

Stage 1 is a correctness reference, not a fast path: it is a torch gather and
softmax over the 584-byte, page-256 paged cache, and it is expected to be slow.

Scope, and it is narrow on purpose: TP1, no speculative decoding, no CUDA
graphs, batch decode with chunked prefill. Anything outside raises with the
profile named rather than falling back to a path that would produce different
numbers.

Import is cheap: nothing is imported from sglang at module import, and
``register()`` only rebinds two names.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

BACKEND_MODULE = "sglang.srt.layers.attention.deepseek_v4_backend"
SPARSE_PREFILL_ENV = "SGLANG_OPT_FLASHMLA_SPARSE_PREFILL"

# The stage-1 profile. Everything here is asserted, not detected: a launch that
# selects this component with speculative decoding or CUDA graphs on gets a
# refusal naming the profile, because the torch reference has not been written
# for either and a silent fallback would produce plausible wrong numbers.
STAGE1_PROFILE = "tp1|no-spec|no-cuda-graph|batch-decode|chunked-prefill"

_HOOKED = False


class Stage1ProfileError(RuntimeError):
    """The configured launch profile is outside stage 1's scope."""


def _force_dense_prefill() -> None:
    """Turn the sparse prefill branch off before any forward can take it."""
    if os.environ.get(SPARSE_PREFILL_ENV, "").strip() in ("", "0", "false", "False"):
        return
    logger.warning(
        "attention_dsv4_sm70: %s was %r; forcing 0 because this backend supplies "
        "no sparse prefill implementation and the branch is selected before the "
        "dense fallback",
        SPARSE_PREFILL_ENV,
        os.environ[SPARSE_PREFILL_ENV],
    )
    os.environ[SPARSE_PREFILL_ENV] = "0"


def _gpu_is_sm70() -> bool:
    """Whether this process has an SM70 device. Kept separate so tests can fake it."""
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def register() -> None:
    """Install the torch decode/prefill replacements; idempotent."""
    import importlib

    from sglang_restomod import attest

    global _HOOKED
    if _HOOKED:
        return

    try:
        if not _gpu_is_sm70():
            import torch

            cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
            attest.inactive("attention_dsv4_sm70", f"needs SM70, found cap {cap}")
            return
    except Exception as exc:  # noqa: BLE001 - no torch is a state, not a crash
        attest.inactive("attention_dsv4_sm70", f"torch is not usable ({exc})")
        return

    try:
        module: Any = importlib.import_module(BACKEND_MODULE)
    except Exception as exc:  # noqa: BLE001 - absent module means the pin moved
        attest.inactive("attention_dsv4_sm70", f"{BACKEND_MODULE} is not importable ({exc})")
        return

    resolver = getattr(module, "resolve_flash_mla", None)
    if resolver is None:
        logger.error(
            "attention_dsv4_sm70: %s has no resolve_flash_mla; patch 0007 is not "
            "applied, and without it the base constructor imports flash_mla and "
            "cannot run here",
            BACKEND_MODULE,
        )
        attest.inactive("attention_dsv4_sm70", "patch 0007 (resolve_flash_mla) is not applied")
        return

    from . import torch_attn

    _force_dense_prefill()

    # resolve_flash_mla receives the pool's interleaved flag; the torch
    # replacement needs it because the two page arrangements differ.
    module.resolve_flash_mla = lambda swa_interleaved: (
        functools.partial(torch_attn.paged_mla, _interleaved=bool(swa_interleaved)),
        0,
    )
    backend = getattr(module, "DeepseekV4AttnBackend", None)
    if backend is None:
        attest.inactive("attention_dsv4_sm70", f"{BACKEND_MODULE} has no DeepseekV4AttnBackend")
        return
    backend._forward_prefill_sparse = torch_attn.forward_prefill_dense

    _HOOKED = True
    logger.info(
        "attention_dsv4_sm70: decode and dense prefill route to torch; %s forced off; profile %s",
        SPARSE_PREFILL_ENV,
        STAGE1_PROFILE,
    )
