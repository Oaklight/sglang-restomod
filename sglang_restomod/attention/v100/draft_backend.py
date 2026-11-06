"""Serve an EAGLE/NEXTN draft worker with this component's backend.

The pin builds a draft worker's attention backends by looking the configured
name up in a fixed map (``speculative/draft_utils.py``) that does not hold
``tilelang_fa_v100``, so a 27B MTP launch has to name ``triton`` for the draft.
That is what puts the draft extend on the Triton kernel: at 32k it costs 96.7 ms
of a 168 ms step, against the 10.7 ms the native grouped verifier takes for the
target verify (issue #75).

This module makes the name resolve for the draft:

* draft **extend** gets the component backend. That is the pass over the
  accepted tokens, and it is the expensive one.
* draft **decode** gets the component's native multi-step container
  (``draft_decode``) when the model, KV dtype and page layout fit it; otherwise
  it stays on Triton. The native container keeps the pin's per-step state and
  hands the split-KV kernel a page table per step.

Installing changes nothing unless the launch asks for
``--speculative-draft-attention-backend tilelang_fa_v100``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

FLAG = "_restomod_v100_draft"
DRAFT_DECODE_FALLBACK = "triton"


def install_on(factory, build, wrap, build_native=None) -> bool:
    """Patch ``DraftBackendFactory``. Returns True if anything changed.

    ``build(runner)`` returns the component backend, ``wrap(runner, backend)``
    is the engine's draft-extend wrapper, and ``build_native(runner, topk,
    steps, stamp)`` returns the native draft-decode container or None. The
    last is injected so the patch can be tested without importing sglang.
    """
    from .backend_registry import FLASH_ATTN_V100

    if getattr(factory, FLAG, False):
        return False

    original_extend = factory.create_draft_extend_backend
    original_decode = factory.create_decode_backend

    def create_draft_extend_backend(self):
        if self.draft_attn_backend != FLASH_ATTN_V100 or self.qsa_profile is not None:
            return original_extend(self)
        backend = build(self.draft_model_runner)
        if backend is None:
            return None
        wrapped = wrap(self.draft_model_runner, backend)
        if wrapped is not None:
            wrapped.prefill_attention_backend_str = FLASH_ATTN_V100
            wrapped.decode_attention_backend_str = FLASH_ATTN_V100
        return wrapped

    def create_decode_backend(self):
        if self.draft_attn_backend != FLASH_ATTN_V100 or self.qsa_profile is not None:
            return original_decode(self)
        if self.speculative_num_steps > 1:
            builder = build_native
            if builder is None:
                from .draft_decode import build_native as builder
            native = builder(
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
                FLASH_ATTN_V100,
            )
            if native is not None:
                return native
        saved = self.draft_attn_backend
        self.draft_attn_backend = DRAFT_DECODE_FALLBACK
        try:
            return original_decode(self)
        finally:
            self.draft_attn_backend = saved

    factory.create_draft_extend_backend = create_draft_extend_backend
    factory.create_decode_backend = create_decode_backend
    setattr(factory, FLAG, True)
    return True


def install_topk_fallback() -> bool:
    """Point flashinfer's radix top-k at torch on sm70.

    ``sglang.srt.models.dflash`` binds ``flashinfer.top_k`` at import time and
    only takes its own ``torch.topk`` fallback when that import fails. On sm70 the
    import succeeds and the kernel refuses at call time
    (``FlashInfer requires GPUs with sm75 or higher``), so replace the module
    attribute before the model module is imported.

    The replacement is process-wide because the binding happens at import time.
    Three other callers share the symbol: ``speculative/uno_tree.py``,
    ``speculative/uno_utils.py`` and ``layers/attention/dsa/dsa_topk_backend.py``.
    On sm70 the kernel refuses there too, so the fallback replaces a failure; on
    sm75 and newer the original is called unchanged.
    """
    try:
        import flashinfer
        from sglang.srt.utils.common import get_device_sm
    except Exception as exc:  # noqa: BLE001 - report, never abort loading
        logger.warning("attention_v100: top-k fallback skipped: %s", exc)
        return False
    if getattr(flashinfer, "_restomod_sm70_topk", False):
        return False
    original = getattr(flashinfer, "top_k", None)
    if original is None:
        return False

    def top_k(scores, k, *args, **kwargs):
        if get_device_sm() < 75:
            import torch

            return torch.topk(scores, k, dim=-1)
        return original(scores, k, *args, **kwargs)

    flashinfer.top_k = top_k
    flashinfer._restomod_sm70_topk = True
    logger.info("attention_v100: flashinfer top_k falls back to torch on sm70")
    return True


def install() -> bool:
    """Component hook: patch the engine's draft factory if it is importable."""
    try:
        from sglang.srt.layers.attention.attention_registry import (
            attn_backend_wrapper_for_draft_extend,
        )
        from sglang.srt.speculative import draft_utils
    except Exception as exc:  # noqa: BLE001 - report, never abort loading
        logger.warning("attention_v100: draft backend hook skipped: %s", exc)
        return False

    from .backend_registry import FLASH_ATTN_V100, build_backend

    install_topk_fallback()
    changed = install_on(
        draft_utils.DraftBackendFactory,
        lambda runner: build_backend(FLASH_ATTN_V100, runner),
        attn_backend_wrapper_for_draft_extend,
    )
    if changed:
        logger.info("attention_v100: a draft worker can now select tilelang_fa_v100")
    return changed
