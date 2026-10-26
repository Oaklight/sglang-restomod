"""CUDA graph metadata delegation for the V100 full-attention backend.

The graph runner calls ``init_forward_metadata_out_graph`` instead of the
legacy capture/replay hooks. The V100 backend inherits a no-op from
``AttentionBackend``, so the Triton backend it delegates decode to never gets
its metadata built during capture, and ``forward_decode`` dies on
``swa_out_cache_loc``.

The fix was validated as a patch against the fork
(``patches/sglang-sxm2/graph-metadata.patch`` in the lab repository:
27B-FP8 decode 10.3 -> 30-35 tok/s). Here it is injected at import time, so the
engine checkout stays untouched, and it becomes a no-op once the engine ships
the method itself.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

METHOD = "init_forward_metadata_out_graph"


def _init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
    """Delegate capture/replay metadata to whichever backend owns the mode."""
    mode = forward_batch.forward_mode
    if (
        (
            self._uses_sm70_fp8_kv
            and not self._uses_native_linear_verify(mode)
            and not self._uses_fp8_prefill_scratch(mode)
        )
        or mode.is_decode_or_idle()
        or mode.is_draft_extend_v2()
        or (mode.is_target_verify() and not self._uses_native_linear_verify(mode))
    ):
        # Decode and the speculative paths run on the Triton backend.
        self._triton.init_forward_metadata_out_graph(forward_batch, in_capture)
        self.forward_metadata = None
        return

    if in_capture:
        self.init_forward_metadata_capture_cuda_graph(
            forward_batch.batch_size,
            forward_batch.input_ids.shape[0],
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.encoder_lens,
            mode,
            forward_batch.spec_info,
        )
    else:
        self.init_forward_metadata_replay_cuda_graph(
            forward_batch.batch_size,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.seq_lens_sum,
            forward_batch.encoder_lens,
            mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu,
        )


def install_on(cls) -> bool:
    """Add the delegation to ``cls`` unless it already defines it."""
    if METHOD in vars(cls):
        return False
    setattr(cls, METHOD, _init_forward_metadata_out_graph)
    return True


def install() -> bool:
    """Patch the engine's V100 full-attention backend.

    Raises when the engine has no such backend: selecting this component names
    a capability the engine does not provide, and failing at startup is better
    than a silent no-op.
    """
    from sglang.srt.layers.attention.flash_attn_v100_backend import (
        FlashAttnV100Backend,
    )

    added = install_on(FlashAttnV100Backend)
    if added:
        logger.info("attention_v100: installed %s on FlashAttnV100Backend", METHOD)
    else:
        logger.info(
            "attention_v100: FlashAttnV100Backend already defines %s; leaving it",
            METHOD,
        )
    return added
