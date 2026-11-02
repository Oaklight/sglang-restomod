"""SM70 GDN prefill through the TileLang chunked kernel.

The pin ships three GDN prefill backends and none of them is fast on Volta:
``triton`` (the default), ``cutedsl`` and ``flashinfer``. A trace of the 27B
prefill shows the Triton chunk kernels taking 63.9% of every 4096-token
prefill chunk, and the fork still carries a TileLang implementation that
computes the same thing 23x faster on the same shapes.

This module binds that kernel behind the pin's own ``TritonGDNKernel.extend``,
so the engine keeps its dispatcher, its argument parsing and its fallbacks: the
component only answers the prefill call, and only when the call matches what
the TileLang kernel supports. Everything else -- decode, verify, non-FP16
activations, a non-SM70 device, a short batch, or a caller that asks for
``inplace_update=False`` -- falls through to the Triton kernel untouched.

The first ``extend`` call of a process always runs on Triton: it is the call
that installs the tracking guard (see ``_ensure_track_guard``), and until that
guard exists the kernel cannot tell whether its per-chunk states will be read.
The server's own warmup takes that hit, so a served request starts on the fast
path; a launch with ``--skip-server-warmup`` pays Triton for its first chunk.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

FLAG = "_restomod_gdn_tilelang"
TRACK_FLAG = "_restomod_gdn_tilelang_track"

# Set for the duration of one `GDNAttnBackend.forward_extend` call when that call
# needs the per-chunk state snapshots. Single-threaded per scheduler process,
# and the wrapper restores the previous value in a `finally`.
_TRACK_ACTIVE = False
_GUARD_INSTALLED = False

# Below this many tokens the recurrence is too short for the chunked kernel's
# setup to pay off; Triton wins and skips a JIT compile.
MIN_TOKENS_ENV = "SGLANG_V100_GDN_TILELANG_MIN_TOKENS"
DEFAULT_MIN_TOKENS = 256

_KEY_DIM = 128
_VALUE_DIM = 128


def _min_tokens() -> int:
    raw = os.environ.get(MIN_TOKENS_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MIN_TOKENS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{MIN_TOKENS_ENV} must be an integer, got {raw!r}.") from exc
    if value < 0:
        raise ValueError(f"{MIN_TOKENS_ENV} must be non-negative, got {value}.")
    return value


def _unsqueeze_gate(tensor):
    """The pin's gating kernel returns ``[tokens, heads]``; the TileLang
    kernels declare ``[1, tokens, heads]``."""
    return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor


def _supported(q, k, v, ssm_states, inplace_update: bool) -> bool:
    import torch

    if not inplace_update:
        return False
    if not torch.cuda.is_available() or not q.is_cuda:
        return False
    if torch.cuda.get_device_capability(q.device) != (7, 0):
        return False
    if q.dtype is not torch.float16 or k.dtype is not torch.float16:
        return False
    if v.dtype is not torch.float16:
        return False
    if q.ndim != 4 or q.shape[0] != 1:
        return False
    if q.shape[-1] != _KEY_DIM or k.shape[-1] != _KEY_DIM:
        return False
    if v.ndim != 4 or v.shape[-1] != _VALUE_DIM:
        return False
    # The pool is [slot, heads, value_dim, key_dim], so the last axis is K.
    if ssm_states.shape[-1] != _KEY_DIM or ssm_states.shape[-2] != _VALUE_DIM:
        return False
    return q.shape[1] >= _min_tokens()


def _needs_engine_features(kwargs) -> bool:
    """True when the call wants something only the engine's Triton kernel does.

    The pin's GDN backend lets a caller hand in an output buffer and request
    per-chunk state tracking. The TileLang kernel does neither, so either sends
    the call to Triton. Mamba state checkpoints are handled separately: the
    TileLang kernel can produce them, so ``_tracking_active()`` turns them on
    instead of falling back.
    """
    if kwargs.get("store_checkpoints"):
        return True
    if kwargs.get("state_checkpoint_cu_starts") is not None:
        return True
    if kwargs.get("num_state_checkpoints"):
        return True
    if kwargs.get("track_state") is not None:
        return True
    return kwargs.get("output") is not None


def _tracking_active() -> bool:
    """True while a prefill needs the per-chunk state snapshots ``h``.

    The pin's GDN backend reads them from the third return value of ``extend``
    for requests whose mamba radix cache rows are unaligned to the checkpoint
    interval. That decision lives in ``forward_metadata``, which ``extend``
    never sees, so ``install_track_guard`` records it around the call.
    """
    return _TRACK_ACTIVE


def _ensure_track_guard() -> bool:
    """Install the ``forward_extend`` wrapper on first use.

    Returns False until the wrapper is in place, so the call that triggers the
    import falls back to Triton: its tracking state is unknown to us.
    """
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return True
    try:
        from sglang.srt.layers.attention.linear import gdn_backend
    except Exception as exc:  # noqa: BLE001 - no engine means no guard
        logger.warning(
            "sglang-restomod: GDN tracking guard unavailable, staying on Triton: %s", exc
        )
        return False

    _GUARD_INSTALLED = True
    install_track_guard(gdn_backend.GDNAttnBackend)
    logger.info("sglang-restomod: GDN prefill tracking guard installed")
    return False


def _tracking_needed(backend) -> bool:
    """Mirror the condition the engine uses to read the third return value.

    ``gdn_backend.py`` requires a track mask, at least one unaligned row, and a
    kernel that cannot snapshot them itself. The snapshot term is included even
    though ``_needs_engine_features`` would already divert such a call, because
    a reader of this predicate has to be able to compare it with the engine's.
    """
    meta = getattr(backend, "forward_metadata", None)
    if meta is None or not getattr(meta, "has_mamba_track_mask", False):
        return False
    dispatcher = getattr(backend, "kernel_dispatcher", None)
    kernel = getattr(dispatcher, "extend_kernel", None)
    if getattr(kernel, "supports_track_state_snapshot", False):
        return False
    source = getattr(meta, "track_ssm_h_src", None)
    return source is not None and source.numel() > 0


def tilelang_extend(
    q,
    k,
    v,
    g,
    beta,
    *,
    ssm_states,
    cache_indices,
    query_start_loc,
    inplace_update: bool = True,
    **kwargs,
):
    """Run the chunked TileLang GDN kernel, or return ``None`` to fall back."""
    if _needs_engine_features(kwargs):
        return None
    if not _ensure_track_guard():
        return None
    if not _supported(q, k, v, ssm_states, inplace_update):
        return None

    from .gdn_chunked_tilelang import chunked_gdn_sm70

    store_checkpoints = _tracking_active()
    output, checkpoints = chunked_gdn_sm70(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        _unsqueeze_gate(g),
        _unsqueeze_gate(beta),
        scale=q.shape[-1] ** -0.5,
        state=ssm_states,
        state_indices=cache_indices,
        cu_seqlens=query_start_loc,
        store_checkpoints=store_checkpoints,
    )
    return output, None, (checkpoints if store_checkpoints else None)


def install_on(kernel_cls) -> bool:
    """Wrap ``kernel_cls.extend`` so the TileLang path gets first refusal.

    ``extend`` resolves ``tilelang_extend`` from this module at call time, so a
    test can replace it without touching the engine.
    """
    if getattr(kernel_cls, FLAG, False):
        return False

    original = kernel_cls.extend

    def extend(self, q, k, v, g, beta, **kwargs):
        result = tilelang_extend(q, k, v, g, beta, **kwargs)
        if result is not None:
            return result
        return original(self, q, k, v, g, beta, **kwargs)

    kernel_cls.extend = extend
    setattr(kernel_cls, FLAG, True)
    return True


def install_track_guard(backend_cls) -> bool:
    """Record, for the duration of each prefill, whether it needs snapshots.

    ``GDNAttnBackend.forward_extend`` reads the track mask from
    ``forward_metadata`` and then calls ``extend`` without it, so the kernel
    cannot tell whether the third return value will be read. This wrapper
    answers that question before the call and clears it afterwards.
    """
    if getattr(backend_cls, TRACK_FLAG, False):
        return False

    original = backend_cls.forward_extend

    def forward_extend(self, *args, **kwargs):
        global _TRACK_ACTIVE
        previous = _TRACK_ACTIVE
        _TRACK_ACTIVE = _tracking_needed(self)
        try:
            return original(self, *args, **kwargs)
        finally:
            _TRACK_ACTIVE = previous

    backend_cls.forward_extend = forward_extend
    setattr(backend_cls, TRACK_FLAG, True)
    return True


def install() -> bool:
    """Route the pin's GDN prefill through the TileLang kernel on SM70.

    Deliberately does not import ``gdn_backend``: the package entry point runs
    components before it installs the SM70 overrides, and importing the model
    runner this early binds the original ``sm75`` capability gate in
    ``model_runner`` for the rest of the process. The track guard is installed
    on the first prefill instead, once the engine is fully loaded.
    """
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    installed = install_on(TritonGDNKernel)
    if installed:
        logger.info("sglang-restomod: GDN prefill routed to the SM70 TileLang kernel")
    return installed
