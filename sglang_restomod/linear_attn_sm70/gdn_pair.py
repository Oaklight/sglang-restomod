"""SM70 TileLang replacement for the two Triton GDN prefill kernels.

The pin's chunked gated-delta-rule prefill runs four Triton kernels. Two of
them dominate the prefill: ``chunk_o.py::chunk_fwd_o`` (the chunk output) and
``chunk_delta_h.py::chunk_gated_delta_rule_fwd_h`` (the sequential state
pass). Triton 3.5 has no SM70 MMA lowering, so both compile to scalar FMA and
spill; on a V100 the TileLang pair in ``gdn_pair_tilelang`` runs the same math
108x and 33x faster at 4096 tokens (sm70-kernels d75987a).

This module rebinds those two functions. ``fla.chunk`` binds both names at
import time, so patching the source modules alone is a silent no-op: every
already-imported holder is rebound and the result is asserted to be reachable
from ``fla.chunk`` before the component reports success.

A call is served by TileLang only when it is bit-compatible with the vendored
kernel. Every other call -- varlen, a token count that is not a multiple of
64, a per-channel gate, a checkpoint snapshot, a scale the kernel does not
hardcode, a head dim the tiles do not cover -- is refused and runs the
original Triton function unchanged. Refusal is a fallback, never an error and
never an approximation.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

logger = logging.getLogger(__name__)

ENV_KILL = "SM70_GDN_TILELANG"

BT = 64
_BV_O = 64
_BV_H = 32

_INSTALLED = False
_KERNELS = None
_KERNEL_IMPORT_FAILED = False
_ORIGINALS: dict[str, object] = {}
_NOTES: set[str] = set()


def enabled() -> bool:
    """False when ``SM70_GDN_TILELANG`` asks for the Triton control arm.

    The comparison harness flips this switch between arms, so it has to leave
    the process byte-identical to one that never selected the component: when
    it is off nothing is rebound and no kernel is imported.
    """
    raw = os.environ.get(ENV_KILL, "").strip().lower()
    if not raw:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{ENV_KILL} must be 0/1, got {raw!r}.")


def _note(key: str, level: int, message: str, *args) -> None:
    """Log once per distinct key; a per-layer refusal must not flood the log."""
    if key in _NOTES:
        return
    _NOTES.add(key)
    logger.log(level, "sglang-restomod: GDN pair " + message, *args)


def _load_kernels():
    """Import the two vendored kernel modules on first use; ``None`` without TileLang.

    ``tilelang`` is an engine dependency, not a declared one of this package, so a
    host can enable the component without it. The kernels live in the source
    modules synced by ``scripts/sync_kernels.py``, so nothing is re-exported here.
    """
    global _KERNELS, _KERNEL_IMPORT_FAILED
    if _KERNELS is not None or _KERNEL_IMPORT_FAILED:
        return _KERNELS
    try:
        from ._gdn_chunk_h import chunk_h_kernel
        from ._gdn_chunk_o import chunk_o_kernel
    except Exception as exc:  # noqa: BLE001 - no kernel means Triton is correct
        _KERNEL_IMPORT_FAILED = True
        logger.warning("sglang-restomod: TileLang GDN pair unavailable, staying on Triton: %s", exc)
        return None
    _KERNELS = (chunk_o_kernel, chunk_h_kernel)
    return _KERNELS


@lru_cache(maxsize=8)
def _chunk_o_for(b, t_len, h, hg, k, v):
    kernels = _load_kernels()
    if kernels is None:
        return None
    return kernels[0](b, t_len, h, hg, k, v, BT, _BV_O, 128)


@lru_cache(maxsize=8)
def _chunk_h_for(b, t_len, h, hg, k, v):
    kernels = _load_kernels()
    if kernels is None:
        return None
    return kernels[1](b, t_len, h, hg, k, v, BT, _BV_H, 128)


def _cuda_device_id(tensor) -> int:
    return tensor.device.index if tensor.device.index is not None else 0


@lru_cache(maxsize=8)
def _sm70(index: int) -> bool:
    import torch

    return torch.cuda.get_device_capability(index) == (7, 0)


def _device_reason(tensor, what: str) -> str:
    """Device half of the gate, kept out of the shape half so both are testable."""
    import torch

    if not torch.cuda.is_available() or tensor.device.type != "cuda":
        return f"refused {what}: not on CUDA"
    if not _sm70(_cuda_device_id(tensor)):
        return f"refused {what}: not an SM70 device"
    return ""


def _refusal_key(reason: str) -> str:
    """Drop the per-shape value so one refusal logs one line, not one per shape."""
    head = reason.split("(", 1)[0]
    return head.split("=", 1)[0].strip()


# --------------------------------------------------------------------------
# chunk_o
# --------------------------------------------------------------------------


def _gate_chunk_o(q, k, v, h, g, scale, cu_seqlens, chunk_size) -> str:
    """Return "" when the vendored chunk_o kernel computes this call exactly."""
    import torch

    if cu_seqlens is not None:
        return "refused chunk_o: varlen (cu_seqlens is set)"
    if chunk_size != BT:
        return f"refused chunk_o: chunk_size={chunk_size}, the kernel is BT={BT}"
    if g is None:
        return "refused chunk_o: no gate tensor"
    for name, tensor in (("q", q), ("k", k), ("v", v), ("h", h), ("g", g)):
        if tensor is None:
            return f"refused chunk_o: {name} is None"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or h.ndim != 5 or g.ndim != 3:
        return "refused chunk_o: unexpected rank"

    B, T, Hg, K = q.shape
    H, V = v.shape[2], v.shape[3]
    if k.shape != q.shape:
        return f"refused chunk_o: k.shape={tuple(k.shape)} != q.shape={tuple(q.shape)}"
    if tuple(v.shape[:2]) != (B, T):
        return "refused chunk_o: v does not share [B, T] with q"
    if tuple(g.shape) != (B, T, H):
        return f"refused chunk_o: g.shape={tuple(g.shape)} != {(B, T, H)}"
    if Hg == 0 or H % Hg != 0:
        return f"refused chunk_o: H={H} is not a multiple of Hg={Hg}"
    if T % BT != 0:
        return f"refused chunk_o: T={T} is not a multiple of {BT}"
    if K % 16 != 0:
        return f"refused chunk_o: K={K} is not a multiple of 16"
    if V % _BV_O != 0:
        return f"refused chunk_o: V={V} is not a multiple of {_BV_O}"
    if tuple(h.shape) != (B, T // BT, H, V, K):
        return f"refused chunk_o: h.shape={tuple(h.shape)} != {(B, T // BT, H, V, K)}"
    if (
        q.dtype is not torch.float16
        or k.dtype is not torch.float16
        or v.dtype is not torch.float16
        or h.dtype is not torch.float16
    ):
        return "refused chunk_o: activations are not float16"
    if g.dtype is not torch.float32:
        return f"refused chunk_o: g dtype {g.dtype} is not float32"
    if scale is not None:
        if not isinstance(scale, (int, float)):
            return f"refused chunk_o: scale is {type(scale).__name__}, not a float"
        if scale != K**-0.5:
            # The kernel bakes K**-0.5 into the store; a rescale would change
            # the fp16 rounding against the reference.
            return f"refused chunk_o: scale={scale!r} != K**-0.5={K**-0.5!r}"
    return ""


def _run_chunk_o(q, k, v, h, g):
    B, T, Hg, K = q.shape
    H, V = v.shape[2], v.shape[3]
    kernel = _chunk_o_for(B, T, H, Hg, K, V)
    if kernel is None:
        return None
    return kernel(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        h.contiguous().view(B * (T // BT), H, V, K),
        g.contiguous(),
    )


def chunk_fwd_o(q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_size=64):
    """Drop-in for ``fla.chunk_o.chunk_fwd_o``; falls back on any mismatch."""
    original = _ORIGINALS.get("chunk_fwd_o")
    reason = _gate_chunk_o(q, k, v, h, g, scale, cu_seqlens, chunk_size) or _device_reason(
        q, "chunk_o"
    )
    if reason:
        _note(_refusal_key(reason), logging.INFO, "%s; using Triton", reason)
        return original(q, k, v, h, g=g, scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    try:
        result = _run_chunk_o(q, k, v, h, g)
    except Exception as exc:  # noqa: BLE001 - serving keeps a correct fallback
        _note(
            f"chunk_o fault {type(exc).__name__}",
            logging.ERROR,
            "chunk_o fault (%s), using Triton: %s",
            type(exc).__name__,
            exc,
        )
        result = None
    if result is not None:
        return result
    return original(q, k, v, h, g=g, scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size)


# --------------------------------------------------------------------------
# chunk_gated_delta_rule_fwd_h
# --------------------------------------------------------------------------


def _gate_fwd_h(
    k,
    w,
    u,
    g,
    gk,
    initial_state,
    initial_state_indices,
    save_new_value,
    cu_seqlens,
    chunk_indices,
    use_exp2,
    inplace_update,
    track_state,
    track_chunk_idx,
) -> str:
    """Return "" when the vendored chunk_h kernel computes this call exactly."""
    import torch

    if cu_seqlens is not None or chunk_indices is not None:
        return "refused chunk_h: varlen (cu_seqlens/chunk_indices is set)"
    if gk is not None:
        return "refused chunk_h: per-channel gate (gk)"
    if use_exp2:
        return "refused chunk_h: use_exp2"
    if track_state is not None or track_chunk_idx is not None:
        return "refused chunk_h: checkpoint tracking"
    if g is None:
        return "refused chunk_h: no gate tensor"
    if not save_new_value:
        return "refused chunk_h: save_new_value=False"
    for name, tensor in (("k", k), ("w", w), ("u", u), ("g", g)):
        if tensor is None:
            return f"refused chunk_h: {name} is None"
    if k.ndim != 4 or w.ndim != 4 or u.ndim != 4 or g.ndim != 3:
        return "refused chunk_h: unexpected rank"

    B, T, Hg, K = k.shape
    H, V = u.shape[2], u.shape[3]
    if tuple(w.shape) != (B, T, H, K):
        return f"refused chunk_h: w.shape={tuple(w.shape)} != {(B, T, H, K)}"
    if tuple(u.shape[:2]) != (B, T):
        return "refused chunk_h: u does not share [B, T] with k"
    if tuple(g.shape) != (B, T, H):
        return f"refused chunk_h: g.shape={tuple(g.shape)} != {(B, T, H)}"
    if Hg == 0 or H % Hg != 0:
        return f"refused chunk_h: H={H} is not a multiple of Hg={Hg}"
    if T % BT != 0:
        return f"refused chunk_h: T={T} is not a multiple of {BT}"
    if K % 16 != 0:
        return f"refused chunk_h: K={K} is not a multiple of 16"
    if V % _BV_H != 0:
        return f"refused chunk_h: V={V} is not a multiple of {_BV_H}"
    if k.dtype is not torch.float16 or w.dtype is not torch.float16 or u.dtype is not torch.float16:
        return "refused chunk_h: activations are not float16"
    if g.dtype is not torch.float32:
        return f"refused chunk_h: g dtype {g.dtype} is not float32"
    if initial_state is not None:
        if initial_state.ndim != 4 or tuple(initial_state.shape[1:]) != (H, V, K):
            return f"refused chunk_h: initial_state.shape={tuple(initial_state.shape)}"
        if initial_state.dtype is not torch.float16:
            return f"refused chunk_h: initial_state dtype {initial_state.dtype} is not float16"
        if initial_state_indices is None:
            return "refused chunk_h: initial_state without initial_state_indices"
        if initial_state_indices.ndim != 1 or initial_state_indices.shape[0] != B:
            return "refused chunk_h: initial_state_indices does not have one entry per sequence"
        if initial_state_indices.dtype not in (torch.int32, torch.int64):
            return f"refused chunk_h: initial_state_indices dtype {initial_state_indices.dtype}"
    return ""


def _run_fwd_h(k, w, u, g, initial_state, initial_state_indices, inplace_update):
    import torch

    B, T, Hg, K = k.shape
    H, V = u.shape[2], u.shape[3]
    kernel = _chunk_h_for(B, T, H, Hg, K, V)
    if kernel is None:
        return None

    valid = None
    if initial_state is None:
        s0 = torch.zeros(B, H, V, K, device=k.device, dtype=k.dtype)
    else:
        # index_select follows the pool's own slot stride, so an envelope-strided
        # or page-major pool needs no special case. A negative entry is the pin's
        # "no state" sentinel: the reference starts those rows at zero and does
        # not write them back.
        valid = (initial_state_indices >= 0).view(B, 1, 1, 1)
        slots = initial_state_indices.clamp_min(0)
        s0 = torch.where(valid, initial_state.index_select(0, slots), 0.0).contiguous()

    h, v_new = kernel(
        k.contiguous(),
        w.contiguous(),
        u.contiguous(),
        g.contiguous(),
        s0,
    )

    if initial_state is not None and inplace_update:
        current = initial_state.index_select(0, slots)
        initial_state.index_copy_(0, slots, torch.where(valid, s0, current))
    return h.view(B, T // BT, H, V, K), v_new


def chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    initial_state_indices=None,
    save_new_value=True,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=False,
    inplace_update=True,
    track_state=None,
    track_chunk_idx=None,
    chunk_offsets=None,
):
    """Drop-in for ``fla.chunk_delta_h.chunk_gated_delta_rule_fwd_h``.

    ``chunk_offsets`` only selects varlen chunk offsets; the reference ignores
    it when ``cu_seqlens`` is None, and so does this path.
    """
    original = _ORIGINALS.get("chunk_gated_delta_rule_fwd_h")
    reason = _gate_fwd_h(
        k,
        w,
        u,
        g,
        gk,
        initial_state,
        initial_state_indices,
        save_new_value,
        cu_seqlens,
        chunk_indices,
        use_exp2,
        inplace_update,
        track_state,
        track_chunk_idx,
    ) or _device_reason(k, "chunk_h")
    if reason:
        _note(_refusal_key(reason), logging.INFO, "%s; using Triton", reason)
        return original(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=use_exp2,
            inplace_update=inplace_update,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
            chunk_offsets=chunk_offsets,
        )
    try:
        result = _run_fwd_h(k, w, u, g, initial_state, initial_state_indices, inplace_update)
    except Exception as exc:  # noqa: BLE001 - serving keeps a correct fallback
        _note(
            f"chunk_h fault {type(exc).__name__}",
            logging.ERROR,
            "chunk_h fault (%s), using Triton: %s",
            type(exc).__name__,
            exc,
        )
        result = None
    if result is not None:
        return result
    return original(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        inplace_update=inplace_update,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
        chunk_offsets=chunk_offsets,
    )


# --------------------------------------------------------------------------
# installation
# --------------------------------------------------------------------------

_REPLACEMENTS = {
    "chunk_fwd_o": chunk_fwd_o,
    "chunk_gated_delta_rule_fwd_h": chunk_gated_delta_rule_fwd_h,
}


def _rebind(name: str, old, new) -> list[str]:
    """Swap every module-level holder of ``name`` that still points at ``old``."""
    holders = []
    for module in list(sys.modules.values()):
        if module is None or module is sys.modules[__name__]:
            continue
        try:
            current = getattr(module, name, None)
        except Exception:  # noqa: BLE001 - a partially initialised module
            continue
        if current is not old:
            continue
        try:
            setattr(module, name, new)
        except Exception:  # noqa: BLE001 - read-only module attribute
            continue
        holders.append(getattr(module, "__name__", repr(module)))
    return holders


def _bind(chunk_o_module, chunk_delta_h_module, fla_chunk) -> bool:
    """Rebind both wrappers and prove the engine's caller sees them."""
    _ORIGINALS["chunk_fwd_o"] = chunk_o_module.chunk_fwd_o
    _ORIGINALS["chunk_gated_delta_rule_fwd_h"] = chunk_delta_h_module.chunk_gated_delta_rule_fwd_h

    rebound = {}
    for name, replacement in _REPLACEMENTS.items():
        source = chunk_o_module if name == "chunk_fwd_o" else chunk_delta_h_module
        rebound[name] = _rebind(name, _ORIGINALS[name], replacement)
        setattr(source, name, replacement)

    # A module imported after this point binds from the source module, which now
    # carries the replacement. What matters is that the caller the engine
    # actually uses sees it; anything less is a silent no-op.
    for name, replacement in _REPLACEMENTS.items():
        if getattr(fla_chunk, name, None) is not replacement:
            logger.error(
                "sglang-restomod: GDN pair did not reach fla.chunk.%s, staying on Triton", name
            )
            # Undo both names. A holder that never carried our replacement -- the
            # one that failed the check included -- is left as it is.
            for other, repl in _REPLACEMENTS.items():
                source = chunk_o_module if other == "chunk_fwd_o" else chunk_delta_h_module
                setattr(source, other, _ORIGINALS[other])
                _rebind(other, repl, _ORIGINALS[other])
            _ORIGINALS.clear()
            return False

    logger.info(
        "sglang-restomod: GDN prefill chunk_o and chunk_h routed to TileLang (rebound %s)",
        rebound,
    )
    return True


def install() -> bool:
    """Bind the TileLang pair; False when disabled, already bound, or unavailable."""
    global _INSTALLED
    if _INSTALLED:
        return False
    if not enabled():
        logger.info(
            "sglang-restomod: %s=0, GDN prefill stays on the pinned Triton kernels", ENV_KILL
        )
        return False

    try:
        from sglang.kernels.ops.attention.fla import chunk as fla_chunk
        from sglang.kernels.ops.attention.fla import chunk_delta_h, chunk_o
    except Exception as exc:  # noqa: BLE001 - no engine means nothing to patch
        logger.warning(
            "sglang-restomod: GDN pair cannot reach the FLA kernels, staying on Triton: %s", exc
        )
        return False

    _INSTALLED = _bind(chunk_o, chunk_delta_h, fla_chunk)
    return _INSTALLED
