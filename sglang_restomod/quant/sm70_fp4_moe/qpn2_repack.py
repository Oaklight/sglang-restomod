"""Prefill fallback for a QPN2-resident model: repack one layer to Marlin.

A QPN2 weight cannot feed the prefill GEMM directly (the shared B tile is a
transpose buffer, #224). The fallback keeps QPN2 resident in HBM and repacks the
layer into a reused scratch buffer when a prefill shape arrives, then hands the
Marlin layout to the unchanged Marlin call.

One launch per prefix: the fused kernel reads QPN2 and writes both the Marlin
int32 words and the [E, N, K/16] E4M3 scales. Measured 1.26 ms (w13) + 0.81 ms
(w2) per layer on a V100, against a 2.07 ms budget for the pair.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_CSRC = Path(__file__).resolve().parent / "csrc" / "repack_qpn2_marlin.cu"
_LOCK = threading.Lock()
_MODULE = None
_FAILED = False
_SCRATCH: dict = {}
# Call-order record: entries are appended in prepare order (gate_up, down, ...)
# so each call can prefetch the next one while the current GEMM runs.
_ORDER: list = []
_INDEX: dict = {}
_STREAM = None
_EVENT: dict = {}
_OWNER: dict = {}
_LOGGED: set = set()


def _side_stream():
    global _STREAM
    if _STREAM is None:
        import torch

        _STREAM = torch.cuda.Stream()
    return _STREAM


def register(weight, global_scale, size_n: int, size_k: int, prefix: str) -> None:
    """Record a converted prefix so the next prefill call can prefetch it."""
    key = weight.data_ptr()
    if key in _INDEX:
        return
    _INDEX[key] = len(_ORDER)
    _ORDER.append((key, weight, global_scale, size_n, size_k, prefix))


def reset() -> None:
    global _EVENT
    _ORDER.clear()
    _INDEX.clear()
    _EVENT = {}


def _repack(weight, global_scale, size_n, size_k, stream=None):
    """Launch both repack kernels; returns the three buffers for this shape.

    Two orderings matter, and both were missing before this guard:

    * On the prefetch (side-stream) path, the buffers being overwritten belong
      to the shape used two calls back, whose GEMM may still be reading them on
      the default stream. Without an edge from that stream the repack can
      overwrite a weight buffer under a live GEMM, which is why the side stream
      waits on the current one before it writes.
    * The reverse edge is the event recorded here: the consumer waits on it in
      ``prefill_shim`` before the GEMM reads the new content.
    """
    import torch

    module = _load()
    if module is None:
        return None
    per_expert = (size_n // 32) * (size_k // 16) * 288
    experts = int(weight.numel() // per_expert)
    weight_buf, raw, encoded = _scratch(experts, size_n, size_k, weight.device)
    key = (size_n, size_k)
    flat = weight.reshape(-1)

    if stream is None:
        # Synchronous path on the default stream. A prefetch of this same shape
        # may still be writing these buffers on the side stream, so wait for it
        # rather than racing two writers.
        _wait_for_prefetch(key)
        module.repack_qpn2_marlin_into(flat, weight_buf, raw, size_n, size_k)
        module.repack_qpn2_scales_encoded_into(flat, encoded, size_n, size_k)
        return weight_buf, encoded

    # Prefetch: order after every read already enqueued on the default stream,
    # then record the event the consumer will wait on.
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module.repack_qpn2_marlin_into(flat, weight_buf, raw, size_n, size_k)
        module.repack_qpn2_scales_encoded_into(flat, encoded, size_n, size_k)
        event = torch.cuda.Event()
        event.record(stream)
    _EVENT[key] = event
    return weight_buf, encoded


def _wait_for_prefetch(key) -> None:
    """Make the current stream wait for an outstanding prefetch of ``key``."""
    import torch

    event = _EVENT.pop(key, None)
    if event is not None:
        torch.cuda.current_stream().wait_event(event)


def _load():
    global _MODULE, _FAILED
    with _LOCK:
        if _MODULE is not None or _FAILED:
            return _MODULE
        try:
            from torch.utils.cpp_extension import load

            _MODULE = load(
                name="repack_qpn2_marlin",
                sources=[str(_CSRC)],
                extra_cuda_cflags=[
                    "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                ],
                verbose=False,
            )
            logger.info("SM70 QPN2: fused prefill repack ready")
        except Exception:  # pragma: no cover - build failure keeps the raise
            _FAILED = True
            logger.warning("SM70 QPN2: prefill repack unavailable", exc_info=True)
        return _MODULE


def _scratch(experts: int, size_n: int, size_k: int, device):
    """One pair of buffers per shape, reused by every layer in call order."""
    import torch

    key = (experts, size_n, size_k, str(device))
    buf = _SCRATCH.get(key)
    if buf is None:
        weight = torch.empty(
            (experts, size_k // 16, size_n * 2), dtype=torch.int32, device=device
        )
        scales = torch.empty(
            (experts, size_n, size_k // 16), dtype=torch.uint8, device=device
        )
        encoded = torch.empty(
            (experts, size_k // 16, size_n), dtype=torch.float8_e4m3fn, device=device
        )
        _SCRATCH[key] = buf = (weight, scales, encoded)
    return buf


def prefill_shim(qpn2_weight, global_scale, size_n: int, size_k: int):
    """QPN2 prefix -> ``(marlin_weight, marlin_scales, marlin_global_scale)``.

    Two-slot pipeline: the current prefix's buffers are waited on, then the next
    entry in the call order is repacked on a side stream while this prefix's
    GEMM runs. A miss (call order differs from prepare order) degrades to a
    synchronous repack - bandwidth lost, never correctness.

    Returns ``None`` when the extension is unavailable, so the caller can keep
    its original raise instead of serving a shape it cannot read.
    """
    import torch

    module = _load()
    if module is None:
        return None
    key = (size_n, size_k)
    ptr = qpn2_weight.data_ptr()
    idx = _INDEX.get(ptr)
    experts = int(qpn2_weight.numel() // ((size_n // 32) * (size_k // 16) * 288))
    # One global scale per expert. A mismatch means the caller's size_n/size_k
    # do not describe this bank (for example the fused gate+up width used as if
    # it were the weight's N), and the repack would silently serve the wrong
    # expert count to the Marlin reader.
    if global_scale is not None and int(global_scale.numel()) != experts:
        raise RuntimeError(
            "SM70 QPN2: the bank holds %d experts but global_scale has %d "
            "entries (n=%d k=%d); the caller's size_n/size_k do not describe "
            "this bank" % (experts, global_scale.numel(), size_n, size_k)
        )
    shape = _scratch(experts, size_n, size_k, qpn2_weight.device)
    owner = _OWNER.get(key)

    if owner != ptr:
        # Not prefetched (or predicted wrong): repack synchronously.
        _repack(qpn2_weight, global_scale, size_n, size_k)
        _OWNER[key] = ptr
    elif key in _EVENT:
        # Prefetch issued on the side stream: make the default stream wait.
        _wait_for_prefetch(key)

    weight, _raw, encoded = shape
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(
            "SM70 QPN2 prefill shim: repacking N=%d K=%d into the Marlin scratch",
            size_n, size_k,
        )
    from .method import encode_global_scale

    encoded_global = encode_global_scale(global_scale, torch.float16).contiguous()

    # Prefetch the next conversion in call order, if it has the other shape.
    # It runs on the side stream under this prefix's GEMM. The shape test is
    # not cosmetic: with only one projection converted every entry shares one
    # shape, and prefetching it would overwrite the buffer this call's GEMM is
    # about to read.
    if idx is not None and idx + 1 < len(_ORDER):
        nxt = _ORDER[idx + 1]
        nkey = (nxt[3], nxt[4])
        if nkey != key and _OWNER.get(nkey) != nxt[0]:
            if _repack(nxt[1], nxt[2], nxt[3], nxt[4], stream=_side_stream()) is not None:
                _OWNER[nkey] = nxt[0]
    return weight, encoded, encoded_global
