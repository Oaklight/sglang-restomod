"""A fused ``tf32_hc_prenorm_gemm`` for SM70: one pass instead of a torch chain.

The DSV4 HC pre step is ``out = x @ fn.T`` and ``sqrsum = sum(x**2, -1)``, both
of which the caller consumes as raw sums -- the ``rsqrt(sqrsum / k + eps)`` is
applied downstream, which is why the sum is what gets written. The pin's
reference arm (``deep_gemm_sm70._hc_prenorm_reference``) is three launches: a
bf16-to-fp32 cast of a ``[T, 20480]`` tensor, an fp32 cuBLAS ``mm``, and a
square-and-sum over the same tensor. This module does it in two Triton launches
and never materializes the fp32 cast.

**Where it wins and where it does not.** The gate
(``docs/runs/2026-11-17-prenorm-kernel-gate.md``) measured this kernel against
the torch chain and against the shim's own reference arm:

* decode shapes (M = 1-32), inside a captured CUDA graph: it **loses by 4-7x**
  -- 0.189 ms against torch's 0.027. It is not launch-bound (capture shrank it
  15%); it is real GPU work on a grid of 8 CTAs. Decode must use torch.
* prefill shapes, eager, against the shim's own 3-launch reference arm: it
  wins 1.32x at M=4096 (2.6378 -> 1.9932 ms) and 1.41x at M=8192
  (5.1610 -> 3.6511), ties at M=1024 (1.05x), and loses at M=256 (0.84x).
  Prefill is not graph-captured, which is why the eager number is the right
  one there.

So this registers the ``fused`` arm and leaves dispatch to the shim's
``SGLANG_SM70_HC_PRENORM=auto`` threshold. Registering it does not make decode
faster; it makes the large-M arm available.

Determinism is a property of the design, not of the run: split-K partials are
written to memory and folded by a second kernel, never atomically accumulated.
Measured replay-to-replay delta 0.0 on both arms.

Import is cheap: no torch and no triton at module import. ``register()``
installs a thunk into the shim, and the kernel module is imported on the first
call.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SHIM_MODULE = "sglang_restomod.kernels.deep_gemm_sm70"

# 8 splits x 32 rows is 8 CTAs for a 32-row call: good for prefill, useless at
# decode, which is why nothing below the auto threshold reaches this kernel.
DEFAULT_SPLITS = 8
BT = 32
BK = 128

_launch = None


def _load_launch():
    """Import the Triton module once, on first call rather than at register()."""
    global _launch
    if _launch is None:
        from . import kernel

        _launch = kernel.launch
    return _launch


def _split_count(hc: int, requested: int | None) -> int:
    """Largest divisor of ``hc`` not above the request, so K splits evenly."""
    want = DEFAULT_SPLITS if requested is None else requested
    if want < 1:
        raise ValueError(f"hc_prenorm: num_splits must be >= 1, got {want}")
    for split in range(min(want, hc), 0, -1):
        if hc % split == 0:
            return split
    return 1


def hc_prenorm(x, fn, out, sqrsum, num_splits: int | None = None) -> None:
    """Write ``out = x @ fn.T`` and ``sqrsum = sum(x**2, -1)``, both fp32.

    The signature and semantics are the shim entry's; it is registered through
    ``install_fused_hc_prenorm`` and called as ``op(x, fn, out, sqrsum,
    num_splits=...)``. Validation happens here on purpose: an input this kernel
    cannot represent must raise, not silently compute a different reduction.
    """
    import torch

    if out is None or sqrsum is None:
        raise ValueError("hc_prenorm: out and sqrsum must be preallocated by the caller")
    if x.dim() != 2:
        raise ValueError(f"hc_prenorm: x must be 2-D [T, hc], got {tuple(x.shape)}")
    if fn.dim() != 2 or fn.shape[1] != x.shape[1]:
        raise ValueError(
            f"hc_prenorm: fn must be [mix, {x.shape[1]}], got {tuple(fn.shape)}"
        )
    t, hc = x.shape
    mix = fn.shape[0]
    if tuple(out.shape) != (t, mix):
        raise ValueError(f"hc_prenorm: out must be {(t, mix)}, got {tuple(out.shape)}")
    if tuple(sqrsum.shape) != (t,):
        raise ValueError(f"hc_prenorm: sqrsum must be {(t,)}, got {tuple(sqrsum.shape)}")
    if out.dtype != torch.float32 or sqrsum.dtype != torch.float32:
        raise ValueError("hc_prenorm: out and sqrsum must be float32")
    if hc % BK:
        # The caller's K is the HC width (20480 in DSV4); a width that is not a
        # BK multiple would make every split's last iteration a masked tail.
        raise ValueError(f"hc_prenorm: hc must be a multiple of {BK}, got {hc}")
    if not out.is_cuda or not sqrsum.is_cuda or not x.is_cuda:
        raise ValueError("hc_prenorm: tensors must be on a CUDA device")
    if t == 0:
        return
    _load_launch()(x, fn, out, sqrsum, _split_count(hc, num_splits))


def register() -> None:
    """Install :func:`hc_prenorm` as the shim's ``fused`` arm; idempotent."""
    import importlib

    from sglang_restomod import attest

    try:
        shim: Any = importlib.import_module(SHIM_MODULE)
    except Exception as exc:  # noqa: BLE001 - the shim is a package module; absent is a state
        logger.warning("hc_prenorm_sm70: %s is not importable (%s)", SHIM_MODULE, exc)
        attest.inactive("hc_prenorm_sm70", f"{SHIM_MODULE} is not importable ({exc})")
        return

    installer = getattr(shim, "install_fused_hc_prenorm", None)
    if installer is None:
        logger.warning(
            "hc_prenorm_sm70: %s has no install_fused_hc_prenorm; the pin moved and "
            "this component needs updating",
            SHIM_MODULE,
        )
        attest.inactive("hc_prenorm_sm70", "the shim has no install_fused_hc_prenorm")
        return

    installer(hc_prenorm)
    logger.info(
        "hc_prenorm_sm70: fused prenorm registered; it is the large-M arm, and "
        "decode belongs to torch (see the module docstring)"
    )
