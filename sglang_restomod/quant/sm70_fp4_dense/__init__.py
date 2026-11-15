"""Dense NVFP4 on SM70 through the marlin_v100 extension.

The pin already has the dense NVFP4-A16 Marlin path: with ``--fp4-gemm-backend
marlin``, ``ModelOptFp4LinearMethod.apply`` returns into
``apply_fp4_marlin_linear`` before it quantizes an activation, so a layer the
checkpoint tags ``NVFP4`` runs as W4A16 on Volta. What the pin does not have is
a kernel for it. Its own Marlin build stops below SM80, and ``marlin_utils_fp4``
imports ``gptq_marlin_gemm`` and ``gptq_marlin_repack`` into its own module
namespace, so the routing ``sm70_marlin_v100`` performs for ``marlin_utils``
never reaches this path.

This component replaces those names in ``marlin_utils_fp4`` -- plus the scale
permutation and the per-tensor global-scale step -- with calls into
``marlin_v100``. :mod:`.method` explains each one.

Requirements:

* ``SM70_MARLIN_DIR`` points at the directory holding the built ``_C*.so``;
* E2M1 weights, E4M3 group-16 block scales, FP16 activations, no zero-point,
  and a scalar global scale per GEMM (see ``PROVENANCE.md``).

Issue: sglang-restomod#37.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

# The extension signature is the same ``_C.marlin_gemm`` that the WNA16 path
# calls, so the slot mapping is shared rather than copied: two 19-slot builders
# would be two things to keep in step.
from ..sm70_marlin_v100 import gemm_args

logger = logging.getLogger(__name__)

ENV_DIR = "SM70_MARLIN_DIR"

_LOCK = threading.Lock()
_UNRESOLVED = object()
_OPS: object = _UNRESOLVED

__all__ = ["available", "directory", "gemm_args", "install", "ops", "register"]


def directory() -> Path:
    """Return the extension directory named by ``SM70_MARLIN_DIR``."""
    raw = os.environ.get(ENV_DIR)
    if not raw:
        raise RuntimeError(
            f"SM70 NVFP4 dense needs the marlin_v100 extension: set {ENV_DIR} to "
            "the directory that holds _C*.so (see PROVENANCE.md)."
        )
    return Path(raw)


def _resolve() -> tuple[object, object] | None:
    """Load ``_C*.so`` once and return ``(repack, gemm)``, or ``None``."""
    global _OPS
    with _LOCK:
        if _OPS is not _UNRESOLVED:
            return _OPS  # type: ignore[return-value]
        _OPS = None
        try:
            root = directory()
        except RuntimeError:
            return None
        matches = sorted(root.glob("_C*.so"))
        if not matches:
            logger.warning("SM70 NVFP4 dense: no _C*.so under %s (%s)", root, ENV_DIR)
            return None
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            if torch.cuda.get_device_capability() != (7, 0):
                return None
            torch.ops.load_library(str(matches[0]))
            repack = getattr(torch.ops._C, "gptq_marlin_repack", None)
            gemm = getattr(torch.ops._C, "marlin_gemm", None)
            if repack is None or gemm is None:
                logger.warning(
                    "SM70 NVFP4 dense: %s does not export gptq_marlin_repack and marlin_gemm",
                    matches[0],
                )
                return None
        except Exception as exc:  # noqa: BLE001 - availability probe
            logger.warning("SM70 NVFP4 dense: %s could not be loaded (%s)", matches[0], exc)
            return None
        _OPS = (repack, gemm)
        logger.info("SM70 NVFP4 dense: loaded marlin_v100 from %s", matches[0].parent)
        return _OPS  # type: ignore[return-value]


def available() -> bool:
    """Whether marlin_v100 can serve a dense NVFP4 GEMM on this device."""
    return _resolve() is not None


def ops() -> tuple[object, object]:
    """Return ``(gptq_marlin_repack, marlin_gemm)``; raise when unavailable."""
    resolved = _resolve()
    if resolved is None:
        raise RuntimeError(
            "the SM70 NVFP4 dense path needs a loadable marlin_v100 build: set "
            f"{ENV_DIR} to the directory that holds _C*.so (see PROVENANCE.md)"
        )
    return resolved


def install() -> bool:
    """Route ``marlin_utils_fp4`` to the marlin_v100 kernel.

    Returns whether anything was patched, so a second call reports ``False``
    rather than rebinding an already-routed namespace.
    """
    from .method import install as _install

    return _install()


def register() -> None:
    """Component entry point."""
    install()
