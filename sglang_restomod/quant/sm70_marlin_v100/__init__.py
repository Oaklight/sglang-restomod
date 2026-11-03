"""Dense WNA16 on SM70 through the marlin_v100 extension.

The extension is built outside this package from ``zhinianqin/marlin_v100``
(see ``PROVENANCE.md``). Its ``_C`` library exports four operators:

* ``gptq_marlin_repack`` -- pack a GPTQ checkpoint into the marlin layout;
* ``awq_marlin_repack`` -- the AWQ equivalent;
* ``marlin_gemm`` -- the dense WNA16 GEMM;
* ``marlin_int4_fp8_preprocess`` -- the int4/fp8 preprocessing pass.

SM70 needs the first and the third. ``SM70_MARLIN_DIR`` names the directory
that holds the built ``_C*.so``.

This module only resolves the library and reports whether it can serve a dense
WNA16 GEMM; the capability gate in :mod:`sglang_restomod.overrides` reads
:func:`available`. Wiring ``marlin_gemm`` into the GPTQ linear method is issue
#40, so a launch that passes the gate still runs the upstream method until that
lands.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

ENV_DIR = "SM70_MARLIN_DIR"

_LOCK = threading.Lock()
_UNRESOLVED = object()
_OPS: object = _UNRESOLVED


def directory() -> Path:
    """Return the extension directory named by ``SM70_MARLIN_DIR``."""
    raw = os.environ.get(ENV_DIR)
    if not raw:
        raise RuntimeError(
            f"SM70 WNA16 needs the marlin_v100 extension: set {ENV_DIR} to the "
            "directory that holds _C*.so (see PROVENANCE.md)."
        )
    return Path(raw)


def _resolve() -> Optional[Tuple[object, object]]:
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
            logger.warning("SM70 WNA16: no _C*.so under %s (%s)", root, ENV_DIR)
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
                    "SM70 WNA16: %s does not export gptq_marlin_repack and marlin_gemm",
                    matches[0],
                )
                return None
        except Exception as exc:  # noqa: BLE001 - availability probe
            logger.warning("SM70 WNA16: %s could not be loaded (%s)", matches[0], exc)
            return None
        _OPS = (repack, gemm)
        logger.info("SM70 WNA16: loaded marlin_v100 from %s", matches[0].parent)
        return _OPS  # type: ignore[return-value]


def available() -> bool:
    """Whether marlin_v100 can serve a dense WNA16 GEMM on this device."""
    return _resolve() is not None


def ops() -> Tuple[object, object]:
    """Return ``(gptq_marlin_repack, marlin_gemm)``; raise when unavailable."""
    resolved = _resolve()
    if resolved is None:
        raise RuntimeError(
            "the SM70 WNA16 path needs a loadable marlin_v100 build: set "
            f"{ENV_DIR} to the directory that holds _C*.so (see PROVENANCE.md)"
        )
    return resolved
