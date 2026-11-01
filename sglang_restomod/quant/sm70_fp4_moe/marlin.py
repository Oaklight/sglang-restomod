"""Loader for the marlin_v100 SM70 extension.

The extension is built outside this package from
``zhinianqin/marlin_v100`` (see ``PROVENANCE.md``). It exports two torch
libraries:

* ``_C`` -- ``gptq_marlin_repack``, used at load time;
* ``_moe_C`` -- ``moe_wna16_marlin_gemm``, the expert GEMM.

``SM70_MARLIN_DIR`` names the directory that holds the built ``.so`` files.
Import of this module is cheap; torch and the libraries are touched only by
:func:`ensure_loaded`.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_DIR = "SM70_MARLIN_DIR"

_LOCK = threading.Lock()
_LOADED: Ops | None = None


@dataclass(frozen=True)
class Ops:
    """The two extension entry points this layer uses."""

    repack: object
    gemm: object


def directory() -> Path:
    """Return the extension directory named by ``SM70_MARLIN_DIR``."""
    raw = os.environ.get(ENV_DIR)
    if not raw:
        raise RuntimeError(
            f"SM70 NVFP4 MoE needs the marlin_v100 extension: set {ENV_DIR} to the "
            "directory that holds _C*.so and _moe_C*.so (see PROVENANCE.md)."
        )
    return Path(raw)


def _find(pattern: str) -> Path:
    root = directory()
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {pattern} under {root} ({ENV_DIR})")
    return matches[0]


def ensure_loaded() -> Ops:
    """Load both libraries once and return their entry points."""
    global _LOADED
    with _LOCK:
        if _LOADED is not None:
            return _LOADED
        import torch

        c_lib = _find("_C*.so")
        moe_lib = _find("_moe_C*.so")
        torch.ops.load_library(str(c_lib))
        torch.ops.load_library(str(moe_lib))

        repack = getattr(torch.ops._C, "gptq_marlin_repack", None)
        gemm = getattr(torch.ops._moe_C, "moe_wna16_marlin_gemm", None)
        if repack is None or gemm is None:
            raise RuntimeError(
                f"{c_lib} / {moe_lib} do not export gptq_marlin_repack and "
                "moe_wna16_marlin_gemm"
            )
        _LOADED = Ops(repack=repack, gemm=gemm)
        logger.info("SM70 NVFP4 MoE: loaded marlin_v100 from %s", c_lib.parent)
        return _LOADED


def available() -> bool:
    """Whether the extension can be loaded in this environment."""
    try:
        ensure_loaded()
        return True
    except Exception as exc:  # noqa: BLE001 - availability probe
        logger.debug("SM70 NVFP4 MoE extension unavailable: %s", exc)
        return False
