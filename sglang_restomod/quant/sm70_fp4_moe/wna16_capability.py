"""Let compressed-tensors WNA16 reach Marlin on SM70.

``CompressedTensorsWNA16.get_min_capability`` reports 80, so a W4A16 checkpoint
is refused on Volta before the Marlin backend is consulted at all. The
``marlin_v100`` extension that this component loads is exactly that backend, so
the gate is lowered to 70 when the device is SM70 and the extension can be
loaded.

Every other device keeps the upstream answer, which is what "no behaviour change
on SM80+" means here: on an Ampere box the extension may also be present, and
the check still reports 80.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

FLAG = "_restomod_wna16_sm70"

SM70_CAPABILITY = 70
UPSTREAM_CAPABILITY = 80


def _sm70_with_marlin() -> bool:
    """Whether this process can run the Marlin WNA16 kernel on SM70."""
    import torch

    from .marlin import available

    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 7:
        return False
    return available()


def install_on(scheme_cls) -> bool:
    """Wrap ``get_min_capability`` on ``scheme_cls``; idempotent."""
    if getattr(scheme_cls, FLAG, False):
        return False

    original = scheme_cls.get_min_capability

    def get_min_capability(cls):
        if _sm70_with_marlin():
            return SM70_CAPABILITY
        return original.__func__(cls)

    scheme_cls.get_min_capability = classmethod(get_min_capability)
    setattr(scheme_cls, FLAG, True)
    return True


def install() -> bool:
    """Patch the dense compressed-tensors WNA16 scheme, if the pin has it."""
    try:
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
            CompressedTensorsWNA16,
        )
    except Exception as exc:  # noqa: BLE001 - engine layout may differ
        logger.warning("sglang-restomod: WNA16 scheme not found, gate unchanged: %s", exc)
        return False

    installed = install_on(CompressedTensorsWNA16)
    if installed:
        logger.info("sglang-restomod: WNA16 accepts SM70 when marlin_v100 is loadable")
    return installed
