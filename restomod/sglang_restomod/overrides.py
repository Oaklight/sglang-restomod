"""Single place for deliberate runtime overrides.

Prefer SGLang's public registries (attention backend, quantization, model
registry) over monkey-patching. If a patch is unavoidable:

* target one symbol,
* assert the expected upstream version,
* and record the upstream file in ``v100_plus/core-patches.json``.

Keep this module import-light; it is loaded during server startup.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install() -> None:
    """Install restomod overrides.

    No-op until the first override lands. Call this from
    ``sglang_restomod.register`` when it does.
    """
    logger.debug("sglang-restomod overrides: nothing to install")
