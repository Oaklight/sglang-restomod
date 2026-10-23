"""Deliberate runtime overrides.

Rules:

* prefer SGLang's public registries over monkey-patching;
* keep each override in its own module and import it only when enabled;
* if an upstream file must change, add a patch under ``patches/`` and record it
  in ``patches/manifest.json`` instead of editing the installed package.

Nothing is installed yet; this module is the single seam where selected
components will hook in.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install() -> None:
    """Install selected overrides. No-op at baseline."""
    logger.debug("sglang-restomod: no overrides selected")
