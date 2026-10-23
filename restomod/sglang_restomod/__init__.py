"""Restomod overlay for SGLang.

This package is the additive half of ``sglang-restomod``. It registers through
SGLang's public plugin entry point and keeps upstream-file edits out of the
overlay.

Enable it with::

    SGLANG_PLUGINS=v100_plus,restomod
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Entry point for the ``sglang.srt.plugins`` group.

    Keep this cheap: SGLang loads it during startup. Real hook installation
    belongs in :func:`sglang_restomod.overrides.install`, called from here once
    the first override exists.
    """
    logger.info("sglang-restomod overlay plugin loaded")
