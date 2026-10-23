"""Lightweight downstream layer for SGLang.

``main`` contains only this layer. The upstream SGLang tree is not vendored;
it is pinned in ``UPSTREAM_PIN.toml`` and fetched at build time when patches
exist.

SGLang loads this package through its public plugin entry point::

    SGLANG_PLUGINS=restomod

Keep this package additive. Upstream-file edits belong in ``patches/`` with a
matching entry in ``patches/manifest.json``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Entry point for the ``sglang.srt.plugins`` group.

    This is intentionally a no-op until the first selected component lands.
    """
    logger.info("sglang-restomod plugin loaded")
