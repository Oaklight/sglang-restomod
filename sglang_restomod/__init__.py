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

__version__ = "0.0.0a0.dev0"

logger = logging.getLogger(__name__)


def register() -> None:
    """Entry point for the ``sglang.srt.plugins`` group.

    Loads only the components named in ``SGLANG_RESTOMOD_COMPONENTS``. With the
    variable unset the plugin stays inert, which is the baseline behaviour.
    """
    from .plugins import load_enabled

    loaded = load_enabled()
    if loaded:
        logger.info("sglang-restomod plugin loaded (components: %s)", ", ".join(loaded))
    else:
        logger.info("sglang-restomod plugin loaded (no components selected)")
