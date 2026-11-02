"""SM70 linear-attention component.

Routes the GDN (gated delta rule) prefill through the TileLang chunked kernel
that the pinned engine no longer ships. Decode and verify keep their pinned
kernels; so does every call the TileLang kernel does not support.

Selection: ``SGLANG_RESTOMOD_COMPONENTS=gdn_tilelang_sm70``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Component entry point."""
    from .gdn_tilelang import install

    install()
