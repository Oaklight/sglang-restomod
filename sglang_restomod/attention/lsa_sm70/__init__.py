"""Latent Sparse Attention (LSA) selection for SM70.

LSA clusters keys hierarchically in a learned latent space, has each query
score bucket centroids instead of every key, and reads only the selected keys
plus a local window and sinks. The reference is
`Prefrontal-AI/latent-sparse-attention` (Apache-2.0); this component ports the
selection core and, in stages, the sm70 serving path.

The selection core imports torch only when a caller imports it; this module and
``register()`` stay import-light, as the component contract requires.

Selection: ``SGLANG_RESTOMOD_COMPONENTS=lsa_sm70``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Component entry point: register the LSA attention backend."""
    from .backend_registry import install

    install()
