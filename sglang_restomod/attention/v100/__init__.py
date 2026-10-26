"""V100 full-attention backend hooks.

Component entry point for the V100 full-attention path. It currently injects
the CUDA graph metadata delegation the engine's backend is missing; the rest of
the backend lives in the engine checkout (see ``PROVENANCE.md`` for the source
and the split we intend).

Selection: ``SGLANG_RESTOMOD_COMPONENTS=attention_v100``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Component entry point: install the graph metadata delegation."""
    from .graph_metadata import install

    install()
