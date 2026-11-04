"""V100 attention component.

Registers the component's V100 full-attention backend with the pinned engine
and installs the CUDA graph metadata delegation it needs. Everything the
engine does not provide lives here; ``PROVENANCE.md`` records where it came
from and ``UPSTREAM_PIN.toml`` pins the kernel submodule.

Selection: ``SGLANG_RESTOMOD_COMPONENTS=attention_v100``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Component entry point: register the backend, then patch it."""
    from .backend_registry import install as install_backends
    from .backend_registry import install_draft_choices, install_qsa_swap
    from .draft_backend import install as install_draft_backend
    from .graph_metadata import install as install_graph_metadata

    install_backends()
    install_draft_choices()
    install_qsa_swap()
    install_graph_metadata()
    install_draft_backend()
