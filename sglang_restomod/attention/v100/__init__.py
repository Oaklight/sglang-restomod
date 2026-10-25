"""V100 full-attention backend and its CUDA graph hooks.

Status: scaffold. The backend extraction is E5.1; this module only reserves the
component name and records provenance so the launch profile can select it
without changing later.

The first thing that lands here is the CUDA graph metadata delegation described
in ``PROVENANCE.md`` and validated in
``docs/runs/2026-10-25-qwen3.8-27b-fp8-cuda-graph-lambda12.md``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Component entry point. The backend wiring lands with E5.1."""
    logger.debug("sglang-restomod: attention_v100 component registered (scaffold)")
