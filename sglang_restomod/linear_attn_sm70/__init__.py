"""SM70 linear-attention component.

Routes GDN (gated delta rule) prefill through TileLang kernels and leaves
everything else on the pin.

Two substitutions are available, and both refuse any call they cannot compute
exactly, falling back to the pinned Triton kernel:

``pair``
    Rebinds ``fla.chunk_o.chunk_fwd_o`` and
    ``fla.chunk_delta_h.chunk_gated_delta_rule_fwd_h`` -- the two Triton kernels
    that dominate prefill. Serves every caller of the FLA chunk path.

``whole``
    Wraps ``TritonGDNKernel.extend`` with the fork's single-kernel chunked GDN.
    Serves only the GDN backend's own prefill entry point.

``auto`` (default) installs both: ``whole`` answers the calls it supports and
``pair`` covers the rest.

Selection: ``SGLANG_RESTOMOD_COMPONENTS=gdn_tilelang_sm70``.

Environment:

``SGLANG_V100_GDN_TILELANG_IMPL``
    ``auto`` (default), ``pair`` or ``whole``.

``SM70_GDN_TILELANG=0``
    Control arm: nothing is rebound and no TileLang kernel is imported, so the
    process runs the pinned Triton kernels exactly as one that never selected
    the component. This is the switch an A/B harness toggles.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Component entry point."""
    from .gdn_pair import enabled
    from .gdn_pair import install as install_pair
    from .gdn_tilelang import impl_mode
    from .gdn_tilelang import install as install_whole

    install_pair()
    if enabled() and impl_mode() != "pair":
        install_whole()
