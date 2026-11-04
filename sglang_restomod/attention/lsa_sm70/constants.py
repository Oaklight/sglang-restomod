"""Constants shared by the LSA selection port.

Vendored from Prefrontal-AI/latent-sparse-attention @
97b82d695b74ca341965a16bbc2a951c8b0c567a (src/lsa/constants.py), Apache-2.0.
See PROVENANCE.md.
"""

from __future__ import annotations

# A selected-key slot that names nothing, in a list padded to a fixed width.
PAD: int = -1

# Norm floor that keeps a zero row at zero instead of producing NaN.
NORM_EPS: float = 1e-12
