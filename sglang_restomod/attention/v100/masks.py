"""The per-layer mask this component's paged kernel applies.

Kept free of torch and sglang so it can be unit-tested without either. The
kernel takes a boolean ``causal`` and an ``int`` window where ``-1`` means no
window; this module is the only place that decides them.
"""

from __future__ import annotations

from typing import Optional


def resolve_layer_mask(
    *,
    default_causal: bool,
    is_cross_attention: bool,
    is_encoder_only: bool,
    sliding_window_size: Optional[int],
) -> tuple[bool, int]:
    """Return ``(causal, sliding_window_size)`` for the paged kernel.

    A non-causal layer keeps its window. The kernel applies it as a left bound
    only, so an encoder-only sliding layer sees every key from
    ``q - window`` onward, including the positions after the query. That is what
    a DFLASH draft block needs: its layers are ``sliding_attention`` with
    ``is_causal: false``, and clearing the window there makes the draft attend
    to the whole prefix instead of the 2,047-token window it was trained on.
    """
    causal = bool(default_causal) and not (is_cross_attention or is_encoder_only)
    window = (
        int(sliding_window_size)
        if sliding_window_size is not None and int(sliding_window_size) >= 0
        else -1
    )
    return causal, window
