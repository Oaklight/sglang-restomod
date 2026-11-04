"""A smaller read budget for the queries before a position.

Vendored from Prefrontal-AI/latent-sparse-attention @
97b82d695b74ca341965a16bbc2a951c8b0c567a (src/lsa/budget.py), Apache-2.0,
with the SM70 port's changes. See PROVENANCE.md.

A design may read fewer long-range keys and buckets for the early queries, where
most of the context is not visible yet. The late queries read the design's own
budget. The reference uses the same type in its settings, its oracle and its
kernels, so the port keeps one type too.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class EarlyBudget:
    """The `top_k` and `top_p` a query reads while its position is below `position`."""

    # First position that reads the design's own budget.
    position: int
    # Long-range keys an early query reads at most.
    top_k: int
    # Buckets an early query picks for expansion.
    top_p: int

    def __post_init__(self) -> None:
        """Reject a field that is not a positive whole number."""
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"early_budget {item.name} must be an integer - got {value!r}")
            if value < 1:
                raise ValueError(f"early_budget {item.name} must be at least 1 - got {value}")

    @classmethod
    def read(cls, value: EarlyBudget | Mapping[str, Any] | None) -> EarlyBudget | None:
        """Build the knob from a mapping of `position`, `top_k` and `top_p`, or None."""
        if value is None or isinstance(value, EarlyBudget):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(
                f"early_budget is a mapping of position, top_k and top_p, or null - got {value!r}"
            )
        names = [item.name for item in fields(cls)]
        if sorted(value) != sorted(names):
            raise ValueError(f"early_budget has the keys {names} - got {sorted(value)}")
        return cls(**value)


def budget_at(position: int, top_k: int, top_p: int, early: EarlyBudget | None) -> tuple[int, int]:
    """Return the `top_k` and `top_p` a query at `position` reads."""
    if early is not None and position < early.position:
        return early.top_k, early.top_p
    return top_k, top_p
