"""Component registry.

Every V100 capability lands as an additive component under ``sglang_restomod``
and is enabled by name. Importing this module is deliberately cheap: no torch,
no sglang, no kernel loading. A component is only imported when selected, and
its ``register()`` must stay lightweight (bindings load on first use).

Selection is a single environment variable so that a launch profile fully
describes the stack:

    SGLANG_RESTOMOD_COMPONENTS=fp8_moe,attention_v100

An empty or unset variable means "baseline": the plugin loads and does nothing.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ENV_COMPONENTS = "SGLANG_RESTOMOD_COMPONENTS"


@dataclass(frozen=True)
class Component:
    """A selectable capability."""

    name: str
    module: str
    description: str


COMPONENTS: dict[str, Component] = {
    "fp8_moe": Component(
        name="fp8_moe",
        module="sglang_restomod.quant.sm70_fp8_moe",
        description="Block-FP8 expert GEMM bound from the prebuilt TurboMind library",
    ),
    "attention_v100": Component(
        name="attention_v100",
        module="sglang_restomod.attention.v100",
        description="V100 full-attention backend and its CUDA graph metadata hooks",
    ),
}


def selection(env: Mapping[str, str] | None = None) -> list[str]:
    """Return the selected component names, validated and de-duplicated."""
    raw = (env if env is not None else os.environ).get(ENV_COMPONENTS, "")
    names: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if name and name not in names:
            names.append(name)
    unknown = [name for name in names if name not in COMPONENTS]
    if unknown:
        known = ", ".join(sorted(COMPONENTS))
        raise ValueError(
            f"unknown component(s) in {ENV_COMPONENTS}: {', '.join(unknown)}; "
            f"known: {known}"
        )
    return names


def is_enabled(name: str, env: Mapping[str, str] | None = None) -> bool:
    return name in selection(env)


def load_enabled(env: Mapping[str, str] | None = None) -> Sequence[str]:
    """Import and register every selected component; return the loaded names."""
    loaded: list[str] = []
    for name in selection(env):
        component = COMPONENTS[name]
        module = importlib.import_module(component.module)
        register = getattr(module, "register", None)
        if register is None:
            raise AttributeError(f"component {name!r} ({component.module}) has no register()")
        register()
        loaded.append(name)
        logger.debug("sglang-restomod: component %s registered", name)
    return loaded
