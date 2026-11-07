"""A guarded stand-in for the ``flash_mla`` wheel.

v0.5.23 imports ``sglang.kernels.ops.attention.flash_mla_interface`` whenever a
speculative worker sets up decode CUDA graphs: ``eagle_worker_v2`` appends
``FlashMLABackend`` to the graph-supported backend list, and that module
imports the external ``flash_mla`` wheel at module scope. The wheel ships
prebuilt kernels for SM90 and newer. An SM70 launch never runs FlashMLA, but
without the wheel the import aborts graph capture before the QSA backend is
asked to capture anything.

This module installs a stub for exactly that import, and only when the import
fails: if ``flash_mla`` imports, nothing is installed or shadowed. That is what
keeps the SM70 workaround off a Hopper host. The stub serves the names the
import binds -- ``fused_norm_rope_attn_rope_cast`` and ``FlashMLASchedMeta`` --
and raises for every other one, so a launch that really selects FlashMLA still
fails loudly, with a message that says what to install.

Selection: ``SGLANG_RESTOMOD_COMPONENTS=attention_v100`` installs it at plugin
load, before any worker imports the attention backend.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

# The override logger, so the one line this module prints appears where the rest
# of the SM70 override reports itself.
from sglang_restomod.overrides import logger

# Set on both stub modules. ``is_stub_installed`` reads it instead of comparing
# against a module object, so a real wheel that is already imported is never
# reported as ours.
STUB_ATTR = "__restomod_flash_mla_stub__"

_GUIDANCE = (
    "the flash_mla wheel is not installed; SGLang 0.5.23 only needs the import "
    "to succeed when a speculative worker captures decode CUDA graphs, which is "
    "the one case sglang-restomod stubs. Install sgl-flash-mla (SM90+) if this "
    "launch really selects the FlashMLA attention backend."
)

_INSTALLED = False


def _unavailable(name: str) -> ImportError:
    return ImportError(f"flash_mla.{name} is not implemented by the restomod stub: {_GUIDANCE}")


def _raise_unavailable(name: str) -> Any:
    raise _unavailable(name)


def _module_getattr(name: str) -> Any:
    # Module dunders keep their normal semantics: tools probe modules for
    # ``__path__``, ``__version__`` and so on, and an ImportError there would
    # surface far from the code that caused it.
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    raise _unavailable(name)


class FlashMLASchedMeta:
    """Stands in for the wheel's scheduling metadata type.

    The pin binds the name at import and tests values with ``isinstance``; it
    constructs one only when FlashMLA actually runs, which an SM70 launch never
    does.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise _unavailable("FlashMLASchedMeta()")


def _mega(*args: Any, **kwargs: Any) -> Any:
    raise _unavailable("fused_norm_rope_attn_rope_cast()")


def install_stub() -> bool:
    """Install the stub modules; return True once they are in ``sys.modules``."""
    global _INSTALLED
    if _INSTALLED:
        return True
    package = types.ModuleType("flash_mla")
    interface = types.ModuleType("flash_mla.flash_mla_interface")
    package.__doc__ = "sglang-restomod stub; the flash_mla wheel is not installed."
    interface.__doc__ = package.__doc__
    # A package needs __path__ for ``import flash_mla.flash_mla_interface`` to
    # find the submodule, and the attribute so the pin can also reach it as
    # ``flash_mla.flash_mla_interface``.
    package.__path__ = []  # type: ignore[attr-defined]
    package.flash_mla_interface = interface  # type: ignore[attr-defined]
    package.fused_norm_rope_attn_rope_cast = _mega  # type: ignore[attr-defined]
    interface.FlashMLASchedMeta = FlashMLASchedMeta  # type: ignore[attr-defined]
    package.__getattr__ = _module_getattr  # type: ignore[attr-defined]
    interface.__getattr__ = _module_getattr  # type: ignore[attr-defined]
    setattr(package, STUB_ATTR, True)
    setattr(interface, STUB_ATTR, True)
    sys.modules["flash_mla"] = package
    sys.modules["flash_mla.flash_mla_interface"] = interface
    _INSTALLED = True
    return True


def is_stub_installed() -> bool:
    return bool(getattr(sys.modules.get("flash_mla"), STUB_ATTR, False))


def flash_mla_status() -> str:
    """What an ``import flash_mla`` would do in this process."""
    if is_stub_installed():
        return "stub"
    if "flash_mla" in sys.modules:
        return "imported"
    return "importable" if importlib.util.find_spec("flash_mla") else "absent"


def ensure_flash_mla_stub() -> bool:
    """Install the stub if ``import flash_mla`` fails; return whether it is in.

    Returns False when the real wheel imported, so the caller can tell "left
    alone" from "stubbed". The guard is the import itself, not a device or
    backend name: a host that has a working wheel never gets the stub, and a
    wheel that is present but broken is reported and then stubbed.
    """
    if is_stub_installed():
        return True
    if "flash_mla" in sys.modules:
        return False
    try:
        import flash_mla  # noqa: F401
    except ImportError as exc:
        install_stub()
        logger.info(
            "sglang-restomod: flash_mla is not importable (%s); installed the "
            "guarded stub for the decode-graph import",
            exc,
        )
        return True
    return False
