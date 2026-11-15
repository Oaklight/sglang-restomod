"""Machine-readable record of what actually registered.

Both of the obvious checks can say the wrong thing on these hosts:

* **Logs.** Plugin-load logging does not reach the log file. On 2026-11-15 a
  control-arm server log contained zero hits for every plugin-load line,
  including sglang's own ``Available plugins``, so "the log says nothing" was
  indistinguishable from "nothing loaded".
* **Package metadata.** ``importlib.metadata.version("sglang")`` reports the
  *installed* distribution. When the engine runs from a source tree on
  ``PYTHONPATH`` — which is how this layer is used — that is a different tree
  from the one being imported, so ``versions.txt`` can name the fork while the
  pin is what executes.

And the failure is silent twice over: ``sglang.srt.plugins`` wraps the plugin
call in ``except Exception``, so a component that raises at registration leaves
a server that starts normally, serves on the pinned kernels, and produces
numbers indistinguishable from "the component did nothing".

So this module records what happened, and it records it where a verifier can
read it: in-process through :func:`snapshot`, and optionally as JSON through
``SGLANG_RESTOMOD_ATTEST=/path/attest.json``.

Nothing here may raise. A component that fails is recorded as failed and the
rest still load; an observability hook that can take down the server is worse
than the silence it replaces.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

ENV_ATTEST = "SGLANG_RESTOMOD_ATTEST"

_STATE: dict[str, Any] = {
    "selected": [],
    "components": {},
    "loaded": [],
    "failed": [],
}


def reset() -> None:
    """Clear the record. For tests."""
    _STATE.update({"selected": [], "components": {}, "loaded": [], "failed": []})


def set_selected(names: list[str]) -> None:
    _STATE["selected"] = list(names)


def engine() -> dict[str, Any]:
    """Identify the engine that is actually imported, not the one installed."""
    info: dict[str, Any] = {"python": sys.version.split()[0]}
    module = sys.modules.get("sglang")
    if module is not None:
        info["path"] = getattr(module, "__file__", None)
        # version as the imported tree reports it, which is not the installed
        # distribution when the tree comes from PYTHONPATH
        info["version"] = getattr(module, "__version__", None)
    else:
        info["path"] = None
        info["version"] = None
    return info


def record(name: str, ok: bool, detail: str = "") -> None:
    """Record one component's registration outcome."""
    entry = {"ok": bool(ok)}
    if detail:
        entry["detail"] = detail
    _STATE["components"][name] = entry
    if ok:
        if name not in _STATE["loaded"]:
            _STATE["loaded"].append(name)
        if name in _STATE["failed"]:
            _STATE["failed"].remove(name)
    else:
        if name not in _STATE["failed"]:
            _STATE["failed"].append(name)


def snapshot() -> dict[str, Any]:
    """Everything recorded so far, plus the engine identity."""
    out = {
        "engine": engine(),
        "selected": list(_STATE["selected"]),
        "loaded": list(_STATE["loaded"]),
        "failed": list(_STATE["failed"]),
        "components": {k: dict(v) for k, v in _STATE["components"].items()},
    }
    # A component selected but never recorded did not get a chance to run, which
    # is a different failure from one that raised.
    out["missing"] = [n for n in out["selected"] if n not in out["components"]]
    return out


def write(path: str | None = None) -> str | None:
    """Write the snapshot as JSON; return the path, or None when not configured.

    A write failure is logged and swallowed: the file is for observers, and an
    unwritable path must not stop a server from starting.
    """
    target = path or os.environ.get(ENV_ATTEST)
    if not target:
        return None
    try:
        payload = snapshot()
        tmp = f"{target}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, target)
    except Exception:  # pragma: no cover - never fail a launch for a log file
        logger.warning("sglang-restomod: could not write attestation to %s", target, exc_info=True)
        return None
    logger.info(
        "sglang-restomod: attested %d loaded, %d failed, %d missing -> %s",
        len(payload["loaded"]),
        len(payload["failed"]),
        len(payload["missing"]),
        target,
    )
    return target
