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
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

ENV_ATTEST = "SGLANG_RESTOMOD_ATTEST"

_STATE: dict[str, Any] = {
    "selected": [],
    "components": {},
    "loaded": [],
    "failed": [],
    "inactive": [],
}


def reset() -> None:
    """Clear the record. For tests."""
    _STATE.update(
        {
            "selected": [],
            "components": {},
            "loaded": [],
            "failed": [],
            "inactive": [],
        }
    )


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
    """Record one component's registration outcome.

    A reason the component recorded about itself with :func:`inactive` is kept
    when this call has no detail of its own. ``plugins.load_enabled`` records
    ``ok`` after ``register()`` returns, so overwriting here used to leave the
    name in ``inactive`` with nothing next to it saying why.
    """
    entry = {"ok": bool(ok)}
    if detail:
        entry["detail"] = detail
    else:
        previous = _STATE["components"].get(name) or {}
        if previous.get("detail"):
            entry["detail"] = previous["detail"]
    _STATE["components"][name] = entry
    if ok:
        if name not in _STATE["loaded"]:
            _STATE["loaded"].append(name)
        if name in _STATE["failed"]:
            _STATE["failed"].remove(name)
    else:
        if name not in _STATE["failed"]:
            _STATE["failed"].append(name)


def inactive(name: str, reason: str) -> None:
    """Record that a component registered but will do nothing.

    Neither of the other two states describes this. ``failed`` is wrong: the
    component loaded cleanly and the server is healthy. ``loaded`` alone is
    also wrong: it says the component is being used. On 2026-11-15 three
    components took this path (the SM70 block-FP8 library was not resolvable),
    the server served on bf16 dense linears under the same component list, and
    every gate available at the time reported success.

    Must never raise: this is observability, not control flow.
    """
    try:
        text = str(reason)
    except Exception:  # noqa: BLE001 - a bad reason must not lose the record
        text = "<unprintable reason>"
    try:
        entry = _STATE["components"].setdefault(name, {"ok": True})
        entry["detail"] = "registered, inactive: " + text
        if name not in _STATE["inactive"]:
            _STATE["inactive"].append(name)
        if name in _STATE["failed"]:
            _STATE["failed"].remove(name)
        if name not in _STATE["loaded"]:
            _STATE["loaded"].append(name)
    except Exception:  # noqa: BLE001 - see the module docstring
        logger.debug("attest.inactive failed for %s", name, exc_info=True)


def snapshot() -> dict[str, Any]:
    """Everything recorded so far, plus the engine identity."""
    out = {
        "engine": engine(),
        "selected": list(_STATE["selected"]),
        "loaded": list(_STATE["loaded"]),
        "failed": list(_STATE["failed"]),
        "inactive": list(_STATE["inactive"]),
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
    # A temporary file of its own per write, not one per target. A TP4 launch
    # registers from four processes at once, and a shared ``<target>.tmp`` is
    # consumed by the first ``os.replace``; the rest then log FileNotFoundError
    # and record nothing. That is what four ranks did on 2026-11-15.
    tmp: str | None = None
    try:
        payload = snapshot()
        directory = os.path.dirname(target) or "."
        fd, tmp = tempfile.mkstemp(
            dir=directory, prefix=f"{os.path.basename(target)}.", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, target)
        tmp = None
    except Exception:  # never fail a launch for a log file
        logger.warning("sglang-restomod: could not write attestation to %s", target, exc_info=True)
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return None
    logger.info(
        "sglang-restomod: attested %d loaded, %d failed, %d missing -> %s",
        len(payload["loaded"]),
        len(payload["failed"]),
        len(payload["missing"]),
        target,
    )
    return target
