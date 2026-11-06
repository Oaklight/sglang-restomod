"""LSA on SM70: registration for the standalone ``lsa-sm70`` package.

The implementation is not vendored here. ``Oaklight/lsa-sm70`` is the only
source: it carries the selection core, the Triton programs, the read and the
``LsaSm70Backend``. ``UPSTREAM_PIN.toml`` records the commit. This module keeps
the component name ``lsa_sm70`` and stays import-light, as the component
contract requires: the package is imported only when ``register()`` runs, and
the package itself defers torch, triton and sglang to first use.

Selection: ``SGLANG_RESTOMOD_COMPONENTS=lsa_sm70``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO = "https://github.com/Oaklight/lsa-sm70"


def _install_command() -> str:
    """Name the pinned commit in the error, when the pin file is at hand."""
    pin = Path(__file__).resolve().parents[3] / "UPSTREAM_PIN.toml"
    try:
        import tomllib

        commit = tomllib.loads(pin.read_text())["lsa"]["commit"]
        ref = f"@{commit}"
    except Exception:  # missing pin file or section: still give a usable command
        ref = ""
    return f"pip install 'lsa-sm70 @ git+{REPO}{ref}'"


def register() -> None:
    """Register the LSA attention backend from the ``lsa-sm70`` package."""
    try:
        from lsa_sm70 import register as register_package
    except ModuleNotFoundError as exc:
        if exc.name != "lsa_sm70":
            raise
        raise RuntimeError(
            f"the lsa_sm70 component needs the standalone package: {_install_command()}"
        ) from exc
    register_package()
    logger.debug("sglang-restomod: lsa_sm70 delegated to the lsa-sm70 package")
