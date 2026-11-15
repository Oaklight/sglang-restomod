"""Locate the prebuilt SM70 libraries without depending on a fork module.

The TurboMind ``.so`` files are the one artefact the components cannot build at
import time. On the fork they are resolved by ``sglang.kernels.sm70_paths``,
which the pinned upstream does not have, so on a stock engine every component
fell back to "set an environment variable" — a provisioning step that nothing
announces and that silently disabled the dense kernel on hosts that exported
only the MoE names.

Discovery here is ordered and explicit:

1. an environment variable naming the file itself, when the caller lists one;
2. the fork's resolver, when the engine has it;
3. ``$SGLANG_SM70_PREBUILT_DIR`` (``os.pathsep`` separated);
4. ``<sglang package>/kernels/prebuilt``;
5. directories the caller passes;
6. ``sglang_restomod/prebuilt``, so a wheel can ship the libraries;
7. ``sys.prefix/lib`` and ``sys.prefix/lib64``.

This module is standard-library only and does no I/O beyond ``Path.is_file``, so
it is safe to import from a component's ``register()``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from pathlib import Path

ENV_PREBUILT_DIR = "SGLANG_SM70_PREBUILT_DIR"


def _sglang_root() -> Path | None:
    """Return the directory of the imported ``sglang`` package, if any."""
    module = sys.modules.get("sglang")
    filename = getattr(module, "__file__", None)
    if not filename:
        return None
    return Path(filename).resolve().parent


def search_dirs(extra: Iterable[str | os.PathLike[str]] = ()) -> list[Path]:
    """Return the directories searched for a prebuilt library, in order."""
    dirs: list[Path] = []
    configured = os.environ.get(ENV_PREBUILT_DIR)
    if configured:
        dirs.extend(Path(part) for part in configured.split(os.pathsep) if part)
    root = _sglang_root()
    if root is not None:
        dirs.append(root / "kernels" / "prebuilt")
    dirs.extend(Path(item) for item in extra)
    dirs.append(Path(__file__).resolve().parent / "prebuilt")
    prefix = Path(sys.prefix)
    dirs.extend((prefix / "lib", prefix / "lib64"))
    return dirs


def _from_fork(name: str) -> Path | None:
    """Ask the fork's ``sm70_prebuilt`` helper, when the engine ships one."""
    try:
        from sglang.kernels.sm70_paths import sm70_prebuilt
    except Exception:  # ImportError, or a module that exists but is broken
        return None
    try:
        candidate = Path(str(sm70_prebuilt(name)))
    except Exception:
        return None
    return candidate if candidate.is_file() else None


def find_prebuilt(
    name: str,
    *,
    env_names: Iterable[str] = (),
    extra_dirs: Iterable[str | os.PathLike[str]] = (),
) -> Path | None:
    """Return the first existing path for ``name``, or ``None``.

    An environment variable is only accepted when it names an existing file:
    a stale value must not shadow a library that discovery can still find.
    """
    for variable in env_names:
        value = os.environ.get(variable)
        if value and Path(value).is_file():
            return Path(value)

    from_fork = _from_fork(name)
    if from_fork is not None:
        return from_fork

    for directory in search_dirs(extra_dirs):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def describe_search(
    name: str,
    *,
    env_names: Iterable[str] = (),
    extra_dirs: Iterable[str | os.PathLike[str]] = (),
) -> str:
    """Return a message naming every place that was searched for ``name``."""
    tried: list[str] = []
    for variable in env_names:
        value = os.environ.get(variable)
        tried.append(f"${variable}={value or '<unset>'}")
    tried.append("sglang.kernels.sm70_paths.sm70_prebuilt")
    tried.extend(str(directory / name) for directory in search_dirs(extra_dirs))
    return (
        f"could not find {name}; searched: " + ", ".join(tried) + ". "
        f"Point ${ENV_PREBUILT_DIR} at the directory holding it."
    )
