#!/usr/bin/env python3
"""Sync the pinned SM70 kernel sources into the package, or verify them.

The kernel repository is a git submodule pinned by commit in
``UPSTREAM_PIN.toml``. Each entry under ``[[kernels.files]]`` names one source
in the submodule and the generated copy inside the package. ``--check`` fails
on any drift between the pin, the submodule, a source and its copy.

Only the ``.cu`` destinations get a ``SHA256SUMS`` file; the Python kernel is
verified by the same per-file hash, without a sums file in its package
directory.

Usage:
    python scripts/sync_kernels.py            # copy + write SHA256SUMS
    python scripts/sync_kernels.py --check    # verify only, non-zero on drift
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_FILE = ROOT / "UPSTREAM_PIN.toml"


def _pin() -> dict:
    with PIN_FILE.open("rb") as fh:
        data = tomllib.load(fh)
    try:
        return data["kernels"]
    except KeyError:  # pragma: no cover - configuration error
        raise SystemExit(f"{PIN_FILE} has no [kernels] section")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _submodule_head(path: Path) -> str:
    # `git -C` walks up to the enclosing repository, so an uninitialized
    # submodule directory resolves the *parent's* HEAD and the pin check
    # compares against the wrong commit. Require the submodule's own git dir.
    if not (path / ".git").exists():
        raise SystemExit(
            f"submodule not initialized at {path}; run "
            "`git submodule update --init --recursive`"
        )
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise SystemExit("git is not available")
    except subprocess.CalledProcessError:
        raise SystemExit(
            f"{path} is not a git checkout; run "
            "`git submodule update --init --recursive` first"
        )
    return out.stdout.strip()


def _sums_text(dest_dir: Path) -> str:
    lines = []
    for path in sorted(dest_dir.glob("*.cu")):
        lines.append(f"{_sha256(path)}  {path.name}")
    return "\n".join(lines) + "\n"


def _dest_dirs(files: list[dict]) -> list[Path]:
    return sorted({(ROOT / entry["dest"]).parent for entry in files})


def _verify(kernels: dict, *, require_dest: bool) -> list[tuple[Path, Path]]:
    submodule = ROOT / kernels["path"]
    files = kernels["files"]

    if not submodule.is_dir():
        raise SystemExit(
            f"missing submodule at {submodule}; "
            "run `git submodule update --init --recursive`"
        )

    head = _submodule_head(submodule)
    if head != kernels["commit"]:
        raise SystemExit(
            f"submodule is at {head}, pin says {kernels['commit']}"
        )

    pairs = []
    for entry in files:
        source = submodule / entry["source"]
        dest = ROOT / entry["dest"]

        if not source.is_file():
            raise SystemExit(f"missing kernel source {source}")

        actual = _sha256(source)
        if actual != entry["sha256"]:
            raise SystemExit(
                f"kernel source hash drift: {source} {actual} != {entry['sha256']}"
            )

        if require_dest:
            if not dest.is_file():
                raise SystemExit(f"missing generated copy {dest}; run without --check")
            if _sha256(dest) != entry["sha256"]:
                raise SystemExit(f"generated copy differs from source: {dest}")

        pairs.append((source, dest))

    if require_dest:
        for dest_dir in _dest_dirs(files):
            if not any(dest_dir.glob("*.cu")):
                continue
            sums = dest_dir / "SHA256SUMS"
            if not sums.is_file():
                raise SystemExit(f"missing {sums}; run without --check")
            if sums.read_text() != _sums_text(dest_dir):
                raise SystemExit(f"{sums} does not match the files on disk")

    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the submodule pin, the source hashes and the generated copies",
    )
    args = parser.parse_args()

    kernels = _pin()
    pairs = _verify(kernels, require_dest=args.check)

    if args.check:
        print(f"ok: {kernels['path']}@{kernels['commit'][:12]} ({len(pairs)} files)")
        return 0

    for source, dest in pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        print(f"copied {source.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")

    for dest_dir in _dest_dirs(kernels["files"]):
        if any(dest_dir.glob("*.cu")):
            (dest_dir / "SHA256SUMS").write_text(_sums_text(dest_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
