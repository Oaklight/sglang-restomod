#!/usr/bin/env python3
"""Build the artefacts `BINARIES.toml` marks as buildable.

    python scripts/build_binaries.py --print            # show the commands
    python scripts/build_binaries.py --run --only marlin-dense

The source directory for each artefact comes from the site config:

    [build]
    marlin-dense = "/path/to/marlin_v100"
    sgl-kernel-sm70 = "/path/to/sglang/python/sglang/kernels/aot"

Nothing here is site-specific; paths come from that file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="BINARIES.toml")
    parser.add_argument("--site", default=os.environ.get("RESTOMOD_SITE_CONFIG"))
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    manifest = _load(Path(args.manifest))
    site = _load(Path(args.site)) if args.site else {}
    builds = site.get("build", {})
    wanted = set(args.only) if args.only else None
    failures = 0

    for artefact in manifest.get("artefact", []):
        name = artefact.get("name", "<unnamed>")
        if wanted and name not in wanted:
            continue
        produce = artefact.get("produce") or {}
        if produce.get("kind") != "build":
            continue
        command = produce.get("command") or []
        source = builds.get(name)
        if not command:
            print(f"{name}: no build command in the manifest")
            failures += 1
            continue
        where = f" (in {source})" if source else ""
        print(f"{name}: {' '.join(command)}{where}")
        if args.run:
            if not source:
                print(f"{name}: no [build] entry in the site config", file=sys.stderr)
                failures += 1
                continue
            result = subprocess.run(command, cwd=source)
            if result.returncode:
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
