#!/usr/bin/env python3
"""Fetch the artefacts `BINARIES.toml` marks as fetchable.

    python scripts/fetch_binaries.py --check          # verify what is present
    python scripts/fetch_binaries.py --run            # download what is absent
    python scripts/fetch_binaries.py --only flashinfer --run

The site config (`RESTOMOD_SITE_CONFIG`) maps `resolve.site_path` names to
paths. Nothing here is site-specific; paths come from that file.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="BINARIES.toml")
    parser.add_argument("--site", default=os.environ.get("RESTOMOD_SITE_CONFIG"))
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--check", action="store_true", help="verify present artefacts only")
    parser.add_argument("--run", action="store_true", help="actually fetch (default: print)")
    args = parser.parse_args()

    if not args.site:
        print("error: set --site or RESTOMOD_SITE_CONFIG", file=sys.stderr)
        return 2

    manifest = _load(Path(args.manifest))
    site = _load(Path(args.site))
    paths = site.get("paths", {})
    wanted = set(args.only) if args.only else None
    failures = 0

    for artefact in manifest.get("artefact", []):
        name = artefact.get("name", "<unnamed>")
        if wanted and name not in wanted:
            continue
        produce = artefact.get("produce") or {}
        if produce.get("kind") != "fetch":
            continue
        expected = artefact.get("sha256") or ""
        target = paths.get((artefact.get("resolve") or {}).get("site_path", ""))
        url = produce.get("url")
        if not url:
            print(f"{name}: no url in the manifest")
            failures += 1
            continue
        if url.startswith("git+"):
            spec = url[4:]
            repo, _, commit = spec.partition("@")
            if args.check:
                print(f"{name}: {repo} @ {commit} (git pin; no local hash)")
                continue
            where = f" -> {target}" if target else ""
            print(f"{name}: git clone {repo} && git checkout {commit}{where}")
            if args.run and target:
                subprocess.run(["git", "clone", repo, target], check=False)
                subprocess.run(["git", "-C", target, "checkout", commit], check=False)
            continue
        if args.check:
            if not target or not Path(target).is_file():
                print(f"{name}: missing ({target})")
                failures += 1
            elif expected and _sha256(Path(target)) != expected:
                print(f"{name}: sha256 mismatch")
                failures += 1
            else:
                print(f"{name}: ok")
            continue
        print(f"{name}: GET {url} -> {target}")
        if args.run and target:
            urllib.request.urlretrieve(url, target)
            if expected and _sha256(Path(target)) != expected:
                print(f"{name}: sha256 mismatch after download", file=sys.stderr)
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
