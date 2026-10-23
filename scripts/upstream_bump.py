#!/usr/bin/env python3
"""Update UPSTREAM_PIN.toml and patches/manifest.json together."""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = ROOT / "UPSTREAM_PIN.toml"
MANIFEST = ROOT / "patches" / "manifest.json"


def resolve(ref: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{ref}^{{commit}}"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="upstream tag, e.g. v0.5.22")
    parser.add_argument("--commit", default="", help="commit SHA (resolved if omitted)")
    args = parser.parse_args()

    commit = args.commit or resolve(args.ref)
    version = args.ref.removeprefix("v")

    PIN.write_text(
        "# The upstream baseline this layer targets.\n"
        "#\n"
        "# `upstream` is a reference branch in this repository containing the\n"
        "# pristine SGLang tree at this commit. `main` does not merge it; it only\n"
        "# carries the layer, the patches, and the pin.\n"
        "#\n"
        "# `scripts/upstream_bump.py` updates this file and patches/manifest.json.\n"
        "\n"
        "[upstream]\n"
        'repo = "sgl-project/sglang"\n'
        'kind = "tag"\n'
        f'ref = "{args.ref}"\n'
        f'commit = "{commit}"\n'
        f'pypi_version = "{version}"\n'
        f'pinned_at = "{datetime.date.today().isoformat()}"\n'
    )

    manifest = json.loads(MANIFEST.read_text())
    manifest["upstream"] = commit
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"pinned {args.ref} -> {commit}")
    print(f"updated {PIN.relative_to(ROOT)} and {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
