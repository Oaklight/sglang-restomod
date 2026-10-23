#!/usr/bin/env python3
"""Check that UPSTREAM_PIN.toml and patches/manifest.json agree."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = ROOT / "UPSTREAM_PIN.toml"
MANIFEST = ROOT / "patches" / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-remote",
        action="store_true",
        help="also require origin/upstream to point at the pin",
    )
    args = parser.parse_args()

    pin = tomllib.loads(PIN.read_text())["upstream"]
    manifest = json.loads(MANIFEST.read_text())
    errors: list[str] = []

    if manifest.get("upstream") != pin["commit"]:
        errors.append(
            f"manifest upstream {manifest.get('upstream')} != pin {pin['commit']}"
        )

    if args.check_remote:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "origin/upstream"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            errors.append("origin/upstream is not available")
        elif result.stdout.strip() != pin["commit"]:
            errors.append(
                f"origin/upstream {result.stdout.strip()} != pin {pin['commit']}"
            )

    if errors:
        print("pin check FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"pin OK: {pin['ref']} ({pin['commit']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
