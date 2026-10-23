#!/usr/bin/env python3
"""Verify that the pin file, provenance file, and HEAD agree."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = ROOT / "UPSTREAM_PIN.toml"
PROV = ROOT / "v100_plus" / "provenance.json"


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def main() -> int:
    pin = tomllib.loads(PIN.read_text())["upstream"]
    prov = json.loads(PROV.read_text())
    errors: list[str] = []

    if pin["ref"] != prov["current_upstream"]:
        errors.append(
            "UPSTREAM_PIN.ref "
            f"({pin['ref']}) != provenance.current_upstream "
            f"({prov['current_upstream']})"
        )

    ancestor = git("merge-base", "--is-ancestor", pin["ref"], "HEAD")
    if ancestor.returncode != 0:
        head = git("rev-parse", "--short", "HEAD").stdout.strip()
        errors.append(f"pin {pin['ref']} is not an ancestor of HEAD ({head})")

    # The `upstream` branch is maintained by pin-sync from this file. Do not
    # compare it here: a bump PR would fail before the branch is moved.

    if errors:
        print("pin check FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    label = pin.get("tag") or pin["ref"][:12]
    print(f"pin OK: {label} ({pin['ref']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
