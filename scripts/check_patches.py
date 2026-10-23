#!/usr/bin/env python3
"""Validate the patch queue and its manifest against the pinned source."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = ROOT / "UPSTREAM_PIN.toml"
PATCH_DIR = ROOT / "patches"
MANIFEST = PATCH_DIR / "manifest.json"
VALID_STATUS = {"pending", "accepted", "local-only", "rejected"}
DIFF_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)


def apply_check(source: pathlib.Path, patch: pathlib.Path, reverse: bool = False) -> int:
    cmd = ["git", "-C", str(source), "apply", "--check"]
    if reverse:
        cmd.append("--reverse")
    cmd.append(str(patch))
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=".build/sglang")
    parser.add_argument("--skip-apply", action="store_true")
    parser.add_argument(
        "--retire-report",
        action="store_true",
        help="print RETIRE lines for accepted patches already upstream; do not fail",
    )
    args = parser.parse_args()

    pin = tomllib.loads(PIN.read_text())["upstream"]
    manifest = json.loads(MANIFEST.read_text())
    errors: list[str] = []
    warnings: list[str] = []

    if manifest.get("upstream") != pin["commit"]:
        errors.append(
            f"manifest upstream {manifest.get('upstream')} != pin {pin['commit']}"
        )

    entries: dict[str, dict] = manifest.get("patches", {})
    patch_files = sorted(PATCH_DIR.glob("*.patch"))
    patch_names = {patch.name for patch in patch_files}

    for name, entry in entries.items():
        if name not in patch_names:
            errors.append(f"manifest lists {name} but no such patch file")
        status = entry.get("upstream_status", "pending")
        if status not in VALID_STATUS:
            errors.append(f"{name}: invalid upstream_status {status!r}")
        if not entry.get("paths"):
            errors.append(f"{name}: paths is empty")

    for patch in patch_files:
        entry = entries.get(patch.name)
        if entry is None:
            errors.append(f"{patch.name}: missing manifest entry")
            continue
        touched = {m.group(1) for m in DIFF_RE.finditer(patch.read_text())}
        unlisted = touched - set(entry.get("paths", []))
        if unlisted:
            errors.append(f"{patch.name}: touches unlisted paths: {sorted(unlisted)}")

    source = (ROOT / args.source).resolve()
    if not args.skip_apply:
        if not (source / ".git").exists():
            errors.append(
                f"source {source} is not a git checkout; run scripts/fetch_upstream.sh"
            )
        else:
            for patch in patch_files:
                entry = entries.get(patch.name, {})
                status = entry.get("upstream_status", "pending")
                forward = apply_check(source, patch)
                reverse = apply_check(source, patch, reverse=True)

                if status == "accepted":
                    if reverse == 0:
                        message = (
                            f"{patch.name}: accepted and already present upstream; retire it"
                        )
                        if args.retire_report:
                            print(f"RETIRE {patch.name}")
                        else:
                            errors.append(message)
                    elif forward != 0:
                        errors.append(
                            f"{patch.name}: accepted but neither forward nor reverse applies; rebase it"
                        )
                else:
                    if reverse == 0 and forward != 0:
                        warnings.append(
                            f"{patch.name}: reverse-applies but is marked {status!r}; is it already upstream?"
                        )
                    elif forward != 0:
                        errors.append(
                            f"{patch.name}: does not apply forward to the pin; rebase it"
                        )

    for warning in warnings:
        print(f"WARN {warning}", file=sys.stderr)

    if errors:
        print("patch check FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"patch check OK: {len(patch_files)} patches against {pin['ref']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
