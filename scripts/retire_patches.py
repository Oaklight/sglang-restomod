#!/usr/bin/env python3
"""Retire accepted patches that the new upstream pin already contains.

Run this on a pin bump after scripts/fetch_upstream.sh. A patch is retired when
``git apply --check --reverse`` succeeds against the new source: upstream has
the change, so keeping the patch would double-apply it or mask the upstream
version.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "patches"
MANIFEST = PATCH_DIR / "manifest.json"


def reverse_applies(source: pathlib.Path, patch: pathlib.Path) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(source), "apply", "--check", "--reverse", str(patch)],
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=".build/sglang")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = (ROOT / args.source).resolve()
    if not (source / ".git").exists():
        raise SystemExit(f"source {source} is not a git checkout")

    manifest = json.loads(MANIFEST.read_text())
    entries: dict[str, dict] = manifest.get("patches", {})
    retired: list[str] = []

    for name, entry in list(entries.items()):
        if entry.get("upstream_status") != "accepted":
            continue
        patch = PATCH_DIR / name
        if not patch.exists():
            retired.append(name)
            del entries[name]
            continue
        if reverse_applies(source, patch):
            print(f"RETIRE {name}")
            retired.append(name)
            if not args.dry_run:
                del entries[name]
                patch.unlink()
        else:
            print(f"KEEP   {name} (accepted, not yet in this pin)")

    if retired and not args.dry_run:
        manifest["patches"] = entries
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"retired {len(retired)} patch(es)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
