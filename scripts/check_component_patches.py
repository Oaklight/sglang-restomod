#!/usr/bin/env python3
"""Check component patches against the upstream trees they apply to.

Component patches (``sglang_restomod/**/patches/*.patch``) do not touch the
pinned sglang source, so ``check_patches.py`` never sees them. Without a check
they can rot silently: the top-level guard passes with such a patch deleted.

Each component patch directory carries a ``manifest.json``:

    {
      "repo": "zhinianqin/marlin_v100",
      "commit": "3f16d442cdb4c24dd225bbec196c982a54d9a31c",
      "patches": ["0001-splitk-from-occupancy.patch"]
    }

For every entry this script clones the repository at that commit, applies the
patch with ``git apply --check``, applies it for real, and then checks that it
reverse-applies — which is what proves the file in the tree is the patch we
think it is.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_NAME = "manifest.json"


def run(cmd: list[str], cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def clone_at(repo: str, commit: str, dest: pathlib.Path) -> None:
    """Shallow-clone one commit without pulling the whole history."""
    dest.mkdir(parents=True, exist_ok=True)
    steps = [
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", f"https://github.com/{repo}.git"],
        ["git", "fetch", "-q", "--depth", "1", "origin", commit],
        ["git", "checkout", "-q", "FETCH_HEAD"],
    ]
    for cmd in steps:
        result = run(cmd, cwd=dest)
        if result.returncode != 0:
            raise RuntimeError(
                f"{' '.join(cmd)} failed: {result.stderr.strip() or result.stdout.strip()}"
            )


def check_dir(patch_dir: pathlib.Path, workdir: pathlib.Path) -> list[str]:
    manifest_path = patch_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"{patch_dir}: no {MANIFEST_NAME}"]

    manifest = json.loads(manifest_path.read_text())
    repo = manifest["repo"]
    commit = manifest["commit"]
    patches = manifest.get("patches") or []
    if not patches:
        return [f"{manifest_path}: empty patch list"]

    dest = workdir / patch_dir.name
    clone_at(repo, commit, dest)

    problems: list[str] = []
    for name in patches:
        patch = patch_dir / name
        if not patch.is_file():
            problems.append(f"{patch}: missing")
            continue
        check = run(["git", "apply", "--check", str(patch)], cwd=dest)
        if check.returncode != 0:
            problems.append(f"{patch}: does not apply to {repo}@{commit[:12]}: "
                            f"{check.stderr.strip()}")
            continue
        applied = run(["git", "apply", str(patch)], cwd=dest)
        if applied.returncode != 0:
            problems.append(f"{patch}: apply failed: {applied.stderr.strip()}")
            continue
        reverse = run(["git", "apply", "--check", "--reverse", str(patch)], cwd=dest)
        if reverse.returncode != 0:
            problems.append(f"{patch}: does not reverse-apply after apply: "
                            f"{reverse.stderr.strip()}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="repository root to scan (default: this checkout)",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    patch_dirs = sorted(
        p.parent for p in root.glob("sglang_restomod/**/patches/*.patch")
    )
    if not patch_dirs:
        print("no component patch directories; nothing to check")
        return 0

    problems: list[str] = []
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="component-patches-"))
    try:
        for patch_dir in patch_dirs:
            rel = patch_dir.relative_to(root)
            print(f"checking {rel}")
            try:
                problems.extend(check_dir(patch_dir, workdir))
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{rel}: {exc}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if problems:
        print("\ncomponent patch problems:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"ok: {len(patch_dirs)} component patch directories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
