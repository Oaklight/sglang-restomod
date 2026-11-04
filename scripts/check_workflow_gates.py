#!/usr/bin/env python3
"""Check the GPU workflow against the gates in issue #12.

The rules, all readable from the file without a YAML parser:

- ``on:`` lists only ``workflow_dispatch`` and ``schedule``. A pull-request or
  push trigger would let a fork reach the host.
- No ``secrets.`` reference anywhere: the job gets no credential.
- No ``actions/cache``: a cache written elsewhere is code this host runs.
- Every ``uses:`` is pinned to a full commit SHA.
- The job names an ``environment`` (approval gate) and a ``self-hosted``
  runner with capability labels.

Usage: ``check_workflow_gates.py [file ...]``; the default is the GPU workflow.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT = ROOT / ".github" / "workflows" / "restomod-gpu.yml"
ALLOWED_EVENTS = {"workflow_dispatch", "schedule"}
SHA = re.compile(r"^[0-9a-f]{40}$")


def events(text: str) -> set[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^['\"]?on['\"]?:\s*(.*)$", line)
        if not match:
            continue
        inline = match.group(1).strip()
        if inline:
            return {
                item.strip().strip("'\"")
                for item in inline.strip("[]").split(",")
                if item.strip()
            }
        found = set()
        for follow in lines[index + 1 :]:
            if follow and not follow[0].isspace():
                break
            key = re.match(r"^\s{2}([A-Za-z_]+):", follow)
            if key:
                found.add(key.group(1))
        return found
    return set()


def check(text: str) -> list[str]:
    problems: list[str] = []

    found = events(text)
    if not found:
        problems.append("no on: events")
    extra = found - ALLOWED_EVENTS
    if extra:
        problems.append(f"on: lists {sorted(extra)}; allowed {sorted(ALLOWED_EVENTS)}")

    for number, line in enumerate(text.splitlines(), 1):
        if "secrets." in line and not line.lstrip().startswith("#"):
            problems.append(f"line {number}: references secrets.")
        if "actions/cache" in line and not line.lstrip().startswith("#"):
            problems.append(f"line {number}: uses actions/cache")
        uses = re.search(r"\buses:\s*(\S+)", line)
        if uses and not line.lstrip().startswith("#"):
            target = uses.group(1)
            if not target.startswith("./"):
                _, _, ref = target.partition("@")
                if not SHA.match(ref):
                    problems.append(f"line {number}: uses {target} is not pinned to a SHA")

    if not re.search(r"^\s+environment:\s*\S+", text, re.MULTILINE):
        problems.append("no environment: gate")
    if "self-hosted" not in text:
        problems.append("runs-on does not name a self-hosted runner")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="workflow files; default is the GPU workflow")
    args = parser.parse_args(argv)

    files = [pathlib.Path(p) for p in args.files] or [DEFAULT]
    failed = False
    for path in files:
        if not path.is_file():
            print(f"missing workflow: {path}", file=sys.stderr)
            failed = True
            continue
        problems = check(path.read_text())
        if problems:
            failed = True
            print(f"workflow gates FAILED: {path}", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
        else:
            print(f"workflow gates OK: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
