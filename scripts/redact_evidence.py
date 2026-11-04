#!/usr/bin/env python3
"""Remove machine identity from an evidence directory.

``scripts/gpu_suite.sh`` runs this after every file is written and before the
archive is packed, because the archive is uploaded to a public release asset.
It rewrites absolute paths under a home or site mount, NVIDIA serial numbers,
UUIDs and bus ids, the ``host=`` line, and any name from ``IDENTITY_DENYLIST``.

The pass is idempotent and records only counts in ``redaction.txt``; the
replaced values are not written anywhere.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

_PATH_ROOTS = (  # identity-check: allow
    "home",
    "homes",
    "Users",
    "root",
    "nfs",
    "scratch",
    "gpfs",
    "lustre",
    "beegfs",
    "raid",
    "workspace",
    "mnt",
    "media",
)
_PATH_ALT = "|".join(_PATH_ROOTS)  # identity-check: allow
_PATH_RE = re.compile(  # identity-check: allow
    r"(?<![\w])/(?:" + _PATH_ALT + r")/[^\s\"'`,;:)\]]*"
)

_REDACTIONS = [
    ("path", _PATH_RE),
    ("nvidia-field", re.compile(r"(?m)^(\s*(?:Serial Number|GPU UUID|Bus Id|ClusterUUID)\s*:).*$")),
    ("gpu-bus-header", re.compile(r"(?m)^GPU [0-9a-fA-F:]+")),
    ("host-line", re.compile(r"(?m)^host=.*$")),
    ("uname-nodename", re.compile(r"(?m)^(Linux) \S+(?=[^\n]* #)")),
]

_FIELD_RE = re.compile(r"(?m)^(\s*[A-Za-z][A-Za-z ]*:\s*)(?!<redacted>).*$")


def _path_tail(match: re.Match[str]) -> str:
    parts = [p for p in match.group(0).split("/") if p]
    return "<path>/" + (parts[-1] if parts else "")


def _field_tail(match: re.Match[str]) -> str:
    return match.group(1) + " <redacted>"


def denylist_from_env() -> list[str]:
    raw = os.environ.get("IDENTITY_DENYLIST", "")
    return [p.strip() for p in re.split(r"[,\n]", raw) if len(p.strip()) >= 3]


def redact_text(text: str, denylist: list[str], counts: dict[str, int]) -> str:
    for name, pattern in _REDACTIONS:
        if name == "path":
            text, n = pattern.subn(_path_tail, text)
        elif name == "nvidia-field":
            text, n = pattern.subn(_field_tail, text)
        elif name == "host-line":
            text, n = pattern.subn("host=<redacted>", text)
        elif name == "gpu-bus-header":
            text, n = pattern.subn("GPU <bus>", text)
        elif name == "uname-nodename":
            text, n = pattern.subn(r"\1 <redacted>", text)
        else:
            text, n = pattern.subn("<redacted>", text)
        counts[name] = counts.get(name, 0) + n
    for item in denylist:
        pattern = re.compile(re.escape(item), re.IGNORECASE)
        text, n = pattern.subn("<redacted>", text)
        counts["denylist"] = counts.get("denylist", 0) + n
    return text


def run(directory: pathlib.Path, denylist: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "redaction.txt":
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if "\0" in text:
            continue
        updated = redact_text(text, denylist, counts)
        if updated != text:
            path.write_text(updated)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="evidence directory to rewrite in place")
    parser.add_argument(
        "--denylist",
        default=None,
        help="comma or newline separated names; defaults to $IDENTITY_DENYLIST",
    )
    args = parser.parse_args(argv)

    directory = pathlib.Path(args.directory)
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 2

    if args.denylist is None:
        denylist = denylist_from_env()
    else:
        denylist = [p.strip() for p in re.split(r"[,\n]", args.denylist) if len(p.strip()) >= 3]

    counts = run(directory, denylist)
    lines = [f"{name}\t{count}" for name, count in sorted(counts.items())]
    (directory / "redaction.txt").write_text(
        "fields redacted from this bundle; values are not recorded\n"
        + ("\n".join(lines) + "\n" if lines else "none\n")
    )
    print("redacted: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
