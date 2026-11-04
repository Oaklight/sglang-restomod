#!/usr/bin/env python3
"""Fail when a tracked file carries a machine identity.

The rules are generic shapes: fully-qualified domain names, home-directory
paths, absolute paths under a site mount, long digit-only tokens, and NVIDIA
device identifiers. Site-specific names -- host names, user names, mount roots
-- come from ``IDENTITY_DENYLIST``, which CI fills from a repository secret.
This file never contains them, and a checked-in denylist would itself be the
leak.

A line that legitimately has to hold a pattern (this file, for example) marks
itself with ``identity-check: allow``.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOW_MARK = "identity-check: allow"  # identity-check: allow

# Rules are assembled from parts so the source does not match itself.
_HOME_ROOTS = ("home", "Users", "homes", "root")  # identity-check: allow
_SITE_ROOTS = (  # identity-check: allow
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
_TLDS = (  # identity-check: allow
    "com",
    "net",
    "org",
    "io",
    "ai",
    "dev",
    "cloud",
    "edu",
    "gov",
    "co",
    "uk",
    "de",
    "jp",
    "cn",
    "us",
    "info",
    "app",
    "xyz",
    "tech",
    "systems",
    "services",
    "internal",
    "local",
    "lan",
    "corp",
    "site",
    "online",
)

# Public hosts that may appear in docs and URLs. A name that ends with one of
# these is accepted; everything else shaped like a host is a finding.
PUBLIC_HOSTS = (  # identity-check: allow
    "github.com",
    "githubusercontent.com",
    "pypi.org",
    "pytorch.org",
    "img.shields.io",
    "jsdelivr.net",
    "apache.org",
    "opensource.org",
    "python.org",
    "pytest.org",
    "readthedocs.io",
    "nvidia.com",
    "huggingface.co",
    "arxiv.org",
    "doi.org",
    "anaconda.org",
    "conda.io",
    "example.invalid",
    "example.com",
    "example.org",
)

_HOME_ALT = "|".join(_HOME_ROOTS)  # identity-check: allow
_SITE_ALT = "|".join(_SITE_ROOTS)  # identity-check: allow
_TLD_ALT = "|".join(_TLDS)  # identity-check: allow

RULES = [
    (
        "home-path",
        re.compile(r"(?<![\w])/(?:" + _HOME_ALT + r")/\S*"),  # identity-check: allow
    ),
    (
        "site-path",
        re.compile(r"(?<![\w])/(?:" + _SITE_ALT + r")/\S*"),  # identity-check: allow
    ),
    (
        "windows-home",
        re.compile(r"[A-Za-z]:\\Users\\\S*"),  # identity-check: allow
    ),
    (
        "fqdn",
        re.compile(
            r"(?<![\w@.-])(?:[A-Za-z][A-Za-z0-9-]{0,62}\.){1,8}(?:"
            + _TLD_ALT  # identity-check: allow
            + r")(?![\w-])",
            re.IGNORECASE,
        ),
    ),
    ("gpu-identifier", re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}")),
    ("long-digits", re.compile(r"(?<![\w.])\d{10,}(?![\w.])")),
]


def _public_host(token: str) -> bool:
    host = token.lower().lstrip("/")
    return any(host == p or host.endswith("." + p) for p in PUBLIC_HOSTS)


def _host_shaped(token: str) -> bool:
    """A two-label ``a.tld`` token is also a common attribute pair.

    ``logger.info`` and ``self.local`` must not read as hosts. A name that
    carries a digit or a hyphen in a label, or that has three labels, does not
    occur as an attribute chain here, so those are taken as hosts.
    """
    labels = token.split(".")
    if len(labels) >= 3:
        return True
    return any(ch.isdigit() or ch == "-" for ch in token.rsplit(".", 1)[0])


def denylist_from_env() -> list[str]:
    raw = os.environ.get("IDENTITY_DENYLIST", "")
    out = []
    for part in re.split(r"[,\n]", raw):
        item = part.strip()
        if len(item) >= 3:
            out.append(item)
    return out


def tracked_files(root: pathlib.Path) -> list[pathlib.Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    names = [n for n in result.stdout.decode().split("\0") if n]
    files = []
    for name in names:
        path = root / name
        # A submodule is one index entry whose path is a directory.
        if path.is_file():
            files.append(path)
    return files


def scan_text(text: str, denylist: list[str]) -> list[tuple[int, str, str]]:
    findings = []
    lowered = [(d.lower(), d) for d in denylist]
    for lineno, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARK in line:
            continue
        for name, pattern in RULES:
            for match in pattern.finditer(line):
                token = match.group(0)
                if name == "fqdn" and (_public_host(token) or not _host_shaped(token)):
                    continue
                findings.append((lineno, name, token))
        low = line.lower()
        for needle, original in lowered:
            if needle in low:
                findings.append((lineno, "denylist", original))
    return findings


def scan_file(path: pathlib.Path, root: pathlib.Path, denylist: list[str]):
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return []
    if "\0" in text:
        return []
    try:
        shown = str(path.relative_to(root))
    except ValueError:
        shown = str(path)
    return [
        (shown, lineno, rule, token)
        for lineno, rule, token in scan_text(text, denylist)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="repository root")
    parser.add_argument(
        "--paths",
        nargs="*",
        help="scan these files instead of every tracked file",
    )
    parser.add_argument(
        "--denylist",
        default=None,
        help="comma or newline separated names; defaults to $IDENTITY_DENYLIST",
    )
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root).resolve()
    if args.denylist is None:
        denylist = denylist_from_env()
    else:
        denylist = [p.strip() for p in re.split(r"[,\n]", args.denylist) if len(p.strip()) >= 3]

    if args.paths:
        files = [pathlib.Path(p) for p in args.paths]
        base = root
    else:
        files = tracked_files(root)
        base = root

    findings = []
    for path in files:
        path = path if path.is_absolute() else (base / path)
        if path.is_file():
            findings.extend(scan_file(path, root, denylist))

    if findings:
        print("identity check FAILED", file=sys.stderr)
        for rel, lineno, rule, token in findings:
            print(f"  {rel}:{lineno}: {rule}: {token}", file=sys.stderr)
        print(
            f"{len(findings)} finding(s) in {len({f[0] for f in findings})} file(s)",
            file=sys.stderr,
        )
        return 1

    note = "" if denylist else " (no IDENTITY_DENYLIST supplied)"
    print(f"identity OK: {len(files)} file(s){note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
