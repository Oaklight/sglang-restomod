#!/usr/bin/env python3
"""Remove machine identity from an evidence directory.

``scripts/gpu_suite.sh`` runs this after every file is written and before the
archive is packed, because the archive is uploaded to a public release asset.
It rewrites absolute paths under a home or site mount, NVIDIA serial numbers,
UUIDs and bus ids, the ``host=`` line, and any name from ``IDENTITY_DENYLIST``.

The pass is content-idempotent: a second run rewrites nothing. The record is
idempotent too, by accumulation. ``redaction.txt`` describes the bundle, not the
last pass, so counts from an earlier pass are kept and a later pass that finds
nothing leaves them standing. Rewriting the record with the current pass's
zeros would erase the only account of what the published archive had removed.

Two properties of this tool that the person running it needs to know, so they
are here and not only in ``docs/gpu-receipts.md``:

**The local nodename is redacted unconditionally**, with no flag and no list:
this script runs on the host that produced the bundle, so it knows the name the
host writes about itself. The host name is the commonest leak and needs no
operator input.

**The local account is redacted too, unless it is generic.** A name such as
``pding`` is identity worth removing and cannot match prose; ``root``, ``user``,
``admin`` and a handful of others are ordinary words, and rewriting them would
corrupt the bundle. Generic accounts are named in ``_GENERIC_ACCOUNTS`` and are
left alone rather than causing a refusal.

This is the narrow half of the account problem: it covers the account that ran
the pass, not a co-tenant named in a process table or a scheduler log. No
collector in this repository emits one today -- ``gpu_suite.sh`` calls
``nvidia-smi -q``, one GPU-only csv and ``topo -m``, and no ``ps``, ``pgrep``,
``who`` or ``--query-compute-apps`` -- so the structural rule in #328 is
preventive, and that issue names the change that would make it live.

**A denylist is still needed for other hosts** (``--denylist`` or
``$IDENTITY_DENYLIST``): the bundle builder and the packer are not always the
same machine, so a name that is not this host's cannot be discovered here. When
no list is supplied the pass says so in ``redaction.txt`` as well as on stdout,
because the reviewer reading the artifact needs to know what it did not cover.

**The walk is recursive and reports what it could not inspect.** Symlinks are
reported, never followed: writing through one would edit a file outside the
bundle. Files that are not text are reported, never skipped. A bundle that
examined no files is an error, because an empty report reads exactly like a
clean one.
"""

from __future__ import annotations

import argparse
import getpass
import os
import pathlib
import pwd
import re
import socket
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
# A relative path can walk out of an evidence file and name the mount and the
# account on the way back in: a run of ../ segments followed by a path that
# holds a home root. The absolute rule never sees it, because it does not
# start at a known root.
_TRAVERSAL_RE = re.compile(  # identity-check: allow
    r"(?<![\w])(?:\.\./)+(?:[^\s\"'`,;:)\]]*)"
)

_REDACTIONS = [
    ("path", _TRAVERSAL_RE),
    ("path", _PATH_RE),
    (
        "nvidia-field",
        re.compile(
            r"(?m)^(\s*(?:Serial Number|GPU UUID|Bus Id|ClusterUUID)\s*:)(?!\s*<redacted>).*$"
        ),
    ),
    ("gpu-bus-header", re.compile(r"(?m)^GPU [0-9a-fA-F:]+")),
    # The negative lookaheads keep a rule from re-matching its own output. A
    # rewrite that changes nothing must not be counted either: the record says
    # how much identity this bundle lost, and a second pass that finds nothing
    # has to leave it alone.
    ("host-line", re.compile(r"(?m)^host=(?!<redacted>).*$")),
    ("uname-nodename", re.compile(r"(?m)^(Linux) (?!<redacted>)\S+(?=[^\n]* #)")),
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


def local_nodename() -> str | None:
    """The name this host writes about itself, or None when it has none.

    ``gethostname`` and ``uname().nodename`` are the same value on Linux, but
    either can be missing in an unusual environment, and a host name shorter
    than three characters is too weak a pattern to rewrite safely.
    """
    for probe in (socket.gethostname, lambda: os.uname().nodename):
        try:
            name = probe().strip()
        except Exception:  # noqa: BLE001 - no name is a state, not a crash
            continue
        if len(name) >= 3:
            return name
    return None


def name_pattern(name: str) -> re.Pattern[str]:
    """A name as a bounded token, so ``lab`` does not rewrite ``label``."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", re.IGNORECASE)


# Accounts that are ordinary words. Rewriting one would corrupt the bundle, and
# missing the identity in a bundle produced under ``user`` is the cheaper error.
_GENERIC_ACCOUNTS = frozenset(
    {"root", "user", "admin", "test", "ubuntu", "runner", "nobody", "app", "default"}
)


def local_account() -> str | None:
    """The account that ran the pass, or None when it has no useful name.

    A generic account is reported as None: the caller has nothing to redact and
    must not treat that as a failure.
    """
    for probe in (
        getpass.getuser,
        lambda: pwd.getpwuid(os.getuid()).pw_name,
    ):
        try:
            name = probe().strip()
        except Exception:  # noqa: BLE001 - no name is a state, not a crash
            continue
        if len(name) >= 3 and name.lower() not in _GENERIC_ACCOUNTS:
            return name
    return None


def redact_text(
    text: str,
    denylist: list[str],
    counts: dict[str, int],
    nodename: str | None = None,
    account: str | None = None,
) -> str:
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
    # The local host name first: it is the name most likely to be present, and
    # it needs no operator input. It is counted under its own rule.
    if nodename:
        text, n = name_pattern(nodename).subn("<host>", text)
        counts["nodename"] = counts.get("nodename", 0) + n
    if account:
        text, n = name_pattern(account).subn("<account>", text)
        counts["account"] = counts.get("account", 0) + n
    for item in denylist:
        pattern = re.compile(re.escape(item), re.IGNORECASE)
        text, n = pattern.subn("<redacted>", text)
        counts["denylist"] = counts.get("denylist", 0) + n
    return text


# Anything still matching these after redaction is identity the bundle must not
# be packed with. They are deliberately separate from _REDACTIONS: if a
# substitution stops working, the step fails instead of passing quietly.
_RESIDUAL = [
    ("path", _TRAVERSAL_RE),
    ("path", _PATH_RE),
    ("path", re.compile(r"(?<![\w])(?:home|homes)/[A-Za-z0-9._-]+")),
    (
        "nvidia-field",
        re.compile(
            r"(?m)^\s*(?:Serial Number|GPU UUID|Bus Id|ClusterUUID)\s*:\s*(?!<redacted>)\S"
        ),
    ),
    ("gpu-bus-header", re.compile(r"(?m)^GPU [0-9a-fA-F:]+")),
    ("host-line", re.compile(r"(?m)^host=(?!<redacted>).*$")),
    ("uname-nodename", re.compile(r"(?m)^Linux (?!<redacted>)\S+(?=[^\n]* #)")),
]


def bundle_files(directory: pathlib.Path) -> list[pathlib.Path]:
    """Every regular file in the bundle, at any depth, except the record.

    ``iterdir()`` read only the top level, so a nested directory was neither
    rewritten nor verified while ``verify()`` returned an empty list — the
    condition that means "safe to pack" (issue #320). A gate that reports clean
    on input it never examined is worse than no gate, so the walk is recursive
    and the same filters apply at every depth.

    Symlinks are not returned. Writing through one would modify a file outside
    the bundle, and reading one can leave the directory entirely; ``verify``
    reports them instead of following them.
    """
    files: list[pathlib.Path] = []
    for root, dirs, names in os.walk(directory):
        dirs.sort()
        for name in sorted(names):
            path = pathlib.Path(root) / name
            if path.name == "redaction.txt" or path.is_symlink():
                continue
            files.append(path)
    return files


def bundle_symlinks(directory: pathlib.Path) -> list[pathlib.Path]:
    """Every symlink in the bundle: not followed, and not silently dropped."""
    found: list[pathlib.Path] = []
    for root, dirs, names in os.walk(directory):
        for name in sorted(dirs):
            path = pathlib.Path(root) / name
            if path.is_symlink():
                found.append(path)
        for name in sorted(names):
            path = pathlib.Path(root) / name
            if path.is_symlink():
                found.append(path)
    return found


def _relative(path: pathlib.Path, root: pathlib.Path) -> str:
    """Name a file by its path inside the bundle, so a leak names its file."""
    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover - every path comes from a walk of root
        return path.name


def verify(
    directory: pathlib.Path,
    denylist: list[str],
    nodename: str | None = None,
    account: str | None = None,
) -> list[str]:
    """Return every residual identity marker left in the bundle.

    ``run`` calls it after rewriting; an empty list is the condition for
    packing. A non-empty list is a failure, not a warning.
    """
    leaks: list[str] = [
        f"{_relative(path, directory)}: symlink, not inspected"
        for path in bundle_symlinks(directory)
    ]
    for path in bundle_files(directory):
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            leaks.append(f"{_relative(path, directory)}: unreadable, not inspected")
            continue
        if "\0" in text:
            leaks.append(f"{_relative(path, directory)}: binary, not inspected")
            continue
        leaks.extend(
            f"{_relative(path, directory)}: residual {name}"
            for name, pattern in _RESIDUAL
            if pattern.search(text)
        )
        if nodename and name_pattern(nodename).search(text):
            leaks.append(f"{_relative(path, directory)}: residual host name")
        if account and name_pattern(account).search(text):
            leaks.append(f"{_relative(path, directory)}: residual account name")
        for item in denylist:
            if re.search(re.escape(item), text, re.IGNORECASE):
                leaks.append(f"{_relative(path, directory)}: residual denylist name")
    return leaks


def run(
    directory: pathlib.Path,
    denylist: list[str],
    nodename: str | None = None,
    account: str | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in bundle_files(directory):
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if "\0" in text:
            continue
        updated = redact_text(text, denylist, counts, nodename, account)
        if updated != text:
            path.write_text(updated)
    return counts


def _read_counts(path: pathlib.Path) -> dict[str, int]:
    """The counts an earlier pass recorded, ignoring anything unparseable."""
    counts: dict[str, int] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return counts
    for line in lines:
        name, sep, value = line.partition("\t")
        if not sep:
            continue
        try:
            counts[name] = int(value)
        except ValueError:
            continue
    return counts


def merge_counts(earlier: dict[str, int], later: dict[str, int]) -> dict[str, int]:
    """Add this pass's counts to the bundle's record; counts only grow."""
    merged = dict(earlier)
    for name, value in later.items():
        merged[name] = merged.get(name, 0) + value
    return merged


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

    nodename = local_nodename()
    account = local_account()
    if not denylist and nodename is None:
        # Neither a list nor a resolvable local name: nothing about names can
        # be removed, so a bundle from here would certify what it cannot check.
        print("error: no denylist and no resolvable host name", file=sys.stderr)
        return 2

    this_pass = run(directory, denylist, nodename, account)
    examined = len(bundle_files(directory))
    leaks = verify(directory, denylist, nodename, account)
    # The record describes the bundle, so it accumulates; a pass that finds
    # nothing must not overwrite what an earlier pass removed.
    record = merge_counts(_read_counts(directory / "redaction.txt"), this_pass)
    lines = [f"{name}\t{count}" for name, count in sorted(record.items())]
    # The record travels with the artifact, so a coverage gap belongs in it and
    # not only on the terminal that ran the pass. The line carries no tab, so
    # _read_counts ignores it and the counts still accumulate.
    gaps = (
        []
        if denylist
        else ["no denylist supplied; names other than the local nodename were not redacted"]
    )
    (directory / "redaction.txt").write_text(
        "fields redacted from this bundle; values are not recorded\n"
        + ("\n".join(lines) + "\n" if lines else "none\n")
        + ("\n".join(gaps) + "\n" if gaps else "")
    )
    print("redacted: " + (", ".join(f"{k}={v}" for k, v in sorted(this_pass.items())) or "none"))
    # The other silent skip: a bundle with nothing in it produces no leaks and
    # reads exactly like a clean one. Say what was looked at, and refuse zero.
    print(f"examined: {examined} files")
    if examined == 0:
        print(f"error: no files to examine under {directory}", file=sys.stderr)
        return 1
    if leaks:
        print("\nresidual identity in the evidence bundle:", file=sys.stderr)
        for leak in leaks:
            print(f"  {leak}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
