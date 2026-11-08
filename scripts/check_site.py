#!/usr/bin/env python3
"""Audit an SGLang site tree before it is used for a measurement.

Three separate runs on 2026-11-08 were invalidated by site state rather than by
the change under test: a site missing `patches/0002` silently served with NCCL
ring allreduce (+3.16 ms/step), a site missing the tuned fused-MoE JSON silently
ran the default heuristic (+0.86 ms/step), and a half-patched site went unnoticed
because only one target was ever checked. Each of those showed up as one line in
a server log, or not at all.

This prints what the site actually is, and fails when a patch target is not in
its patched state. Tuned configs are reported but do not fail the check: a site
without them is measurable, just slower, and the run should say so explicitly.

Usage:
    check_site.py <site-dir> [--manifest PATH] [--json]
    check_site.py --record-from <unpatched-tree> <site-dir> [--manifest PATH]

Exit codes: 0 every patch target is patched, 1 any target is not, 2 the site or
the manifest cannot be read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "patches" / "manifest.json"

# The shapes measured on the V100 hosts. The filename carries E, N and the
# device name and no dtype: `get_config_dtype_str` returns None for fp16.
TUNED_CONFIGS = (
    "E=512,N=160,device_name=Tesla_V100-SXM2-32GB.json",
    "E=512,N=160,device_name=Tesla_V100-SXM2-32GB_down.json",
)
CONFIG_DIR = "srt/layers/moe/moe_runner/triton_utils/configs"

PRESENT = "PRESENT"
ABSENT = "ABSENT"
MISMATCH = "MISMATCH"
MISSING = "MISSING"

VERSION_LITERAL = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_root(site: pathlib.Path) -> pathlib.Path | None:
    """The site may be an install root, a source checkout, or either one level down."""
    for candidate in (site / "python" / "sglang", site / "sglang", site):
        if (candidate / "__init__.py").is_file():
            return candidate
    return None


def site_version(site: pathlib.Path) -> tuple[str, str]:
    """Return (version, source). An installed site records it; a source tree does not."""
    for metadata in sorted(site.glob("**/sglang-*.dist-info/METADATA")):
        for line in metadata.read_text(errors="replace").splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip(), str(metadata.relative_to(site))
    for version_file in sorted(site.glob("**/sglang/version.py")):
        match = VERSION_LITERAL.search(version_file.read_text(errors="replace"))
        if match:
            return match.group(1), str(version_file.relative_to(site))
    return "unknown", "no dist-info and no literal in version.py"


def target_of(site: pathlib.Path, package: pathlib.Path | None, relative: str) -> pathlib.Path:
    """Map a repository-relative patch target onto the site's layout.

    The manifest names `python/sglang/...` because that is where the path sits
    in the SGLang repository. An installed site is `<site>/sglang/...` with no
    `python/` level, so the prefix has to come off before the lookup: resolving
    it literally reports MISSING for every target on a correctly patched
    install, which reads like a missing file rather than a layout mismatch.
    """
    for prefix in ("python/sglang/", "sglang/"):
        if relative.startswith(prefix) and package is not None:
            return package / relative[len(prefix):]
    return site / relative


def patch_state(
    site: pathlib.Path,
    package: pathlib.Path | None,
    relative: str,
    hashes: dict[str, str],
) -> tuple[str, str]:
    """Classify one patch target: patched, untouched, or edited by something else."""
    target = target_of(site, package, relative)
    if not target.is_file():
        return MISSING, ""
    actual = sha256(target)
    if actual == hashes.get("patched"):
        return PRESENT, actual
    if actual == hashes.get("unpatched"):
        return ABSENT, actual
    return MISMATCH, actual


def tuned_configs(site: pathlib.Path) -> list[tuple[str, str]]:
    """Which tuned MoE configs the site carries, per versioned config directory."""
    found: list[tuple[str, str]] = []
    for directory in sorted(site.glob(f"**/{CONFIG_DIR}/triton_*")):
        for name in TUNED_CONFIGS:
            if (directory / name).is_file():
                found.append((directory.name, name))
    return found


def tree_hash(site: pathlib.Path) -> str:
    """Composite over the sorted per-file listing, `__pycache__` excluded.

    Same form as `scripts/tree_hash.sh`, so two sites can be compared by one line.
    """
    lines: list[str] = []
    for path in sorted(site.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(site).as_posix()}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def audit(site: pathlib.Path, manifest: pathlib.Path) -> tuple[int, list[str]]:
    data = json.loads(manifest.read_text())
    if data.get("upstream") is None:
        raise ValueError(f"{manifest} has no 'upstream' commit")
    lines = [f"site        {site}", f"manifest    {manifest} (pin {data['upstream']})"]

    package = package_root(site)
    version, source = site_version(site)
    lines.append(f"sglang      {version} ({source})")
    layout = "source" if (site / "python" / "sglang").is_dir() else "installed"
    lines.append(f"layout      {layout}")

    failures = 0
    for name, entry in data["patches"].items():
        for relative, hashes in (entry.get("site_hashes") or {}).items():
            state, actual = patch_state(site, package, relative, hashes)
            failures += state != PRESENT
            lines.append(f"patch       {state:<9} {name} -> {relative}")
            if state == MISMATCH:
                lines.append(f"            actual {actual}")

    found = tuned_configs(site)
    if not found:
        lines.append(
            "config      MISSING tuned MoE configs for E=512,N=160 on V100; "
            "the run will use the default heuristic"
        )
    for directory, name in found:
        lines.append(f"config      present   {directory}/{name}")

    lines.append(f"tree        {tree_hash(site)}")
    return failures, lines


def record(unpatched: pathlib.Path, site: pathlib.Path, manifest: pathlib.Path) -> list[str]:
    """Print the manifest block for a site known to be correctly patched."""
    data = json.loads(manifest.read_text())
    out: list[str] = []
    for name, entry in data["patches"].items():
        fragments: dict[str, dict[str, str]] = {}
        for relative in entry.get("paths", []):
            target = site / relative
            if not target.is_file():
                continue
            fragments[relative] = {
                "unpatched": sha256(unpatched / relative),
                "patched": sha256(target),
            }
        if fragments:
            out.append(f'"{name}": {json.dumps({"site_hashes": fragments}, indent=1)}')
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    parser.add_argument(
        "--record-from",
        type=pathlib.Path,
        metavar="UNPATCHED-TREE",
        help="print manifest site_hashes for <site>, using this tree for the unpatched side",
    )
    args = parser.parse_args(argv)

    site = args.site.resolve()
    if not site.is_dir():
        print(f"not a directory: {site}", file=sys.stderr)
        return 2
    try:
        if args.record_from:
            for block in record(args.record_from.resolve(), site, args.manifest):
                print(block)
            return 0
        failures, lines = audit(site, args.manifest)
    except (OSError, ValueError, KeyError) as exc:
        print(f"cannot read the site or the manifest: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"site": str(site), "lines": lines, "failures": failures}))
    else:
        print("\n".join(lines))
        print("SITE: FAIL" if failures else "SITE: OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
