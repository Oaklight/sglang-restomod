"""``restomod doctor``: one command that answers "is this host the stack we think".

The command is import-light: nothing heavy is imported until a check needs it,
so ``restomod --help`` and a missing site config are cheap. It never requires
our mounts to *import*; a missing site file is a diagnostic, not a crash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

MANIFEST_NAME = "BINARIES.toml"
SITE_ENV = "RESTOMOD_SITE_CONFIG"
MOE_TILE_DIR = "srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1"
MOE_TILES = (
    "E=512,N=160,device_name=Tesla_V100-SXM2-32GB.json",
    "E=512,N=160,device_name=Tesla_V100-SXM2-32GB_down.json",
)
VERSION_PACKAGES = (
    "torch",
    "triton",
    "tilelang",
    "sglang-kernel",
    "flashinfer-python",
    "transformers",
)


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_commit(root: Path) -> str | None:
    """The commit a checkout records, from git or from a ``COMMIT`` file.

    The ``COMMIT`` file is for a deployed tree that is not a git checkout; it
    is written when the tree is built. Either source is authoritative for
    "which code is this".
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 - not a git tree, or git is absent
        pass
    marker = root / "COMMIT"
    try:
        text = marker.read_text().strip()
    except Exception:  # noqa: BLE001
        return None
    return text.splitlines()[0].strip() if text else None


def _commit_matches(recorded: str | None, expected: str) -> bool:
    if not recorded or not expected:
        return False
    return recorded.startswith(expected) or expected.startswith(recorded)


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001 - a missing version is a report line
        return "<missing>"


def _find_manifest(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    candidates = (
        Path.cwd() / MANIFEST_NAME,
        Path(__file__).resolve().parents[1] / MANIFEST_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _find_site(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    named = os.environ.get(SITE_ENV)
    if named:
        return Path(named)
    candidate = Path.cwd() / "restomod.site.toml"
    return candidate if candidate.is_file() else None


def _selected_components(site: dict[str, Any]) -> list[str]:
    raw = site.get("components", {}).get("selected") or os.environ.get(
        "SGLANG_RESTOMOD_COMPONENTS", ""
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_artefact(artefact: dict[str, Any], paths: dict[str, Any]) -> Path | None:
    resolve = artefact.get("resolve") or {}
    key = resolve.get("site_path")
    if not key:
        return None
    value = paths.get(key)
    return Path(value) if value else None


def doctor(args: argparse.Namespace) -> int:
    ok: list[str] = []
    failed: list[str] = []
    notes: list[str] = []

    def check(condition: bool, label: str) -> None:
        (ok if condition else failed).append(label)

    manifest_path = _find_manifest(args.manifest)
    manifest: dict[str, Any] = {}
    if manifest_path and manifest_path.is_file():
        try:
            manifest = _load_toml(manifest_path)
            ok.append(f"manifest: {manifest_path}")
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            failed.append(f"manifest: {manifest_path} does not parse ({exc})")
    else:
        failed.append(f"manifest: not found ({MANIFEST_NAME})")

    site_path = _find_site(args.site)
    site: dict[str, Any] = {}
    if site_path and site_path.is_file():
        try:
            site = _load_toml(site_path)
            ok.append(f"site config: {site_path}")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"site config: {site_path} does not parse ({exc})")
    else:
        notes.append("site config: none (paths and hashes cannot be checked)")

    selected = _selected_components(site)
    if selected:
        os.environ.setdefault("SGLANG_RESTOMOD_COMPONENTS", ",".join(selected))
    os.environ.setdefault("SGLANG_PLUGINS", "restomod")
    if site.get("components", {}).get("attest") and "SGLANG_RESTOMOD_ATTEST" not in os.environ:
        os.environ["SGLANG_RESTOMOD_ATTEST"] = site["components"]["attest"]

    paths = site.get("paths", {})
    turbomind = paths.get("turbomind_so")
    if turbomind and "SGLANG_SM70_TURBOMIND_SO" not in os.environ:
        # The component's discovery accepts a file-naming variable first. The
        # site config is that declaration, so a host does not depend on a
        # shell export that nothing announces.
        os.environ["SGLANG_SM70_TURBOMIND_SO"] = turbomind

    print("restomod doctor")
    print(f"  python: {sys.executable}")
    print(f"  selected components: {','.join(selected) if selected else '<none>'}")

    # Engine resolution. This import also runs the plugin loader.
    sglang = None
    try:
        import sglang as _sglang

        sglang = _sglang
        print(f"  sglang.__file__: {sglang.__file__}")
        engine_version = getattr(sglang, "__version__", "?")
        print(f"  sglang version: {engine_version} (metadata {_version('sglang')})")
        ok.append("engine: sglang imports")
        check("/var/tmp" not in str(sglang.__file__), "engine: not a /var/tmp checkout")
    except Exception as exc:  # noqa: BLE001
        failed.append(f"engine: sglang does not import ({exc})")

    # Plugin entry point: without it, SGLANG_PLUGINS has nothing to load and a
    # backend such as qsa_v100 is rejected as an invalid choice.
    try:
        entry_points = [ep.name for ep in metadata.entry_points(group="sglang.srt.plugins")]
        print(f"  plugin entry points: {entry_points}")
        check("restomod" in entry_points, "plugin: restomod entry point resolves")
    except Exception as exc:  # noqa: BLE001
        failed.append(f"plugin: entry point lookup failed ({exc})")

    # Components that registered but did nothing are the silent failure.
    if sglang is not None:
        try:
            from sglang.srt.plugins import load_plugins

            load_plugins()
            from sglang_restomod import attest

            snapshot = attest.snapshot()
            components = snapshot.get("components", {})
            print(f"  components loaded: {snapshot.get('loaded')}")
            print(f"  components inactive: {snapshot.get('inactive')}")
            print(f"  components failed: {snapshot.get('failed')}")
            print(f"  components missing: {snapshot.get('missing')}")
            for name in snapshot.get("inactive", []):
                detail = components.get(name, {}).get("detail", "")
                print(f"  inactive detail: {name}: {detail}")
                failed.append(f"components: {name} inactive ({detail})")
            if not snapshot.get("inactive"):
                ok.append("components: none inactive")
            for name in snapshot.get("failed", []):
                detail = components.get(name, {}).get("detail", "")
                failed.append(f"components: {name} failed ({detail})")
            if not snapshot.get("failed"):
                ok.append("components: none failed")
            for name in snapshot.get("missing", []):
                failed.append(f"components: {name} selected but never recorded")
            if not snapshot.get("missing"):
                ok.append("components: none unrecorded")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"components: could not load the plugin set ({exc})")

    # The deployed tree must record the commit it was built from, and the run
    # must declare the commit it expects. A stale tree is an error, not a
    # configuration choice: the 2026-11-15 overlay failure served bf16 dense
    # linears under a full component list because the site tree predated
    # `_sm70_paths` and every gate passed.
    try:
        import sglang_restomod as _restomod

        restomod_root = Path(_restomod.__file__).resolve().parents[1]
        recorded = _tree_commit(restomod_root)
        declared = (
            site.get("engine", {}).get("commit")
            or os.environ.get("RESTOMOD_EXPECT_COMMIT", "")
        ).strip()
        print(f"  restomod tree: {restomod_root}")
        print(f"  restomod commit: {recorded or '<not recorded>'}")
        print(f"  declared commit: {declared or '<none>'}")
        if declared:
            if recorded is None:
                failed.append("commit: the tree records no commit (git or COMMIT file)")
            else:
                check(
                    _commit_matches(recorded, declared),
                    f"commit: {recorded} matches {declared}",
                )
        else:
            notes.append(
                "commit: no declared commit ([engine].commit or RESTOMOD_EXPECT_COMMIT)"
            )
    except Exception as exc:  # noqa: BLE001
        failed.append(f"commit: cannot locate the restomod tree ({exc})")

    # Hand-tuned V100 MoE tiles must resolve in the tree the engine resolves from.
    if sglang is not None:
        engine_dir = Path(sglang.__file__).resolve().parent
        for name in MOE_TILES:
            tile = engine_dir / MOE_TILE_DIR / name
            print(f"  moe tile: {tile}")
            check(tile.is_file(), f"moe tiles: {name} resolves in the engine tree")

    for name in VERSION_PACKAGES:
        print(f"  {name}: {_version(name)}")

    # Binary hashes. A missing artefact fails only when a selected component
    # consumes it; otherwise it is a note.
    paths = site.get("paths", {})
    for artefact in manifest.get("artefact", []):
        name = artefact.get("name", "<unnamed>")
        if artefact.get("kind") == "image":
            # The image is checked below, once, with its size and optional hash.
            continue
        consumers = set(artefact.get("consumers", []))
        required = bool(consumers & set(selected))
        expected = artefact.get("sha256")
        probe = _resolve_artefact(artefact, paths)
        if probe is None:
            notes.append(f"{name}: no site path configured")
            continue
        if not probe.is_file():
            if required:
                needed = ",".join(sorted(consumers & set(selected)))
                failed.append(f"{name}: missing at {probe} ({needed})")
            else:
                notes.append(f"{name}: absent at {probe} (no selected consumer)")
            continue
        if not expected:
            notes.append(f"{name}: present, no sha256 in the manifest")
            continue
        actual = _sha256(probe)
        if actual == expected:
            ok.append(f"{name}: sha256 matches")
        else:
            failed.append(f"{name}: sha256 {actual} != {expected}")

    image = paths.get("cuda_image")
    if image:
        image_path = Path(image)
        if not image_path.is_file():
            failed.append(f"cuda-image: missing at {image_path}")
        else:
            size = image_path.stat().st_size
            print(f"  cuda image: {image_path} ({size} bytes)")
            if args.deep:
                actual = _sha256(image_path)
                expected = next(
                    (
                        a.get("sha256")
                        for a in manifest.get("artefact", [])
                        if a.get("name") == "cuda-image"
                    ),
                    None,
                )
                check(actual == expected, "cuda-image: sha256 matches")
            else:
                notes.append("cuda-image: size only; pass --deep to hash it")

    print()
    for line in ok:
        print(f"ok   {line}")
    for line in notes:
        print(f"note {line}")
    for line in failed:
        print(f"FAIL {line}")
    print()
    print(f"restomod doctor: {len(ok)} ok, {len(notes)} notes, {len(failed)} failed")
    if args.json:
        print(
            json.dumps(
                {"ok": ok, "notes": notes, "failed": failed, "selected": selected},
                indent=2,
            )
        )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="restomod")
    sub = parser.add_subparsers(dest="command", required=True)
    doc = sub.add_parser("doctor", help="check that this host is the expected stack")
    doc.add_argument("--manifest", help=f"path to {MANIFEST_NAME}")
    doc.add_argument("--site", help=f"path to a site config (default: ${SITE_ENV})")
    doc.add_argument("--deep", action="store_true", help="hash the container image too")
    doc.add_argument("--json", action="store_true", help="also print the report as JSON")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
