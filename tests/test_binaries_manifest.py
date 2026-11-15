"""CPU tests for the artefact manifest and the doctor's pure helpers."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "restomod_cli", ROOT / "sglang_restomod" / "cli.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict:
    with (ROOT / "BINARIES.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_manifest_entries_are_complete_and_unique():
    entries = _manifest()["artefact"]
    names = [entry["name"] for entry in entries]
    assert len(names) == len(set(names)), "artefact names must be unique"
    for entry in entries:
        assert entry.get("kind"), entry.get("name")
        assert isinstance(entry.get("consumers"), list), entry.get("name")
        assert entry.get("silent_break"), entry.get("name")
        assert entry.get("produce", {}).get("kind") in {
            "fetch",
            "build",
            "copy",
            "ship",
        }, entry.get("name")


def test_tracked_rows_match_their_files():
    entries = {entry["name"]: entry for entry in _manifest()["artefact"]}
    cli = _load_cli()

    tiles = entries["moe-tiles-v100"]
    tile_dir = ROOT / "sglang_restomod" / "configs" / "moe" / "triton_3_5_1"
    tile_files = sorted(tile_dir.glob("E=512,N=160,device_name=Tesla_V100-SXM2-32GB*.json"))
    assert len(tile_files) == 2, tile_files
    for tile in tile_files:
        assert cli._sha256(tile) == tiles["sha256"], tile.name

    qpn8 = entries["qpn8-decode-cu"]
    source = (
        ROOT
        / "sglang_restomod"
        / "quant"
        / "sm70_fp8_dense"
        / "csrc"
        / "sm70_fp8_qpn8_decode.cu"
    )
    assert source.is_file()
    assert cli._sha256(source) == qpn8["sha256"]


def test_site_example_is_path_free():
    text = (ROOT / "restomod.site.example.toml").read_text()
    forbidden = (
        "/nfs",
        "/homes",
        "/home/",  # identity-check: allow
        "/scratch/",  # identity-check: allow
        "/gpfs",
        "/lustre",
        "/workspace",
    )
    for name in forbidden:
        assert name not in text, name
    with (ROOT / "restomod.site.example.toml").open("rb") as handle:
        site = tomllib.load(handle)
    assert "paths" in site
    assert "components" in site


def test_sha256_helper(tmp_path):
    cli = _load_cli()
    target = tmp_path / "blob"
    target.write_bytes(b"restomod")
    import hashlib

    assert cli._sha256(target) == hashlib.sha256(b"restomod").hexdigest()
