"""The identity check fails on planted shapes and passes on this tree."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_identity", ROOT / "scripts" / "check_identity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_identity = _load()


def _plant(tmp_path, text):
    path = tmp_path / "planted.txt"
    path.write_text(text)
    return path


def test_planted_fqdn_fails(tmp_path):
    host = "bench-01." + "example.net"
    path = _plant(tmp_path, f"reached {host} today\n")
    assert check_identity.main(["--paths", str(path)]) == 1


def test_planted_home_path_fails(tmp_path):
    home = "/".join(["", "home", "operator", "run.log"])
    path = _plant(tmp_path, f"log at {home}\n")
    assert check_identity.main(["--paths", str(path)]) == 1


def test_planted_site_path_fails(tmp_path):
    site = "/".join(["", "nfs", "vol1", "checkpoints"])
    path = _plant(tmp_path, f"weights in {site}\n")
    assert check_identity.main(["--paths", str(path)]) == 1


def test_planted_serial_fails(tmp_path):
    serial = "04" + "21519" + "019240"
    path = _plant(tmp_path, f"Serial Number : {serial}\n")
    assert check_identity.main(["--paths", str(path)]) == 1


def test_denylisted_name_fails(tmp_path):
    path = _plant(tmp_path, "signed on node-q7\n")
    assert check_identity.main(["--paths", str(path), "--denylist", "node-q7"]) == 1


def test_public_host_passes(tmp_path):
    path = _plant(tmp_path, "see https://github.com/Oaklight/sglang-restomod\n")
    assert check_identity.main(["--paths", str(path)]) == 0


@pytest.mark.skipif(
    not (ROOT / ".git").exists(),
    reason="not a git checkout: the check lists tracked files with git ls-files",
)
def test_repository_is_clean():
    assert check_identity.main(["--root", str(ROOT)]) == 0
