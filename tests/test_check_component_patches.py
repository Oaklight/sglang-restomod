"""Unit tests for scripts/check_component_patches.py.

Everything here is CPU-only and offline. The patch checks run against a
temporary git repository that stands in for the cloned upstream tree, so no
network access and no real component manifest are involved.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_component_patches.py"

GOOD_PATCH = """\
diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-old
+new
"""

BAD_PATCH = """\
diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-absent
+new
"""


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("check_component_patches", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
        },
    )


def make_tree(tmp_path: Path) -> Path:
    dest = tmp_path / "upstream"
    dest.mkdir()
    git(dest, "init", "-q")
    write(dest / "hello.txt", "old\n")
    git(dest, "add", "hello.txt")
    git(dest, "commit", "-qm", "init")
    return dest


def test_one_entry_per_patch_directory(tmp_path, module):
    """Several .patch files in one directory yield one directory, not N."""

    for name in ("0001-a.patch", "0002-b.patch", "0003-c.patch"):
        write(tmp_path / "sglang_restomod" / "quant" / "thing" / "patches" / name)
    write(tmp_path / "sglang_restomod" / "attention" / "other" / "patches" / "0001.patch")

    assert module.find_patch_dirs(tmp_path) == [
        tmp_path / "sglang_restomod" / "attention" / "other" / "patches",
        tmp_path / "sglang_restomod" / "quant" / "thing" / "patches",
    ]


def test_a_good_patch_applies_and_reverse_applies(tmp_path, module):
    dest = make_tree(tmp_path)
    patch = write(tmp_path / "patches" / "0001-good.patch", GOOD_PATCH)

    assert module.check_patch(patch, dest, "example/repo", "0" * 40) == []
    assert (dest / "hello.txt").read_text() == "new\n"


def test_a_non_applying_patch_is_reported(tmp_path, module):
    dest = make_tree(tmp_path)
    patch = write(tmp_path / "patches" / "0001-bad.patch", BAD_PATCH)

    problems = module.check_patch(patch, dest, "example/repo", "0" * 40)

    assert len(problems) == 1
    assert "does not apply to example/repo@" in problems[0]
    assert (dest / "hello.txt").read_text() == "old\n"
