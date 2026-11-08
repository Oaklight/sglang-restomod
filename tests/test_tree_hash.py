"""`scripts/tree_hash.sh` must hash exactly the engine and nothing else.

The digest is used as a gate on a measurement arm or a receipt, so the two
properties that matter are: `__pycache__` and files outside the covered paths
do not move it, and a change to the pin, the patch queue or the plugin does.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tree_hash.sh"

PIN = """[upstream]
kind = "tag"
ref = "v0.5.23"
commit = "297e54fd105e974cdf401e5b4d20253a22c16f90"
"""


def _tree(root: Path) -> Path:
    (root / "UPSTREAM_PIN.toml").write_text(PIN)
    (root / "patches").mkdir()
    (root / "patches" / "0001-x.patch").write_text("diff --git a/x b/x\n")
    (root / "sglang_restomod").mkdir()
    (root / "sglang_restomod" / "mod.py").write_text("X = 1\n")
    (root / "README.md").write_text("not covered\n")
    return root


def _digest(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), str(root), *args],
        capture_output=True,
        text=True,
    )


def _value(result: subprocess.CompletedProcess[str]) -> str:
    for line in result.stdout.splitlines():
        if line.startswith("digest "):
            return line.split()[1]
    raise AssertionError(f"no digest line in {result.stdout!r}")


def test_the_digest_ignores_what_the_engine_does_not_include(tmp_path):
    root = _tree(tmp_path)
    first = _digest(root)
    assert first.returncode == 0, first.stderr
    before = _value(first)

    (root / "sglang_restomod" / "__pycache__").mkdir()
    (root / "sglang_restomod" / "__pycache__" / "mod.cpython-312.pyc").write_text("junk")
    (root / "README.md").write_text("still not covered\n")

    assert _value(_digest(root)) == before


def test_every_covered_path_moves_the_digest(tmp_path):
    root = _tree(tmp_path)
    before = _value(_digest(root))

    (root / "patches" / "0001-x.patch").write_text("diff --git a/y b/y\n")
    after_patch = _value(_digest(root))
    assert after_patch != before

    (root / "sglang_restomod" / "mod.py").write_text("X = 2\n")
    after_plugin = _value(_digest(root))
    assert after_plugin not in (before, after_patch)

    (root / "UPSTREAM_PIN.toml").write_text(PIN.replace("v0.5.23", "v0.5.24"))
    assert _value(_digest(root)) not in (before, after_patch, after_plugin)


def test_expect_fails_on_a_different_tree(tmp_path):
    root = _tree(tmp_path)
    digest = _value(_digest(root))
    assert _digest(root, "--expect", digest).returncode == 0

    (root / "sglang_restomod" / "mod.py").write_text("X = 3\n")
    failed = _digest(root, "--expect", digest)
    assert failed.returncode == 1
    assert "the tree differs" in failed.stderr


def test_a_directory_without_a_pin_is_refused(tmp_path):
    result = _digest(tmp_path)
    assert result.returncode == 2
    assert "not a pin tree" in result.stderr
