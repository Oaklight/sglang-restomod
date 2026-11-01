"""Pin the module-global ``is_hip`` override to its upstream call sites (CPU).

``overrides.py`` makes ``is_hip()`` answer ``True`` on SM70 so the QSA/MQA
backend takes the ROCm-aligned path. That is correct only while ``mqa.py`` has
a single call site. A second one added upstream would silently route CUDA
through ROCm, so the count is asserted here instead of being found on a
four-card run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MQA = "python/sglang/srt/layers/attention/qsa/mqa.py"


def _upstream_source() -> str:
    for ref in ("upstream", "origin/upstream"):
        try:
            out = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"{ref}:{MQA}"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            pytest.skip("git is not available")
        if out.returncode == 0:
            return out.stdout
    pytest.skip("no upstream branch; run scripts/fetch_upstream.sh")


def test_is_hip_has_a_single_call_site():
    source = _upstream_source()
    calls = [line.strip() for line in source.splitlines() if "is_hip()" in line]
    assert len(calls) == 1, calls
