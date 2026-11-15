"""The custom-allreduce log check used by scripts/gpu_suite.sh (CPU).

A failed custom-allreduce build does not fail the launch: the server warns
once, falls back to NCCL, and serves. #260 shows the cost (about 3.1 ms per
decode step) and the reason it went unnoticed (one warning line, no failure).
At the pin there is no success line to match, so these cases pin the honest
three-way answer instead of a check that claims more than it looked at.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_allreduce.py"

FAILURE_LINE = (
    "WARNING ... Setup Custom allreduce failed with Failed to build JIT module "
    "sgl_kernel_jit_cuda_ipc: ... error: no matching function for call to 'get'"
)


def run(tmp_path, log_text=None):
    out = tmp_path / "custom_allreduce.txt"
    args = [sys.executable, str(SCRIPT), "--out", str(out)]
    if log_text is not None:
        log = tmp_path / "server.log"
        log.write_text(log_text)
        args += ["--server-log", str(log)]
    done = subprocess.run(args, capture_output=True, text=True)
    return done, out


def test_a_failure_marker_is_reported_as_failed(tmp_path):
    done, out = run(tmp_path, f"starting\n{FAILURE_LINE}\nserving\n")

    assert done.returncode == 1
    text = out.read_text()
    assert text.splitlines()[0] == "custom_allreduce=failed"
    assert "line=2" in text
    assert "Setup Custom allreduce failed with" in text


def test_a_failure_late_in_a_long_log_is_still_found(tmp_path):
    padding = "".join(f"noise {i}\n" for i in range(5000))
    done, out = run(tmp_path, padding + FAILURE_LINE + "\n")

    assert done.returncode == 1
    assert "line=5001" in out.read_text()


def test_a_clean_log_is_not_called_a_pass(tmp_path):
    done, out = run(tmp_path, "starting\ncapture begin\nserving\n")

    assert done.returncode == 0
    text = out.read_text()
    assert text.splitlines()[0].startswith("custom_allreduce=failure-absent-only")
    assert "does not confirm" in text
    assert "custom_allreduce=ok" not in text


def test_no_log_supplied_records_that_it_did_not_run(tmp_path):
    done, out = run(tmp_path)

    assert done.returncode == 0
    assert out.read_text().splitlines()[0] == "custom_allreduce=not checked"


def test_an_unreadable_log_is_an_error_not_a_state(tmp_path):
    out = tmp_path / "custom_allreduce.txt"
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out",
            str(out),
            "--server-log",
            str(tmp_path / "absent.log"),
        ],
        capture_output=True,
        text=True,
    )

    assert done.returncode == 2
    assert "cannot read --server-log" in done.stderr
    assert not out.exists()
