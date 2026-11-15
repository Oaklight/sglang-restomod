"""The GPU probe inside scripts/gpu_suite.sh.

The probe is a Python heredoc, so the test runs the text the shell would run,
with a fake nvidia-smi on PATH. It exists because a receipt bundle carried
nvidia-smi's "couldn't find libnvidia-ml.so" text where topology belongs, while
CUDA itself worked and runtime.txt named the card correctly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "scripts" / "gpu_suite.sh"
OPEN = "<<'PY_PROBE'\n"
CLOSE = "\nPY_PROBE\n"


def probe_source() -> str:
    source = SUITE.read_text()
    start = source.index(OPEN) + len(OPEN)
    end = source.index(CLOSE, start)
    return source[start:end] + "\n"


def _run_probe(tmp_path, nvidia_smi_body):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "nvidia-smi"
    stub.write_text("#!/bin/sh\n" + nvidia_smi_body)
    stub.chmod(0o755)

    out = tmp_path / "evidence"
    out.mkdir()
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)}
    done = subprocess.run(
        [sys.executable, "-c", probe_source(), str(out)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    return out


FAILING = """
echo "NVIDIA-SMI couldn't find libnvidia-ml.so library in your system." >&2
exit 1
"""

WORKING = """
case "$1" in
  --query-gpu=name) echo "Tesla V100-SXM2-32GB" ;;
  --query-gpu=index*) echo "index, name, compute_cap"; echo "0, Tesla V100-SXM2-32GB, 7.0" ;;
  -q) echo "GPU 00000000:00:00.0" ;;
  topo) echo "GPU0    CPU Affinity" ;;
  *) exit 1 ;;
esac
exit 0
"""


def test_a_missing_driver_library_is_named_not_dressed_up_as_topology(tmp_path):
    out = _run_probe(tmp_path, FAILING)

    topo = (out / "topo.txt").read_text()
    assert "nvidia-smi unavailable" in topo
    assert "libnvidia-ml.so" in topo
    assert "GPU identity is in runtime.txt" in topo
    assert "GPU0" not in topo

    gpus = (out / "gpus.csv").read_text()
    assert "GPU identity is in runtime.txt" in gpus
    assert "Tesla" not in gpus


def test_a_working_nvidia_smi_writes_the_usual_inventory(tmp_path):
    out = _run_probe(tmp_path, WORKING)

    assert "Tesla V100-SXM2-32GB" in (out / "gpus.csv").read_text()
    assert "GPU0    CPU Affinity" in (out / "topo.txt").read_text()
    assert "GPU 00000000:00:00.0" in (out / "nvidia-smi-q.txt").read_text()
    assert not (out / "topo.txt").read_text().startswith("nvidia-smi unavailable")
