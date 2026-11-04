"""The smoke summary must not carry host, user or mount identity."""

from __future__ import annotations

import getpass
import importlib.util
import json
import pathlib
import socket

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gpu_smoke", ROOT / "scripts" / "gpu_smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu_smoke = _load()


def test_summary_has_no_machine_identity(tmp_path, monkeypatch, capsys):
    fake_host = "bench-99." + "example.net"
    fake_user = "fake-operator"
    fake_home = "/".join(["", "home", fake_user])
    for name, value in {
        "HOSTNAME": fake_host,
        "USER": fake_user,
        "LOGNAME": fake_user,
        "HOME": fake_home,
        "PWD": fake_home,
        "TMPDIR": fake_home,
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(socket, "gethostname", lambda: fake_host)
    monkeypatch.setattr(getpass, "getuser", lambda: fake_user)

    out = tmp_path / "smoke.json"
    assert gpu_smoke.main(["--json-out", str(out)]) == 0

    captured = capsys.readouterr()
    text = out.read_text()
    for needle in (fake_host, fake_user, fake_home):
        assert needle not in captured.out
        assert needle not in text

    payload = json.loads(text)
    assert set(payload) <= {
        "schema",
        "torch",
        "cuda",
        "device_count",
        "devices",
        "timings_ms",
        "error",
    }
    for device in payload["devices"]:
        assert set(device) == {"index", "name", "capability", "memory_mb"}
    assert set(payload["timings_ms"]) <= {"matmul_1024_fp16", "copy_1024_fp16"}
