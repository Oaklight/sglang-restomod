"""The smoke summary and its failure line carry no machine identity."""

from __future__ import annotations

import getpass
import importlib.util
import json
import pathlib
import socket
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gpu_smoke", ROOT / "scripts" / "gpu_smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu_smoke = _load()


def _fake_torch(available=False):
    """A stand-in for torch that reports no usable device."""

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return available

    return types.SimpleNamespace(cuda=_Cuda())


def test_summary_has_no_machine_identity(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())

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
    assert gpu_smoke.main(["--json-out", str(out)]) == 1

    captured = capsys.readouterr()
    text = out.read_text()
    for needle in (fake_host, fake_user, fake_home):
        assert needle not in captured.out
        assert needle not in captured.err
        assert needle not in text

    payload = json.loads(text)
    assert set(payload) <= {"schema", "ok", "device_count", "checks", "error_check"}
    assert payload["ok"] is False
    assert payload["error_check"] == "cuda"
    assert payload["checks"]["cuda"] == {"ok": False}


def test_failure_is_one_line_on_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    out = tmp_path / "smoke.json"
    assert gpu_smoke.main(["--json-out", str(out)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("FAILED: ")


def test_json_out_is_required(capsys):
    assert gpu_smoke.main([]) == 2
    assert "json-out" in capsys.readouterr().err
