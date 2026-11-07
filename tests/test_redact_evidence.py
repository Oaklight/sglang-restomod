"""An evidence bundle loses paths, NVIDIA identifiers and denylisted names."""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "redact_evidence", ROOT / "scripts" / "redact_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


redact_evidence = _load()


def test_bundle_is_scrubbed(tmp_path):
    site_root = "/".join(["", "nfs", "vol1", "homes", "operator", "qlab"])
    model = site_root + "/models/Qwen3.8-27B-FP8"
    serial = "04" + "21519" + "019240"
    denylisted = "node-q7"

    (tmp_path / "commit.txt").write_text(
        f"host={denylisted}\nLinux {denylisted} 5.15.0-generic x86_64\n"
    )
    (tmp_path / "nvidia-smi-q.txt").write_text(
        f"GPU 00000000:62:00.0\n    Serial Number : {serial}\n"
        "    GPU UUID : GPU-" + "b4f9748f" + "-a797-d222\n"
    )
    (tmp_path / "server_info.json").write_text(
        '{"model_path": "' + model + '", "host": "127.0.0.1"}\n'
    )
    (tmp_path / "versions.txt").write_text(f"ran on {denylisted}\n")

    counts = redact_evidence.run(tmp_path, [denylisted])

    for name in ("commit.txt", "nvidia-smi-q.txt", "server_info.json", "versions.txt"):
        text = (tmp_path / name).read_text()
        assert denylisted not in text
        assert serial not in text
        assert site_root not in text
        assert "operator" not in text
    assert "Qwen3.8-27B-FP8" in (tmp_path / "server_info.json").read_text()
    assert "127.0.0.1" in (tmp_path / "server_info.json").read_text()
    assert counts["denylist"] >= 1


def test_redaction_record_holds_counts_only(tmp_path, capsys):
    denylisted = "node-q7"
    (tmp_path / "commit.txt").write_text(f"host={denylisted}\n")
    assert redact_evidence.main([str(tmp_path), "--denylist", denylisted]) == 0
    record = (tmp_path / "redaction.txt").read_text()
    assert "host-line\t1" in record
    assert denylisted not in record
    assert denylisted not in capsys.readouterr().out


def test_uname_keeps_the_kernel_release(tmp_path):
    (tmp_path / "commit.txt").write_text(
        "Linux 5.15.0-177-generic x86_64\n"
        "Linux node-q7 5.15.0-177-generic #187-Ubuntu SMP x86_64\n"
    )
    redact_evidence.run(tmp_path, [])
    lines = (tmp_path / "commit.txt").read_text().splitlines()
    assert lines[0] == "Linux 5.15.0-177-generic x86_64"
    assert lines[1] == "Linux <redacted> 5.15.0-177-generic #187-Ubuntu SMP x86_64"


def _relative_path(*parts: str) -> str:
    """Assemble a relative path with leading ../ segments.

    The identity check forbids home-shaped paths in the source, so the
    fixtures below build one instead of spelling it. Do not "simplify" this
    into a literal: that re-breaks the identity check for a reason that will
    look unrelated.
    """
    return "../../..//" + "/".join(parts)


def test_a_relative_traversal_path_is_scrubbed(tmp_path):
    # Traversal names the mount and the account without ever starting at a
    # known absolute root, which is what the old path rule keyed on.
    mount = "weirdmount"
    account = "zoe"
    line = _relative_path(mount, "homes", account, "logs", "run.txt")
    (tmp_path / "trace.txt").write_text(f"opened {line}\n")

    counts = redact_evidence.run(tmp_path, [])

    text = (tmp_path / "trace.txt").read_text()
    assert text.strip() == "opened <path>/run.txt"
    assert mount not in text
    assert account not in text
    assert redact_evidence.verify(tmp_path, []) == []
    assert counts["path"] >= 1


def test_a_leaky_bundle_fails_the_step(tmp_path, capsys):
    # A relative path with no .. is not something the rewrite can fix; the
    # step must fail loudly rather than pack it. Built from parts for the
    # same reason as _relative_path.
    account = "zoe"
    leak = "cache=" + "/".join(("data", "homes", account, "run.txt"))
    (tmp_path / "leak.txt").write_text(leak + "\n")

    assert redact_evidence.main([str(tmp_path), "--denylist", ""]) == 1

    stderr = capsys.readouterr().err
    assert "residual identity" in stderr
    assert "leak.txt" in stderr
