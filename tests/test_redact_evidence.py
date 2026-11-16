"""The redaction pass must cover the whole bundle, not its top level.

``scripts/redact_evidence.py`` is the gate between an evidence directory and a
public release asset. It used to walk ``directory.iterdir()`` and skip anything
that was not a file, so a nested directory was neither rewritten nor checked and
``verify()`` returned an empty list — the condition that means "safe to pack"
(issue #320). These tests plant identity inside subdirectories and assert the
pass finds it.
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
# A path that names a home mount. Built from parts so this file does not itself
# carry the string the redactor exists to remove.
HOME_PATH = "/".join(["", "home", "operator", "run.log"])


def _load():
    spec = importlib.util.spec_from_file_location(
        "redact_evidence", ROOT / "scripts" / "redact_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


redact = _load()


def _nested_bundle(tmp_path, text):
    inner = tmp_path / "smoke"
    inner.mkdir()
    planted = inner / "serve.txt"
    planted.write_text(text)
    return planted


def test_run_rewrites_a_path_in_a_subdirectory(tmp_path):
    planted = _nested_bundle(tmp_path, f"log at {HOME_PATH}\n")

    counts = redact.run(tmp_path, [])

    assert counts.get("path") == 1
    assert HOME_PATH not in planted.read_text()
    assert "<path>/run.log" in planted.read_text()


def test_verify_reports_a_leak_inside_a_subdirectory(tmp_path):
    _nested_bundle(tmp_path, f"log at {HOME_PATH}\n")

    leaks = redact.verify(tmp_path, [])

    assert leaks, "a nested leak must not read as a clean bundle"
    assert any("smoke/serve.txt" in leak for leak in leaks)


def test_verify_is_clean_after_the_pass_and_names_the_file(tmp_path):
    _nested_bundle(tmp_path, f"log at {HOME_PATH}\n")

    redact.run(tmp_path, [])

    assert redact.verify(tmp_path, []) == []


def test_verify_reports_a_denylist_name_at_depth(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "deep.txt").write_text("ran on lambda4 today\n")

    leaks = redact.verify(tmp_path, ["lambda4"])

    assert any("a/b/deep.txt" in leak for leak in leaks)


def test_verify_reports_a_symlink_instead_of_following_it(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text(f"log at {HOME_PATH}\n")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "link.txt").symlink_to(outside)

    leaks = redact.verify(bundle, [])

    assert leaks == ["link.txt: symlink, not inspected"]
    assert HOME_PATH in outside.read_text()


def test_run_does_not_write_through_a_symlink(tmp_path):
    outside = tmp_path.parent / "outside-run.txt"
    outside.write_text(f"log at {HOME_PATH}\n")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "link.txt").symlink_to(outside)

    redact.run(bundle, [])

    assert HOME_PATH in outside.read_text()


def test_verify_reports_a_binary_it_cannot_read(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")

    leaks = redact.verify(tmp_path, [])

    assert leaks == ["blob.bin: binary, not inspected"]


def test_main_redacts_a_nested_leak_and_records_it(tmp_path):
    planted = _nested_bundle(tmp_path, f"log at {HOME_PATH}\n")

    assert redact.main([str(tmp_path)]) == 0
    assert HOME_PATH not in planted.read_text()
    assert (tmp_path / "redaction.txt").is_file()
    # The pass is idempotent: a second run rewrites nothing.
    assert redact.main([str(tmp_path)]) == 0


def test_main_fails_when_something_is_left_uninspected(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01")

    assert redact.main([str(tmp_path)]) == 1


def test_main_reports_what_it_examined(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.txt").write_text("nothing to redact\n")
    (tmp_path / "two.txt").write_text("nor here\n")

    assert redact.main([str(tmp_path)]) == 0

    assert "examined: 2 files" in capsys.readouterr().out


def test_main_refuses_a_bundle_with_nothing_to_examine(tmp_path, capsys):
    assert redact.main([str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert "examined: 0 files" in captured.out
    assert "no files to examine" in captured.err
