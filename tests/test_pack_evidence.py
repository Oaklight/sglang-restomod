"""``scripts/pack_evidence.sh``: the redaction that runs is main's, and it refuses.

The three bundles packed on 2026-11-16 used whatever ``redact_evidence.py`` the
packer's checkout held; two published releases kept the operator account name and
a host name after the rules that remove them landed on main. A practice did not
catch it, so the packer now fails closed: no denylist, no redaction script from
the declared ref, a stale record, or a residual identity stops the pack.

These tests drive the shell script, because the behaviour under test is its exit
status and its messages, not a Python function.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "pack_evidence.sh"

REDACT_OK = """\
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
for path in directory.rglob("*"):
    if path.is_file() and path.name != "redaction.txt":
        path.write_text(path.read_text().replace("leakname", "<host>"))
print("redacted: nothing")
sys.exit(0)
"""

REDACT_FAIL = """\
import sys

print("residual identity in the evidence bundle:")
print("  evidence.txt: residual path")
sys.exit(1)
"""

FAKE_OOB = """\
#!/usr/bin/env bash
# $1 pack --evidence-dir DIR --out OUT [--json]
while [ $# -gt 0 ]; do
  case "$1" in
    --evidence-dir) dir=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    *) shift ;;
  esac
done
count=$(find "$dir" -type f | wc -l)
printf 'archive %s\\n' "$count" > "$out"
printf '{"count": %s, "digest": "sha256:deadbeef"}\\n' "$count"
"""


def _repo(tmp_path: pathlib.Path, redaction: str = REDACT_OK) -> pathlib.Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "redact_evidence.py").write_text(redaction)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "x"],
        cwd=repo,
        check=True,
    )
    return repo


def _oob(tmp_path: pathlib.Path) -> pathlib.Path:
    binary = tmp_path / "oob"
    binary.write_text(FAKE_OOB)
    binary.chmod(0o755)
    return binary


def _run(repo: pathlib.Path, evidence: pathlib.Path, out: pathlib.Path, **env: str):
    base = {
        "PATH": os.environ["PATH"],
        "OOB": str(_oob(repo.parent)),
        "MAIN_REF": "HEAD",
    }
    base.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), str(evidence), str(out)],
        cwd=repo,
        env=base,
        capture_output=True,
        text=True,
    )


def _evidence(tmp_path: pathlib.Path) -> pathlib.Path:
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "evidence.txt").write_text("host leakname ran this\n")
    return directory


def test_no_denylist_refuses_before_doing_anything(tmp_path):
    repo = _repo(tmp_path)
    evidence = _evidence(tmp_path)
    result = _run(repo, evidence, tmp_path / "out.tgz", IDENTITY_DENYLIST="")
    assert result.returncode == 2
    assert "IDENTITY_DENYLIST" in result.stderr
    assert not (evidence / "redaction-script.txt").exists()


def test_whitespace_only_denylist_refuses(tmp_path):
    repo = _repo(tmp_path)
    evidence = _evidence(tmp_path)
    result = _run(repo, evidence, tmp_path / "out.tgz", IDENTITY_DENYLIST=" , \n ")
    assert result.returncode == 2
    assert "empty" in result.stderr


def test_unresolvable_main_ref_refuses(tmp_path):
    repo = _repo(tmp_path)
    evidence = _evidence(tmp_path)
    result = _run(
        repo, evidence, tmp_path / "out.tgz", IDENTITY_DENYLIST="leakname", MAIN_REF="origin/main"
    )
    assert result.returncode == 2
    assert "cannot resolve" in result.stderr


def test_stale_record_refuses_rather_than_mixing_rule_sets(tmp_path):
    repo = _repo(tmp_path)
    evidence = _evidence(tmp_path)
    (evidence / "redaction-script.txt").write_text("file sha256: " + "0" * 64 + "\n")
    result = _run(repo, evidence, tmp_path / "out.tgz", IDENTITY_DENYLIST="leakname")
    assert result.returncode == 2
    assert "start a fresh directory" in result.stderr


def test_failed_redaction_stops_the_pack(tmp_path):
    repo = _repo(tmp_path, redaction=REDACT_FAIL)
    evidence = _evidence(tmp_path)
    out = tmp_path / "out.tgz"
    result = _run(repo, evidence, out, IDENTITY_DENYLIST="leakname")
    assert result.returncode == 2
    assert "not packing" in result.stderr
    assert not out.exists()
    assert not (evidence / "redaction-script.txt").exists()


def test_happy_path_records_the_script_and_prints_both_numbers(tmp_path):
    repo = _repo(tmp_path)
    evidence = _evidence(tmp_path)
    out = tmp_path / "out.tgz"
    result = _run(repo, evidence, out, IDENTITY_DENYLIST="leakname")
    assert result.returncode == 0, result.stderr
    record = (evidence / "redaction-script.txt").read_text()
    digest = hashlib.sha256((repo / "scripts" / "redact_evidence.py").read_bytes()).hexdigest()
    assert f"file sha256: {digest}" in record
    assert "run from HEAD, not from the packer checkout" in record
    # The two numbers are different and each is labelled; three summaries in one
    # evening quoted the archive hash where the receipt wanted the digest.
    assert "manifest digest: sha256:deadbeef" in result.stdout
    assert f"archive sha256:  {hashlib.sha256(out.read_bytes()).hexdigest()}" in result.stdout
    assert "leakname" not in (evidence / "evidence.txt").read_text()
