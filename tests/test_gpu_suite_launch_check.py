"""The launch check inside scripts/gpu_suite.sh.

The check is a Python heredoc, so the only way to test it is to run the text
the shell would run. Each case writes a server_info the endpoint could return
and asserts on the exit status and the row it printed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sglang_restomod.attention.v100.backend_registry import BACKEND_NAMES

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "scripts" / "gpu_suite.sh"
HEADER = "<<'PY' || check_rc=$?\n"
FOOTER = "\nPY\n"

BASE = "--model-path /m --dtype float16 --quantization modelopt_fp4 --tp-size 4 --ep-size 1"
ALL = "attention_v100,fp4_moe,ple_sm70,spec_sm70,fp8_dense,fp8_lm_head,fp8_dense_quant"


def check_script() -> str:
    source = SUITE.read_text()
    start = source.index(HEADER) + len(HEADER)
    end = source.index(FOOTER, start)
    return source[start:end] + "\n"


def installed_without_repo() -> bool:
    """True when the package imports from site-packages alone."""
    probe = subprocess.run(
        [sys.executable, "-I", "-c", "import sglang_restomod"],
        capture_output=True,
    )
    return probe.returncode == 0


def run(
    tmp_path,
    launch_command,
    attention_backend,
    draft_backend,
    components=ALL,
    repo_root=None,
    cwd=None,
):
    info = tmp_path / "server_info.json"
    payload = {
        "attention_backend": attention_backend,
        "speculative_draft_attention_backend": draft_backend,
    }
    if launch_command is not None:
        payload["launch_command"] = launch_command
    info.write_text(json.dumps(payload))
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            check_script(),
            str(info),
            components,
            str(ROOT if repo_root is None else repo_root),
        ],
        capture_output=True,
        text=True,
        # Deliberately not the checkout: the heredoc has to find the package
        # from the path it is given, not from the caller's directory.
        cwd=str(tmp_path if cwd is None else cwd),
    )
    return proc.returncode, proc.stdout


def test_qsa_launch_passes(tmp_path):
    # The Qwen4-Exp profiles launch the sparse backend; it is a registered name.
    rc, out = run(tmp_path, f"{BASE} --attention-backend qsa_v100", "qsa_v100", None)
    assert rc == 0
    assert "ok\tattention_backend\twant=qsa_v100\tgot=qsa_v100" in out


@pytest.mark.parametrize("name", BACKEND_NAMES)
def test_every_registered_name_is_accepted(tmp_path, name):
    # The registry, not a literal in the script, decides what is accepted.
    rc, out = run(tmp_path, f"{BASE} --attention-backend {name}", name, None)
    assert rc == 0
    assert f"want={name}\tgot={name}" in out


def test_tilelang_launch_with_draft_passes(tmp_path):
    rc, out = run(
        tmp_path,
        f"{BASE} --attention-backend tilelang_fa_v100 "
        "--speculative-draft-attention-backend tilelang_fa_v100",
        "tilelang_fa_v100",
        "tilelang_fa_v100",
    )
    assert rc == 0
    assert "ok\tspeculative_draft_attention_backend" in out


@pytest.mark.parametrize("served", BACKEND_NAMES)
def test_no_flag_accepts_either_registered_name(tmp_path, served):
    # The component installs choices, not a default: it promises only that one
    # of its names served, not which one.
    rc, out = run(tmp_path, BASE, served, None)
    assert rc == 0
    assert "ok\tattention_backend\twant=" in out


def test_no_flag_rejects_an_engine_backend(tmp_path):
    rc, out = run(tmp_path, BASE, "triton", None)
    assert rc == 1
    assert "MISMATCH\tattention_backend" in out


def test_missing_launch_command_fails(tmp_path):
    # Without the launch, every other row would compare against a guess.
    rc, out = run(tmp_path, None, "qsa_v100", None)
    assert rc == 1
    assert "MISMATCH\tlaunch_command" in out


def test_engine_backend_is_rejected(tmp_path):
    # attention_v100 is selected but the launch asked the engine for triton.
    rc, out = run(tmp_path, f"{BASE} --attention-backend triton", "triton", None)
    assert rc == 1
    assert "MISMATCH\t--attention-backend\twant=" in out
    assert "got=triton" in out


def test_served_backend_must_match_the_flag(tmp_path):
    rc, out = run(tmp_path, f"{BASE} --attention-backend qsa_v100", "triton", None)
    assert rc == 1
    assert "want=qsa_v100\tgot=triton" in out


def test_requested_draft_that_did_not_install_fails(tmp_path):
    rc, out = run(
        tmp_path,
        f"{BASE} --attention-backend qsa_v100 "
        "--speculative-draft-attention-backend tilelang_fa_v100",
        "qsa_v100",
        None,
    )
    assert rc == 1
    assert "MISMATCH\tspeculative_draft_attention_backend" in out


def test_unrequested_draft_is_recorded_not_asserted(tmp_path):
    rc, out = run(tmp_path, f"{BASE} --attention-backend qsa_v100", "qsa_v100", None)
    assert rc == 0
    assert "not requested by the launch" in out


def test_no_mapped_component_passes(tmp_path):
    rc, out = run(tmp_path, BASE, "triton", None, components="fp4_moe")
    assert rc == 0
    assert "no component in the selection" in out


@pytest.mark.parametrize("separator", [" ", "="])
def test_flag_value_forms(tmp_path, separator):
    flag = f"--attention-backend{separator}qsa_v100"
    rc, out = run(tmp_path, f"{BASE} {flag}", "qsa_v100", None)
    assert rc == 0
    assert "want=qsa_v100" in out


@pytest.mark.parametrize("quote", ["'", '"'])
def test_a_quoted_flag_value_is_read(tmp_path, quote):
    # launch_command is argv joined by spaces, so a shell that quoted the value
    # leaves the quotes in the token.
    rc, out = run(tmp_path, f"{BASE} --attention-backend {quote}qsa_v100{quote}", "qsa_v100", None)
    assert rc == 0
    assert "want=qsa_v100\tgot=qsa_v100" in out


def test_the_checkout_is_read_from_its_own_path(tmp_path):
    # The operator's working directory is not the checkout; the path argument
    # is what makes the import resolve.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    rc, out = run(
        tmp_path,
        f"{BASE} --attention-backend qsa_v100",
        "qsa_v100",
        None,
        cwd=elsewhere,
    )
    assert rc == 0
    assert "want=qsa_v100" in out


@pytest.mark.skipif(
    installed_without_repo(),
    reason="sglang_restomod is installed, so the import cannot fail",
)
def test_unimportable_registry_fails_loudly(tmp_path):
    # A wrong path must name the import, not the served backend.
    rc, out = run(
        tmp_path,
        f"{BASE} --attention-backend qsa_v100",
        "qsa_v100",
        None,
        repo_root=tmp_path,
    )
    assert rc == 1
    assert "MISMATCH\tBACKEND_NAMES" in out
