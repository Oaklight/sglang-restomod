"""The GPU workflow gates from issue #12 fail on a planted violation."""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHA = "0" * 40

VALID = f"""\
name: gpu
on:
  workflow_dispatch:
permissions:
  contents: read
jobs:
  smoke:
    runs-on: [self-hosted, v100]
    environment: gpu
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@{SHA}
        with:
          persist-credentials: false
      - run: python scripts/gpu_smoke.py
"""


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_workflow_gates", ROOT / "scripts" / "check_workflow_gates.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_workflow_gates = _load()


def test_valid_workflow_passes():
    assert check_workflow_gates.check(VALID) == []


def test_third_event_fails():
    text = VALID.replace(
        "  workflow_dispatch:\n", "  workflow_dispatch:\n  pull_request:\n"
    )
    assert any("pull_request" in p for p in check_workflow_gates.check(text))


def test_secret_reference_fails():
    text = VALID.replace(
        "          persist-credentials: false\n",
        "          token: ${{ secrets.TOKEN }}\n",
    )
    assert any("secrets." in p for p in check_workflow_gates.check(text))


def test_cache_fails():
    text = VALID.replace(
        "      - run: python scripts/gpu_smoke.py\n",
        "      - uses: actions/cache@0\n        with:\n          path: x\n          key: y\n",
    )
    assert any("actions/cache" in p for p in check_workflow_gates.check(text))


def test_unpinned_action_fails():
    text = VALID.replace(f"actions/checkout@{SHA}", "actions/checkout@v4")
    assert any("not pinned" in p for p in check_workflow_gates.check(text))


def test_missing_environment_fails():
    text = VALID.replace("    environment: gpu\n", "")
    assert any("environment" in p for p in check_workflow_gates.check(text))


def test_missing_self_hosted_fails():
    text = VALID.replace("[self-hosted, v100]", "ubuntu-latest")
    assert any("self-hosted" in p for p in check_workflow_gates.check(text))


def test_repository_workflow_passes():
    workflow = ROOT / ".github" / "workflows" / "restomod-gpu.yml"
    assert workflow.is_file()
    assert check_workflow_gates.check(workflow.read_text()) == []
