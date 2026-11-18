"""The receipt trigger: an exclusion rule, and the summary that must match it.

The trigger opened the same hole twice while it was an allow-list: first it
demanded hardware evidence for `cli.py` and tests, then it silently exempted
`sglang_restomod/kernels/` and `overrides.py` — the kernels and the swap that
installs them. An allow-list fails open on a path nobody thought of, so the rule
now says what is exempt and lets everything else in the package trigger.

These tests read the workflow as text, because the failure mode worth catching is
drift between the `grep` in the policy step and the rule the job summary prints
in prose. A rule that only exists in a comment cannot be wrong; one that exists
twice can.
"""

from __future__ import annotations

import pathlib
import re

WORKFLOW = (
    pathlib.Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "restomod-gpu-receipt.yml"
)
TEXT = WORKFLOW.read_text()


def _pipeline() -> tuple[re.Pattern[str], list[re.Pattern[str]]]:
    """The include and exclude patterns the policy step feeds to grep."""
    include = None
    excludes: list[re.Pattern[str]] = []
    for line in TEXT.splitlines():
        stripped = line.strip()
        found = re.match(r"\| grep -E '(.+)' \\$", stripped)
        if found:
            include = re.compile(found.group(1))
            continue
        found = re.match(r"\| grep -vE '(.+)' \\$", stripped)
        if found:
            excludes.append(re.compile(found.group(1)))
    assert include is not None, "no include pattern found in the policy step"
    assert excludes, "no exclude pattern found in the policy step"
    return include, excludes


def _triggers(path: str) -> bool:
    include, excludes = _pipeline()
    if path.endswith(".md"):
        return False
    if not include.match(path):
        return False
    return not any(pattern.match(path) for pattern in excludes)


def test_the_rule_is_an_exclusion_over_the_package():
    include, _ = _pipeline()
    # One prefix plus the two directories; the exemptions are the -vE list. If
    # this ever becomes an enumeration again, a new directory fails open.
    assert include.pattern == "^(sglang_restomod/|patches/|recipes/)"


def test_paths_that_decide_gpu_behaviour_require_a_receipt():
    for path in (
        "sglang_restomod/kernels/deep_gemm_sm70/entrypoint.py",
        "sglang_restomod/overrides.py",
        "sglang_restomod/configs/triton_3_5_1/E512.json",
        "sglang_restomod/_sm70_paths.py",
        "sglang_restomod/plugins.py",
        "sglang_restomod/attest.py",
        "sglang_restomod/quant/sm70_fp4_moe/method.py",
        "sglang_restomod/attention/qsa_v100.py",
        "sglang_restomod/__init__.py",
        # A directory nobody has created yet is exactly the case that failed
        # open: it must trigger without anyone remembering to add it.
        "sglang_restomod/some_new_tree/kernel.py",
        "patches/0009-something.patch",
        "recipes/flashnext.md".replace(".md", ".toml"),
    ):
        assert _triggers(path), f"{path} must require a receipt"


def test_paths_that_cannot_decide_behaviour_are_exempt():
    for path in (
        "sglang_restomod/cli.py",
        "sglang_restomod/tests/test_attest.py",
        "docs/gpu-receipts.md",
        "README.md",
        "sglang_restomod/quant/PROVENANCE.md",
        "scripts/pack_evidence.sh",
        "tests/test_receipt_policy.py",
        ".github/workflows/restomod-ci.yml",
    ):
        assert not _triggers(path), f"{path} must not require a receipt"


def test_the_summary_states_the_same_rule_as_the_grep():
    rule = [line for line in TEXT.splitlines() if line.strip().startswith("echo 'Rule:")]
    assert len(rule) == 1, "the job summary must print exactly one rule line"
    text = rule[0]
    for exempt in ("sglang_restomod/cli.py", "sglang_restomod/tests/", "`.md`"):
        assert exempt in text, f"the summary does not name the exemption {exempt}"
    for triggering in ("sglang_restomod/", "patches/", "recipes/"):
        assert triggering in text, f"the summary does not name {triggering}"
    # The words a reader relies on when the check is green.
    assert "nothing in the change can alter what the GPU computes" in text
