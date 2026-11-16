"""``restomod doctor`` states a failed check instead of contradicting itself.

The commit pin is the case that bit: a mismatch used to print
``FAIL commit: <recorded> matches <declared>`` because the label described the
passing state and the failure reused it.
"""

from __future__ import annotations

import pathlib

import pytest

from sglang_restomod import cli

ROOT = pathlib.Path(cli.__file__).resolve().parents[1]


def test_commit_mismatch_says_it_does_not_match(tmp_path, capsys):
    recorded = cli._tree_commit(ROOT)
    if recorded is None:
        pytest.skip("the tree records no commit, so there is nothing to compare")

    site = tmp_path / "site.toml"
    site.write_text('[engine]\ncommit = "deadbeef"\n')

    rc = cli.main(["doctor", "--site", str(site), "--manifest", str(tmp_path / "absent.toml")])
    out = capsys.readouterr().out

    assert rc == 1
    assert f"FAIL commit: {recorded} does not match deadbeef" in out
    assert f"FAIL commit: {recorded} matches deadbeef" not in out


def test_commit_match_says_it_matches(tmp_path, capsys):
    recorded = cli._tree_commit(ROOT)
    if recorded is None:
        pytest.skip("the tree records no commit, so there is nothing to compare")

    site = tmp_path / "site.toml"
    site.write_text(f'[engine]\ncommit = "{recorded}"\n')

    rc = cli.main(["doctor", "--site", str(site), "--manifest", str(tmp_path / "absent.toml")])
    out = capsys.readouterr().out

    # The manifest and the engine are separate failures; the commit line must
    # not be one of them.
    assert f"ok   commit: {recorded} matches {recorded}" in out
    assert "does not match" not in out
    assert rc == 1
