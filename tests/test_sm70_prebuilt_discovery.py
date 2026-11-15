"""Prebuilt-library discovery must work on a stock upstream engine.

The point of these tests is the case the project exists for: an engine with no
``sglang.kernels.sm70_paths``. They are CPU-only and touch nothing outside a
temporary directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sglang_restomod import _sm70_paths


@pytest.fixture
def fake_so(tmp_path: Path) -> Path:
    library = tmp_path / "_sm70_turbomind_v100.so"
    library.write_bytes(b"\x7fELF")
    return library


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        _sm70_paths.ENV_PREBUILT_DIR,
        "SM70_TURBOMIND_SO",
        "SGLANG_SM70_TURBOMIND_SO",
    ):
        monkeypatch.delenv(name, raising=False)


def test_finds_the_library_through_the_directory_variable(
    fake_so: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_sm70_paths.ENV_PREBUILT_DIR, str(fake_so.parent))
    assert _sm70_paths.find_prebuilt(fake_so.name) == fake_so


def test_directory_variable_accepts_a_list(tmp_path: Path, fake_so: Path, monkeypatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(
        _sm70_paths.ENV_PREBUILT_DIR, os.pathsep.join([str(empty), str(fake_so.parent)])
    )
    assert _sm70_paths.find_prebuilt(fake_so.name) == fake_so


def test_a_named_variable_wins(fake_so: Path, tmp_path: Path, monkeypatch) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / fake_so.name).write_bytes(b"stale")
    monkeypatch.setenv("SM70_TURBOMIND_SO", str(fake_so))
    monkeypatch.setenv(_sm70_paths.ENV_PREBUILT_DIR, str(other))
    found = _sm70_paths.find_prebuilt(fake_so.name, env_names=("SM70_TURBOMIND_SO",))
    assert found == fake_so


def test_a_stale_named_variable_does_not_shadow_discovery(
    fake_so: Path, tmp_path: Path, monkeypatch
) -> None:
    """A path that does not exist must not hide a library discovery can find.

    This is the failure that made "set the environment variable" advice lie:
    the operator exported a path from a previous build and the real one was
    never looked for.
    """
    monkeypatch.setenv("SM70_TURBOMIND_SO", str(tmp_path / "gone" / fake_so.name))
    monkeypatch.setenv(_sm70_paths.ENV_PREBUILT_DIR, str(fake_so.parent))
    found = _sm70_paths.find_prebuilt(fake_so.name, env_names=("SM70_TURBOMIND_SO",))
    assert found == fake_so


def test_extra_directories_are_searched(tmp_path: Path, fake_so: Path) -> None:
    assert _sm70_paths.find_prebuilt(fake_so.name, extra_dirs=(tmp_path,)) == fake_so


def test_returns_none_when_nothing_has_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(_sm70_paths.ENV_PREBUILT_DIR, str(tmp_path))
    assert _sm70_paths.find_prebuilt("_absent_kernel.so") is None


def test_describe_search_names_the_variables_and_directories(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(_sm70_paths.ENV_PREBUILT_DIR, str(tmp_path))
    message = _sm70_paths.describe_search(
        "_absent_kernel.so", env_names=("SM70_TURBOMIND_SO",)
    )
    assert "$SM70_TURBOMIND_SO=<unset>" in message
    assert "sm70_prebuilt" in message
    assert str(tmp_path / "_absent_kernel.so") in message
    assert _sm70_paths.ENV_PREBUILT_DIR in message


def test_a_working_tree_is_found_without_the_fork_module(
    tmp_path: Path, fake_so: Path, monkeypatch
) -> None:
    """The stock-engine case: no sglang.kernels.sm70_paths, no exported path."""
    monkeypatch.setenv(_sm70_paths.ENV_PREBUILT_DIR, str(fake_so.parent))
    monkeypatch.setitem(__import__("sys").modules, "sglang", type("m", (), {"__file__": None})())
    assert _sm70_paths.find_prebuilt(fake_so.name) == fake_so
