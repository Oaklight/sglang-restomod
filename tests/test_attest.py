"""Component attestation (CPU).

The point of these tests is the failure nobody sees: a component that raises at
registration used to take every other component down with it, inside a loader
that wraps the whole call in ``except Exception``. The server then started
normally and served on the pinned kernels, which is indistinguishable from
having selected nothing.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import types

import pytest

from sglang_restomod import attest, plugins


@pytest.fixture(autouse=True)
def _clean():
    attest.reset()
    yield
    attest.reset()


def test_registration_is_recorded_and_survives_a_failure(monkeypatch):
    """A failing component is recorded, and the others still load."""
    good = types.ModuleType("fake_good")
    good.register = lambda: None

    def boom():
        raise RuntimeError("no kernel for this shape")

    bad = types.ModuleType("fake_bad")
    bad.register = boom

    monkeypatch.setitem(sys.modules, "fake_good", good)
    monkeypatch.setitem(sys.modules, "fake_bad", bad)
    monkeypatch.setitem(
        plugins.COMPONENTS,
        "fake_good",
        plugins.Component("fake_good", "fake_good", "test"),
    )
    monkeypatch.setitem(
        plugins.COMPONENTS,
        "fake_bad",
        plugins.Component("fake_bad", "fake_bad", "test"),
    )

    loaded = plugins.load_enabled({"SGLANG_RESTOMOD_COMPONENTS": "fake_bad,fake_good"})

    assert list(loaded) == ["fake_good"], "the good component must still load"
    snap = attest.snapshot()
    assert snap["loaded"] == ["fake_good"]
    assert snap["failed"] == ["fake_bad"]
    assert "no kernel for this shape" in snap["components"]["fake_bad"]["detail"]
    assert snap["missing"] == []


def test_component_without_register_is_inactive_not_a_crash(monkeypatch):
    """A module that imports but cannot register is inactive, with the reason."""
    empty = types.ModuleType("fake_empty")
    monkeypatch.setitem(sys.modules, "fake_empty", empty)
    monkeypatch.setitem(
        plugins.COMPONENTS, "fake_empty", plugins.Component("fake_empty", "fake_empty", "test")
    )

    loaded = plugins.load_enabled({"SGLANG_RESTOMOD_COMPONENTS": "fake_empty"})

    assert list(loaded) == []
    snap = attest.snapshot()
    assert snap["inactive"] == ["fake_empty"]
    assert snap["failed"] == []
    assert "no register()" in snap["components"]["fake_empty"]["detail"]


def test_attestation_file_is_written_and_names_the_engine(tmp_path, monkeypatch):
    target = tmp_path / "attest.json"
    monkeypatch.setenv(attest.ENV_ATTEST, str(target))

    attest.set_selected(["attention_v100"])
    attest.record("attention_v100", True)
    assert attest.write() == str(target)

    payload = json.loads(target.read_text())
    assert payload["loaded"] == ["attention_v100"]
    assert payload["failed"] == []
    # Path and version come from the imported module, not importlib.metadata,
    # which reports the installed distribution while a different tree executes.
    assert "engine" in payload and "path" in payload["engine"]


def test_selected_but_never_attempted_is_reported_as_missing(tmp_path, monkeypatch):
    """A component that never got a chance to run is a distinct state."""
    monkeypatch.setenv(attest.ENV_ATTEST, str(tmp_path / "a.json"))
    attest.set_selected(["wanted", "got"])
    attest.record("got", True)
    assert attest.snapshot()["missing"] == ["wanted"]


def test_no_path_configured_writes_nothing(monkeypatch):
    monkeypatch.delenv(attest.ENV_ATTEST, raising=False)
    assert attest.write() is None


def test_unwritable_path_does_not_raise(monkeypatch):
    """Observability must not be able to stop a launch."""
    monkeypatch.setenv(attest.ENV_ATTEST, "/proc/definitely/not/writable/attest.json")
    attest.set_selected([])
    assert attest.write() is None


def test_concurrent_writes_do_not_share_one_temporary_file(tmp_path, monkeypatch, caplog):
    """Every rank writes its own snapshot, and there is one rank per process.

    A TP4 launch has four processes registering at once. They used to share
    ``<target>.tmp``: the first ``os.replace`` consumed the file and the other
    three logged ``FileNotFoundError`` and recorded nothing, so the attestation
    could be written by one rank out of four. Observability that fails silently
    at the moment it matters reads as a complete answer when it is not one.
    """
    target = tmp_path / "attest.json"
    monkeypatch.setenv(attest.ENV_ATTEST, str(target))
    attest.set_selected(["attention_v100"])
    attest.record("attention_v100", True)

    ranks = 4
    barrier = threading.Barrier(ranks, timeout=10)
    real_dump = json.dump

    def synchronized_dump(*args, **kwargs):
        # Hold every writer at the same point, so they interleave the way four
        # ranks do when they register together rather than one after another.
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return real_dump(*args, **kwargs)

    monkeypatch.setattr(attest.json, "dump", synchronized_dump)

    results: list = []

    def worker():
        results.append(attest.write())

    with caplog.at_level(logging.WARNING, logger=attest.__name__):
        threads = [threading.Thread(target=worker) for _ in range(ranks)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

    assert results.count(str(target)) == ranks, "every rank must record a write"
    assert "could not write attestation" not in caplog.text
    assert json.loads(target.read_text())["loaded"] == ["attention_v100"]
