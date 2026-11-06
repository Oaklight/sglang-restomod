"""The lsa_sm70 component is a registration shim for the standalone package."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from sglang_restomod.attention import lsa_sm70 as component


def test_importing_the_component_loads_nothing_heavy():
    code = (
        "import sys\n"
        "import sglang_restomod.attention.lsa_sm70 as component\n"
        "assert callable(component.register)\n"
        "assert 'torch' not in sys.modules, 'torch loaded'\n"
        "assert 'lsa_sm70' not in sys.modules, 'package loaded'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_register_delegates_to_the_package(monkeypatch):
    calls = []
    package = types.ModuleType("lsa_sm70")
    package.register = lambda: calls.append("registered")
    monkeypatch.setitem(sys.modules, "lsa_sm70", package)

    component.register()

    assert calls == ["registered"]


def test_a_missing_package_names_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "lsa_sm70", None)

    with pytest.raises(RuntimeError, match=r"git\+https://github\.com/Oaklight/lsa-sm70"):
        component.register()
