"""Import-cost checks for the attention_v100 tilelang package (#245).

Golden rule 1: importing a component must not pull torch, tilelang or kernels.
The package used to import ``_paged_adapter`` eagerly, which imports torch and
builds TileLang kernels at package import.

The subprocess test blocks ``tilelang`` so it fails if the adapter is imported
eagerly again. The second test enumerates every public name so back-compat is
checked for all of them, not a spot-check.
"""

import subprocess
import sys
import textwrap

import pytest

PUBLIC_NAMES = ["gather_fp8_paged_kv", "grouped_decode_forward", "paged_forward"]
PUBLIC_MODULE = "sglang_restomod.attention.v100.tilelang"


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True
    )


def test_package_imports_without_tilelang():
    proc = _run(
        """
        import sys

        class BlockTilelang:
            def find_spec(self, name, path=None, target=None):
                if name == "tilelang" or name.startswith("tilelang."):
                    raise ImportError(f"{name} blocked for the import-cost test")
                return None

        sys.meta_path.insert(0, BlockTilelang())
        import importlib
        importlib.import_module("sglang_restomod.attention.v100.tilelang")
        assert "tilelang" not in sys.modules, "package import pulled tilelang"
        """
    )
    assert proc.returncode == 0, proc.stderr


def test_public_names_still_importable():
    tilelang = pytest.importorskip("tilelang")
    assert tilelang is not None
    import importlib

    module = importlib.import_module(PUBLIC_MODULE)
    assert sorted(module.__all__) == sorted(PUBLIC_NAMES)
    for name in module.__all__:
        value = getattr(module, name)
        assert callable(value), f"{name} is not callable"
        # `from module import name` is the back-compat form that must keep working.
        from_import = _run(
            f"from {PUBLIC_MODULE} import {name}\nassert callable({name})\n"
        )
        assert from_import.returncode == 0, from_import.stderr
