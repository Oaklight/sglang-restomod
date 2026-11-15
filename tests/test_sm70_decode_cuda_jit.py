"""CPU tests for the SM70 CUDA decode JIT harness (issue #236).

Two environment defects, no GPU needed: the loader logged through
``logger.info_once`` before SGLang installs it, and a full ``TMPDIR`` surfaced
as an nvcc signal instead of an error. Nothing here compiles: the build entry
point is stubbed, and the preflight fails before it would be reached.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "sglang_restomod" / "attention" / "v100" / "tilelang" / "_decode_cuda.py"


def _load_module():
    """Load ``_decode_cuda`` by path.

    Importing it through the package pulls ``_paged_adapter`` and ``tilelang``,
    which restomod-ci does not install; the module itself needs only torch.
    """
    spec = importlib.util.spec_from_file_location("sm70_decode_cuda_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_decode_cuda = _load_module()


@pytest.fixture(autouse=True)
def _reset_loader():
    _decode_cuda._EXT = None
    _decode_cuda._OPS_LOAD_ATTEMPTED = False
    yield
    _decode_cuda._EXT = None
    _decode_cuda._OPS_LOAD_ATTEMPTED = False


class _PlainLogger:
    """A ``logging.Logger`` stand-in without SGLang's ``info_once``."""

    def __init__(self):
        self.lines = []

    def info(self, message, *args):
        self.lines.append(message % args if args else message)


class _OnceLogger(_PlainLogger):
    def __init__(self):
        super().__init__()
        self.once_lines = []

    def info_once(self, message, *args):
        self.once_lines.append(message % args if args else message)


def test_info_once_falls_back_to_logger_info(monkeypatch):
    stand = _PlainLogger()
    monkeypatch.setattr(_decode_cuda, "logger", stand)
    _decode_cuda._info_once("loaded %s", "ok")
    assert stand.lines == ["loaded ok"]


def test_info_once_prefers_the_sglang_helper(monkeypatch):
    stand = _OnceLogger()
    monkeypatch.setattr(_decode_cuda, "logger", stand)
    _decode_cuda._info_once("loaded %s", "ok")
    assert stand.once_lines == ["loaded ok"]
    assert stand.lines == []


# The child process is the reproduction from the issue: a bare interpreter,
# no sglang imported, so ``logging.Logger`` has no ``info_once``.
_BARE_INTERPRETER = r"""
import logging
import os
import sys
import tempfile
import types

for name in list(sys.modules):
    assert not name.startswith("sglang"), f"sglang already imported: {name}"
assert not hasattr(logging.getLogger("probe"), "info_once")

import importlib.util

spec = importlib.util.spec_from_file_location("sm70_decode_cuda_under_test", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

assert not hasattr(m.logger, "info_once"), "the logger already has info_once"
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

os.environ["SGLANG_V100_DECODE_CUDA_BUILD_DIR"] = tempfile.mkdtemp()
m._BUILD_HEADROOM_BYTES = 0

# 1. the cached fast path
m._load_built_extension = lambda build_dir: types.SimpleNamespace(
    sm70_longctx_decode=None
)
assert m._load_sm70_cuda_decode_ops() is not None
m._EXT = None
m._OPS_LOAD_ATTEMPTED = False

# 2. the build path, with the compiler entry point stubbed
import torch.utils.cpp_extension as cpp_extension

cpp_extension.load_inline = lambda **kwargs: types.SimpleNamespace(
    sm70_longctx_decode=None
)
assert m._load_sm70_cuda_decode_ops() is not None

print("BARE_INTERPRETER_OK")
"""


def test_loader_logs_in_a_bare_interpreter():
    proc = subprocess.run(
        [sys.executable, "-c", _BARE_INTERPRETER, str(MODULE_PATH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "BARE_INTERPRETER_OK" in proc.stdout
    assert "AttributeError" not in proc.stderr
    assert proc.stdout.count("CUDA long-context decode partial loaded") == 2


def test_build_preflight_names_a_full_tmpdir(tmp_path, monkeypatch):
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("SGLANG_V100_DECODE_CUDA_BUILD_DIR", str(build))

    free = {str(tmpdir): 32 << 20, str(build): 64 << 30}

    def disk_usage(path):
        return types.SimpleNamespace(free=free[str(path)])

    monkeypatch.setattr(shutil, "disk_usage", disk_usage)

    with pytest.raises(RuntimeError) as excinfo:
        _decode_cuda._load_sm70_cuda_decode_ops()

    message = str(excinfo.value)
    assert f"TMPDIR={tmpdir}" in message
    assert "0.03 GiB free" in message
    assert "SGLANG_V100_DECODE_CUDA_BUILD_DIR" in message


def test_build_preflight_rejects_a_read_only_build_directory(tmp_path, monkeypatch):
    if getattr(os, "geteuid", lambda: 1)() == 0:
        pytest.skip("root ignores directory permissions")
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    build = tmp_path / "readonly"
    build.mkdir()
    build.chmod(0o500)
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("SGLANG_V100_DECODE_CUDA_BUILD_DIR", str(build))
    monkeypatch.setattr(_decode_cuda, "_BUILD_HEADROOM_BYTES", 0)
    try:
        with pytest.raises(RuntimeError, match="not writable") as excinfo:
            _decode_cuda._load_sm70_cuda_decode_ops()
        assert str(build) in str(excinfo.value)
    finally:
        build.chmod(0o700)


def test_build_proceeds_when_both_directories_have_space(tmp_path, monkeypatch):
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("SGLANG_V100_DECODE_CUDA_BUILD_DIR", str(build))
    monkeypatch.setattr(_decode_cuda, "_BUILD_HEADROOM_BYTES", 0)

    import torch.utils.cpp_extension as cpp_extension

    monkeypatch.setattr(
        cpp_extension,
        "load_inline",
        lambda **kwargs: types.SimpleNamespace(sm70_longctx_decode=None),
    )
    assert _decode_cuda._load_sm70_cuda_decode_ops() is not None
