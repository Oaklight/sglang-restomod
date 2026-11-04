"""Contract tests for the SM70 EAGLE renorm replacements (CPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _reference_top_k(probs, k):
    out = torch.zeros_like(probs)
    for row in range(probs.shape[0]):
        index = torch.topk(probs[row], int(k)).indices
        out[row, index] = probs[row, index]
    return out / out.sum(dim=-1, keepdim=True)


def _reference_top_p(probs, top_p):
    out = torch.zeros_like(probs)
    for row in range(probs.shape[0]):
        values, index = torch.sort(probs[row], descending=True)
        cumulative = values.cumsum(dim=-1)
        crossing = int((cumulative >= float(top_p)).nonzero()[0])
        keep = index[: crossing + 1]
        out[row, keep] = probs[row, keep]
    return out / out.sum(dim=-1, keepdim=True)


def test_top_k_matches_reference():
    from sglang_restomod.spec.sm70_sampling import top_k_renorm_probs

    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(3, 64), dim=-1)
    k = torch.tensor([1, 5, 20])
    got = top_k_renorm_probs(probs, k)
    want = torch.stack([_reference_top_k(probs[i : i + 1], k[i])[0] for i in range(probs.shape[0])])
    assert torch.allclose(got, want, atol=1e-6)
    assert torch.allclose(got.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_top_k_scalar():
    from sglang_restomod.spec.sm70_sampling import top_k_renorm_probs

    torch.manual_seed(1)
    probs = torch.softmax(torch.randn(2, 32), dim=-1)
    got = top_k_renorm_probs(probs, 4)
    want = _reference_top_k(probs, 4)
    assert torch.allclose(got, want, atol=1e-6)


def test_top_p_matches_reference():
    from sglang_restomod.spec.sm70_sampling import top_p_renorm_probs

    torch.manual_seed(2)
    probs = torch.softmax(torch.randn(4, 128), dim=-1)
    top_p = torch.tensor([0.1, 0.5, 0.9, 1.0])
    got = top_p_renorm_probs(probs, top_p)
    want = torch.stack([_reference_top_p(probs[i : i + 1], top_p[i])[0] for i in range(4)])
    assert torch.allclose(got, want, atol=1e-6)
    assert torch.allclose(got.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_dtype_is_preserved():
    from sglang_restomod.spec.sm70_sampling import top_k_renorm_probs

    probs = torch.softmax(torch.randn(2, 16), dim=-1).to(torch.float16)
    got = top_k_renorm_probs(probs, 3)
    assert got.dtype == torch.float16


def test_top_p_returns_float32():
    # FlashInfer's top_p_renorm_probs casts to float() before allocating its
    # output, while top_k_renorm_probs uses empty_like. The replacement mirrors
    # both, so a chained call keeps the dtype upstream would produce.
    from sglang_restomod.spec.sm70_sampling import (
        top_k_renorm_probs,
        top_p_renorm_probs,
    )

    probs = torch.softmax(torch.randn(2, 16), dim=-1).to(torch.float16)
    assert top_k_renorm_probs(probs, 3).dtype == torch.float16
    assert top_p_renorm_probs(probs, 0.9).dtype == torch.float32


def _write_stub_module(root, source):
    package = root / "stubpkg"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "sampling.py").write_text(source, encoding="utf-8")
    return package


def test_jit_cccl_compat_wraps_on_import(monkeypatch, tmp_path):
    import importlib
    import sys

    from sglang_restomod.spec import sm70_sampling

    _write_stub_module(
        tmp_path,
        "calls = {}\n\n\ndef load_jit(*args, **kwargs):\n"
        "    calls['args'] = args\n    calls['kwargs'] = kwargs\n    return 'module'\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sm70_sampling, "_JIT_SAMPLING_MODULE", "stubpkg.sampling")
    monkeypatch.setattr(sm70_sampling, "_JIT_CCCL_INSTALLED", False)
    monkeypatch.delitem(sys.modules, "stubpkg", raising=False)
    monkeypatch.delitem(sys.modules, "stubpkg.sampling", raising=False)

    assert sm70_sampling.install_jit_cccl_compat()
    module = importlib.import_module("stubpkg.sampling")
    assert getattr(module.load_jit, "_restomod_cccl_shim", False)

    module.load_jit("speculative_sampling", 4, True, extra_include_paths=["/elsewhere"])
    paths = module.calls["kwargs"]["extra_include_paths"]
    assert paths[0] == sm70_sampling._JIT_CCCL_DIR
    assert paths[1] == "/elsewhere"
    assert module.calls["args"] == ("speculative_sampling", 4, True)


def test_jit_cccl_compat_patches_an_imported_module(monkeypatch):
    import sys
    import types

    from sglang_restomod.spec import sm70_sampling

    calls = {}

    def fake_load_jit(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "module"

    module = types.ModuleType("already.imported.sampling")
    module.load_jit = fake_load_jit
    monkeypatch.setitem(sys.modules, "already.imported.sampling", module)
    monkeypatch.setattr(sm70_sampling, "_JIT_SAMPLING_MODULE", "already.imported.sampling")
    monkeypatch.setattr(sm70_sampling, "_JIT_CCCL_INSTALLED", False)

    assert sm70_sampling.install_jit_cccl_compat()
    assert module.load_jit is not fake_load_jit
    module.load_jit("speculative_sampling", 4, True)
    assert calls["kwargs"]["extra_include_paths"] == [sm70_sampling._JIT_CCCL_DIR]


def test_jit_cccl_compat_is_idempotent(monkeypatch):
    import sys
    import types

    from sglang_restomod.spec import sm70_sampling

    module = types.ModuleType("already.imported.sampling")
    module.load_jit = lambda *args, **kwargs: "module"
    monkeypatch.setitem(sys.modules, "already.imported.sampling", module)
    monkeypatch.setattr(sm70_sampling, "_JIT_SAMPLING_MODULE", "already.imported.sampling")
    monkeypatch.setattr(sm70_sampling, "_JIT_CCCL_INSTALLED", False)

    assert sm70_sampling.install_jit_cccl_compat()
    first = module.load_jit
    assert sm70_sampling.install_jit_cccl_compat()
    assert module.load_jit is first


def test_jit_cccl_compat_arms_one_finder(monkeypatch):
    import sys

    from sglang_restomod.spec import sm70_sampling

    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sm70_sampling, "_JIT_SAMPLING_MODULE", "never.imported.module")
    monkeypatch.setattr(sm70_sampling, "_JIT_CCCL_INSTALLED", False)

    assert sm70_sampling.install_jit_cccl_compat()
    finders = [f for f in sys.meta_path if isinstance(f, sm70_sampling._JitSamplingFinder)]
    assert len(finders) == 1

    monkeypatch.setattr(sm70_sampling, "_JIT_CCCL_INSTALLED", False)
    assert sm70_sampling.install_jit_cccl_compat()
    finders = [f for f in sys.meta_path if isinstance(f, sm70_sampling._JitSamplingFinder)]
    assert len(finders) == 1


def test_cccl_shim_ships_with_the_package():
    import os

    from sglang_restomod.spec import sm70_sampling

    header = os.path.join(sm70_sampling._JIT_CCCL_DIR, "cuda", "functional")
    assert os.path.isfile(header)
    with open(header, encoding="utf-8") as handle:
        text = handle.read()
    assert "#include_next <cuda/functional>" in text
    assert "struct maximum" in text and "struct minimum" in text
