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

    # The second call above returns through _JIT_CCCL_INSTALLED, so it does not
    # exercise the guard inside _wrap_sampling_module. Reset the flag to take
    # that path: a second wrap would install a second shim on an already
    # wrapped function.
    monkeypatch.setattr(sm70_sampling, "_JIT_CCCL_INSTALLED", False)
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


def test_jit_cccl_compat_forwards_a_positional_include_path(monkeypatch):
    """A ``load_jit`` that takes the list positionally gets it back edited.

    The pin's own ``load_jit(*args, extra_include_paths=None)`` makes the
    parameter keyword-only, so this is the other shape a pin bump can produce:
    there the injected keyword would collide with the caller's positional
    value and raise a TypeError inside the JIT path.
    """
    import sys
    import types

    from sglang_restomod.spec import sm70_sampling

    calls = {}

    def fake_load_jit(name, arg, extra_include_paths=None):
        calls["extra"] = extra_include_paths
        return "module"

    module = types.ModuleType("positional.sampling")
    module.load_jit = fake_load_jit
    monkeypatch.setitem(sys.modules, "positional.sampling", module)
    monkeypatch.setattr(sm70_sampling, "_JIT_SAMPLING_MODULE", "positional.sampling")
    monkeypatch.setattr(sm70_sampling, "_JIT_CCCL_INSTALLED", False)

    assert sm70_sampling.install_jit_cccl_compat()
    module.load_jit("speculative_sampling", 4, ["/opt/other/include"])
    assert calls["extra"] == [sm70_sampling._JIT_CCCL_DIR, "/opt/other/include"]


def test_jit_cccl_compat_reports_failure_when_the_module_has_no_load_jit(monkeypatch):
    """A present module without ``load_jit`` is a pin rename, not success."""
    import sys
    import types

    from sglang_restomod.spec import sm70_sampling

    module = types.ModuleType("renamed.sampling")
    monkeypatch.setitem(sys.modules, "renamed.sampling", module)
    monkeypatch.setattr(sm70_sampling, "_JIT_SAMPLING_MODULE", "renamed.sampling")
    monkeypatch.setattr(sm70_sampling, "_JIT_CCCL_INSTALLED", False)

    assert sm70_sampling.install_jit_cccl_compat() is False
    assert sm70_sampling._JIT_CCCL_INSTALLED is False


# --- the binding, not the module attribute ----------------------------------
#
# The pin binds ``LazyImport("flashinfer.sampling", "top_k_renorm_probs")`` in
# ``sglang/kernels/ops/sampling/renorm.py`` and ``eagle_utils`` takes it with a
# module-level ``from ... import``. On a host without flashinfer, patching the
# attribute on ``flashinfer.sampling`` cannot work: that module does not import.
# These tests pin the replacement to the binding.

_LAZY_STUB = '''\
class LazyImport:
    """Stand-in for sglang.utils.LazyImport: two attributes and a load."""

    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name

    def __call__(self, *args, **kwargs):
        raise RuntimeError("the flashinfer handle was called, not replaced")

    def __getattr__(self, name):
        raise RuntimeError("the flashinfer handle was read, not replaced")


top_k_renorm_prob = LazyImport("flashinfer.sampling", "top_k_renorm_probs")
top_p_renorm_prob = LazyImport("flashinfer.sampling", "top_p_renorm_probs")
'''


def _write_renorm_stub(root):
    package = root / "lazypkg"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "renorm.py").write_text(_LAZY_STUB, encoding="utf-8")
    # The eagle_utils shape: a module-level ``from ... import`` of the handle.
    (package / "eagle.py").write_text(
        "from lazypkg.renorm import top_k_renorm_prob, top_p_renorm_prob\n",
        encoding="utf-8",
    )
    return package


def test_flashinfer_lazy_handle_is_recognised():
    from sglang_restomod.spec import sm70_sampling

    class LazyImport:
        module_name = "flashinfer.sampling"
        class_name = "top_k_renorm_probs"

    assert sm70_sampling._is_flashinfer_lazy(LazyImport())
    # A different module, a different name, or a plain function is not a match.
    LazyImport.module_name = "flashinfer.other"
    assert not sm70_sampling._is_flashinfer_lazy(LazyImport())
    LazyImport.module_name = "flashinfer.sampling"
    LazyImport.class_name = "something_else"
    assert not sm70_sampling._is_flashinfer_lazy(LazyImport())
    assert not sm70_sampling._is_flashinfer_lazy(lambda *a: None)
    assert not sm70_sampling._is_flashinfer_lazy(None)


def test_rebind_replaces_a_handle_in_an_already_imported_module(monkeypatch):
    import sys
    import types

    from sglang_restomod.spec import sm70_sampling

    class LazyImport:
        module_name = "flashinfer.sampling"
        class_name = "top_k_renorm_probs"

    module = types.ModuleType("lazy.in.sys.modules")
    module.top_k_renorm_prob = LazyImport()
    monkeypatch.setitem(sys.modules, "lazy.in.sys.modules", module)

    assert sm70_sampling._rebind_module(module) == 1
    assert module.top_k_renorm_prob is sm70_sampling.top_k_renorm_probs
    # Nothing left to replace, so a second sweep is a no-op.
    assert sm70_sampling._rebind_module(module) == 0


def test_rebind_finder_patches_both_modules_on_import(monkeypatch, tmp_path):
    import importlib
    import sys

    from sglang_restomod.spec import sm70_sampling

    _write_renorm_stub(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sm70_sampling, "_RENORM_MODULES", ("lazypkg.renorm", "lazypkg.eagle"))
    monkeypatch.delitem(sys.modules, "lazypkg", raising=False)
    monkeypatch.delitem(sys.modules, "lazypkg.renorm", raising=False)
    monkeypatch.delitem(sys.modules, "lazypkg.eagle", raising=False)

    sm70_sampling._arm_rebind_finder()
    renorm = importlib.import_module("lazypkg.renorm")
    eagle = importlib.import_module("lazypkg.eagle")

    assert renorm.top_k_renorm_prob is sm70_sampling.top_k_renorm_probs
    assert renorm.top_p_renorm_prob is sm70_sampling.top_p_renorm_probs
    # The ``from ... import`` in the second module took the replaced function,
    # which is the binding the pin's eagle_utils actually calls.
    assert eagle.top_k_renorm_prob is sm70_sampling.top_k_renorm_probs
    assert eagle.top_p_renorm_prob is sm70_sampling.top_p_renorm_probs

    # One finder, however many times it is armed.
    sm70_sampling._arm_rebind_finder()
    finders = [f for f in sys.meta_path if isinstance(f, sm70_sampling._RenormRebindFinder)]
    assert len(finders) == 1


def test_unpatchable_binding_reports_a_pin_rename(monkeypatch):
    import sys
    import types

    from sglang_restomod.spec import sm70_sampling

    monkeypatch.setattr(sm70_sampling, "_RENORM_MODULES", ("renamed.renorm",))
    assert sm70_sampling._unpatchable_binding() is None

    module = types.ModuleType("renamed.renorm")
    module.top_k_renorm_prob = lambda probs, k: probs
    module.top_p_renorm_prob = sm70_sampling.top_p_renorm_probs
    monkeypatch.setitem(sys.modules, "renamed.renorm", module)

    reason = sm70_sampling._unpatchable_binding()
    assert reason is not None and "renamed.renorm.top_k_renorm_prob" in reason


def test_install_arms_the_finder_even_with_no_binding_present(monkeypatch):
    """No binding at registration time is the normal case, not a failure.

    The renorm modules are imported by the speculative worker after the plugin
    loads. Returning a reason here is what made the component report inactive
    on a host where it would have worked.
    """
    import sys

    from sglang_restomod.spec import sm70_sampling

    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sm70_sampling, "_INSTALLED", False)
    monkeypatch.setattr(sm70_sampling, "_rebind_lazy_imports", lambda: 0)
    monkeypatch.setattr(sm70_sampling, "_patch_flashinfer_attributes", lambda: 0)
    monkeypatch.setattr(sm70_sampling, "_unpatchable_binding", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 0))

    assert sm70_sampling.install() is None
    finders = [f for f in sys.meta_path if isinstance(f, sm70_sampling._RenormRebindFinder)]
    assert len(finders) == 1


def test_register_records_inactive_when_install_cannot_bind(monkeypatch):
    from sglang_restomod import attest
    from sglang_restomod.spec import sm70_sampling

    monkeypatch.setattr(attest, "reset", attest.reset)
    attest.reset()
    monkeypatch.setattr(sm70_sampling, "install", lambda: "no flashinfer binding")
    monkeypatch.setattr(sm70_sampling, "install_jit_cccl_compat", lambda: True)

    sm70_sampling.register()

    snapshot = attest.snapshot()
    assert "spec_sm70" in snapshot["inactive"]
    assert "no flashinfer binding" in snapshot["components"]["spec_sm70"]["detail"]
    attest.reset()


def test_register_records_nothing_when_install_succeeds(monkeypatch):
    from sglang_restomod import attest
    from sglang_restomod.spec import sm70_sampling

    attest.reset()
    monkeypatch.setattr(sm70_sampling, "install", lambda: None)
    monkeypatch.setattr(sm70_sampling, "install_jit_cccl_compat", lambda: True)

    sm70_sampling.register()

    assert attest.snapshot()["inactive"] == []
    attest.reset()
