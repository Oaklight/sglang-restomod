"""SM70 replacements for the FlashInfer renorm kernels used by EAGLE verify.

Non-greedy speculative decoding renormalizes the target distribution before
tree verification. The pinned EAGLE path calls FlashInfer directly, whatever
``--sampling-backend`` says:

    eagle_utils.py:984 eagle_sample -> top_k_renorm_prob
    flashinfer/sampling.py:2095 -> get_sampling_module
    RuntimeError: FlashInfer requires GPUs with sm75 or higher.

These two functions keep the FlashInfer contract and run on any CUDA device.
The output dtype is mirrored per function, because the two upstream kernels do
not agree in the build this runs against (0.6.12, the ``flashinfer-sm70``
source cache): the ``top_k_renorm_probs`` custom op allocates
``empty_like(probs)`` and keeps the input dtype, while ``top_p_renorm_probs``
does ``probs = probs.float()`` before allocating and therefore always returns
float32. On the EAGLE path both inputs come from a float32 softmax, so the
difference is invisible there -- but a drop-in replacement that normalised it
away would silently change the dtype of the chained call.

``is_deterministic`` is accepted for signature compatibility and does not
change the result. Upstream's flag asks for reproducible *accumulation* (fixed
point integer atomics instead of a float reduction whose order depends on CTA
arrival); a sort and a sequential sum are already bitwise reproducible for a
given input, so the property holds without a branch.

* ``top_k`` keeps the ``k`` largest entries per row.
* ``top_p`` keeps the smallest set of the largest entries whose sum reaches
  ``top_p`` (the crossing entry is kept), then renormalizes.

The cost is one ``topk`` or ``sort`` over the vocabulary per row. At verify
shapes that is a few rows, so it does not dominate the step.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import sys
from typing import TYPE_CHECKING

from sglang_restomod import attest

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_INSTALLED = False

#: Directory holding the ``cuda/functional`` backport for CUDA 12.x toolkits.
_JIT_CCCL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cccl")

#: Module that builds the FlashInfer tree-sampling kernel on first call.
_JIT_SAMPLING_MODULE = "sglang.kernels.ops.speculative.sampling"

_JIT_CCCL_INSTALLED = False


def _per_row(value, batch: int, device) -> torch.Tensor:
    import torch

    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32).reshape(-1)
    return torch.full((batch,), float(value), device=device, dtype=torch.float32)


def top_k_renorm_probs(probs, top_k, is_deterministic: bool = False):
    """Keep the ``top_k`` largest entries per row and renormalize."""
    import torch

    if probs.dim() != 2:
        raise ValueError(f"top_k_renorm_probs expects [rows, vocab], got {probs.shape}")
    out_dtype = probs.dtype
    dense = probs.float()
    rows, vocab = dense.shape
    k = _per_row(top_k, rows, dense.device).clamp(1, vocab).to(torch.int64)
    k_max = int(k.max().item())
    values, indices = torch.topk(dense, k_max, dim=-1)
    rank = torch.arange(k_max, device=dense.device).unsqueeze(0)
    kept = torch.where(rank < k.unsqueeze(1), values, torch.zeros_like(values))
    out = torch.zeros_like(dense)
    out.scatter_(1, indices, kept)
    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out.to(out_dtype)


def top_p_renorm_probs(probs, top_p, is_deterministic: bool = False):
    """Keep the smallest prefix of the sorted row that reaches ``top_p``.

    Returns float32 whatever the input dtype: FlashInfer's version casts to
    ``float()`` before allocating its output.
    """
    import torch

    if probs.dim() != 2:
        raise ValueError(f"top_p_renorm_probs expects [rows, vocab], got {probs.shape}")
    dense = probs.float()
    rows, _ = dense.shape
    threshold = _per_row(top_p, rows, dense.device)

    # A stable sort is not needed: no atomics are involved, so the result is
    # bitwise reproducible for a given input whatever ``is_deterministic`` says.
    sorted_probs, indices = torch.sort(dense, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Keep an entry while the mass before it is still below top_p. This keeps
    # the entry that crosses the threshold, matching FlashInfer's contract.
    keep = (cumulative - sorted_probs) < threshold.unsqueeze(1)
    keep[:, 0] = True
    kept = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    out = torch.zeros_like(dense)
    out.scatter_(1, indices, kept)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)


#: FlashInfer exports both spellings. ``eagle_utils`` imports the plural one.
_ALIASES = (
    "top_k_renorm_probs",
    "top_p_renorm_probs",
    "top_k_renorm_prob",
    "top_p_renorm_prob",
)

_REPLACEMENTS = {
    "top_k_renorm_probs": top_k_renorm_probs,
    "top_p_renorm_probs": top_p_renorm_probs,
    "top_k_renorm_prob": top_k_renorm_probs,
    "top_p_renorm_prob": top_p_renorm_probs,
}


#: Modules that bind the renorm entry points before ``eagle_utils`` uses them.
_RENORM_MODULES = (
    "sglang.kernels.ops.sampling.renorm",
    "sglang.srt.speculative.eagle_utils",
)


def _bound(module, alias):
    """``getattr`` that survives a module-level ``__getattr__`` that raises."""
    try:
        return getattr(module, alias, None)
    except Exception:  # noqa: BLE001 - partial modules raise anything
        return None


def _is_flashinfer_lazy(obj) -> bool:
    """True when ``obj`` is a lazy handle onto one of the names we replace.

    At this pin ``sglang/kernels/ops/sampling/renorm.py`` binds
    ``LazyImport("flashinfer.sampling", "top_k_renorm_probs")`` and
    ``eagle_utils`` takes it with a module-level ``from ... import``, so the
    handle itself is what has to be replaced: patching the attribute on
    ``flashinfer.sampling`` needs that module to import, and on the host this
    component exists for it does not.

    Duck-typed on the class name rather than ``isinstance``, because the class
    lives in ``sglang.utils`` and asking the object's type is the only question
    that cannot trigger a load. ``module_name`` and ``class_name`` are instance
    attributes, so reading them does not go through ``LazyImport.__getattr__``
    either -- that one imports flashinfer.
    """
    if type(obj).__name__ != "LazyImport":
        return False
    if _bound(obj, "module_name") != "flashinfer.sampling":
        return False
    return _bound(obj, "class_name") in _REPLACEMENTS


def _rebind_module(module) -> int:
    """Replace the flashinfer ``LazyImport`` handles bound in one module."""
    rebound = 0
    for alias, replacement in _REPLACEMENTS.items():
        if not _is_flashinfer_lazy(_bound(module, alias)):
            continue
        try:
            setattr(module, alias, replacement)
        except Exception:  # noqa: BLE001 - read-only module attribute
            continue
        rebound += 1
    return rebound


def _rebind_lazy_imports() -> int:
    """Replace every binding that is already in ``sys.modules``."""
    rebound = 0
    for module in list(sys.modules.values()):
        if module is None:
            continue
        rebound += _rebind_module(module)
    return rebound


def _unpatchable_binding() -> str | None:
    """A reason when a loaded module holds a binding this cannot replace.

    The one case the sweep cannot fix and the finder cannot see: the module was
    imported before this ran and its entry point is not a flashinfer lazy
    handle, which is a pin rename. Reporting success there is the failure this
    component is being repaired for.
    """
    for name in _RENORM_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        for alias in ("top_k_renorm_prob", "top_p_renorm_prob"):
            if _bound(module, alias) is _REPLACEMENTS[alias]:
                continue
            return (
                f"{name}.{alias} is not a flashinfer LazyImport and was not "
                "replaced: the pin renamed the renorm entry point"
            )
    return None


def _arm_rebind_finder() -> None:
    """Patch the renorm modules on import, for a launch where they import later."""
    if any(isinstance(finder, _RenormRebindFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _RenormRebindFinder())


def _patch_flashinfer_attributes() -> int:
    """Replace the names on ``flashinfer.sampling`` when that module exists.

    Needed for a call site that imports the function after this runs. Returns
    the number of attributes replaced, or 0 when flashinfer is absent -- which
    is the normal case on a host where it was never installed, and is not a
    failure on its own.
    """
    try:
        import flashinfer.sampling as sampling
    except Exception:  # noqa: BLE001 - absent flashinfer is a supported case
        return 0
    patched = 0
    for alias in _ALIASES:
        if getattr(sampling, alias, None) is None:
            continue
        try:
            setattr(sampling, alias, _REPLACEMENTS[alias])
            patched += 1
        except Exception:  # noqa: BLE001
            continue
    return patched


def install() -> str | None:
    """Replace the EAGLE renorm entry points on SM70.

    Returns ``None`` when the replacement bound, or the reason it did not. The
    reason is what ``register()`` hands to ``attest.inactive``: a component that
    reports success while doing nothing is the failure this whole module exists
    to avoid, and the previous signature -- a bool nobody read -- is how this one
    stayed invisible on a host where flashinfer has never been installed.
    """
    global _INSTALLED
    if _INSTALLED:
        return None
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device: the renorm kernels have nothing to run on"
    if torch.cuda.get_device_capability()[0] >= 8:
        # Not a failure: on sm80+ the FlashInfer kernels are the intended path.
        return "not SM70, FlashInfer's own kernels are the intended path"

    rebound = _rebind_lazy_imports()
    patched = _patch_flashinfer_attributes()
    stale = _unpatchable_binding()
    if stale is not None:
        return stale
    # A finding at registration time is not required. ``renorm`` and
    # ``eagle_utils`` are imported by the speculative worker long after the
    # plugin loads, so zero bindings here is the normal case, and the earlier
    # version of this function called it a failure -- which is the same shape
    # as the bug it now avoids. The finder is what covers those imports.
    _arm_rebind_finder()
    _INSTALLED = True
    # The count belongs in the attestation, not only in a log line: on these
    # hosts plugin log lines do not reach the log file, so the JSON is the only
    # place a launch can say what it bound. Zero now is normal -- the renorm
    # modules import later -- and the finder is what covers them.
    detail = (
        f"armed: {rebound} binding(s) replaced at registration, {patched} "
        f"flashinfer attribute(s); rebinds {', '.join(_RENORM_MODULES)} on import"
    )
    attest.record("spec_sm70", True, detail)
    logger.info("sglang-restomod: EAGLE renorm kernels replaced for SM70 (%s)", detail)
    return None


def _positional_index(func, name: str) -> int | None:
    """Index of ``name`` in ``func``'s positional parameters, or None.

    None when the parameter is absent, is keyword-only, or sits behind
    ``*args``: in those cases the caller cannot pass it positionally and the
    keyword path is the only one that exists.
    """
    try:
        parameters = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return None
    for index, parameter in enumerate(parameters):
        if parameter.name == name:
            return index
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return None
    return None


def _wrap_sampling_module(module) -> bool:
    """Put the shim on ``module.load_jit``. False when there is nothing to wrap."""
    original = getattr(module, "load_jit", None)
    if original is None or getattr(original, "_restomod_cccl_shim", False):
        return False

    # A caller that passes ``extra_include_paths`` positionally gets a
    # ``TypeError`` from the keyword below, inside the JIT path the shim exists
    # to fix. The pin does not, but a wrapper whose failure mode is that error
    # is worse than one that forwards in the form it received.
    extra_index = _positional_index(original, "extra_include_paths")

    @functools.wraps(original)
    def load_jit(*args, **kwargs):
        if extra_index is not None and len(args) > extra_index:
            args = list(args)
            paths = list(args[extra_index] or ())
            if _JIT_CCCL_DIR not in paths:
                paths.insert(0, _JIT_CCCL_DIR)
            args[extra_index] = paths
            args = tuple(args)
        else:
            paths = list(kwargs.get("extra_include_paths") or ())
            if _JIT_CCCL_DIR not in paths:
                paths.insert(0, _JIT_CCCL_DIR)
            kwargs["extra_include_paths"] = paths
        return original(*args, **kwargs)

    load_jit._restomod_cccl_shim = True
    module.load_jit = load_jit
    logger.info("sglang-restomod: CCCL shim added to the speculative JIT includes")
    return True


class _PatchingLoader:
    """Run the real loader, then hand the module to ``patch``."""

    def __init__(self, loader, patch) -> None:
        self._loader = loader
        self._patch = patch

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        self._patch(module)


class _PatchingFinder:
    """Patch named modules right after the import system builds them.

    ``targets`` is a method, not a class attribute, so a test can point it at a
    stub module by patching the global it reads.
    """

    def __init__(self, patch) -> None:
        self._patch = patch

    def targets(self) -> tuple[str, ...]:
        return ()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.targets():
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None:
                spec.loader = _PatchingLoader(spec.loader, self._patch)
                return spec
        return None


class _JitSamplingFinder(_PatchingFinder):
    """Patch the speculative sampling module when it is first imported.

    The component must not import ``torch``-heavy SGLang modules at
    registration time, and a profile without speculative decoding never needs
    this one, so the patch is deferred to the import itself.
    """

    def __init__(self) -> None:
        super().__init__(_wrap_sampling_module)

    def targets(self) -> tuple[str, ...]:
        return (_JIT_SAMPLING_MODULE,)


class _RenormRebindFinder(_PatchingFinder):
    """Rebind the renorm entry points when their modules first import."""

    def __init__(self) -> None:
        super().__init__(_rebind_module)

    def targets(self) -> tuple[str, ...]:
        return _RENORM_MODULES


def install_jit_cccl_compat() -> bool:
    """Add the CCCL backport to the speculative JIT include path.

    The pinned ``flashinfer/sampling.cuh`` uses ``cuda::maximum`` and
    ``cuda::minimum``, which exist only in CCCL 3.x (CUDA 13). Under CUDA 12.x
    the JIT build of the tree-sampling kernel therefore fails before it starts:

        flashinfer/sampling.cuh(40): error: namespace "cuda" has no member "maximum"

    ``sglang/kernels/ops/speculative/sampling.py`` calls ``load_jit`` without
    ``extra_include_paths``, and resolves that name from its own module
    namespace at call time, so replacing the attribute is enough -- the pin
    stays unedited. ``cccl/`` next to this file holds the shim; it defers to
    the toolkit on CUDA 13 and later.

    Returns True once the hook is armed, including when it already was. False
    when the module is already imported and has no ``load_jit`` to wrap: that
    is a pin rename, and reporting success there would send the next step back
    to the original "namespace cuda has no member maximum" error with a
    component that logged that it had installed.
    """
    global _JIT_CCCL_INSTALLED
    if _JIT_CCCL_INSTALLED:
        return True

    # A worker that imported the module before the plugin loaded still has to
    # be patched, and one that has not gets patched by the finder on import.
    module = sys.modules.get(_JIT_SAMPLING_MODULE)
    if module is not None:
        load_jit = getattr(module, "load_jit", None)
        if load_jit is None:
            logger.warning(
                "sglang-restomod: %s has no load_jit; the CCCL shim is not armed",
                _JIT_SAMPLING_MODULE,
            )
            return False
        if not getattr(load_jit, "_restomod_cccl_shim", False) and not _wrap_sampling_module(
            module
        ):
            return False
    elif not any(isinstance(finder, _JitSamplingFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _JitSamplingFinder())

    _JIT_CCCL_INSTALLED = True
    return True


def register() -> None:
    """Component entry point.

    ``install()`` used to return a bool that this function dropped on the floor,
    so a component that could not bind anything still recorded as loaded: the
    attestation said healthy, ``inactive`` was empty, and every non-greedy
    request died in ``eagle_sample``. The reason now goes to ``attest.inactive``
    so a launch that cannot replace these kernels says so.
    """
    reason = install()
    if reason is not None:
        attest.inactive("spec_sm70", reason)
        logger.warning("sglang-restomod: spec_sm70 inactive: %s", reason)
    install_jit_cccl_compat()
