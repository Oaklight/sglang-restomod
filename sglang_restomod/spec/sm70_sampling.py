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


def install() -> bool:
    """Replace the FlashInfer renorm entry points on SM70.

    ``eagle_utils`` imports the name inside ``eagle_sample``, so the attribute
    is resolved per call and replacing it on ``flashinfer.sampling`` is enough
    today. A module that hoisted that import would keep the original binding,
    so every already-imported module that still points at the original is
    rewritten as well -- the same sweep ``sm70_fp4_moe`` uses for
    ``moe_sum_reduce``.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    import sys

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 8:
        return False
    try:
        import flashinfer.sampling as sampling
    except Exception:  # noqa: BLE001 - absent flashinfer means nothing to patch
        return False

    originals = {}
    for alias in _ALIASES:
        original = getattr(sampling, alias, None)
        if original is not None:
            originals[alias] = original
    if not originals:
        return False

    for alias in _ALIASES:
        setattr(sampling, alias, _REPLACEMENTS[alias])

    rebound = 0
    for module in list(sys.modules.values()):
        if module is None or module is sampling:
            continue
        for alias, original in originals.items():
            try:
                bound = getattr(module, alias, None)
            except Exception:  # noqa: BLE001 - partial modules raise anything
                continue
            if bound is not original:
                continue
            try:
                setattr(module, alias, _REPLACEMENTS[alias])
            except Exception:  # noqa: BLE001 - read-only module attribute
                continue
            rebound += 1

    _INSTALLED = True
    logger.info("sglang-restomod: EAGLE renorm kernels replaced for SM70 (%d rebound)", rebound)
    return True


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
    """Run the real loader, then patch the module it produced."""

    def __init__(self, loader) -> None:
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        _wrap_sampling_module(module)


class _JitSamplingFinder:
    """Patch the speculative sampling module when it is first imported.

    The component must not import ``torch``-heavy SGLang modules at
    registration time, and a profile without speculative decoding never needs
    this one, so the patch is deferred to the import itself.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _JIT_SAMPLING_MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None:
                spec.loader = _PatchingLoader(spec.loader)
                return spec
        return None


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
    """Component entry point."""
    install()
    install_jit_cccl_compat()
