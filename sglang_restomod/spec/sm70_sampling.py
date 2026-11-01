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

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_INSTALLED = False


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
    logger.info(
        "sglang-restomod: EAGLE renorm kernels replaced for SM70 (%d rebound)", rebound
    )
    return True


def register() -> None:
    """Component entry point."""
    install()
