"""SM70 replacements for the FlashInfer renorm kernels used by EAGLE verify.

Non-greedy speculative decoding renormalizes the target distribution before
tree verification. The pinned EAGLE path calls FlashInfer directly, whatever
``--sampling-backend`` says:

    eagle_utils.py:984 eagle_sample -> top_k_renorm_prob
    flashinfer/sampling.py:2095 -> get_sampling_module
    RuntimeError: FlashInfer requires GPUs with sm75 or higher.

These two functions keep the FlashInfer contract (per-row ``top_k``/``top_p``,
same dtype out, renormalized rows) and run on any CUDA device:

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

    del is_deterministic
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
    """Keep the smallest prefix of the sorted row that reaches ``top_p``."""
    import torch

    del is_deterministic
    if probs.dim() != 2:
        raise ValueError(f"top_p_renorm_probs expects [rows, vocab], got {probs.shape}")
    out_dtype = probs.dtype
    dense = probs.float()
    rows, _ = dense.shape
    threshold = _per_row(top_p, rows, dense.device)

    sorted_probs, indices = torch.sort(dense, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Keep an entry while the mass before it is still below top_p. This keeps
    # the entry that crosses the threshold, matching FlashInfer's contract.
    keep = (cumulative - sorted_probs) < threshold.unsqueeze(1)
    keep[:, 0] = True
    kept = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    out = torch.zeros_like(dense)
    out.scatter_(1, indices, kept)
    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out.to(out_dtype)


def install() -> bool:
    """Replace the FlashInfer renorm entry points on SM70."""
    global _INSTALLED
    if _INSTALLED:
        return True
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 8:
        return False
    try:
        import flashinfer.sampling as sampling
    except Exception:  # noqa: BLE001 - absent flashinfer means nothing to patch
        return False

    sampling.top_k_renorm_probs = top_k_renorm_probs
    sampling.top_p_renorm_probs = top_p_renorm_probs
    # The public aliases are imported by name in some call sites.
    sampling.top_k_renorm_prob = top_k_renorm_probs
    sampling.top_p_renorm_prob = top_p_renorm_probs
    _INSTALLED = True
    logger.info("sglang-restomod: EAGLE renorm kernels replaced for SM70")
    return True


def register() -> None:
    """Component entry point."""
    install()
