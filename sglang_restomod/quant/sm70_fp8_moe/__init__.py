"""SM70 block-FP8 MoE GEMM.

The prebuilt TurboMind library that SGLang ships for SM70 exports
``fp8_moe_gemm_sm70_out`` but never registers it with torch, and the kernel
sources are not part of the SGLang tree. This component resolves the exported
symbol with a small ``dlopen``/``dlsym`` shim (``shim.cpp``) and exposes a
typed Python surface.

Import of this module is cheap. Torch, sglang and the shim are touched only by
:func:`ensure_loaded`, which the public helpers call on first use.

Contract, validated on lambda4 against an FP16-dequantized reference
(relative error 6.5e-4, gated, four experts):

* weights are per expert ``[N, K]`` e4m3 with scales ``[N/128, K/128]`` fp32;
* ``prepare_experts`` runs ``fp8_prepare(..., interleave_gated_silu=True)`` per
  expert and builds the pointer arrays with the AWQ pointer builder;
* ``apply`` consumes tokens in expert-sorted order and writes ``[tokens, N/2]``
  when ``gated_silu`` is set.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SO_ENV = "SM70_TURBOMIND_SO"
_SO_ENV_ALIAS = "SGLANG_SM70_TURBOMIND_SO"
_CACHE_ENV = "SGLANG_RESTOMOD_EXT_CACHE"

_LOCK = threading.Lock()
_LOADED = False
_SO_PATH: str | None = None


@dataclass(frozen=True)
class Experts:
    """Prepared expert weights and their device pointer arrays."""

    tm_weight: object
    tm_scales: object
    ptr_w: object
    ptr_s: object
    k_ld: int
    q_ld: int
    num_experts: int


def _default_so_path() -> str:
    from sglang.kernels.sm70_paths import sm70_prebuilt

    return str(sm70_prebuilt("_sm70_turbomind_v100.so"))


def ensure_loaded(so_path: str | None = None) -> str:
    """Load the TurboMind library and the binding shim once. Returns the path."""
    global _LOADED, _SO_PATH
    with _LOCK:
        if _LOADED:
            return _SO_PATH or ""
        path = (
            so_path
            or os.environ.get(_SO_ENV)
            or os.environ.get(_SO_ENV_ALIAS)
            or _default_so_path()
        )
        if not Path(path).is_file():
            raise FileNotFoundError(f"SM70 TurboMind library not found: {path}")
        os.environ[_SO_ENV] = path

        import torch

        torch.ops.load_library(path)

        from torch.utils.cpp_extension import load

        cache = os.environ.get(_CACHE_ENV)
        if cache:
            os.environ.setdefault("TORCH_EXTENSIONS_DIR", cache)
        load(
            name="sglang_sm70_fp8_moe_shim",
            sources=[str(Path(__file__).with_name("shim.cpp"))],
            extra_cflags=["-O2"],
            is_python_module=False,
            verbose=False,
        )
        _LOADED = True
        _SO_PATH = path
        logger.info("SM70 FP8 MoE binding loaded from %s", path)
        return path


def available() -> bool:
    """Whether the binding can be loaded in this environment."""
    try:
        ensure_loaded()
        return True
    except Exception as exc:  # noqa: BLE001 - availability probe
        logger.debug("SM70 FP8 MoE binding unavailable: %s", exc)
        return False


def prepare_experts(
    weights,
    scales,
    *,
    group_size: int = 128,
    interleave_gated_silu: bool = True,
) -> Experts:
    """Prepare per-expert block-FP8 weights for :func:`apply`.

    ``weights`` is ``[experts, N, K]`` e4m3, ``scales`` is
    ``[experts, N/group_size, K/group_size]`` fp32.
    """
    ensure_loaded()
    import torch

    experts = int(weights.shape[0])
    prepared = [
        torch.ops.sglang_sm70_turbomind.fp8_prepare(
            weights[e].contiguous(),
            scales[e].contiguous(),
            group_size,
            interleave_gated_silu,
        )
        for e in range(experts)
    ]
    tm_weight = torch.stack([item[0] for item in prepared])
    tm_scales = torch.stack([item[1] for item in prepared])
    k_ld, q_ld = int(prepared[0][2][0]), int(prepared[0][2][1])
    ptr_w, ptr_s = torch.ops.sglang_sm70_turbomind.build_ptrs(
        tm_weight, tm_scales, k_ld, q_ld, experts
    )
    return Experts(
        tm_weight=tm_weight,
        tm_scales=tm_scales,
        ptr_w=ptr_w,
        ptr_s=ptr_s,
        k_ld=k_ld,
        q_ld=q_ld,
        num_experts=experts,
    )


def gemm(out, sorted_input, expert_offsets, experts: Experts, *, k, n, gated_silu=True):
    """Run the expert GEMM into ``out`` (sorted order)."""
    ensure_loaded()
    import torch

    torch.ops.sglang_sm70_fp8_moe.fp8_moe_gemm(
        out,
        sorted_input,
        expert_offsets,
        experts.ptr_w,
        experts.ptr_s,
        experts.num_experts,
        int(k),
        int(n),
        128,
        bool(gated_silu),
    )


def apply(out, sorted_input, expert_offsets, experts: Experts, *, gated_silu=True):
    """Like :func:`gemm`, inferring ``k`` and ``n`` from the tensors."""
    k = int(sorted_input.shape[1])
    n = int(out.shape[1] * 2 if gated_silu else out.shape[1])
    gemm(
        out,
        sorted_input,
        expert_offsets,
        experts,
        k=k,
        n=n,
        gated_silu=gated_silu,
    )


def register() -> None:
    """Component entry point. Weight preparation stays lazy; only the quant
    method selection is installed here (a device check plus a file probe)."""
    from .method import register as _install

    _install()
