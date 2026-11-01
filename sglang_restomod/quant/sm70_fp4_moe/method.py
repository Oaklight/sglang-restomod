"""SM70 NVFP4 MoE through the marlin_v100 extension.

Three upstream hooks are replaced:

1. ``prepare_moe_nvfp4_layer_for_marlin`` -- the checkpoint weights are repacked
   into the Marlin layout the SM70 kernel reads, and the E4M3 scales are
   encoded as the S0E5M3 bytes its integer dequantizer expects. Upstream's
   version permutes the scales with ``marlin_permute_scales``; the SM70 iterator
   indexes them by logical ``n``, so that permutation must not run.
2. ``moe_wna16_marlin_gemm`` -- upstream's JIT Marlin kernel is an empty-body
   stub below SM80. The call is forwarded to the prebuilt SM70 kernel with the
   two signature differences bridged here (``a_scales`` and the extra launch
   hints).
3. ``FusedMoE`` selection is unchanged: launch with
   ``--moe-runner-backend marlin`` and ``--quantization modelopt_fp4``.

The extension is built outside the package; see ``PROVENANCE.md``.
"""

from __future__ import annotations

import logging
import os

import torch

from .marlin import ENV_DIR, available, ensure_loaded
from .scales import encode_global_scale, encode_scales

logger = logging.getLogger(__name__)

ENV_MODE = "SGLANG_SM70_FP4_MOE"


def _mode() -> str:
    return os.environ.get(ENV_MODE, "auto").strip().lower()


def _is_sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def _repack_experts(weight: torch.Tensor) -> torch.Tensor:
    """Packed NVFP4 ``[E, N, K/2]`` uint8 -> Marlin ``[E, K/16, N*2]`` int32."""
    ops = ensure_loaded()
    experts, size_n, packed_k = (int(d) for d in weight.shape)
    size_k = packed_k * 2
    if size_k % 16 or size_n % 64:
        raise RuntimeError(
            "SM70 NVFP4 Marlin needs K divisible by 16 and N by 64, got "
            f"N={size_n} K={size_k}"
        )
    perm = torch.empty(0, dtype=torch.int32, device=weight.device)
    out = torch.empty(
        (experts, size_k // 16, size_n * 2), dtype=torch.int32, device=weight.device
    )
    for expert in range(experts):
        qweight = weight[expert].view(torch.int32).T.contiguous()
        out[expert] = ops.repack(qweight, perm, size_k, size_n, 4, False)
    return out


def prepare_moe_nvfp4(layer, group_size: int | None = None) -> None:
    """Replace the checkpoint expert layout with the SM70 Marlin layout."""
    from sglang.srt.layers.utils import copy_or_rebind_param

    if layer.params_dtype != torch.float16:
        raise RuntimeError(
            "The SM70 NVFP4 profile requires FP16 activations, got "
            f"{layer.params_dtype}"
        )
    if hasattr(layer, "moe_runner_config") and not layer.moe_runner_config.is_gated:
        raise RuntimeError("The SM70 NVFP4 profile requires gated MoE experts")
    if group_size is None:
        group_size = getattr(getattr(layer, "quant_config", None), "group_size", 16)
    if group_size != 16:
        raise RuntimeError(f"SM70 NVFP4 Marlin requires group_size=16, got {group_size}")
    if any(getattr(layer, name, None) is not None for name in ("w13_bias", "w2_bias")):
        raise RuntimeError("The SM70 NVFP4 profile does not support expert bias")

    ensure_loaded()  # fail before touching the parameters

    device = layer.w13_weight.device
    layer.workspace = torch.zeros(
        torch.cuda.get_device_properties(device).multi_processor_count * 4,
        device=device,
        dtype=torch.int32,
    )

    for prefix in ("w13", "w2"):
        weight = getattr(layer, prefix + "_weight")
        scales = getattr(layer, prefix + "_weight_scale")
        global_scales = getattr(layer, prefix + "_weight_scale_2")

        if global_scales.ndim == 2:
            if not torch.equal(global_scales[:, 0], global_scales[:, 1]):
                raise RuntimeError(
                    "Unequal NVFP4 gate/up global scales are unsupported on SM70"
                )
            global_scales = global_scales[:, 0]

        encoded, factor = encode_scales(
            scales.transpose(1, 2).contiguous(), torch.float16
        )
        global_scales = (
            encode_global_scale(global_scales, torch.float16) / factor
        ).contiguous()

        copy_or_rebind_param(layer, prefix + "_weight", _repack_experts(weight))
        copy_or_rebind_param(layer, prefix + "_weight_scale", encoded)
        copy_or_rebind_param(layer, prefix + "_weight_scale_2", global_scales)

        # ModelOpt allocates a swizzled copy for other backends. Marlin reads
        # only weight_scale; keep the unused attribute as an alias so the
        # second allocation is released.
        swizzled = prefix + "_blockscale_swizzled"
        if hasattr(layer, swizzled):
            setattr(layer, swizzled, getattr(layer, prefix + "_weight_scale"))


def moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c_or_none: torch.Tensor | None,
    b_q_weight: torch.Tensor,
    b_bias_or_none: torch.Tensor | None,
    b_scales: torch.Tensor,
    global_scale_or_none: torch.Tensor | None,
    b_zeros_or_none: torch.Tensor | None,
    g_idx_or_none: torch.Tensor | None,
    perm_or_none: torch.Tensor | None,
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    is_ep: bool,
    b_q_type,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    """Forward the pinned Marlin call to the SM70 extension.

    Signature differences against the extension:

    * it takes an ``a_scales`` tensor after ``b_scales`` (unused here);
    * it takes the scalar-type id, not the ``ScalarType`` object and ``is_ep``;
    * it takes three launch hints, which the SM70 kernel ignores.
    """
    del is_ep
    op = ensure_loaded().gemm
    return op(
        a,
        c_or_none,
        b_q_weight,
        b_bias_or_none,
        b_scales,
        None,
        global_scale_or_none,
        b_zeros_or_none,
        g_idx_or_none,
        perm_or_none,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        int(b_q_type.id),
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
        -1,
        -1,
        -1,
    )


_INSTALLED = False
_ORIGINALS: dict[str, object] = {}


def _patch(module, name: str, replacement) -> None:
    original = getattr(module, name, None)
    if original is None:
        return
    _ORIGINALS[f"{module.__name__}.{name}"] = original
    setattr(module, name, replacement)


def _install_sum_reduce() -> bool:
    """Supply ``moe_sum_reduce`` when the installed sgl_kernel lacks it.

    The function is imported into several modules by name, so replacing the
    attribute on ``sgl_kernel`` alone misses the bindings made at import time.
    Every already-imported module that still points at the original function is
    rewritten as well.
    """
    import sys

    from . import sum_reduce

    if not sum_reduce.available():
        return False
    try:
        import sgl_kernel
        import sgl_kernel.moe as sgl_moe
    except ImportError:
        return False

    original = sgl_moe.moe_sum_reduce
    replacement = sum_reduce.moe_sum_reduce
    sgl_kernel.moe_sum_reduce = replacement
    sgl_moe.moe_sum_reduce = replacement
    for module in list(sys.modules.values()):
        if module is None:
            continue
        if getattr(module, "moe_sum_reduce", None) is original:
            setattr(module, "moe_sum_reduce", replacement)
    logger.info("SM70 NVFP4 MoE: replaced missing sgl_kernel.moe_sum_reduce")
    return True


def install() -> bool:
    """Install the SM70 NVFP4 hooks. Returns True when they were installed."""
    global _INSTALLED
    if _INSTALLED:
        return True
    mode = _mode()
    if mode == "off":
        return False
    if mode not in ("auto", "on"):
        raise ValueError(f"{ENV_MODE} must be auto, on or off; got {mode!r}")
    if not _is_sm70():
        if mode == "on":
            raise RuntimeError(f"{ENV_MODE}=on requires an SM70 device")
        return False
    if mode == "auto" and not available():
        logger.warning(
            "SM70 NVFP4 MoE: %s does not hold a usable marlin_v100 build; "
            "leaving upstream Marlin in place",
            os.environ.get(ENV_DIR, "<unset>"),
        )
        return False

    _install_sum_reduce()

    from sglang.kernels.ops.moe import moe_wna16_marlin
    from sglang.srt.layers.quantization import marlin_utils_fp4, modelopt_quant

    _patch(marlin_utils_fp4, "prepare_moe_nvfp4_layer_for_marlin", prepare_moe_nvfp4)
    # modelopt_quant imported the name at module load.
    _patch(modelopt_quant, "prepare_moe_nvfp4_layer_for_marlin", prepare_moe_nvfp4)
    _patch(moe_wna16_marlin, "moe_wna16_marlin_gemm", moe_wna16_marlin_gemm)
    try:
        from sglang.srt.layers.moe.fused_moe_triton import fused_marlin_moe

        _patch(fused_marlin_moe, "moe_wna16_marlin_gemm", moe_wna16_marlin_gemm)
    except ImportError:
        pass

    _INSTALLED = True
    logger.info("SM70 NVFP4 MoE method installed (marlin_v100)")
    return True


def uninstall() -> None:
    """Test helper: restore the upstream hooks."""
    global _INSTALLED
    if not _INSTALLED:
        return
    for key, original in list(_ORIGINALS.items()):
        module_name, attr = key.rsplit(".", 1)
        import importlib

        setattr(importlib.import_module(module_name), attr, original)
    _ORIGINALS.clear()
    _INSTALLED = False


def register() -> None:
    """Component entry point."""
    install()
