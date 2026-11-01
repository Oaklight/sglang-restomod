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


def check_expert_shape(size_n: int, size_k: int) -> None:
    """Reject an expert shard the SM70 Marlin tile cannot cover.

    Runs before the extension loads, so it is reachable without a GPU.
    """
    if size_k % 16 or size_n % 64:
        raise RuntimeError(
            "SM70 NVFP4 Marlin needs K divisible by 16 and N by 64, got "
            f"N={size_n} K={size_k}"
        )


def _repack_experts(weight: torch.Tensor) -> torch.Tensor:
    """Packed NVFP4 ``[E, N, K/2]`` uint8 -> Marlin ``[E, K/16, N*2]`` int32.

    ``sglang-v100-plus`` pads a narrow shard before repacking
    (``pad_nvfp4_expert_width``). That helps its eight-way expert sharding,
    where ``inter`` is 80 per rank: ``w13``'s ``N`` is 160 and ``w2``'s ``K``
    is 80, and the gate/up halves grow 80 -> 96. ``160 % 64 == 32``, so
    ``check_expert_shape`` rejects that profile; this component does not
    cover it.

    The four-card profile it does ship is ``moe_intermediate_size=640`` over
    ``TP=4``, i.e. ``inter=160`` per rank: ``w13`` is ``N=320 K=2560`` and
    ``w2`` is ``N=2560 K=160``. Both satisfy the gate, so no padding is needed.
    """
    experts, size_n, packed_k = (int(d) for d in weight.shape)
    size_k = packed_k * 2
    check_expert_shape(size_n, size_k)
    ops = ensure_loaded()
    perm = torch.empty(0, dtype=torch.int32, device=weight.device)
    out = torch.empty(
        (experts, size_k // 16, size_n * 2), dtype=torch.int32, device=weight.device
    )
    for expert in range(experts):
        qweight = weight[expert].view(torch.int32).T.contiguous()
        out[expert] = ops.repack(qweight, perm, size_k, size_n, 4, False)
    return out


def _ep_size() -> int | None:
    """The launch's expert-parallel width, or None when it is not readable.

    ``moe_ep_size`` is a ``ParallelContext`` field that is absent until the
    parallel context is published; before that the read fails with
    ``AttributeError`` and the check is skipped.
    """
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().moe_ep_size)
    except Exception:  # noqa: BLE001 - an unpublished context is not an error
        return None


def prepare_moe_nvfp4(layer, group_size: int | None = None) -> None:
    """Replace the checkpoint expert layout with the SM70 Marlin layout."""
    from sglang.srt.layers.utils import copy_or_rebind_param

    if layer.params_dtype != torch.float16:
        raise RuntimeError(
            "The SM70 NVFP4 profile requires FP16 activations, got "
            f"{layer.params_dtype}"
        )
    ep_size = _ep_size()
    if ep_size is not None and ep_size != 1:
        # The kernel indexes this rank's weights by global expert id, so an EP
        # launch would read the wrong expert. Checked here, during weight
        # loading, so the server never binds; the GEMM raises again as a
        # backstop.
        raise RuntimeError(
            "SM70 NVFP4 Marlin does not support expert parallelism; "
            f"launch with --ep-size 1 (got moe_ep_size={ep_size})"
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
    # The SM70 entry point ignores ``workspace`` (``(void)workspace`` in
    # marlin_v100's ``moe/marlin_moe_wna16/ops.cu``), so this is a
    # compatibility tensor, not a lock array. The size follows upstream
    # Marlin's own cap in ``fused_marlin_moe.py`` (``min(max_workspace_size,
    # sms * 4)``) so the attribute stays valid if a non-SM70 path reads it.
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

        # ModelOpt allocates a swizzled copy for the TRT-LLM backend. Marlin
        # reads only weight_scale, and that attribute now holds encoded Q0.7
        # bytes rather than blockscales, so aliasing it here would hand a
        # reader plausible garbage. Set it to None, as upstream does wherever
        # the backend does not read it (``modelopt_quant.py``), and let the
        # allocation go.
        swizzled = prefix + "_blockscale_swizzled"
        if hasattr(layer, swizzled):
            setattr(layer, swizzled, None)


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
    if is_ep:
        # ``is_ep`` is ``expert_map is not None``. The SM70 kernel indexes this
        # rank's weights by global expert id, so an EP launch would read the
        # wrong expert without either layer noticing. Fail at startup instead.
        raise RuntimeError(
            "SM70 NVFP4 Marlin does not support expert parallelism; "
            "launch with --ep-size 1"
        )
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


def _patch(module, name: str, replacement) -> bool:
    """Replace ``module.name``; False when the symbol is not there to replace.

    A missing symbol is not a no-op on this component: the pinned Marlin kernel
    is an empty body below SM80, so a hook that fails to bind leaves routed
    experts contributing zeros while the server keeps serving. The caller turns
    False into a startup failure.
    """
    original = getattr(module, name, None)
    if original is None:
        logger.warning(
            "SM70 NVFP4 MoE: %s.%s not found; the hook did not bind",
            module.__name__,
            name,
        )
        return False
    _ORIGINALS[f"{module.__name__}.{name}"] = original
    setattr(module, name, replacement)
    return True


def _install_sum_reduce() -> bool:
    """Supply ``moe_sum_reduce`` when the installed sgl_kernel lacks it.

    The function is imported into several modules by name, so replacing the
    attribute on ``sgl_kernel`` alone misses the bindings made at import time.
    Every already-imported module that still points at the original function is
    rewritten as well.

    The sweep only covers modules imported before this runs. Anything imported
    later binds whatever ``sgl_kernel`` exposes at that point, which is the
    replacement. ``torch.ops.sgl_kernel.moe_sum_reduce`` is deliberately left
    alone: the Marlin runner resolves the Python function, so patching that is
    enough, but a direct operator-namespace call would still raise.
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

    original = getattr(sgl_moe, "moe_sum_reduce", None)
    if original is None:
        # ``available()`` probes the operator namespace; this reads the Python
        # wrapper. A build with one but not the other is not ours to patch, so
        # leave it alone rather than raising out of install().
        return False

    replacement = sum_reduce.moe_sum_reduce
    sgl_kernel.moe_sum_reduce = replacement
    sgl_moe.moe_sum_reduce = replacement
    for module in list(sys.modules.values()):
        if module is None:
            continue
        # A module that is mid-import can raise anything from __getattr__, and
        # a raise here would leave the server with no moe_sum_reduce at all --
        # exactly the failure this function exists to prevent.
        try:
            bound = getattr(module, "moe_sum_reduce", None)
        except Exception:  # noqa: BLE001 - partial modules raise arbitrary errors
            continue
        if bound is original:
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
    if not available():
        # ``sglang.srt.plugins`` wraps plugin execution in
        # ``except Exception: logger.exception(...)`` at the pin, so a raise
        # here would only be logged and the engine would keep loading with the
        # empty SM80 kernel in the path. Install a hook that refuses at
        # weight-load time instead: ``modelopt_quant`` calls the prepare
        # function while loading, before the server binds.
        from sglang.srt.layers.quantization import marlin_utils_fp4, modelopt_quant

        def _refuse(*_args, **_kwargs):
            raise RuntimeError(
                f"{ENV_MODE}={mode} needs a loadable marlin_v100 build on SM70; "
                f"{os.environ.get(ENV_DIR, '<unset>')} does not hold one. "
                f"Set {ENV_DIR}, or set {ENV_MODE}=off to disable the layer."
            )

        _patch(marlin_utils_fp4, "prepare_moe_nvfp4_layer_for_marlin", _refuse)
        _patch(modelopt_quant, "prepare_moe_nvfp4_layer_for_marlin", _refuse)
        logger.error(
            "SM70 NVFP4 MoE: no usable marlin_v100 build in %s; weight loading "
            "will fail",
            os.environ.get(ENV_DIR, "<unset>"),
        )
        return False

    from . import sum_reduce

    if sum_reduce.available() and not _install_sum_reduce():
        # Not raised: ``register()`` runs under ``except Exception`` in
        # ``sglang.srt.plugins``, so a raise would only be logged. The runner
        # resolves the absent op at the first expert layer and fails there,
        # which is the loud outcome this needs.
        logger.error(
            "SM70 NVFP4 MoE: sgl_kernel exports no moe_sum_reduce and the "
            "replacement could not be bound; the Marlin runner will fail at "
            "the first expert layer"
        )

    from sglang.kernels.ops.moe import moe_wna16_marlin
    from sglang.srt.layers.quantization import marlin_utils_fp4, modelopt_quant

    required = {
        "marlin_utils_fp4.prepare_moe_nvfp4_layer_for_marlin": _patch(
            marlin_utils_fp4, "prepare_moe_nvfp4_layer_for_marlin", prepare_moe_nvfp4
        ),
        # modelopt_quant imported the name at module load.
        "modelopt_quant.prepare_moe_nvfp4_layer_for_marlin": _patch(
            modelopt_quant, "prepare_moe_nvfp4_layer_for_marlin", prepare_moe_nvfp4
        ),
        "moe_wna16_marlin.moe_wna16_marlin_gemm": _patch(
            moe_wna16_marlin, "moe_wna16_marlin_gemm", moe_wna16_marlin_gemm
        ),
    }
    missing = sorted(name for name, bound in required.items() if not bound)
    if missing:
        # Same reasoning as the missing-extension branch: ``register()`` runs
        # under ``except Exception`` in ``sglang.srt.plugins``, so raising here
        # would only be logged. Refuse at weight-load time on whichever
        # prepare hook still exists.
        names = ", ".join(missing)

        def _refuse_binding(*_args, **_kwargs):
            raise RuntimeError(
                f"SM70 NVFP4 MoE: hook(s) did not bind: {names}. The pin moved; "
                "update sglang_restomod/quant/sm70_fp4_moe/method.py."
            )

        for module in (marlin_utils_fp4, modelopt_quant):
            _patch(module, "prepare_moe_nvfp4_layer_for_marlin", _refuse_binding)
        logger.error("SM70 NVFP4 MoE: hook(s) did not bind: %s", names)
        return False
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
