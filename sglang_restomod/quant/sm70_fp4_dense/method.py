"""The bridge from the pin's dense NVFP4 path into ``marlin_v100``.

``marlin_utils_fp4`` imports three names into its own module namespace at load
time, and its dense functions look them up there:

1. ``gptq_marlin_gemm`` -- forwarded to ``_C.marlin_gemm``. The extension has a
   bias slot and an activation-scale slot the engine does not carry, so the call
   is rebuilt positionally by ``gemm_args``. ``b_q_type.id`` carries the
   ``float4_e2m1f`` scalar type that selects the NVFP4 kernel, and
   ``global_scale`` / ``b_scales`` are passed through untouched.
2. ``gptq_marlin_repack`` -- the SM70 extension packs its own tile layout, so
   the weight has to be packed by the build that reads it.
3. ``marlin_permute_scales`` -- identity. The SM70 NVFP4 iterator indexes the
   block scales by logical ``n``; the Ampere permutation returns numbers that
   look plausible and are wrong.

A fourth name is patched that is not a kernel binding, because the w4a4 method
leaves it unhandled: ``nvfp4_marlin_process_global_scale``. ``weight_scale_2``
is one entry per shard of a fused layer, so ``gate_up_proj`` reaches the kernel
with ``numel() == 2`` in fp32, while the op requires a half or bfloat16 scalar.
``ModelOptNvFp4A16LinearMethod`` collapses it with ``max()``; this repeats that
and warns. The result then goes back to float32: the pin's prepare step asserts
a 16-bit input, but the SM70 NVFP4 kernel reads a float32 scalar
(``marlin.cu:281``), and every value that survives the 16-bit round trip is
exactly representable again, so the widening is lossless.
"""

from __future__ import annotations

import logging

from ..sm70_marlin_v100 import gemm_args
from . import ENV_DIR, available, ops

logger = logging.getLogger(__name__)

_NVFP4_BITS = 4
_COLLAPSE_WARNED = False


def _patch(module, name: str, replacement) -> bool:
    """Replace ``module.name`` once; return whether it was replaced now."""
    original = getattr(module, name, None)
    if original is None:
        return False
    if getattr(original, "_restomod_sm70", False):
        return False
    replacement._restomod_sm70 = True  # type: ignore[attr-defined]
    setattr(module, name, replacement)
    return True


def _make_gemm(gemm):
    def gptq_marlin_gemm(
        a,
        c,
        b_q_weight,
        b_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type,
        size_m,
        size_n,
        size_k,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=False,
        is_zp_float=False,
    ):
        if size_m == 0:
            import torch

            if c is None:
                c = torch.empty((size_m, size_n), dtype=a.dtype, device=a.device)
            return c
        return gemm(
            *gemm_args(
                a,
                c,
                b_q_weight,
                b_scales,
                global_scale,
                b_zeros,
                g_idx,
                perm,
                workspace,
                b_q_type,
                size_m,
                size_n,
                size_k,
                is_k_full,
                use_atomic_add,
                use_fp32_reduce,
                is_zp_float,
            )
        )

    return gptq_marlin_gemm


def _make_repack(repack):
    def gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits=_NVFP4_BITS):
        # The extension packs one tile shape; a different width would pack a
        # layout the kernel cannot read, so it is refused rather than trusted.
        if num_bits != _NVFP4_BITS:
            raise ValueError(
                f"SM70 NVFP4 dense: marlin_v100 packs 4-bit tiles, got num_bits={num_bits}"
            )
        return repack(b_q_weight, perm, size_k, size_n, _NVFP4_BITS, False)

    return gptq_marlin_repack


def marlin_permute_scales(s, size_k, size_n, group_size):
    """Keep the block scales in their natural ``[K/group, N]`` order.

    The SM70 NVFP4 iterator reads them unpermuted; the Ampere permutation that
    upstream applies here reorders an array this kernel indexes by logical
    ``n``.
    """
    return s.reshape((-1, size_n)).contiguous()


def _make_global_scale(original):
    def nvfp4_marlin_process_global_scale(global_scale):
        global _COLLAPSE_WARNED

        import torch

        if global_scale.numel() != 1:
            # A fused layer calibrated gate and up separately. One alpha for
            # both halves degrades whichever was not the largest, which is the
            # same approximation ModelOptNvFp4A16LinearMethod makes.
            if not _COLLAPSE_WARNED:
                logger.warning(
                    "SM70 NVFP4 dense: collapsing %d per-shard global scales with max()",
                    global_scale.numel(),
                )
                _COLLAPSE_WARNED = True
            global_scale = global_scale.max()
        if global_scale.dtype not in (torch.float16, torch.bfloat16):
            global_scale = global_scale.to(torch.float16)
        processed = original(global_scale)
        # The pin prepares a 16-bit scalar; the SM70 NVFP4 kernel reads float32
        # (marlin.cu:281) and rejects anything else. Widening is exact.
        if processed.dtype != torch.float32:
            processed = processed.to(torch.float32)
        return processed

    return nvfp4_marlin_process_global_scale


def _refuse_on_load() -> bool:
    """Fail at weight-load time when the extension is not usable.

    ``sglang.srt.plugins`` wraps component ``register()`` in
    ``except Exception``, so raising here would be logged and the engine would
    keep loading with the pin's own kernel in the path -- which is a stub below
    SM80, so the GEMM would return zeros. ``prepare_nvfp4_layer_for_marlin``
    runs while weights load, well before the server binds, so the refusal sits
    there instead.
    """
    try:
        from sglang.srt.layers.quantization import marlin_utils_fp4
    except Exception as exc:  # noqa: BLE001 - absent module means nothing to patch
        logger.warning("SM70 NVFP4 dense: marlin_utils_fp4 is not importable (%s)", exc)
        return False

    def _refuse(*_args, **_kwargs):
        raise RuntimeError(
            "SM70 NVFP4 dense needs a loadable marlin_v100 build on SM70, and "
            f"{ENV_DIR} does not hold one. Set {ENV_DIR} to the directory with "
            "_C*.so, or remove fp4_dense from SGLANG_RESTOMOD_COMPONENTS."
        )

    _patch(marlin_utils_fp4, "prepare_nvfp4_layer_for_marlin", _refuse)
    logger.error(
        "SM70 NVFP4 dense: no usable marlin_v100 build in %s; weight loading will fail",
        ENV_DIR,
    )
    return False


def install() -> bool:
    """Patch every name the dense NVFP4 path resolves; idempotent."""
    if not available():
        return _refuse_on_load()

    repack, gemm = ops()

    try:
        from sglang.srt.layers.quantization import marlin_utils_fp4
    except Exception as exc:  # noqa: BLE001 - absent module means nothing to patch
        logger.warning("SM70 NVFP4 dense: marlin_utils_fp4 is not importable (%s)", exc)
        return False

    bridges = {
        "gptq_marlin_gemm": _make_gemm(gemm),
        "gptq_marlin_repack": _make_repack(repack),
        "marlin_permute_scales": marlin_permute_scales,
    }
    missing = sorted(name for name in bridges if getattr(marlin_utils_fp4, name, None) is None)
    if missing:
        logger.error(
            "SM70 NVFP4 dense: marlin_utils_fp4 has no %s; the pin moved and this "
            "component needs updating",
            ", ".join(missing),
        )
        return False

    changed = False
    for name, replacement in bridges.items():
        if _patch(marlin_utils_fp4, name, replacement):
            changed = True
            logger.info("SM70 NVFP4 dense: marlin_utils_fp4.%s routed to marlin_v100", name)

    original_global_scale = getattr(marlin_utils_fp4, "nvfp4_marlin_process_global_scale", None)
    if original_global_scale is None:
        logger.warning(
            "SM70 NVFP4 dense: marlin_utils_fp4 has no nvfp4_marlin_process_global_scale; "
            "a fused layer will reach the kernel with a per-shard scale"
        )
    elif _patch(
        marlin_utils_fp4,
        "nvfp4_marlin_process_global_scale",
        _make_global_scale(original_global_scale),
    ):
        changed = True
        logger.info("SM70 NVFP4 dense: per-shard global scales collapsed for the SM70 kernel")

    return changed
