"""Let an FP8 checkpoint with FP4 experts use the SM70 Marlin expert GEMM.

DeepSeek-V4's FP8 checkpoints store expert weights as E2M1 packed two per byte
with one E8M0 scale per 32 values. The engine already has a method for that
shape -- ``Mxfp4MarlinMoEMethod`` allocates exactly this layout and stores the
scales as E8M0 -- but its prepare and runner are gated to SM90/SM120, and the
sgl-kernel it calls has no SM70 build.

Our ``marlin_v100`` kernel reads the same E2M1 nibbles with E4M3 scales per 16
and a per-expert global. The weights need no conversion at all; only the scale
grid changes, and ``expand_mxfp4_scales`` does that exactly for this checkpoint
(measured: no block clamps across 8.8M slots). This hook therefore replaces one
method's ``process_weights_after_loading`` and leaves ``apply`` alone: the
upstream ``apply`` builds its quant info from the layer's prepared tensors and
calls ``fused_marlin_moe``, which this component already points at our kernel.

Nothing here is a fallback: if the shapes or dtypes are not the ones this
conversion produces, the hook raises instead of computing something else.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_INSTALLED = False
_ORIGINAL = None


def _expand(layer, prefix: str, device) -> int:
    """Rewrite ``<prefix>_weight_scale`` in place; return the clamp count."""
    from .mxfp4_scales import expand_mxfp4_scales

    scale = getattr(layer, prefix + "_weight_scale_inv", None)
    if scale is None:
        scale = getattr(layer, prefix + "_weight_scale", None)
    if scale is None:
        raise RuntimeError(
            f"SM70 MXFP4 MoE: {prefix} has no weight scale; the checkpoint is "
            "not FP4 experts and this hook should not have been selected"
        )
    e4, global_scale, clamped = expand_mxfp4_scales(scale)
    setattr(layer, prefix + "_weight_scale", e4)
    setattr(layer, prefix + "_weight_scale_2", global_scale.to(device))
    return clamped


def _process_weights_after_loading(self, layer) -> None:
    """fp8 normalization, then the scale grid, then our Marlin prepare."""
    from .method import prepare_moe_nvfp4

    self._fp8.process_weights_after_loading(layer)
    if getattr(layer, "_mega_moe_weights_built", False):
        return

    device = layer.w13_weight.device
    clamped = _expand(layer, "w13", device) + _expand(layer, "w2", device)
    if clamped:
        # A block below E4M3's 2**-9 floor lost its scale. Say so rather than
        # computing a silently different model.
        raise RuntimeError(
            f"SM70 MXFP4 MoE: {clamped} scale blocks of {self.prefix} fall "
            "outside E4M3's exact range; this checkpoint needs a wider scale "
            "container than per-16 E4M3 on SM70"
        )

    prepare_moe_nvfp4(layer)
    layer._dsv4_mxfp4_backend = "marlin"
    logger.info(
        "SM70 MXFP4 MoE: %s prepared for marlin_v100 (E8M0 per 32 -> E4M3 per 16)",
        self.prefix,
    )


def install() -> bool:
    """Point the MXFP4 Marlin method's prepare at our kernel. Idempotent."""
    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return True
    try:
        from sglang.srt.layers.quantization import mxfp4_marlin_moe as upstream
    except ImportError:
        logger.warning(
            "SM70 MXFP4 MoE: the pin has no mxfp4_marlin_moe module; "
            "FP8 checkpoints with FP4 experts stay on the upstream SM90 path"
        )
        return False
    method = getattr(upstream, "Mxfp4MarlinMoEMethod", None)
    if method is None:
        logger.warning(
            "SM70 MXFP4 MoE: Mxfp4MarlinMoEMethod is absent from the pin; "
            "FP8 checkpoints with FP4 experts stay on the upstream path"
        )
        return False
    _ORIGINAL = method.process_weights_after_loading
    method.process_weights_after_loading = _process_weights_after_loading
    _INSTALLED = True
    return True
