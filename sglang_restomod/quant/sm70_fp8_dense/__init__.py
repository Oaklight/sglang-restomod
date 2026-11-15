"""SM70 dense FP8 (W8A16) component.

The pinned upstream routes every FP8 linear through kernels that need E4M3
support. Volta has no E4M3 float and Triton refuses to emit it
(``type fp8e4nv not supported in this architecture``), so the dense path cannot
run at the pin without this component.

``_method.py`` is the fork's TurboMind/kernel module, copied apart from the
``sglang.kernels.sm70_paths`` import, which does not exist upstream.

The three hooks below mirror the fork's ``fp8.py`` (init at 542, prepare at
1249, apply at 1285). They are runtime hooks because the pinned engine is
installed as a wheel.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def make_block_fp8_method(weight_block_size: list[int]):
    """Build a block-FP8 method through its constructor.

    A method built with ``__new__`` lacks every attribute the constructor
    sets; the pin then reads one (v0.5.23: ``block_fp8_as_mxfp8``) and fails
    at model init. Construct it properly, then opt into the SM70 kernel.
    """
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=list(weight_block_size),
    )
    method = Fp8LinearMethod(config)
    method.use_sm70_turbomind = True
    return method


def install() -> bool:
    """Route SM70 block-FP8 linears to the TurboMind W8A16 kernels."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    from . import _method as kernels

    if getattr(Fp8LinearMethod.apply, "_restomod_sm70", False):
        return False

    orig_init = Fp8LinearMethod.__init__
    orig_prepare = Fp8LinearMethod.process_weights_after_loading
    orig_apply = Fp8LinearMethod.apply

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.use_sm70_turbomind = False
        try:
            self.use_sm70_turbomind = kernels.can_use_sm70_turbomind_fp8(
                getattr(self.quant_config, "weight_block_size", None)
            )
        except Exception as exc:  # noqa: BLE001 - stay on the upstream path
            logger.warning("sm70_fp8_dense: disabled (%s)", exc)
        if self.use_sm70_turbomind:
            logger.info("sm70_fp8_dense: block-FP8 linears use the TurboMind kernel")

    def process_weights_after_loading(self, layer):
        if getattr(self, "use_sm70_turbomind", False):
            layer.weight_block_size = self.quant_config.weight_block_size
            kernels.prepare_sm70_turbomind_fp8_linear(layer)
            if hasattr(layer, "input_scale"):
                del layer.input_scale
            return None
        return orig_prepare(self, layer)

    def apply(self, *args, **kwargs):
        if getattr(self, "use_sm70_turbomind", False):
            layer, x = args[0], args[1]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            if isinstance(x, tuple):
                x = x[0]
            if x.dtype != torch.float16:
                # A bf16 draft calls the shared lm_head; the TurboMind kernel
                # is fp16 only.
                x = x.to(torch.float16)
            return kernels.apply_sm70_turbomind_fp8_linear(layer, x, bias)
        return orig_apply(self, *args, **kwargs)

    apply._restomod_sm70 = True
    Fp8LinearMethod.__init__ = __init__
    Fp8LinearMethod.process_weights_after_loading = process_weights_after_loading
    Fp8LinearMethod.apply = apply
    return True


def register() -> None:
    """Component entry point."""
    install()
