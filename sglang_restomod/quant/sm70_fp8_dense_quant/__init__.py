"""Quantize a BF16 checkpoint's dense linears to block FP8 after load.

A BF16 model on SM70 reads two bytes per dense weight on every decode step,
and those GEMMs are bandwidth-bound at M=1..4, so the byte count is the cost.
The block-FP8 path in ``fp8_dense`` already serves the 27B; this component
extends it to checkpoints that ship dense weights in BF16.

Measured on one V100, K=2560, random weights, block 128 (issue #95):

| M | N | fp16 linear | TurboMind FP8 | speedup |
|---:|---:|---:|---:|---:|
| 1 | 3072 | 25.9 us | 19.6 us | 1.32 |
| 4 | 3072 | 30.2 us | 19.4 us | 1.56 |
| 4 | 1280 | 23.9 us | 15.8 us | 1.51 |
| 1 | 62080 | 476.9 us | 198.8 us | 2.40 |

In the engine on Flash-Next NVFP4 (4x V100, TP4): the four 473 us LM-head
launches per step become one 0.84 ms TurboMind GEMM, 32.79 -> 31.54 ms/step.

Admission is deliberately narrow: a layer is converted only when its weight is
2-D BF16/FP16, both sides are a multiple of the 128 block, the layer is not an
expert and not the LM head, and its quantization method is the engine's
unquantized one. Everything else keeps its original path.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

BLOCK = 128
SKIP_PREFIX = ("experts",)
SKIP_SUFFIX = ("lm_head",)


def _admitted(name: str, module) -> bool:
    """Return whether this module is a dense linear this component may convert."""
    if not name:
        return False
    if any(part in name for part in SKIP_PREFIX):
        return False
    if any(name.endswith(suffix) for suffix in SKIP_SUFFIX):
        return False
    weight = getattr(module, "weight", None)
    if weight is None or weight.dim() != 2:
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16):
        return False
    if getattr(module, "weight_block_size", None) == [BLOCK, BLOCK]:
        return False
    if not (_is_linear_base(module) or isinstance(module, torch.nn.Linear)):
        return False
    if _is_linear_base(module) and "Unquantized" not in type(
        module.quant_method
    ).__name__:
        return False
    n, k = weight.shape
    return n % BLOCK == 0 and k % BLOCK == 0


def _is_linear_base(module) -> bool:
    """LinearBase exposes these two; a plain ``nn.Linear`` has neither."""
    return hasattr(module, "quant_method") and hasattr(
        module, "output_size_per_partition"
    )


def _plain_linear_forward(self, x):
    """Forward for a converted ``nn.Linear``: route through the FP8 method."""
    return self.quant_method.apply(self, x, self.bias)


def convert(module) -> bool:
    """Quantize one admitted layer in place and point it at the FP8 method."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    from sglang_restomod.quant.sm70_fp8_lm_head import _quantize

    plain = not _is_linear_base(module)
    q, scale = _quantize(module.weight.data.to(torch.float16))

    class _Config:
        weight_block_size = [BLOCK, BLOCK]

    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.quant_config = _Config()
    method.use_sm70_turbomind = True

    module.weight = torch.nn.Parameter(q, requires_grad=False)
    module.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
    module.weight_block_size = [BLOCK, BLOCK]
    module.orig_dtype = torch.float16
    module.output_size_per_partition = q.shape[0]
    module.input_size_per_partition = q.shape[1]
    module.quant_method = method
    method.process_weights_after_loading(module)
    if plain:
        module.forward = _plain_linear_forward.__get__(module, type(module))
    return True


def quantize_dense_linears(model) -> tuple[int, int]:
    """Convert every admitted layer; return (converted, refused)."""
    converted = refused = 0
    for name, module in model.named_modules():
        if not _admitted(name, module):
            continue
        try:
            convert(module)
            converted += 1
        except Exception as exc:  # noqa: BLE001 - one bad layer must not stop the rest
            refused += 1
            logger.warning(
                "sm70_fp8_dense_quant: %s stays on its original path (%s)", name, exc
            )
    return converted, refused


def weight_holder_types(model) -> dict[str, int]:
    """Count the module types that carry a 2-D float weight.

    Logged once so a launch says which layer classes the admission saw, instead
    of only how many it converted.
    """
    counts: dict[str, int] = {}
    for _name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        if weight.dtype not in (torch.float16, torch.bfloat16):
            continue
        key = type(module).__name__
        counts[key] = counts.get(key, 0) + 1
    return counts


def install() -> bool:
    """Quantize the dense linears once, after the model weights are loaded."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.model_executor.model_runner import ModelRunner

    from sglang_restomod.quant.sm70_fp8_dense import _method as kernels

    if not getattr(Fp8LinearMethod.apply, "_restomod_sm70", False):
        logger.warning("sm70_fp8_dense_quant: fp8_dense is not installed, skipped")
        return False
    if not kernels.can_use_sm70_turbomind_fp8([BLOCK, BLOCK]):
        logger.warning("sm70_fp8_dense_quant: the TurboMind FP8 kernel is unavailable")
        return False

    original = ModelRunner.load_model
    if getattr(original, "_restomod_fp8_dense_quant", False):
        return False

    def load_model(self):
        result = original(self)
        try:
            converted, refused = quantize_dense_linears(self.model)
            logger.info(
                "sm70_fp8_dense_quant: %d dense linears quantized to block FP8, "
                "%d refused; weight holders by type: %s",
                converted,
                refused,
                weight_holder_types(self.model),
            )
        except Exception as exc:  # noqa: BLE001 - a BF16 server beats a dead one
            logger.warning("sm70_fp8_dense_quant: skipped (%s)", exc)
        return result

    load_model._restomod_fp8_dense_quant = True
    ModelRunner.load_model = load_model
    return True


def register() -> None:
    """Component entry point."""
    install()
