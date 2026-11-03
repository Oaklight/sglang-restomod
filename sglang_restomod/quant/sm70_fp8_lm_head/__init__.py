"""SM70 block-FP8 LM head.

The 27B checkpoint keeps ``lm_head`` in ``ignored_layers``, so the head stays
f16 and every draft step reads 1.27 GB of weights per card. The head is a
plain [vocab, hidden] matrix that the block-FP8 path already covers, so this
component quantizes it once after the model loads and hands it to the
``fp8_dense`` method, which routes it to the TurboMind FP8 GEMM.

Measured on one V100, M=6, the TP2 shard shape: f16 linear 2.001 ms against
fp8 0.793 ms. Five head calls a step. In the engine, against the same launch
without this component: 8k decode 63.08 -> 70.37 tok/s (three fresh reps,
512 output tokens), and 32k unchanged within the no-head spread.

The component needs ``fp8_dense``: it reuses that component's patched
``process_weights_after_loading`` and ``apply``. Without it the head would be
quantized into a method that has no kernel for the prepared layout, so the
install refuses instead.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

BLOCK = 128


def _quantize(weight: torch.Tensor):
    """Return (fp8 weight [N, K], scale_inv [N/128, K/128]) for a block-FP8 head."""
    n, k = weight.shape
    if n % BLOCK or k % BLOCK:
        raise ValueError(f"lm_head shape {tuple(weight.shape)} is not a multiple of {BLOCK}")
    wb = weight.view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = (wb.abs().amax(dim=(1, 3)) / 448.0).clamp(min=1e-8)
    q = (wb / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n, k)
    return q.contiguous(), scale.to(torch.float32).contiguous()


def quantize_lm_head(model) -> bool:
    """Quantize ``model.lm_head`` in place; return True when it was converted."""
    head = getattr(model, "lm_head", None)
    if head is None or not hasattr(head, "weight"):
        return False
    weight = head.weight.data
    if weight.dim() != 2 or weight.dtype not in (torch.float16, torch.bfloat16):
        return False
    if getattr(head, "weight_block_size", None) == [BLOCK, BLOCK]:
        return False

    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    q, scale = _quantize(weight.to(torch.float16))

    class _Config:
        weight_block_size = [BLOCK, BLOCK]

    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.quant_config = _Config()
    method.use_sm70_turbomind = True  # install() checked fp8_dense and the kernel

    head.weight = torch.nn.Parameter(q, requires_grad=False)
    head.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
    head.weight_block_size = [BLOCK, BLOCK]
    head.orig_dtype = torch.float16
    # ParallelLMHead is a weight holder, not a Linear; the fp8 apply path reads
    # these two sizes.
    head.output_size_per_partition = q.shape[0]
    head.input_size_per_partition = q.shape[1]
    head.quant_method = method
    method.process_weights_after_loading(head)
    logger.info(
        "sm70_fp8_lm_head: %s quantized to block FP8 (%d x %d)",
        getattr(head, "prefix", "lm_head"), q.shape[0], q.shape[1],
    )
    return True


def install() -> bool:
    """Quantize the LM head once, after the model weights are loaded."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.model_executor.model_runner import ModelRunner

    from sglang_restomod.quant.sm70_fp8_dense import _method as kernels

    if not getattr(Fp8LinearMethod.apply, "_restomod_sm70", False):
        logger.warning("sm70_fp8_lm_head: fp8_dense is not installed, skipped")
        return False
    if not kernels.can_use_sm70_turbomind_fp8([BLOCK, BLOCK]):
        logger.warning("sm70_fp8_lm_head: the TurboMind FP8 kernel is unavailable, skipped")
        return False

    original = ModelRunner.load_model
    if getattr(original, "_restomod_fp8_lm_head", False):
        return False

    def load_model(self):
        result = original(self)
        try:
            quantize_lm_head(self.model)
        except Exception as exc:  # noqa: BLE001 - a slow head beats a dead server
            logger.warning("sm70_fp8_lm_head: skipped (%s)", exc)
        return result

    load_model._restomod_fp8_lm_head = True
    ModelRunner.load_model = load_model
    return True


def register() -> None:
    """Component entry point."""
    install()
