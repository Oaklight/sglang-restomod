"""Put the speculative draft's experts on the SM70 NVFP4 Marlin GEMM.

Why the draft never reaches our kernel, in two engine facts:

* ``modelopt_quant.py`` returns ``None`` from the quant-method lookup for a
  ``FusedMoE`` whose prefix the checkpoint excludes, and this checkpoint's
  ``quantization_config.ignore`` lists ``mtp.*``. The layer therefore gets
  ``UnquantizedFusedMoEMethod``.
* ``unquant.py`` builds that method's runner as
  ``MoeRunner(MoeRunnerBackend.TRITON, ...)`` unconditionally, whatever
  ``--moe-runner-backend`` says. The draft is on Triton by construction, not by
  a launch mistake.

This component closes the gap from the weight side: after load, it quantizes
the draft layer's FP16 experts to NVFP4 (see :mod:`.quantize`), gives the layer
the tensors an NVFP4 checkpoint would have supplied, and hands it to the same
``ModelOptNvFp4FusedMoEMethod`` the target uses. ``fp4_moe`` then repacks and
runs them through the Marlin kernel it already binds, so no kernel code is
added here.

Gating: a launch selects this component by name, and nothing else changes.
Without the component, an unquantized draft layer keeps its Triton path. The
component also refuses unless ``fp4_moe`` is installed, because the pinned
Marlin kernel is an empty-body stub below SM80: converting without the
replacement would zero the draft's experts rather than speed them up.

Selection is by observed state, not by prefix. Inside a model whose class is
the MTP class, every ``FusedMoE`` whose method is ``UnquantizedFusedMoEMethod``
is a candidate; the target's layers are NVFP4 already and never match.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Names of the four tensors a converted layer owns.
_W13 = ("w13_weight", "w13_weight_scale", "w13_weight_scale_2")
_W2 = ("w2_weight", "w2_weight_scale", "w2_weight_scale_2")


def _is_draft_model(model) -> bool:
    """True for the MTP draft module; the target model is a different class."""
    return "MTP" in type(model).__name__


def _admits(module, fused_moe_type, unquantized_type) -> bool:
    """Whether ``module`` is an unquantized expert layer this component may take."""
    return isinstance(module, fused_moe_type) and isinstance(
        getattr(module, "quant_method", None), unquantized_type
    )


def candidate_layers(model):
    """Yield ``(name, layer)`` for the draft's unquantized expert layers."""
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    for name, module in model.named_modules():
        if _admits(module, FusedMoE, UnquantizedFusedMoEMethod):
            yield name, module


def convert_layer(layer) -> None:
    """Quantize one draft expert layer in place and point it at the NVFP4 method.

    Raises before mutating the layer when any gate refuses, and restores the
    original tensors and method if the method swap fails after mutation.
    """
    import torch
    from sglang.srt.layers.moe.utils import get_moe_runner_backend
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    from .quantize import quantize_experts, refusal

    backend = get_moe_runner_backend()
    if not backend.is_marlin():
        raise RuntimeError(
            f"the draft is only converted under --moe-runner-backend marlin; got {backend.value}"
        )
    quant_config = getattr(layer, "quant_config", None)
    nvfp4_config = getattr(quant_config, "nvfp4_config", None)
    if nvfp4_config is None:
        raise RuntimeError(f"layer carries no NVFP4 sub-config ({type(quant_config).__name__})")

    reason = refusal(layer.w13_weight) or refusal(layer.w2_weight)
    if reason is not None:
        raise RuntimeError(reason)

    converted = {
        "w13": quantize_experts(layer.w13_weight),
        "w2": quantize_experts(layer.w2_weight),
    }

    touched = _W13 + _W2 + ("_mega_moe_nvfp4",)
    saved = {name: getattr(layer, name) for name in touched if hasattr(layer, name)}
    created = [name for name in touched if not hasattr(layer, name)]
    saved_method = layer.quant_method
    saved_runner = getattr(layer, "runner", None)
    try:
        for prefix, bank in converted.items():
            setattr(
                layer,
                f"{prefix}_weight",
                torch.nn.Parameter(bank.weight, requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}_weight_scale",
                torch.nn.Parameter(bank.scales, requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}_weight_scale_2",
                torch.nn.Parameter(bank.global_scale, requires_grad=False),
            )
        # Read directly (not via getattr) by ModelOptNvFp4FusedMoEMethod.apply.
        layer._mega_moe_nvfp4 = False

        method = ModelOptNvFp4FusedMoEMethod(nvfp4_config)
        # Same call FusedMoE.__init__ makes, so the layer's own contract is
        # unchanged; it sets the runner the layer will dispatch through.
        method.create_moe_runner(layer, layer.moe_runner_config)
        layer.quant_method = method
        layer.runner = method.runner
        # The marlin branch collapses the w13 scale pair and calls the
        # ``prepare_moe_nvfp4_layer_for_marlin`` that ``fp4_moe`` replaced.
        method.process_weights_after_loading(layer)
    except Exception:
        for name in created:
            if hasattr(layer, name):
                delattr(layer, name)
        for name, value in saved.items():
            setattr(layer, name, value)
        layer.quant_method = saved_method
        if saved_runner is not None:
            layer.runner = saved_runner
        raise

    logger.info(
        "sm70_mtp_fp4_moe: experts -> NVFP4 Marlin, w13 %s w2 %s, %d experts",
        tuple(layer.w13_weight.shape),
        tuple(layer.w2_weight.shape),
        layer.w13_weight.shape[0],
    )


def convert_model(model) -> tuple[int, int]:
    """Convert every candidate in ``model``; return ``(converted, refused)``."""
    converted = refused = 0
    for name, layer in candidate_layers(model):
        try:
            convert_layer(layer)
            converted += 1
        except Exception as exc:  # noqa: BLE001 - one layer must not stop the rest
            refused += 1
            logger.warning("sm70_mtp_fp4_moe: %s stays on its original path (%s)", name, exc)
    return converted, refused


def _fp4_moe_installed() -> bool:
    """Whether the SM70 Marlin prepare/GEMM hooks are ours."""
    from sglang.srt.layers.quantization import marlin_utils_fp4

    owner = getattr(marlin_utils_fp4.prepare_moe_nvfp4_layer_for_marlin, "__module__", "")
    return owner.startswith("sglang_restomod")


def install() -> bool:
    """Convert the draft's experts once, after the model weights are loaded."""
    from sglang.srt.model_executor.model_runner import ModelRunner

    if not _fp4_moe_installed():
        logger.warning(
            "sm70_mtp_fp4_moe: fp4_moe is not installed, skipped; the draft keeps "
            "its Triton expert path"
        )
        return False

    original = ModelRunner.load_model
    if getattr(original, "_restomod_mtp_fp4_moe", False):
        return False

    def load_model(self):
        result = original(self)
        try:
            if not _is_draft_model(self.model):
                return result
            converted, refused = convert_model(self.model)
            if converted or refused:
                logger.info(
                    "sm70_mtp_fp4_moe: %d draft expert layers on NVFP4 Marlin, %d refused",
                    converted,
                    refused,
                )
        except Exception as exc:  # noqa: BLE001 - a serving draft beats a dead server
            logger.warning("sm70_mtp_fp4_moe: skipped (%s)", exc)
        return result

    load_model._restomod_mtp_fp4_moe = True
    ModelRunner.load_model = load_model
    return True


def register() -> None:
    """Component entry point."""
    install()
