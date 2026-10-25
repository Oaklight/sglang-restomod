"""SM70 block-FP8 MoE method.

Replaces ``Fp8MoEMethod`` for SM70 expert layers. The checkpoint's block-FP8
weights are prepared for the TurboMind expert GEMM (validated numerically in
``selftest.py``) and ``apply`` gathers each token's routed slots into
expert-sorted order, runs the FP8 expert GEMMs, and scatters the result back.

Under expert parallelism the dispatcher has already mapped global expert ids to
this rank's local ids and marks routes owned by another rank with -1
(``StandardDispatcher``). The TurboMind permute helper does not honor that
sentinel - it buckets negative ids into the last expert - so the routing slots
are compacted here before the GEMM, and the same mask drops their weights.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from torch.nn.parameter import Parameter

from . import apply as fp8_apply
from . import available, ensure_loaded, prepare_experts

logger = logging.getLogger(__name__)

ENV_MODE = "SGLANG_SM70_FP8_MOE"
_SO_NAME = "_sm70_turbomind_v100.so"

_LOGGED_CONFIGS: set[tuple[int, int, int, int]] = set()


def _mode() -> str:
    return os.environ.get(ENV_MODE, "auto").strip().lower()


def _is_sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def _library_present() -> bool:
    """Cheap check: does not import or compile the shim."""
    try:
        from sglang.kernels.sm70_paths import sm70_prebuilt

        return Path(sm70_prebuilt(_SO_NAME)).is_file()
    except Exception:  # noqa: BLE001 - a missing path is not an error here
        return False


class SM70Fp8MoEMethod(Fp8MoEMethod):
    """Block-FP8 experts through the TurboMind W8A16 GEMM."""

    def __init__(self, quant_config):
        if not _is_sm70():
            raise RuntimeError("SM70Fp8MoEMethod requires compute capability 7.0")
        super().__init__(quant_config)
        self.use_turbomind = False
        self.output_scale: float | None = None

    def create_moe_runner(self, layer, moe_runner_config):
        # The Triton runner stays as the fallback for unsupported configs.
        super().create_moe_runner(layer, moe_runner_config)
        self.use_turbomind = False

    @staticmethod
    def _unsupported_reason(cfg) -> str | None:
        # The kernel's gated epilogue is a plain silu(gate) * up; the scorer
        # below applies the router weights.
        if cfg.activation != "silu" or not cfg.is_gated:
            return f"activation={cfg.activation} is_gated={cfg.is_gated}"
        if cfg.swiglu_limit is not None or cfg.gemm1_clamp_limit is not None:
            return "clamped SwiGLU"
        if cfg.gemm1_alpha is not None or cfg.gemm1_beta is not None:
            return "SwiGLU alpha/beta"
        if cfg.apply_router_weight_on_input or cfg.no_combine:
            return "router weight on input or no combine"
        if cfg.wide_output_scale != 1.0 or cfg.gate_up_input_scale != 1.0:
            return "FP8 range scaling"
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        cfg = self.moe_runner_config
        reason = self._unsupported_reason(cfg)
        if reason is not None:
            # SM70 has no Triton FP8 expert kernel to fall back to.
            raise RuntimeError(f"SM70 FP8 MoE does not support {reason}")
        try:
            ensure_loaded()
        except Exception as exc:  # noqa: BLE001 - report the reason to the user
            raise RuntimeError(
                "SM70 FP8 MoE was selected but the TurboMind binding cannot be "
                f"loaded: {exc}. Set {ENV_MODE}=off to use the Triton runner."
            ) from exc

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        s13 = layer.w13_weight_scale_inv
        s2 = layer.w2_weight_scale_inv
        if w13 is None or w2 is None or s13 is None or s2 is None:
            raise RuntimeError("FP8 MoE weights or scales are missing")

        debug = bool(os.environ.get("SGLANG_RESTOMOD_DEBUG"))
        if debug:
            _, free = torch.cuda.mem_get_info()
            logger.info(
                "SM70 FP8 MoE: preparing experts, gpu allocated %.2f GiB, free %.2f GiB",
                torch.cuda.memory_allocated() / 2**30,
                free / 2**30,
            )

        num_experts, w13_n, hidden_size = (int(d) for d in w13.shape)
        intermediate_size = w13_n // 2
        if w13_n % 256 or hidden_size % 128 or intermediate_size % 128:
            raise RuntimeError(
                "block-128 FP8 MoE needs hidden and per-expert intermediate "
                f"multiples of 128, got {hidden_size}/{intermediate_size}"
            )

        e13 = prepare_experts(w13, s13.float(), interleave_gated_silu=True)
        e2 = prepare_experts(w2, s2.float(), interleave_gated_silu=False)


        layer.w13_tm_weight = Parameter(e13.tm_weight, requires_grad=False)
        layer.w13_tm_scales = Parameter(e13.tm_scales, requires_grad=False)
        layer.w2_tm_weight = Parameter(e2.tm_weight, requires_grad=False)
        layer.w2_tm_scales = Parameter(e2.tm_scales, requires_grad=False)
        layer.w13_weight_ptrs = Parameter(e13.ptr_w, requires_grad=False)
        layer.w13_scale_ptrs = Parameter(e13.ptr_s, requires_grad=False)
        layer.w2_weight_ptrs = Parameter(e2.ptr_w, requires_grad=False)
        layer.w2_scale_ptrs = Parameter(e2.ptr_s, requires_grad=False)
        layer.w13_quant = e13
        layer.w2_quant = e2

        # The loader keeps a snapshot dict of every parameter for the whole
        # post-load pass, so dropping the attribute alone would leave the
        # block-FP8 source resident and double the expert footprint. Release
        # the storage through the Parameter objects themselves.
        for param in (w13, w2, s13, s2):
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        layer.w13_weight = None
        layer.w2_weight = None
        layer.w13_weight_scale_inv = None
        layer.w2_weight_scale_inv = None
        del w13, w2, s13, s2
        if debug:
            logger.info(
                "SM70 FP8 MoE: layer converted, gpu allocated %.2f GiB",
                torch.cuda.memory_allocated() / 2**30,
            )

        scale = cfg.routed_scaling_factor
        self.output_scale = (
            None
            if scale in (None, 1.0) or layer.should_fuse_routed_scaling_factor_in_topk
            else float(scale)
        )
        layer.sm70_num_experts = num_experts
        layer.sm70_hidden_size = hidden_size
        layer.sm70_intermediate_size = intermediate_size
        layer.sm70_top_k = cfg.top_k
        self._allocate_buffers(layer, 32)
        self.use_turbomind = True

        config = (num_experts, hidden_size, intermediate_size, cfg.top_k)
        if config not in _LOGGED_CONFIGS:
            _LOGGED_CONFIGS.add(config)
            logger.info(
                "SM70 FP8 MoE: using TurboMind block-FP8 expert GEMM "
                "(experts=%d hidden=%d intermediate=%d topk=%d)",
                *config,
            )

    @staticmethod
    def _allocate_buffers(layer: torch.nn.Module, max_tokens: int) -> None:
        e, h, i, k = (
            layer.sm70_num_experts,
            layer.sm70_hidden_size,
            layer.sm70_intermediate_size,
            layer.sm70_top_k,
        )
        slots = max_tokens * k
        device = layer.w13_tm_weight.device
        layer.sm70_max_tokens = max_tokens
        layer.sm70_output = torch.empty(max_tokens, h, dtype=torch.float16, device=device)
        layer.sm70_permuted = torch.empty(slots, h, dtype=torch.float16, device=device)
        layer.sm70_gate_up = torch.empty(slots, i, dtype=torch.float16, device=device)
        layer.sm70_sorted_output = torch.empty(slots, h, dtype=torch.float16, device=device)
        layer.sm70_offsets = torch.empty(e + 1, dtype=torch.int32, device=device)

    def _workspace(self, layer, tokens: int):
        k = layer.sm70_top_k
        slots = tokens * k
        if tokens <= layer.sm70_max_tokens:
            return (
                layer.sm70_output[:tokens],
                layer.sm70_permuted[:slots],
                layer.sm70_gate_up[:slots],
                layer.sm70_sorted_output[:slots],
                layer.sm70_offsets,
            )
        # Prefill workspaces are transient; retaining them per layer exhausts VRAM.
        dev = layer.w13_tm_weight.device
        h, i, e = layer.sm70_hidden_size, layer.sm70_intermediate_size, layer.sm70_num_experts
        return (
            torch.empty(tokens, h, dtype=torch.float16, device=dev),
            torch.empty(slots, h, dtype=torch.float16, device=dev),
            torch.empty(slots, i, dtype=torch.float16, device=dev),
            torch.empty(slots, h, dtype=torch.float16, device=dev),
            torch.empty(e + 1, dtype=torch.int32, device=dev),
        )

    def apply(self, layer, dispatch_output):
        if not self.use_turbomind:
            raise RuntimeError("SM70 FP8 MoE was not prepared for this layer")
        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        tokens = x.shape[0]
        output, permuted, gate_up, sorted_output, offsets = self._workspace(layer, tokens)
        output.zero_()
        if tokens == 0:
            return StandardCombineInput(hidden_states=output)

        ids = topk.topk_ids
        weights = topk.topk_weights
        valid = ids >= 0
        if bool(valid.all()):
            flat_ids = ids.reshape(-1)
            token_index = torch.arange(tokens, device=ids.device).repeat_interleave(
                layer.sm70_top_k
            )
            flat_weights = weights.reshape(-1)
        else:
            # EP: keep only the routes this rank owns.
            token_index, slot_index = valid.nonzero(as_tuple=True)
            flat_ids = ids[token_index, slot_index]
            flat_weights = weights[token_index, slot_index]
        rows = flat_ids.numel()
        if rows == 0:
            return StandardCombineInput(hidden_states=output)

        order = torch.argsort(flat_ids, stable=True)
        token_index = token_index[order]
        flat_weights = flat_weights[order].to(torch.float16)
        sorted_ids = flat_ids[order]

        counts = torch.bincount(sorted_ids.long(), minlength=layer.sm70_num_experts)
        offsets[0] = 0
        offsets[1:] = counts.cumsum(0).to(torch.int32)

        sorted_x = permuted[:rows]
        torch.index_select(x, 0, token_index, out=sorted_x)
        gate = gate_up[:rows]
        y2 = sorted_output[:rows]
        fp8_apply(gate, sorted_x, offsets, layer.w13_quant, gated_silu=True)
        fp8_apply(y2, gate, offsets, layer.w2_quant, gated_silu=False)
        y2.mul_(flat_weights.unsqueeze(1))
        output.index_add_(0, token_index, y2)
        if self.output_scale is not None:
            output.mul_(self.output_scale)
        return StandardCombineInput(hidden_states=output)


_INSTALLED = False


def install() -> bool:
    """Route block-FP8 MoE layers to this method on SM70.

    ``SGLANG_SM70_FP8_MOE=off`` leaves the upstream method in place,
    ``auto`` (default) patches when the TurboMind library is present, and
    ``on`` additionally requires the binding to load.
    """
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
    if mode == "auto" and not _library_present():
        logger.warning("SM70 TurboMind library not found; keeping the upstream FP8 MoE method")
        return False
    if mode == "on" and not available():
        raise RuntimeError(f"{ENV_MODE}=on but the TurboMind binding cannot be loaded")

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod

    original = Fp8Config.get_quant_method
    Fp8Config._restomod_original_get_quant_method = original

    def get_quant_method(self, layer, prefix):
        method = original(self, layer, prefix)
        block = self.weight_block_size
        if (
            isinstance(layer, FusedMoE)
            and isinstance(method, Fp8MoEMethod)
            and block is not None
            and list(block) == [128, 128]
        ):
            return SM70Fp8MoEMethod(self)
        return method

    Fp8Config.get_quant_method = get_quant_method
    _INSTALLED = True
    logger.info("SM70 FP8 MoE method installed")
    return True


def uninstall() -> None:
    """Test helper: restore the upstream method selection."""
    global _INSTALLED
    if not _INSTALLED:
        return
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    original = getattr(Fp8Config, "_restomod_original_get_quant_method", None)
    if original is not None:
        Fp8Config.get_quant_method = original
        del Fp8Config._restomod_original_get_quant_method
    _INSTALLED = False


def register() -> None:
    """Component entry point: install the method selection for this process."""
    install()
