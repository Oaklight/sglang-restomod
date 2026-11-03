"""Deliberate runtime overrides.

Rules:

* prefer SGLang's public registries over monkey-patching;
* keep each override in its own module and import it only when enabled;
* if an upstream file must change, add a patch under ``patches/`` and record it
  in ``patches/manifest.json`` instead of editing the installed package.

The one override here is a capability gate. It is a runtime hook rather than a
patch because the pinned engine is installed as a wheel in our current
deployment, and a wheel cannot be patched at install time. Issue #92 tracks
converting it to ``patches/0002`` once we build from source.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False


def _install_sm70_capability_gate() -> bool:
    """Permit compute capability 7.0 at model load.

    ``maybe_downgrade_dtype_for_legacy_gpu`` does two things: it forces float16
    below sm80 (exactly what V100 needs) and then raises below sm75. V100 is
    (7, 0), so the second check aborts every load.

    The replacement keeps the float16 fallback and lowers the floor to sm70.
    """
    from sglang.srt.model_executor import model_runner as runner_module
    from sglang.srt.model_executor.model_runner_components import load_model_utils

    original = getattr(runner_module, "maybe_downgrade_dtype_for_legacy_gpu", None)
    if original is None:
        logger.warning(
            "sglang-restomod: maybe_downgrade_dtype_for_legacy_gpu not found; "
            "the engine layout changed, SM70 load will fail"
        )
        return False
    if getattr(original, "_restomod_sm70", False):
        return False

    def maybe_downgrade_dtype_for_legacy_gpu(*, model_config) -> None:
        import torch

        from sglang.srt.runtime_context import get_context

        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            logger.info(
                "Compute capability below sm80. Use float16 due to lack of "
                "bfloat16 support."
            )
            get_context().override("ModelRunner._sm80_dtype_fallback", dtype="float16")
            model_config.dtype = torch.float16
            if (major, minor) < (7, 0):
                raise RuntimeError(
                    "sglang-restomod supports sm70 and above; this device is "
                    f"sm{major}{minor}."
                )

    maybe_downgrade_dtype_for_legacy_gpu._restomod_sm70 = True
    runner_module.maybe_downgrade_dtype_for_legacy_gpu = (
        maybe_downgrade_dtype_for_legacy_gpu
    )
    load_model_utils.maybe_downgrade_dtype_for_legacy_gpu = (
        maybe_downgrade_dtype_for_legacy_gpu
    )
    return True


def _install_fp8_min_capability() -> bool:
    """Let the ``fp8`` quantization config load on sm70.

    Upstream reports a floor of 80 for fp8 and refuses to build the config
    below it. Our fp8 path is a W8A16 GEMM, which runs on sm70, so the floor
    does not describe this layer.
    """
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    original = Fp8Config.get_min_capability
    if getattr(original, "_restomod_sm70", False):
        return False

    def get_min_capability(self) -> int:
        return 70

    get_min_capability._restomod_sm70 = True
    Fp8Config.get_min_capability = get_min_capability
    return True


def _sm70_marlin_v100_available() -> bool:
    """Whether the marlin_v100 extension can serve a dense WNA16 GEMM here."""
    from sglang_restomod.quant.sm70_marlin_v100 import available

    return available()


def _install_wna16_min_capability() -> bool:
    """Let a WNA16 (int4) quantization config load on sm70.

    ``CompressedTensorsWNA16.get_min_capability`` reports 80, so the config is
    refused before any weight is read. The V100 build of marlin_v100 exports
    ``gptq_marlin_repack`` and ``marlin_gemm``, which is what the WNA16 path
    needs, so the floor does not describe that build.

    This only admits the config: the dense GEMM still has to be wired into the
    linear method (issue #40). On a host without the extension, or on anything
    that is not an SM70 device, the upstream answer is returned unchanged.
    """
    try:
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_wNa16,
        )
    except Exception:  # noqa: BLE001 - absent module means nothing to patch
        return False

    config_cls = getattr(compressed_tensors_wNa16, "CompressedTensorsWNA16", None)
    if config_cls is None:
        return False
    original = config_cls.get_min_capability
    if getattr(original, "_restomod_sm70", False):
        return False

    def get_min_capability(cls) -> int:
        if _sm70_marlin_v100_available():
            return 70
        return original(cls)

    get_min_capability._restomod_sm70 = True
    config_cls.get_min_capability = classmethod(get_min_capability)
    return True


def _install_disable_small_copy() -> bool:
    """Disable the Triton small-copy fast path on sm70.

    ``try_small_copy`` fails to build its kernel with the pinned Triton on
    sm70: ``AttributeError: 'NoneType' object has no attribute 'type'`` during
    ``make_ir``. Returning False makes every caller fall back to the ordinary
    copy, which is what the registry used before that path existed.
    """
    from sglang.kernels.ops.memory import small_copy
    from sglang.srt.model_executor import cuda_graph_buffer_registry

    original = small_copy.try_small_copy
    if getattr(original, "_restomod_sm70", False):
        return False

    def try_small_copy(dsts, srcs):
        return False

    try_small_copy._restomod_sm70 = True
    small_copy.try_small_copy = try_small_copy
    if hasattr(cuda_graph_buffer_registry, "try_small_copy"):
        cuda_graph_buffer_registry.try_small_copy = try_small_copy
    return True


def _install_cutlass_shim() -> bool:
    """Provide ``FastDivmodDivisorV2`` for flashinfer's Blackwell modules.

    flashinfer 0.7.1 imports ``cutlass.cute.FastDivmodDivisorV2`` at module
    import time from ``gdn_kernels/blackwell/``. The installed cutlass dsl does
    not define that name. Those kernels are tcgen05/Blackwell only and never
    run on sm70, so aliasing the older class is enough to let the import finish.
    """
    try:
        import cutlass.cute as cute
    except Exception:  # noqa: BLE001 - absent cutlass is not our problem
        return False
    if hasattr(cute, "FastDivmodDivisorV2"):
        return False
    base = getattr(cute, "FastDivmodDivisor", None)
    if base is None:
        return False
    cute.FastDivmodDivisorV2 = base
    return True


def _install_force_fp16_cast() -> bool:
    """Cast bfloat16 weights to float16 after the model loads.

    The checkpoint declares bfloat16 for the layers it does not quantise, and the
    pin has no equivalent of the fork's ``SGLANG_SM70_FORCE_FP16``. The runner is
    fp16, so those layers stay bf16 and the first matmul fails
    (``qwen4_exp._prepare_attn_stage``: Half != BFloat16).

    Only bf16 is converted: float8, int8 and the packed quantised tensors must be
    left exactly as loaded.
    """
    from sglang.srt.model_executor.model_runner import ModelRunner

    original = ModelRunner.load_model
    if getattr(original, "_restomod_fp16", False):
        return False

    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        import torch

        if torch.cuda.get_device_capability()[0] >= 8:
            return result
        model = getattr(self, "model", None)
        if model is None:
            return result
        converted = 0
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor.dtype == torch.bfloat16:
                tensor.data = tensor.data.to(torch.float16)
                converted += 1
        if converted:
            logger.info("sglang-restomod: cast %d bf16 tensors to fp16", converted)
        return result

    load_model._restomod_fp16 = True
    ModelRunner.load_model = load_model
    return True


def _install_hyperconnection_fp16() -> bool:
    """Run the hyper-connection residual stream in float16 on sm70.

    ``GatedResidual`` casts every activation it touches with
    ``.to(self.params_dtype)``, and ``qwen4_exp`` builds both configs with a
    hard-coded ``params_dtype=torch.bfloat16``. On sm70 the runner is float16, so
    the residual stream turns bf16 and the next projection (fp16 weight) aborts
    with ``BFloat16 != Half`` in ``qwen3_5._forward_input_proj``.

    Rewriting the dtype on the config collects every such cast at once; the
    branch that requires bf16 (``hyperconnection.py:191``) then takes its
    generic path.
    """
    try:
        import msgspec
        from sglang.srt.layers import hyperconnection
    except Exception:  # noqa: BLE001 - absent module means nothing to patch
        return False

    original = hyperconnection.GatedResidual.__init__
    if getattr(original, "_restomod_fp16", False):
        return False

    def __init__(self, config, *args, **kwargs):
        import torch

        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] < 8
            and config.params_dtype == torch.bfloat16
        ):
            config = msgspec.structs.replace(config, params_dtype=torch.float16)
        original(self, config, *args, **kwargs)

    __init__._restomod_fp16 = True
    hyperconnection.GatedResidual.__init__ = __init__
    return True


def _install_linear_dtype_guard() -> bool:
    """Match an unquantised linear's input dtype to its weight on sm70.

    Mixed dtypes appear wherever the model hard-codes bf16: the PLE gather kernel
    emits bf16 rows, and multiplying them by an fp16 ``weight_scale`` promotes
    the result to fp32 before it reaches ``key_proj``. Casting at the single
    ``UnquantizedLinearMethod.apply`` seam covers every such site, including the
    ones this layer has not seen yet, and costs one comparison per call once the
    model is consistent.
    """
    try:
        from sglang.srt.layers.quantization import unquant
    except Exception:  # noqa: BLE001 - absent module means nothing to patch
        return False

    original = unquant.UnquantizedLinearMethod.apply
    if getattr(original, "_restomod_dtype_guard", False):
        return False

    def apply(self, layer, x, bias=None):
        weight = getattr(layer, "weight", None)
        if weight is not None and x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        return original(self, layer, x, bias)

    apply._restomod_dtype_guard = True
    unquant.UnquantizedLinearMethod.apply = apply
    return True


class _Fp16TorchView:
    """``torch`` with ``bfloat16`` replaced by ``float16``.

    ``QSATokenToKVPool.__init__`` validates the indexer cache dtype against a
    literal ``torch.bfloat16`` and then allocates its buffers from that same
    value. Swapping the module global for this view during the call moves the
    check and the allocation together, so they cannot disagree.
    """

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        import torch

        if name == "bfloat16":
            return torch.float16
        return getattr(self._real, name)


def _install_qsa_indexer_fp16() -> bool:
    """Store the QSA compressed indexer cache in float16 on sm70.

    ``resolve_qsa_indexer_dtype("auto")`` yields bfloat16, and the indexer JIT
    (``kernels/ops/attention/qsa_indexer.py:46-50``) only accepts an output dtype
    equal to the input's or float8_e4m3. The input follows the runner (float16),
    so bf16 output is rejected. float8_e4m3 needs SM90, so float16 is the only
    dtype that satisfies both.
    """
    try:
        import torch
        from sglang.srt.mem_cache import qsa_kv_pool
    except Exception:  # noqa: BLE001 - absent module means nothing to patch
        return False

    original_resolve = qsa_kv_pool.resolve_qsa_indexer_dtype
    if getattr(original_resolve, "_restomod_fp16", False):
        return False

    def resolve_qsa_indexer_dtype(name: str):
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            if name in ("auto", "bfloat16"):
                return torch.float16
        return original_resolve(name)

    resolve_qsa_indexer_dtype._restomod_fp16 = True
    qsa_kv_pool.resolve_qsa_indexer_dtype = resolve_qsa_indexer_dtype

    pool = qsa_kv_pool.QSATokenToKVPool
    # The pending ring is documented as bf16; on sm70 it must follow the runner.
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        pool.index_state_dtype = torch.float16

    original_init = pool.__init__

    def __init__(self, *args, **kwargs):
        real = qsa_kv_pool.torch
        qsa_kv_pool.torch = _Fp16TorchView(real)
        try:
            original_init(self, *args, **kwargs)
        finally:
            qsa_kv_pool.torch = real

    __init__._restomod_fp16 = True
    pool.__init__ = __init__
    return True


def _install_qsa_mqa_head_alignment() -> bool:
    """Pad the QSA decode head count to 16 on sm70.

    ``qsa/mqa.py:420`` picks a head alignment of 8 on CUDA and 16 on ROCm. The
    TileLang GEMM warp partition on sm70 rejects N=8 (``N must be divisible by
    16``), so the CUDA choice cannot compile here. The padding is a zero-filled
    head slice, and the kernel's head sum is weight-free, so 16 changes no
    arithmetic. ``is_hip`` is referenced only at that line in this module.
    """
    try:
        from sglang.srt.layers.attention.qsa import mqa
    except Exception:  # noqa: BLE001 - absent module means nothing to patch
        return False

    original = mqa.is_hip
    if getattr(original, "_restomod_sm70", False):
        return False

    def is_hip() -> bool:
        import torch

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            return True
        return original()

    is_hip._restomod_sm70 = True
    mqa.is_hip = is_hip
    return True


def _install_qsa_mqa_scoring_dtype() -> bool:
    """Score QSA QK in float16 on sm70.

    ``qsa/mqa.py:119 _scoring_dtype`` returns bfloat16 for every non-fp8 pair.
    The sm70 MMA template only accepts FP16 operands (``mma_sm70.h:349``:
    "SM70 only supports m16n16k4 with FP16 inputs"), so a bf16 kernel does not
    compile. float8_e4m3 is passed through untouched.

    ``_TILELANG_DTYPES`` has no float16 entry, so it is extended here; both
    kernels take that name as their operand dtype parameter.
    """
    try:
        import torch
        from sglang.srt.layers.attention.qsa import mqa
    except Exception:  # noqa: BLE001 - absent module means nothing to patch
        return False

    mqa._TILELANG_DTYPES.setdefault(torch.float16, "float16")
    original = mqa._scoring_dtype
    if getattr(original, "_restomod_sm70", False):
        return False

    def _scoring_dtype(q, k):
        dtype = original(q, k)
        if dtype == torch.bfloat16 and torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] < 8:
                return torch.float16
        return dtype

    _scoring_dtype._restomod_sm70 = True
    mqa._scoring_dtype = _scoring_dtype
    return True


def _install_profiler() -> bool:
    """Profile ``SGLANG_RESTOMOD_PROFILE`` forward passes and dump a kernel table.

    Host ``nsys`` is CUDA 11.x and cannot inject into the 12.8 container, and the
    image ships no profiler, so the only capture path is in-process. The window
    skips step 0 (graph warm-up) and writes one file per device.

    Env: ``SGLANG_RESTOMOD_PROFILE=<steps>``, ``SGLANG_RESTOMOD_PROFILE_OUT``
    (default ``/tmp/restomod_profile_<device>.txt``).
    """
    steps_text = os.environ.get("SGLANG_RESTOMOD_PROFILE")
    if not steps_text:
        return False
    try:
        steps = int(steps_text)
    except ValueError:
        logger.warning("sglang-restomod: bad SGLANG_RESTOMOD_PROFILE=%r", steps_text)
        return False
    if steps <= 0:
        return False

    import torch

    from sglang.srt.model_executor.model_runner import ModelRunner

    original = ModelRunner.forward
    if getattr(original, "_restomod_profile", False):
        return False

    state = {"seen": 0, "prof": None}

    def forward(self, *args, **kwargs):
        seen = state["seen"]
        if seen == 1:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            )
            prof.__enter__()
            state["prof"] = prof
        result = original(self, *args, **kwargs)
        state["seen"] = seen + 1
        if state["prof"] is not None and state["seen"] > steps + 1:
            prof = state["prof"]
            state["prof"] = None
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            device = torch.cuda.current_device()
            path = os.environ.get(
                "SGLANG_RESTOMOD_PROFILE_OUT", f"/tmp/restomod_profile_{device}.txt"
            )
            table = prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=40
            )
            with open(path, "w") as handle:
                handle.write(table)
            logger.info("sglang-restomod: profile written to %s", path)
        return result

    forward._restomod_profile = True
    ModelRunner.forward = forward
    return True


def _install_disable_hc_mix_persistent() -> bool:
    """Keep the hyper-connection mix off the persistent grid-barrier kernel.

    ``kernels/ops/gemm/hc_mix.py`` runs one CTA per SM (80 on this part) and
    synchronises them with an atomic spin barrier. Profiled on 8xV100 that kernel
    is 78.7% of all CUDA time: 1.135 ms per call, ~97 calls per decode step.

    Its fast alternative (``_jit_mix_ok``) is gated on compute capability 10, so
    it never applies here. Returning False from ``fused_hc_mix_supported`` sends
    the layer to ``GatedResidual._mix_compute``, five small torch kernels that are
    far cheaper at one row.

    ``hyperconnection`` binds the name at import time, so both modules are
    patched when the second one is already loaded.
    """
    try:
        from sglang.kernels.ops.gemm import hc_mix
    except Exception:  # noqa: BLE001 - absent module means nothing to patch
        return False

    original = hc_mix.fused_hc_mix_supported
    if getattr(original, "_restomod_sm70", False):
        return False

    def fused_hc_mix_supported(*args, **kwargs):
        import torch

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            return False
        return original(*args, **kwargs)

    fused_hc_mix_supported._restomod_sm70 = True
    hc_mix.fused_hc_mix_supported = fused_hc_mix_supported
    try:
        from sglang.srt.layers import hyperconnection

        hyperconnection.fused_hc_mix_supported = fused_hc_mix_supported
    except Exception:  # noqa: BLE001 - imported later, picks up the patched name
        pass
    return True


def _install_fp4_min_capability() -> bool:
    """Let the NVFP4 quantization config load on sm70.

    ``ModelOptFp4Config.get_min_capability`` reports 80. The scale format and the
    packing are load-time concerns, and the resident footprint of an NVFP4
    checkpoint is what makes a 4-card, single-island deployment possible, so the
    floor has to come off before anything else can be measured.

    This only admits the config; the expert GEMM still has to exist for sm70.
    """
    try:
        from sglang.srt.layers.quantization import modelopt_quant
    except Exception:  # noqa: BLE001 - absent module means nothing to patch
        return False

    config_cls = getattr(modelopt_quant, "ModelOptFp4Config", None)
    if config_cls is None:
        return False
    original = config_cls.get_min_capability
    if getattr(original, "_restomod_sm70", False):
        return False

    def get_min_capability(cls) -> int:
        return 70

    get_min_capability._restomod_sm70 = True
    config_cls.get_min_capability = classmethod(get_min_capability)
    return True


def install() -> None:
    """Install selected overrides. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    changed = _install_sm70_capability_gate()
    changed = _install_fp8_min_capability() or changed
    changed = _install_disable_small_copy() or changed
    changed = _install_cutlass_shim() or changed
    changed = _install_force_fp16_cast() or changed
    changed = _install_hyperconnection_fp16() or changed
    changed = _install_linear_dtype_guard() or changed
    changed = _install_qsa_indexer_fp16() or changed
    changed = _install_qsa_mqa_head_alignment() or changed
    changed = _install_qsa_mqa_scoring_dtype() or changed
    changed = _install_profiler() or changed
    changed = _install_disable_hc_mix_persistent() or changed
    changed = _install_fp4_min_capability() or changed
    changed = _install_wna16_min_capability() or changed
    _INSTALLED = True
    if changed:
        logger.info("sglang-restomod: capability floor lowered to sm70")
