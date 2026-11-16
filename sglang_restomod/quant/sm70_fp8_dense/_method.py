"""TurboMind W8A16 block-FP8 linear kernels for NVIDIA SM70."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)


def _resolve_sm70_paths():
    """Return the engine's SM70 path helpers, or ``(None, None)``.

    ``sglang.kernels.sm70_paths`` is a fork module; the pinned upstream has no
    such file. When it is absent the TurboMind library has to be named through
    ``SM70_TURBOMIND_SO`` and the QPN8 source through
    ``SGLANG_SM70_QPN8_SRC``.
    """
    try:
        from sglang.kernels.sm70_paths import sm70_csrc, sm70_prebuilt
    except ImportError:
        return None, None
    return sm70_csrc, sm70_prebuilt


_sm70_csrc, _sm70_prebuilt = _resolve_sm70_paths()

def _discover_default_ops_path() -> str | None:
    """Locate the TurboMind library without requiring a fork module.

    The fork resolves it through ``sglang.kernels.sm70_paths``, which the pinned
    upstream does not ship. Discovery in :mod:`sglang_restomod._sm70_paths`
    covers the environment and the standard directories, so a stock engine
    finds the library without anyone exporting a variable.
    """
    from sglang_restomod._sm70_paths import find_prebuilt

    found = find_prebuilt(
        "_sm70_turbomind_v100.so",
        env_names=("SM70_TURBOMIND_SO", "SGLANG_SM70_TURBOMIND_SO"),
    )
    return str(found) if found is not None else None


_DEFAULT_OPS_PATH = _discover_default_ops_path()
_OPS_LOAD_ATTEMPTED = False
_OPS_AVAILABLE = False

_PREFILL_BACKENDS = ("auto", "turbomind", "fp16")

# The packed E4M3 kernel spans complete K quantization groups and writes N in
# 32-column HMMA-884 tiles. A partial K group can wedge split-K completion and
# a partial N tile corrupts the packing, so both shapes are refused before
# packing or dispatch: the kernel's own failure is a stall or a wrong result.
_GROUP_SIZE = 128
_N_PACK = 32


def _check_dense_weight_shape(
    n: int, k: int, group_size: int = _GROUP_SIZE
) -> None:
    """Reject a block-FP8 weight the packed E4M3 kernel cannot represent."""
    if n <= 0 or k <= 0:
        raise RuntimeError(
            f"SM70 TurboMind FP8 needs positive N and K, got N={n} K={k}."
        )
    if n % _N_PACK:
        raise RuntimeError(
            f"SM70 TurboMind FP8 needs N a multiple of {_N_PACK}, got N={n}."
        )
    if k % group_size:
        raise RuntimeError(
            f"SM70 TurboMind FP8 needs K a multiple of the {group_size}-wide "
            f"quantization group, got K={k}."
        )


def _check_activation_k(layer: torch.nn.Module, x_2d: torch.Tensor) -> None:
    """Reject an activation whose K differs from the prepared weight's K."""
    expected = getattr(layer, "sm70_fp8_in_features", None)
    if expected is not None and x_2d.shape[-1] != expected:
        raise RuntimeError(
            "SM70 TurboMind FP8: activation K disagrees with the prepared "
            f"weight (prepared K={expected}, got {x_2d.shape[-1]})."
        )


# --- SM70 QPN8 decode backend (W8A16, M<=8, default on) ----------------
# Adapted from v100-skinny's QPN8 kernel for our block-wise [N/128][K/128]
# fp32 scale checkpoint format. Packed codes + per-K-block scales are built
# once at weight load; decode GEMMs (M<=8) route to the QPN8 kernel instead
# of the TurboMind fp8_gemm when enabled. The exact TP4 gate/up shape
# additionally fuses its paired projections with the SiLU multiply at decode.
# The fork resolves the source through `sglang.kernels.sm70_paths`; the pinned
# upstream has no such module. The packaged copy is the default so a stock
# engine works without an export; `SGLANG_SM70_QPN8_SRC` remains the override.
_PACKAGED_QPN8_SRC = Path(__file__).with_name("csrc") / "sm70_fp8_qpn8_decode.cu"
_QPN8_SRC_PATH = (
    _sm70_csrc("sm70_fp8_qpn8_decode.cu")
    if _sm70_csrc
    else (
        os.environ.get("SGLANG_SM70_QPN8_SRC")
        or (str(_PACKAGED_QPN8_SRC) if _PACKAGED_QPN8_SRC.is_file() else None)
    )
)
_QPN8_EXT = None
_QPN8_MISSING_LOGGED = False
_QPN8_EXACT_DENSE_SHAPES = {
    "down_proj": (5120, 4352),
    "gate_up_proj": (8704, 5120),
    "in_proj_qkvz": (4096, 5120),
    "out_proj": (5120, 1536),
    "o_proj": (5120, 1536),
}
_QPN8_SPLITK = 16
_QPN8_NACC = 1
_QPN8_FASTDEC_NACC_ARG = 3  # NACC=1 plus the extension's word-parallel decoder
_QPN8_FASTDEC_SUFFIXES = {"down_proj", "gate_up_proj", "in_proj_qkvz"}
_KORDER8 = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]

# Qwen3.8-27B-FP8 TP4 shapes admitted by 1Cat's real-weight, bitwise gates.
# The TurboMind tensors have already been converted to logical KxN here.
_SM70_FP8_PREFILL_DENSE_MIN_M = 3920
_SM70_FP8_PREFILL_DENSE_SHAPES = {
    "gate_up_proj": (5120, 8704),
    "down_proj": (4352, 5120),
    "out_proj": (1536, 5120),
    "o_proj": (1536, 5120),
}
# Qwen3.8-Flash-Next TP4 shapes, per rank, as (K, N). Off by default: the
# 27B table above is the one validated against real weights and bitwise gates,
# and these have only been checked against the checkpoint and the launch, not
# against an output comparison. Enable with
# SGLANG_SM70_FP8_PREFILL_EXACT_DENSE_SHAPES=flashnext.
#
# Every shape is 128-aligned on both sides, which both the block-128 scale
# layout and the dispatch require, and each is at most 4096 x 2560 in the
# logical KxN form, so the 85 MiB workspace sized by the 27B entry still holds.
_SM70_FP8_PREFILL_DENSE_SHAPES_FLASHNEXT = {
    # 12 full-attention layers, QKV fused by the loader. The first derivation
    # said 3328 from the config's q/k/v head counts; the checkpoint's fused
    # tensor is 3584 wide per rank, which is what the candidate log measured.
    "qkv_proj": (2560, 3584),
    # both attention families' output projections: K = 6144 / 4 = 1536
    "o_proj": (1536, 2560),
    "out_proj": (1536, 2560),
    # 36 linear-attention layers, in_proj_qkv + in_proj_z fused per layer
    "in_proj_qkvz": (2560, 4096),
    "ple.key_proj": (2560, 2560),
    "ple.value_proj": (2560, 2560),
}
ENV_EXACT_DENSE_SHAPES = "SGLANG_SM70_FP8_PREFILL_EXACT_DENSE_SHAPES"

_SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS = max(
    k * n for k, n in _SM70_FP8_PREFILL_DENSE_SHAPES.values()
)
_SM70_FP8_PREFILL_DENSE_WORKSPACE_BYTES = (
    _SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS * torch.float16.itemsize
)
# Layers deliberately retain only a CUDA address. This strong cache owns the
# allocation and keeps the 85 MiB buffer out of torch.compile/CUDA-graph inputs.
_SM70_FP8_PREFILL_DENSE_WORKSPACES: dict[tuple[int, torch.dtype], torch.Tensor] = {}
_SM70_FP8_PREFILL_DENSE_OOM_WARNED = False


def _get_sm70_fp8_prefill_backend() -> str:
    backend = os.environ.get("SGLANG_SM70_FP8_PREFILL_BACKEND", "auto").lower()
    if backend not in _PREFILL_BACKENDS:
        raise ValueError(
            "SGLANG_SM70_FP8_PREFILL_BACKEND must be auto, turbomind, or "
            f"fp16; got {backend!r}."
        )
    return backend


def _get_sm70_fp8_prefill_min_tokens() -> int:
    value = int(
        os.environ.get(
            "SGLANG_SM70_FP8_PREFILL_MIN_TOKENS",
            str(_SM70_FP8_PREFILL_DENSE_MIN_M),
        )
    )
    if value <= 0:
        raise ValueError(
            f"SGLANG_SM70_FP8_PREFILL_MIN_TOKENS must be positive; got {value}."
        )
    return value


def _env_flag(name: str, default: str) -> bool:
    value = os.environ.get(name, default).strip().lower()
    if value not in ("0", "false", "off", "no", "1", "true", "on", "yes"):
        raise ValueError(f"{name} must be a boolean value, got {value!r}.")
    return value in ("1", "true", "on", "yes")


def _get_sm70_fp8_decode_qpn8_enabled() -> bool:
    # Default on for SM70 FP8 decode; set SGLANG_SM70_FP8_DECODE_QPN8=0 to
    # fall back to the TurboMind decode GEMM path.
    return _env_flag("SGLANG_SM70_FP8_DECODE_QPN8", "1")


def _get_sm70_fp8_qpn8_fastdec_enabled() -> bool:
    return _env_flag("SGLANG_SM70_FP8_QPN8_FASTDEC", "1")


def _get_sm70_fp8_qpn8_fused_gate_enabled() -> bool:
    return _env_flag("SGLANG_SM70_FP8_QPN8_FUSED_GATE", "1")


def _load_sm70_qpn8_ops():
    """Lazy-load the standalone SM70 QPN8 decode extension (JIT-built)."""
    global _QPN8_EXT
    if _QPN8_EXT is not None:
        return _QPN8_EXT
    if not _QPN8_SRC_PATH.is_file():
        raise RuntimeError(f"SM70 QPN8 decode source not found: {_QPN8_SRC_PATH}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("SM70 QPN8 decode requires an NVIDIA SM70 GPU.")
    from torch.utils.cpp_extension import load_inline

    build_directory = os.environ.get(
        "SGLANG_SM70_QPN8_BUILD_DIR", "/tmp/sglang_qpn8_build"
    )
    os.makedirs(build_directory, exist_ok=True)
    _QPN8_EXT = load_inline(
        name="sglang_sm70_qpn8_v100",
        cpp_sources="",
        cuda_sources=_QPN8_SRC_PATH.read_text(),
        functions=None,
        is_python_module=True,
        verbose=False,
        build_directory=build_directory,
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
    )
    logger.info_once(
        "SM70 (V100): QPN8 decode extension loaded from %s.",
        _QPN8_SRC_PATH,
    )
    return _QPN8_EXT


def _qpn8_prepack(w8: torch.Tensor) -> torch.Tensor:
    """(N,K) uint8 -> [tile N/32][group K/16][lane 32][16B] contiguous codes."""
    n, k = w8.shape
    dev = w8.device
    tiles, groups = n // 32, k // 16
    lane = torch.arange(32, device=dev)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    korder = torch.tensor(_KORDER8, device=dev)
    g = torch.arange(groups, device=dev)
    kidx = g.view(groups, 1) * 16 + korder.view(1, 16)
    ncol = torch.arange(tiles, device=dev).view(tiles, 1) * 32 + col.view(1, 32)
    out = w8[
        ncol.view(tiles, 1, 32, 1).expand(tiles, groups, 32, 16),
        kidx.view(1, groups, 1, 16).expand(tiles, groups, 32, 16),
    ]
    return out.contiguous().view(-1)


def _qpn8_gscales(scale_inv: torch.Tensor, k: int) -> torch.Tensor:
    """[N/128][K/128] fp32 -> half[K/128][N/32], decoder's *256 folded."""
    nb, _ = scale_inv.shape
    gs = scale_inv.t().reshape(k // 128, nb) * 256.0
    gs = gs.repeat_interleave(4, dim=1).to(torch.float16).contiguous()
    return gs


def _is_sm70_fp8_qpn8_exact_dense_layer(layer: torch.nn.Module) -> bool:
    if getattr(layer, "tp_size", 1) != 4:
        return False
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    expected = _QPN8_EXACT_DENSE_SHAPES.get(suffix)
    return expected is not None and tuple(layer.weight.shape) == expected


def _maybe_prepare_sm70_fp8_qpn8(layer: torch.nn.Module) -> None:
    if not _get_sm70_fp8_decode_qpn8_enabled():
        return
    # The decoder is JIT-built from a .cu that this distribution does not ship
    # and that the pinned upstream has no path helper for. Without it the gate
    # below would admit a layer and the build would raise at startup.
    global _QPN8_MISSING_LOGGED
    if _QPN8_SRC_PATH is None or not Path(_QPN8_SRC_PATH).is_file():
        if not _QPN8_MISSING_LOGGED:
            _QPN8_MISSING_LOGGED = True
            logger.warning(
                "sm70_fp8_dense: the QPN8 decode source is unavailable "
                "(SGLANG_SM70_QPN8_SRC), decode GEMMs stay on TurboMind"
            )
        return
    if not _is_sm70_fp8_qpn8_exact_dense_layer(layer):
        return
    weight = layer.weight.data
    scales = layer.weight_scale_inv.data.to(torch.float32).contiguous()
    n, k = weight.shape
    if n % 128 or k % 128:
        return
    _load_sm70_qpn8_ops()
    raw = weight.view(torch.uint8).contiguous()  # already [N,K]
    layer.sm70_fp8_qpn8_codes = _qpn8_prepack(raw)
    layer.sm70_fp8_qpn8_gscales = _qpn8_gscales(scales, k)
    layer.sm70_fp8_qpn8_shape = (k, n)
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    layer.sm70_fp8_qpn8_gated = suffix == "gate_up_proj"
    layer.sm70_fp8_qpn8_fused_gate = (
        layer.sm70_fp8_qpn8_gated and _get_sm70_fp8_qpn8_fused_gate_enabled()
    )
    layer.sm70_fp8_qpn8_nacc_arg = (
        _QPN8_FASTDEC_NACC_ARG if suffix in _QPN8_FASTDEC_SUFFIXES else _QPN8_NACC
    )
    if not _get_sm70_fp8_qpn8_fastdec_enabled():
        layer.sm70_fp8_qpn8_nacc_arg = _QPN8_NACC
    layer.sm70_fp8_qpn8_ready = True
    logger.info_once(
        "SM70 (V100): QPN8 decode backend prepared for %s K=%d N=%d.",
        getattr(layer, "prefix", "?"),
        k,
        n,
    )


def _sm70_fp8_layer_tp_size(layer: torch.nn.Module) -> tuple[int | None, str]:
    """Return ``(tp_size, source)`` for ``layer``.

    ``LinearBase`` in the pinned engine carries no ``tp_size`` attribute: every
    occurrence in ``sglang/srt/layers/linear.py`` is a local from
    ``get_group_rank_size``, and nothing in ``sglang/srt`` assigns one to a
    linear module. The gate below used ``getattr(layer, "tp_size", 1)``, so on
    every site and every model it evaluated ``1 != 4`` and refused every layer.
    Both exact-dense tables admitted nothing, silently, because the only log
    line was the per-layer one this feature adds.

    The process-wide value the engine does expose is ``get_parallel().tp_size``.
    The source is returned with it so a log line can say which accessor
    answered; ``None`` means both were unavailable and the caller should fall
    back to the shape tables, which are TP4-specific already.
    """
    value = getattr(layer, "tp_size", None)
    if value is not None:
        return int(value), "layer.tp_size"
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().tp_size), "get_parallel().tp_size"
    except Exception:  # noqa: BLE001 - a probe, not control flow
        return None, "unavailable"


def _is_sm70_fp8_prefill_exact_dense_layer(layer: torch.nn.Module) -> bool:
    tp_size, _source = _sm70_fp8_layer_tp_size(layer)
    if tp_size is not None and tp_size != 4:
        return False
    shape = tuple(layer.weight.shape)
    # Modules converted by ``fp8_dense_quant`` keep no ``prefix``, so the suffix
    # is empty and a name-matched gate can never admit them. The dispatched
    # kernel is bit-identical to the TurboMind GEMM on every admitted shape, so
    # a shape match cannot change the result, only the kernel that computes it.
    # This fallback is confined to the env-gated Flash-Next table: the default
    # 27B path keeps its name-and-shape admission exactly as it was.
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    expected = _SM70_FP8_PREFILL_DENSE_SHAPES.get(suffix)
    if expected is not None and shape == expected:
        return True
    if os.environ.get(ENV_EXACT_DENSE_SHAPES, "").strip().lower() != "flashnext":
        return False
    extra = _SM70_FP8_PREFILL_DENSE_SHAPES_FLASHNEXT.get(suffix) if suffix else None
    # The TurboMind tensor is already in logical KxN here, the same layout the
    # 27B table above is written in.
    if extra is not None and shape == extra:
        return all(dim % 128 == 0 for dim in extra)
    admitted_shapes = set(_SM70_FP8_PREFILL_DENSE_SHAPES_FLASHNEXT.values())
    return shape in admitted_shapes and all(dim % 128 == 0 for dim in shape)


def _get_sm70_fp8_prefill_exact_dense_workspace(
    weight: torch.Tensor,
) -> Optional[torch.Tensor]:
    global _SM70_FP8_PREFILL_DENSE_OOM_WARNED
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, torch.float16)
    workspace = _SM70_FP8_PREFILL_DENSE_WORKSPACES.get(key)
    if workspace is not None:
        return workspace
    try:
        workspace = torch.empty(
            _SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS,
            dtype=torch.float16,
            device=weight.device,
        )
    except torch.OutOfMemoryError:
        if not _SM70_FP8_PREFILL_DENSE_OOM_WARNED:
            logger.warning(
                "Insufficient memory for the bounded SM70 FP8 prefill "
                "workspace; falling back to TurboMind W8A16."
            )
            _SM70_FP8_PREFILL_DENSE_OOM_WARNED = True
        return None
    _SM70_FP8_PREFILL_DENSE_WORKSPACES[key] = workspace
    return workspace


def _use_sm70_fp8_prefill_dispatch(layer: torch.nn.Module) -> bool:
    return getattr(
        layer, "sm70_fp8_prefill_exact_dense_workspace_ptr", 0
    ) != 0 and hasattr(torch.ops.sglang_sm70_turbomind, "fp8_prefill_dispatch")


def _resolve_ops_path() -> str | None:
    """Name the TurboMind library, or ``None`` when nothing names it.

    The dense kernel path predates the MoE one and used its own variable, but
    the site launchers only export the MoE names. A host that serves the MoE
    experts and leaves ``SGLANG_SM70_TURBOMIND_OPS_PATH`` alone therefore
    disabled the dense kernel without saying so. Accept all three names:
    the explicit dense variable first, then the fork's own resolver, then the
    MoE aliases.
    """
    explicit = os.environ.get("SGLANG_SM70_TURBOMIND_OPS_PATH")
    if explicit:
        return explicit
    if _DEFAULT_OPS_PATH:
        return _DEFAULT_OPS_PATH
    for name in ("SM70_TURBOMIND_SO", "SGLANG_SM70_TURBOMIND_SO"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _load_sm70_turbomind_fp8_ops() -> bool:
    global _OPS_AVAILABLE, _OPS_LOAD_ATTEMPTED
    if _OPS_LOAD_ATTEMPTED:
        return _OPS_AVAILABLE
    _OPS_LOAD_ATTEMPTED = True

    ops_path = _resolve_ops_path()
    if not ops_path:
        # Nothing names the library. That is a provisioning gap, not a load
        # failure: report it the same way a failed load is reported, so every
        # caller can keep its fallback.
        logger.warning(
            "SM70 TurboMind FP8 operators unavailable: no library path "
            "(set SGLANG_SM70_TURBOMIND_OPS_PATH, SM70_TURBOMIND_SO or "
            "SGLANG_SM70_TURBOMIND_SO)"
        )
        return _OPS_AVAILABLE
    try:
        torch.ops.load_library(str(ops_path))
        _OPS_AVAILABLE = hasattr(
            torch.ops.sglang_sm70_turbomind, "fp8_prepare"
        ) and hasattr(torch.ops.sglang_sm70_turbomind, "fp8_gemm")
        if not _OPS_AVAILABLE:
            raise RuntimeError("the required fp8_prepare/fp8_gemm operators are absent")
    except Exception as exc:
        logger.warning("SM70 TurboMind FP8 operators unavailable: %s", exc)
        _OPS_AVAILABLE = False
    return _OPS_AVAILABLE


def can_use_sm70_turbomind_fp8(
    weight_block_size: Optional[list[int]],
) -> bool:
    """Return whether the local SM70 W8A16 kernel covers this quantization."""
    backend = os.environ.get("SGLANG_SM70_FP8_BACKEND", "auto").lower()
    if backend not in ("auto", "turbomind", "marlin"):
        raise ValueError(
            "SGLANG_SM70_FP8_BACKEND must be auto, turbomind, or marlin; "
            f"got {backend!r}."
        )
    if backend == "marlin":
        return False
    if (
        weight_block_size != [128, 128]
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (7, 0)
    ):
        if backend == "turbomind":
            raise RuntimeError(
                "The SM70 TurboMind FP8 backend requires an NVIDIA SM70 GPU "
                "and block-wise FP8 weights with block_size=[128, 128]."
            )
        return False

    available = _load_sm70_turbomind_fp8_ops()
    if backend == "turbomind" and not available:
        raise RuntimeError(
            "SGLANG_SM70_FP8_BACKEND=turbomind was requested, but the "
            "TurboMind FP8 extension could not be loaded."
        )
    return available


def prepare_sm70_turbomind_fp8_linear(layer: torch.nn.Module) -> None:
    if layer.orig_dtype != torch.float16:
        raise RuntimeError(
            "SM70 TurboMind FP8 requires FP16 activations, "
            f"but the layer dtype is {layer.orig_dtype}."
        )
    if getattr(layer, "weight_block_size", None) != [128, 128]:
        raise RuntimeError("SM70 TurboMind FP8 requires weight_block_size=[128, 128].")

    stored = layer.weight
    if stored.dim() != 2:
        raise RuntimeError(
            "SM70 TurboMind FP8 requires a 2-D weight, got shape "
            f"{tuple(stored.shape)}."
        )
    n, k = (int(dim) for dim in stored.shape)
    _check_dense_weight_shape(n, k)
    expected_scales = (
        (n + _GROUP_SIZE - 1) // _GROUP_SIZE,
        (k + _GROUP_SIZE - 1) // _GROUP_SIZE,
    )
    stored_scales = tuple(layer.weight_scale_inv.shape)
    if stored_scales != expected_scales:
        raise RuntimeError(
            "SM70 TurboMind FP8 requires scales of "
            f"[ceil(N/{_GROUP_SIZE}), ceil(K/{_GROUP_SIZE})] = "
            f"{expected_scales}, got {stored_scales}."
        )

    # QPN8 decode codes must be captured from the raw checkpoint weight before
    # the TurboMind repack replaces it.
    _maybe_prepare_sm70_fp8_qpn8(layer)

    weight = layer.weight.data
    scales = layer.weight_scale_inv.data.to(torch.float32).contiguous()
    logical_widths = getattr(layer, "logical_widths", None)
    is_gated_silu = (
        getattr(layer, "prefix", "").rsplit(".", 1)[-1] == "gate_up_proj"
        and isinstance(logical_widths, list)
        and len(logical_widths) == 2
        and logical_widths[0] == logical_widths[1]
    )
    tm_weight, tm_scales, meta = torch.ops.sglang_sm70_turbomind.fp8_prepare(
        weight,
        scales,
        128,
        is_gated_silu,
    )
    layer.weight = Parameter(tm_weight, requires_grad=False)
    layer.weight_scale_inv = Parameter(tm_scales, requires_grad=False)
    layer.sm70_fp8_k_ld = int(meta[0].item())
    layer.sm70_fp8_q_ld = int(meta[1].item())
    layer.sm70_fp8_turbomind = True
    layer.sm70_fp8_gated_silu = is_gated_silu
    layer.sm70_fp8_in_features = k
    prefill_backend = _get_sm70_fp8_prefill_backend()
    exact_dense_enabled = _env_flag("SGLANG_SM70_FP8_PREFILL_EXACT_DENSE", "1")
    if (
        prefill_backend != "turbomind"
        and exact_dense_enabled
        and hasattr(torch.ops.sglang_sm70_turbomind, "fp8_prefill_dispatch")
        and _is_sm70_fp8_prefill_exact_dense_layer(layer)
    ):
        workspace = _get_sm70_fp8_prefill_exact_dense_workspace(layer.weight)
        if workspace is not None:
            layer.sm70_fp8_prefill_exact_dense_workspace_ptr = workspace.data_ptr()
            layer.sm70_fp8_prefill_min_tokens = _get_sm70_fp8_prefill_min_tokens()
            # The gate is silent by construction: a suffix or shape that misses
            # admits nothing and says nothing, and the arm then measures the
            # baseline again. One line per admitted layer makes the count
            # greppable from the launch log.
            logger.info(
                "sm70_fp8_dense: exact-dense prefill layer admitted: %s %s",
                getattr(layer, "prefix", "<no-prefix>"),
                tuple(layer.weight.shape),
            )
    if os.environ.get(ENV_EXACT_DENSE_SHAPES, "").strip().lower() == "flashnext":
        # A zero admitted count must name its cause. Without this line "the
        # shapes are absent", "the suffix is empty" and "the tp guard refused"
        # all look the same in the log -- and on 2026-11-15 the third was the
        # true one for 408 of 416 candidates on Flash-Next.
        tp_size, tp_source = _sm70_fp8_layer_tp_size(layer)
        logger.info(
            "sm70_fp8_dense: candidate prefix=%s shape=%s tp=%s (%s) admitted=%s",
            getattr(layer, "prefix", "<no-prefix>"),
            tuple(layer.weight.shape),
            tp_size,
            tp_source,
            _is_sm70_fp8_prefill_exact_dense_layer(layer),
        )
    logger.info_once("SM70 (V100): using TurboMind W8A16 block-FP8 dense GEMM.")
    if is_gated_silu:
        logger.info_once(
            "SM70 (V100): using the TurboMind fused gate/up SiLU epilogue."
        )
    if _use_sm70_fp8_prefill_dispatch(layer):
        logger.info_once(
            "SM70 (V100): using the graph-safe shared 85 MiB exact-dense "
            "FP8 prefill workspace for admitted TP4 projections at M >= %d.",
            layer.sm70_fp8_prefill_min_tokens,
        )


def apply_sm70_turbomind_fp8_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    _check_activation_k(layer, x_2d)
    k, n = (
        layer.sm70_fp8_qpn8_shape
        if getattr(layer, "sm70_fp8_qpn8_ready", False)
        else (None, layer.output_size_per_partition)
    )
    m = x_2d.shape[0]
    if (
        getattr(layer, "sm70_fp8_qpn8_ready", False)
        and x_2d.dtype == torch.float16
        and 0 < m <= 8
        and x_2d.shape[-1] == k
    ):
        out_2d = _load_sm70_qpn8_ops().qpn8_linear(
            x_2d,
            layer.sm70_fp8_qpn8_codes,
            layer.sm70_fp8_qpn8_gscales,
            n,
            k,
            _QPN8_SPLITK,
            layer.sm70_fp8_qpn8_nacc_arg,
        )
    else:
        out_2d = torch.empty(
            (m, layer.output_size_per_partition),
            dtype=x.dtype,
            device=x.device,
        )
        if _use_sm70_fp8_prefill_dispatch(layer) and x_2d.dtype == torch.float16:
            torch.ops.sglang_sm70_turbomind.fp8_prefill_dispatch(
                out_2d,
                layer.sm70_fp8_prefill_exact_dense_workspace_ptr,
                x_2d,
                layer.weight,
                layer.weight_scale_inv,
                128,
                layer.sm70_fp8_k_ld,
                layer.sm70_fp8_q_ld,
                False,
                layer.sm70_fp8_prefill_min_tokens,
            )
        else:
            torch.ops.sglang_sm70_turbomind.fp8_gemm(
                out_2d,
                x_2d,
                layer.weight,
                layer.weight_scale_inv,
                128,
                layer.sm70_fp8_k_ld,
                layer.sm70_fp8_q_ld,
                False,
            )
    if getattr(layer, "sm70_fp8_gated_silu", False):
        # The packed weight is interleaved for the fused epilogue. Preserve the
        # ordinary linear contract when a caller does not request fusion.
        out_features = layer.output_size_per_partition // 2
        out_2d = (
            out_2d.reshape(x_2d.shape[0], out_features, 2)
            .transpose(1, 2)
            .reshape(x_2d.shape[0], layer.output_size_per_partition)
        )
    if bias is not None:
        out_2d.add_(bias)
    return out_2d.reshape(*x.shape[:-1], layer.output_size_per_partition)


def apply_sm70_turbomind_fp8_fused_silu_and_mul(
    layer: torch.nn.Module,
    x: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Apply a packed gate/up projection and SiLU multiplication in one GEMM."""
    if not getattr(layer, "sm70_fp8_gated_silu", False):
        return None

    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    _check_activation_k(layer, x_2d)
    out_features = layer.output_size_per_partition // 2
    k, n = (
        layer.sm70_fp8_qpn8_shape
        if getattr(layer, "sm70_fp8_qpn8_gated", False)
        else (None, None)
    )
    m = x_2d.shape[0]
    if (
        getattr(layer, "sm70_fp8_qpn8_gated", False)
        and x_2d.dtype == torch.float16
        and 0 < m <= 8
        and x_2d.shape[-1] == k
    ):
        if getattr(layer, "sm70_fp8_qpn8_fused_gate", False):
            fused = _load_sm70_qpn8_ops().qpn8_gated_silu(
                x_2d,
                layer.sm70_fp8_qpn8_codes,
                layer.sm70_fp8_qpn8_gscales,
                out_features,
                k,
            )
            return fused.reshape(*x.shape[:-1], out_features)
        gate_up = _load_sm70_qpn8_ops().qpn8_linear(
            x_2d,
            layer.sm70_fp8_qpn8_codes,
            layer.sm70_fp8_qpn8_gscales,
            n,
            k,
            _QPN8_SPLITK,
            layer.sm70_fp8_qpn8_nacc_arg,
        )
        gate = gate_up[:, :out_features]
        up = gate_up[:, out_features:]
        fused = torch.nn.functional.silu(gate) * up
        return fused.reshape(*x.shape[:-1], out_features)
    out_2d = torch.empty(
        (x_2d.shape[0], out_features),
        dtype=x.dtype,
        device=x.device,
    )
    if _use_sm70_fp8_prefill_dispatch(layer) and x_2d.dtype == torch.float16:
        torch.ops.sglang_sm70_turbomind.fp8_prefill_dispatch(
            out_2d,
            layer.sm70_fp8_prefill_exact_dense_workspace_ptr,
            x_2d,
            layer.weight,
            layer.weight_scale_inv,
            128,
            layer.sm70_fp8_k_ld,
            layer.sm70_fp8_q_ld,
            True,
            layer.sm70_fp8_prefill_min_tokens,
        )
    else:
        torch.ops.sglang_sm70_turbomind.fp8_gemm(
            out_2d,
            x_2d,
            layer.weight,
            layer.weight_scale_inv,
            128,
            layer.sm70_fp8_k_ld,
            layer.sm70_fp8_q_ld,
            True,
        )
    return out_2d.reshape(*x.shape[:-1], out_features)
