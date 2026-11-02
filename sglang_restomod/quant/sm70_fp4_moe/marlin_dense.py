"""Dense W4A16 through marlin_v100 on SM70.

The pin's GPTQ/AWQ Marlin kernels are compiled empty below SM80: the repack
writes zeros and the GEMM computes nothing
(``jit/csrc/gemm/marlin/marlin_template.h`` and ``gptq_marlin_repack.cuh`` both
guard on ``__CUDA_ARCH__ < 800``). The ``marlin_v100`` extension this component
already loads registers real ones — ``_C.gptq_marlin_repack`` and
``_C.marlin_gemm`` — and the fork wires them into the dense path.

This module does the same by rebinding the two names the dense path resolves at
call time, so no engine file changes:

* ``sglang.kernels.ops.gemm.gptq_marlin.gptq_marlin_gemm``, which the marlin
  utils import by name;
* ``sglang.kernels.ops.quantization.gptq_marlin_repack.gptq_marlin_repack`` in
  every module that bound it.

Off SM70, or without the extension, both wrappers forward to the pinned
functions unchanged.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

GEMM_FLAG = "_restomod_marlin_dense_gemm"
REPACK_FLAG = "_restomod_marlin_dense_repack"

_STATE: dict = {"gemm": None, "repack": None, "probed": False}


def _ops():
    """Return ``(marlin_gemm, gptq_marlin_repack)`` or ``(None, None)``.

    Probed once: the extension load is a ``dlopen`` and the answer cannot change
    within a process.
    """
    if _STATE["probed"]:
        return _STATE["gemm"], _STATE["repack"]
    _STATE["probed"] = True

    import torch

    from .marlin import available

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
        return None, None
    if not available():
        return None, None

    gemm = getattr(torch.ops._C, "marlin_gemm", None)
    repack = getattr(torch.ops._C, "gptq_marlin_repack", None)
    if gemm is None or repack is None:
        logger.warning(
            "sglang-restomod: marlin_v100 loaded without a dense gemm/repack op; "
            "W4A16 stays on the pinned kernels"
        )
        return None, None

    _STATE["gemm"] = gemm
    _STATE["repack"] = repack
    return gemm, repack


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
    """The pinned signature, answered by the SM70 kernel when it is available."""
    gemm, _ = _ops()
    if gemm is None:
        return _ORIGINALS["gemm"](
            a, c, b_q_weight, b_scales, global_scale, b_zeros, g_idx, perm,
            workspace, b_q_type, size_m, size_n, size_k, is_k_full,
            use_atomic_add, use_fp32_reduce, is_zp_float,
        )
    return gemm(
        a, c, b_q_weight, None, b_scales, None, global_scale, b_zeros, g_idx,
        perm, workspace, b_q_type.id, size_m, size_n, size_k, is_k_full,
        use_atomic_add, use_fp32_reduce, is_zp_float,
    )


def gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits):
    """The pinned five-argument signature; marlin_v100 adds a trailing flag."""
    _, repack = _ops()
    if repack is None:
        return _ORIGINALS["repack"](b_q_weight, perm, size_k, size_n, num_bits)
    return repack(b_q_weight, perm, size_k, size_n, num_bits, False)


_ORIGINALS: dict = {}


def _rebind(original, replacement) -> int:
    """Replace every module attribute still bound to ``original``."""
    import sys

    rebound = 0
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            bound = getattr(module, original.__name__, None)
        except Exception:  # noqa: BLE001 - partial modules raise anything
            continue
        if bound is not original:
            continue
        try:
            setattr(module, original.__name__, replacement)
            rebound += 1
        except Exception:  # noqa: BLE001 - read-only module attribute
            continue
    return rebound


def install() -> bool:
    """Point the dense W4A16 path at marlin_v100 on SM70."""
    from sglang.kernels.ops.gemm import gptq_marlin as gemm_module
    from sglang.kernels.ops.quantization import gptq_marlin_repack as repack_module

    if getattr(gemm_module, GEMM_FLAG, False):
        return False

    original_gemm = gemm_module.gptq_marlin_gemm
    original_repack = repack_module.gptq_marlin_repack
    _ORIGINALS["gemm"] = original_gemm
    _ORIGINALS["repack"] = original_repack

    gemm_module.gptq_marlin_gemm = gptq_marlin_gemm
    repack_module.gptq_marlin_repack = gptq_marlin_repack
    gems = _rebind(original_gemm, gptq_marlin_gemm)
    reps = _rebind(original_repack, gptq_marlin_repack)
    setattr(gemm_module, GEMM_FLAG, True)
    setattr(repack_module, REPACK_FLAG, True)
    logger.info(
        "sglang-restomod: dense W4A16 routed to marlin_v100 on SM70 (%d+%d rebinds)",
        gems,
        reps,
    )
    return True
