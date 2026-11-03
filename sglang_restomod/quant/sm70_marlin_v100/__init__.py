"""Dense WNA16 on SM70 through the marlin_v100 extension.

The extension is built outside this package from ``zhinianqin/marlin_v100``
(see ``PROVENANCE.md``). Its ``_C`` library exports four operators:

* ``gptq_marlin_repack`` -- pack a GPTQ checkpoint into the marlin layout;
* ``awq_marlin_repack`` -- the AWQ equivalent;
* ``marlin_gemm`` -- the dense WNA16 GEMM;
* ``marlin_int4_fp8_preprocess`` -- the int4/fp8 preprocessing pass.

SM70 needs the first and the third. ``SM70_MARLIN_DIR`` names the directory
that holds the built ``_C*.so``.

The capability gate in :mod:`sglang_restomod.overrides` reads
:func:`available`. :func:`install` then routes the engine's GPTQ call to this
kernel: the pinned engine JIT-builds its own marlin kernel, and that build does
not compile for sm70 (``namespace "sglang::device::marlin" has no member
"marlin_mm"``).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

ENV_DIR = "SM70_MARLIN_DIR"

_LOCK = threading.Lock()
_UNRESOLVED = object()
_OPS: object = _UNRESOLVED


def directory() -> Path:
    """Return the extension directory named by ``SM70_MARLIN_DIR``."""
    raw = os.environ.get(ENV_DIR)
    if not raw:
        raise RuntimeError(
            f"SM70 WNA16 needs the marlin_v100 extension: set {ENV_DIR} to the "
            "directory that holds _C*.so (see PROVENANCE.md)."
        )
    return Path(raw)


def _resolve() -> Optional[Tuple[object, object]]:
    """Load ``_C*.so`` once and return ``(repack, gemm)``, or ``None``."""
    global _OPS
    with _LOCK:
        if _OPS is not _UNRESOLVED:
            return _OPS  # type: ignore[return-value]
        _OPS = None
        try:
            root = directory()
        except RuntimeError:
            return None
        matches = sorted(root.glob("_C*.so"))
        if not matches:
            logger.warning("SM70 WNA16: no _C*.so under %s (%s)", root, ENV_DIR)
            return None
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            if torch.cuda.get_device_capability() != (7, 0):
                return None
            torch.ops.load_library(str(matches[0]))
            repack = getattr(torch.ops._C, "gptq_marlin_repack", None)
            gemm = getattr(torch.ops._C, "marlin_gemm", None)
            if repack is None or gemm is None:
                logger.warning(
                    "SM70 WNA16: %s does not export gptq_marlin_repack and marlin_gemm",
                    matches[0],
                )
                return None
        except Exception as exc:  # noqa: BLE001 - availability probe
            logger.warning("SM70 WNA16: %s could not be loaded (%s)", matches[0], exc)
            return None
        _OPS = (repack, gemm)
        logger.info("SM70 WNA16: loaded marlin_v100 from %s", matches[0].parent)
        return _OPS  # type: ignore[return-value]


def available() -> bool:
    """Whether marlin_v100 can serve a dense WNA16 GEMM on this device."""
    return _resolve() is not None


def ops() -> Tuple[object, object]:
    """Return ``(gptq_marlin_repack, marlin_gemm)``; raise when unavailable."""
    resolved = _resolve()
    if resolved is None:
        raise RuntimeError(
            "the SM70 WNA16 path needs a loadable marlin_v100 build: set "
            f"{ENV_DIR} to the directory that holds _C*.so (see PROVENANCE.md)"
        )
    return resolved


def gemm_args(
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
) -> tuple:
    """Map the engine's GPTQ call onto ``_C::marlin_gemm``.

    The extension takes a bias and an activation-scale slot the engine does not
    have, at positions 3 and 5, so the arguments cannot be passed through.
    """
    return (
        a,
        c,
        b_q_weight,
        None,
        b_scales,
        None,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )


def install() -> bool:
    """Route the engine's GPTQ marlin call to the marlin_v100 kernel.

    ``marlin_utils.apply_gptq_marlin_linear`` calls the module-global
    ``gptq_marlin_gemm``, which JIT-builds the upstream marlin kernel. That
    build has no sm70 instantiation, so the call is replaced here. The two
    signatures differ only in the extra bias/activation-scale slots of the
    extension, which the engine leaves empty.
    """
    if not available():
        logger.warning("SM70 WNA16: the marlin_v100 extension did not load; GPTQ stays on the JIT path")
        return False

    from sglang.srt.layers.quantization import marlin_utils

    original = getattr(marlin_utils, "gptq_marlin_gemm", None)
    if original is None:
        logger.warning("SM70 WNA16: marlin_utils has no gptq_marlin_gemm; nothing routed")
        return False
    if getattr(original, "_restomod_sm70", False):
        return False

    repack, gemm = ops()

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
                a, c, b_q_weight, b_scales, global_scale, b_zeros, g_idx, perm,
                workspace, b_q_type, size_m, size_n, size_k, is_k_full,
                use_atomic_add, use_fp32_reduce, is_zp_float,
            )
        )

    gptq_marlin_gemm._restomod_sm70 = True
    marlin_utils.gptq_marlin_gemm = gptq_marlin_gemm
    logger.info("SM70 WNA16: GPTQ marlin GEMM routed to marlin_v100")

    _install_repack(repack)
    _install_scales()
    return True


def _install_scales() -> bool:
    """Keep the WNA16 scales in their natural [K/group, N] order.

    ``marlin_permute_scales`` reorders them for the Ampere tile layout. The
    SM70 kernel reads them unpermuted: a GEMM fed the permuted tensor returns
    values that look plausible and are wrong, which is how this was found.
    """
    try:
        from sglang.srt.layers.quantization import marlin_utils
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_wNa16 as scheme,
        )
    except Exception as exc:  # noqa: BLE001 - absent module means nothing to patch
        logger.warning("SM70 WNA16: scale permute not routed (%s)", exc)
        return False

    original = getattr(scheme, "marlin_permute_scales", None)
    if original is None or getattr(original, "_restomod_sm70", False):
        return False

    def marlin_permute_scales(s, size_k, size_n, group_size):
        return s.reshape((-1, size_n)).contiguous()

    marlin_permute_scales._restomod_sm70 = True
    scheme.marlin_permute_scales = marlin_permute_scales
    marlin_utils.marlin_permute_scales = marlin_permute_scales
    logger.info("SM70 WNA16: scales kept in natural order for the SM70 kernel")
    return True


def _install_repack(repack) -> bool:
    """Route the weight repack to the extension as well.

    The engine packs the WNA16 weight with its own ``gptq_marlin_repack``,
    whose output layout is the one its GEMM reads. The marlin_v100 build packs
    for SM70 (``_pack_sm70_native_tile`` in its tests), so a weight packed by
    the engine's kernel and read by this one is garbage. Both sides have to
    come from the same build.
    """
    try:
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_wNa16 as scheme,
        )
    except Exception as exc:  # noqa: BLE001 - absent module means nothing to patch
        logger.warning("SM70 WNA16: the WNA16 scheme module is not importable (%s)", exc)
        return False

    original = getattr(scheme, "gptq_marlin_repack", None)
    if original is None:
        logger.warning("SM70 WNA16: the WNA16 scheme has no gptq_marlin_repack; repack not routed")
        return False
    if getattr(original, "_restomod_sm70", False):
        return False

    def gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits):
        return repack(b_q_weight, perm, size_k, size_n, num_bits, False)

    gptq_marlin_repack._restomod_sm70 = True
    scheme.gptq_marlin_repack = gptq_marlin_repack
    logger.info("SM70 WNA16: WNA16 repack routed to marlin_v100")
    return True


def register() -> None:
    """Component entry point."""
    install()
