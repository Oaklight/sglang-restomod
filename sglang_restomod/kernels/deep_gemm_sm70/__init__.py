"""A stand-in for DeepGEMM on SM70, installed only when nothing else defines it.

DeepGEMM is compiled for the architectures its kernels target, and Volta is not
one of them. Two things therefore have to be true before this module does
anything: the device must be SM70, and a real ``deep_gemm`` must not resolve.
Neither is overridable by an environment variable — ``register()`` checks both
and returns, so a platform or an image that ships the real package keeps it.

What it installs is deliberately incomplete. The pin reaches DeepGEMM through
about thirty names, and a substitute that answered all of them with a best
effort would turn a missing kernel into a wrong number. So every name this
module does not genuinely implement is a stub that raises ``NotImplementedError``
naming the entry and the shapes it was called with. A loud failure at the call
site is the whole safety property.

Names that upstream probes with ``hasattr`` for capability selection are a
separate case and stay **absent** rather than stubbed: ``fp8_fp4_mqa_logits``
gates the FP4 dense indexer, and ``fp8_mega_moe`` / ``mega_moe_pre_dispatch_sm90``
gate the mega-MoE path. Defining them as stubs would switch those paths on and
land in the stub later; leaving them out keeps the selection decision the one
the pin already makes.

Implemented here (semantics from the pin's own fallbacks):
``bf16_gemm_nt``, ``fp8_paged_mqa_logits``, ``fp8_mqa_logits``,
``tf32_hc_prenorm_gemm``, and the configuration surface
(``set_pdl``, ``set_num_sms``/``get_num_sms``,
``set_compile_mode``/``get_compile_mode``,
``get_mk_alignment_for_contiguous_layout``).

Everything else raises. Three of those refusals are permanent, not pending:
``fp8_gemm_nt``, the grouped pair and ``fp8_fp4_gemm_nt``. A kernel-level entry
does exist under each of them — ``torch.ops.sglang_sm70_turbomind.fp8_gemm``
for the dense case, ``torch.ops.sgl_kernel.moe_wna16_marlin`` and
``torch.ops._C.marlin_gemm`` for the grouped ones — but those take pre-packed
W8A16 weights with per-layer leading dimensions, or routing metadata, while
DeepGEMM's signature is ``(A fp8 + scale, B fp8 + scale, recipe)``. Forwarding
would mean a re-pack plus a scale-format conversion *inside* the entry: a
second implementation of the quantisation, which is how two paths drift apart.
They stay refusals unless someone accepts that duplication deliberately.

The entry that does earn its place is ``tf32_hc_prenorm_gemm``. It has two arms,
selected by ``SGLANG_SM70_HC_PRENORM``: ``torch`` (default) is the
``hc_pre_torch_impl`` math, and ``fused`` calls a kernel registered with
``install_fused_hc_prenorm()``. The fused arm raises when nothing is registered
rather than falling back, so an A/B cannot silently measure the same arm twice.
That switch is how the engine reaches a fused prenorm through a DeepGEMM-shaped
entry; its acceptance gate belongs to the kernel's own design document.

The two logits entries route to the pin's torch twins first. Measured on
2026-11-16, ``fp8_paged_mqa_logits_torch`` costs 0.287 ms per call and is flat
from 2048 to 8192 pages (overhead-bound, 0.9-3.7 GB/s), so a kernel is expected
to replace that body later — with torch kept as the reference arm, because top-k
selection tolerance has to be judged against it, not merely dropped. Two
caveats on reusing those numbers: they were taken **without CUDA graphs**, while
serving decode is graph-captured, so a per-call overhead figure does not
translate into a per-step one; and the metadata argument is ``None`` today and
ignored by the torch twin, which is why both entries accept and ignore it rather
than requiring it.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from typing import Any, Optional

logger = logging.getLogger(__name__)

SUBSTITUTE_FLAG = "__restomod_substitute__"

# DeepGEMM requires the contiguous (m-grouped) layout's M to be a multiple of
# this; the pin reads the value to build its layouts, so the number must stay
# the real one even though the grouped GEMM below is unimplemented.
MK_ALIGNMENT = 128

_num_sms: Optional[int] = None
_compile_mode: int = 0
_pdl: bool = False


def _real_deep_gemm_installed() -> bool:
    """True when a real ``deep_gemm`` resolves or was already imported."""
    if sys.modules.get("deep_gemm") is not None:
        return True
    try:
        return importlib.util.find_spec("deep_gemm") is not None
    except (ImportError, ValueError):  # a broken parent package still counts
        return True


def _device_sm() -> Optional[int]:
    """The current device's SM number, or None when it cannot be read."""
    try:
        from sglang.srt.utils import get_device_sm

        return get_device_sm()
    except Exception:
        return None


def _describe(value: Any) -> str:
    if isinstance(value, tuple):
        return "(" + ", ".join(_describe(v) for v in value) + ")"
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return f"{tuple(shape)}:{dtype}"
    return type(value).__name__


def _called_with(args: tuple, kwargs: dict) -> str:
    parts = [_describe(a) for a in args]
    parts += [f"{k}={_describe(v)}" for k, v in kwargs.items()]
    return ", ".join(parts) if parts else "no arguments"


def _unimplemented(name: str, reason: str):
    def stub(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"deep_gemm_sm70: '{name}' is not implemented on SM70. {reason} "
            f"Called with: {_called_with(args, kwargs)}"
        )

    stub.__name__ = name
    stub.__doc__ = reason
    return stub


# --------------------------------------------------------------------------
# Implemented entries
# --------------------------------------------------------------------------


def bf16_gemm_nt(lhs, rhs, out) -> None:
    """``out = lhs @ rhs.T`` through cuBLAS. Shapes: lhs [m, k], rhs [n, k]."""
    import torch

    if lhs.shape[-1] != rhs.shape[-1]:
        raise ValueError(
            f"bf16_gemm_nt: NT product needs lhs[..., k] and rhs[..., k], got "
            f"{tuple(lhs.shape)} and {tuple(rhs.shape)}"
        )
    torch.mm(lhs.to(out.dtype), rhs.t(), out=out)


HC_PRENORM_ENV = "SGLANG_SM70_HC_PRENORM"

_fused_hc_prenorm = None


def install_fused_hc_prenorm(op) -> None:
    """Register the fused prenorm kernel for the ``fused`` arm of the switch.

    Called by whichever component ships the kernel, at its ``register()``. The
    signature it must accept is the entry's own:
    ``op(x, fn, out, sqrsum, num_splits=...)``.
    """
    global _fused_hc_prenorm
    _fused_hc_prenorm = op


def _hc_prenorm_reference(x, fn, out, sqrsum) -> None:
    """The math of the pin's ``hc_pre_torch_impl`` (``deepseek_v4.py:3358``).

    ``out = x @ fn.T`` in fp32 and ``sqrsum = sum(x**2, -1)``; the caller turns
    ``sqrsum / k + eps`` into the rsqrt, which is why the sum, not the mean, is
    written. Volta has no TF32, so this is fp32 cuBLAS — the same reduction the
    torch fallback performs, not the tf32 kernel's rounding.
    """
    import torch

    if x.shape[0] == 0:
        return
    xf = x.float()
    torch.mm(xf, fn.float().t(), out=out)
    torch.sum(xf * xf, dim=-1, out=sqrsum)


def tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_splits=None) -> None:
    """Two arms, chosen by ``SGLANG_SM70_HC_PRENORM``.

    ``torch`` (the default) runs the reference above; ``fused`` calls the kernel
    registered through :func:`install_fused_hc_prenorm`. A ``fused`` request with
    no kernel registered raises: a silent fallback would make an A/B measure the
    same arm twice and report it as a speedup.
    """
    import os

    arm = os.environ.get(HC_PRENORM_ENV, "torch").strip().lower()
    if arm in ("", "torch", "reference"):
        _hc_prenorm_reference(x, fn, out, sqrsum)
        return
    if arm in ("fused", "kernel"):
        if _fused_hc_prenorm is None:
            raise NotImplementedError(
                f"deep_gemm_sm70: {HC_PRENORM_ENV}=fused, but no fused prenorm kernel "
                "is registered. The component that ships it must call "
                "sglang_restomod.kernels.deep_gemm_sm70.install_fused_hc_prenorm(op) "
                "from its register()."
            )
        _fused_hc_prenorm(x, fn, out, sqrsum, num_splits=num_splits)
        return
    raise ValueError(
        f"deep_gemm_sm70: {HC_PRENORM_ENV}={arm!r}; expected 'torch' or 'fused'"
    )


def fp8_paged_mqa_logits(
    q_fp8, kvcache_fp8, weight, seq_lens, page_table, metadata, max_seq_len, clean_logits=False
):
    """Forward to the pin's torch paged-logits twin (``dsv4/indexer.py:82``)."""
    from sglang.srt.layers.attention.dsv4.indexer import fp8_paged_mqa_logits_torch

    return fp8_paged_mqa_logits_torch(
        q_fp8, kvcache_fp8, weight, seq_lens, page_table, metadata, max_seq_len, clean_logits
    )


def fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits=False, max_seqlen_k=None):
    """Non-paged MQA logits in torch, row by row.

    ``kv`` is ``(k_fp8 [tokens, 128], scale [tokens] fp32)``. Row ``i`` scores the
    contiguous range ``[ks[i], ke[i])`` into output columns ``0 .. ke-ks-1``; the
    remaining columns are ``-inf`` when ``clean_logits`` and zero otherwise, which
    is the distinction the kernel's ``clean_logits`` draws. Slow by construction:
    it exists so the nonpaged indexer can be exercised, not so it can be served.
    """
    import torch

    k, k_scale = kv
    rows = q.shape[0]
    width = int(max_seqlen_k) if max_seqlen_k is not None else int((ke - ks).max())
    fill = torch.finfo(torch.float32).min if clean_logits else 0.0
    out = torch.full((rows, width), fill, dtype=torch.float32, device=q.device)
    for i in range(rows):
        lo, hi = int(ks[i]), int(ke[i])
        if hi <= lo:
            continue
        qf = q[i].to(torch.float32)  # [heads, 128]
        kf = k[lo:hi].to(torch.float32)  # [n, 128]
        scores = torch.relu(kf @ qf.t()) * weights[i].to(torch.float32).unsqueeze(0)
        out[i, : hi - lo] = scores.sum(dim=1) * k_scale[lo:hi].to(torch.float32)
    return out


def set_num_sms(num_sms: int) -> None:
    """DeepGEMM sizes its grids by SM count. This substitute has no such grids;
    the value is kept so ``get_num_sms`` round-trips as callers expect."""
    global _num_sms
    _num_sms = int(num_sms)


def get_num_sms() -> int:
    global _num_sms
    if _num_sms is None:
        try:
            import torch

            _num_sms = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        except Exception:
            _num_sms = 0
    return _num_sms


def set_compile_mode(mode: int) -> None:
    global _compile_mode
    _compile_mode = int(mode)


def get_compile_mode() -> int:
    return _compile_mode


def set_pdl(enabled: bool) -> None:
    """Volta has no programmatic dependent launch; accepted and ignored."""
    global _pdl
    _pdl = bool(enabled)


def get_mk_alignment_for_contiguous_layout() -> int:
    return MK_ALIGNMENT


# --------------------------------------------------------------------------
# Stubs: present in the real package, not implemented here
# --------------------------------------------------------------------------

_NO_DENSE_GEMM = (
    "The SM70 components expose layer-level entry points, not a raw "
    "GEMM: see sglang_restomod.quant.sm70_fp8_dense.apply_sm70_turbomind_fp8_linear. "
    "Wiring that into this signature is the first kernel decision, not a wrapper."
)
_NO_GROUPED_GEMM = (
    "The SM70 grouped path is sglang_restomod.quant.sm70_fp4_moe.moe_wna16_marlin_gemm, "
    "which takes routing metadata (sorted_token_ids, expert_ids, topk_weights) "
    "rather than DeepGEMM's lhs/rhs/m_indices, so it cannot be forwarded positionally."
)
_NO_FP4_GEMM = (
    "The SM70 FP4 expert kernel is reached through fused_marlin_moe "
    "(sglang_restomod.quant.sm70_fp4_moe), not as a standalone NT GEMM."
)
_NO_MEGA = "This is a Blackwell mega-MoE entry; there is no SM70 implementation or fallback."
_NO_LAYOUT = "Layout/helper entry; not needed by any SM70 path that this substitute answers."

_STUBS = {
    "fp8_gemm_nt": _NO_DENSE_GEMM,
    "fp8_fp4_gemm_nt": _NO_FP4_GEMM,
    "fp8_m_grouped_gemm_nt_masked": _NO_GROUPED_GEMM,
    "m_grouped_fp8_gemm_nt_contiguous": _NO_GROUPED_GEMM,
    "m_grouped_bf16_gemm_nt_masked": _NO_GROUPED_GEMM,
    "m_grouped_bf16_gemm_nt_contiguous": _NO_GROUPED_GEMM,
    "bf16_gemm_nn": _NO_DENSE_GEMM,
    "fp8_einsum": _NO_MEGA,
    "mega_moe_pre_dispatch": _NO_MEGA,
    "get_block_m_for_mega_moe": _NO_MEGA,
    "transform_weights_for_mega_moe": _NO_MEGA,
    "get_symm_buffer_for_mega_moe": _NO_MEGA,
    "fp8_fp4_mega_moe": _NO_MEGA,
    "nvfp4_mega_moe": _NO_MEGA,
    "bf16_mega_moe": _NO_MEGA,
    "mega_mhc": _NO_MEGA,
    "bf16_mega_gate": _NO_MEGA,
    "fp8_fp4_paged_sparse_mqa_logits": _NO_MEGA,
    "get_paged_mqa_logits_metadata": _NO_LAYOUT,
    "get_paged_sparse_mqa_logits_metadata": _NO_LAYOUT,
    "transform_sf_into_required_layout": _NO_LAYOUT,
    "per_block_cast_to_fp8": _NO_LAYOUT,
    "ceil_to_ue8m0": _NO_LAYOUT,
    "use_deterministic_algorithms": _NO_LAYOUT,
    "SymmBuffer": _NO_MEGA,
}

_IMPLEMENTED = {
    "bf16_gemm_nt": bf16_gemm_nt,
    "tf32_hc_prenorm_gemm": tf32_hc_prenorm_gemm,
    "fp8_paged_mqa_logits": fp8_paged_mqa_logits,
    "fp8_mqa_logits": fp8_mqa_logits,
    "set_num_sms": set_num_sms,
    "get_num_sms": get_num_sms,
    "set_compile_mode": set_compile_mode,
    "get_compile_mode": get_compile_mode,
    "set_pdl": set_pdl,
    "get_mk_alignment_for_contiguous_layout": get_mk_alignment_for_contiguous_layout,
}


_UTILS_ENTRIES = {
    "deep_gemm.utils.layout": {
        "get_mn_major_tma_aligned_packed_ue8m0_tensor": _NO_LAYOUT,
    },
    "deep_gemm.utils.math": {
        "transform_ue4m3_sf_into_required_layout": _NO_LAYOUT,
    },
}


def _submodule(name: str, entries: dict) -> types.ModuleType:
    """A submodule whose known helpers are call-time stubs and nothing else.

    ``__getattr__`` must raise ``AttributeError``, not ``NotImplementedError``:
    the import system and half the ecosystem probe for ``__file__``,
    ``__path__`` and friends, and an exception of another type escapes the
    ``except ImportError`` guards that probe this package (torch's import chain
    is one). Loudness belongs at the call, which is where ``entries`` land.
    """
    mod = types.ModuleType(name)
    mod.__doc__ = f"{name}: part of the SM70 deep_gemm stand-in; no helper is implemented."
    for entry, reason in entries.items():
        mod.__dict__[entry] = _unimplemented(f"{name}.{entry}", reason)

    def __getattr__(attr: str, _name=name):
        raise AttributeError(
            f"deep_gemm_sm70: '{_name}.{attr}' is absent; the known helpers here are "
            "stubs that raise NotImplementedError when called"
        )

    mod.__getattr__ = __getattr__  # type: ignore[attr-defined]
    return mod


def build() -> types.ModuleType:
    """The module object installed as ``deep_gemm``."""
    module = types.ModuleType("deep_gemm")
    module.__doc__ = "SM70 stand-in installed by sglang_restomod.kernels.deep_gemm_sm70."
    setattr(module, SUBSTITUTE_FLAG, True)
    for name, fn in _IMPLEMENTED.items():
        setattr(module, name, fn)
    for name, reason in _STUBS.items():
        setattr(module, name, _unimplemented(name, reason))

    utils = _submodule("deep_gemm.utils", {})
    utils.layout = _submodule(
        "deep_gemm.utils.layout", _UTILS_ENTRIES["deep_gemm.utils.layout"]
    )
    utils.math = _submodule(
        "deep_gemm.utils.math", _UTILS_ENTRIES["deep_gemm.utils.math"]
    )
    module.utils = utils

    def __getattr__(attr: str):
        raise AttributeError(
            f"deep_gemm_sm70: no attribute '{attr}'. Names that are stubs raise "
            "NotImplementedError at the call; names that are absent are absent on "
            "purpose, because the pin's hasattr probes gate paths this substitute "
            "cannot serve."
        )

    module.__getattr__ = __getattr__  # type: ignore[attr-defined]
    return module


def register() -> None:
    """Install the substitute when, and only when, nothing else provides one."""
    if _real_deep_gemm_installed():
        logger.info(
            "sglang-restomod: a real deep_gemm resolves; the SM70 stand-in was not installed"
        )
        return
    sm = _device_sm()
    if sm != 70:
        logger.info("sglang-restomod: device sm %s is not 70; no deep_gemm stand-in", sm)
        return
    module = build()
    sys.modules["deep_gemm"] = module
    sys.modules["deep_gemm.utils"] = module.utils
    sys.modules["deep_gemm.utils.layout"] = module.utils.layout
    sys.modules["deep_gemm.utils.math"] = module.utils.math
    logger.info(
        "sglang-restomod: installed the SM70 deep_gemm stand-in "
        "(%d entries implemented, %d stubs raising)",
        len(_IMPLEMENTED), len(_STUBS),
    )
